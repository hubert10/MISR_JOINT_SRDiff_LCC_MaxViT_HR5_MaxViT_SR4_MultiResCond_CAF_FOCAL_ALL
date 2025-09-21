import torch
import torch.nn.functional as F
import torchvision.transforms as T
from functools import partial
from torch import nn
from tqdm import tqdm
import numpy as np
from .module_util import default
from utils.sr_utils import SSIM, PerceptualLoss
from utils.hparams import hparams
from losses.srdiff_loss import (
    pixel_wise_closest_sr_sits_aer_loss,
    grad_pixel_wise_closest_sr_sits_aer_loss,
    temp_gradient_magnitude_consistency_loss,
    gray_value_consistency_loss,
    cross_entropy_loss,
)
from losses.focal_smooth import FocalLossWithSmoothing


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(
        beta_start, beta_end, warmup_time, dtype=np.float64
    )
    return betas


def get_beta_schedule(
    num_diffusion_timesteps, beta_schedule="linear", beta_start=0.0001, beta_end=0.02
):
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


# gaussian diffusion trainer class
class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, cond_net, timesteps=1000, loss_type="l1"):
        super().__init__()
        self.denoise_fn = denoise_fn
        # condition network, which is either RRDBLTAENet or HighResLTAENet
        # for satelite image time series integration
        self.cond_net = cond_net
        self.criterion_sat = FocalLossWithSmoothing(
            hparams["num_classes"], gamma=2, alpha=1, lb_smooth=0.2
        )
        self.ssim_loss = SSIM(window_size=11)
        if hparams["aux_percep_loss"]:
            self.percep_loss_fn = [PerceptualLoss()]
        if hparams["beta_schedule"] == "cosine":
            betas = cosine_beta_schedule(timesteps, s=hparams["beta_s"])
        if hparams["beta_schedule"] == "linear":
            betas = get_beta_schedule(timesteps, beta_end=hparams["beta_end"])
            if hparams["res"]:
                betas[-1] = 0.999

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )
        self.sample_tqdm = True

    # Calculates the mean and variance of the noised sample x_t
    #  at timestep t in the forward diffusion process
    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # Estimates the orginal clean image x_0 from a noised sample x_t at timestep t
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # Computes the mean and variance of the posterior distribution in the reverse
    # (denoising) process of a diffusion model; which is used in the denoising step
    # to predict a less noisy image from x_t at timestep t
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # Computes the mean and variance of the reverse (denoising) step
    # in a diffusion model
    def p_mean_variance(self, x, t, noise_pred, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    # Add gradient difference loss function to support l1-loss
    def closest_lr_sits_aer(self, img_lr_up, closest_indices):
        B, T, C, H, W = img_lr_up.shape
        closest_indices = closest_indices.to(torch.long)

        # Ensure closest_indices is a tensor
        if not torch.is_tensor(closest_indices):
            closest_indices = torch.tensor(closest_indices, device=img_lr_up.device)

        # Gather closest satellite images
        closest_sat_image = img_lr_up[
            torch.arange(B), closest_indices
        ]  # shape (B, C, H, W)
        return closest_sat_image

    # TRAINING STEP

    def forward(
        self,
        img_hr,  # High-resolution (HR) image
        img_lr,  # Low-resolution (LR) input
        img_lr_up,  # Upsampled LR image
        labels_sr,  # Ground truth labels for classification
        t=None,  # Diffusion timestep
        dates=None,  # Additional temporal information (for LTAE)
        closest_idx=None,  # Closest index for temporal features
        alphas=None,  # Auxiliary parameters (for HighResNet)
        *args,
        **kwargs
    ):
        x = img_hr
        b, *_, device = *x.shape, x.device

        # Randomly select a timestep:
        # if t is None(Not given), then randomly choose a timestep from 0 to num_timesteps
        # t is a tensor of shape batch size tensor([1, 3, 4, 1, 0]) for 5 images in a batch size

        t = (
            torch.randint(0, self.num_timesteps, (b,), device=device).long()
            if t is None
            else torch.LongTensor([t]).repeat(b).to(device)
        )

        # cond: extracted temporal features from the low-resolution image
        # encoded with HighResnet-LTAE temporal encoder
        self.cond_net.eval()
        # Prevents PyTorch from tracking gradients, reducing memory
        # usage and speeding up inference.
        with torch.no_grad():
            _, cond = self.cond_net.backbone(img_lr)
            cond_net_out, _, _ = self.cond_net(img_lr)

        # cond: shape (B, T, C, H, W) where T is the number of time steps
        # img_lr_up: shape (B, T, C, H, W) where T is the number of time steps

        p_losses, x_tp1, noise_pred, x_t, x_t_gt, x_0 = self.p_losses(
            x, t, cond, img_lr, img_lr_up, closest_idx, dates, *args, **kwargs
        )
        
        # p_losses: main loss computed based on the predicted noise and the ground truth noise
        ret = p_losses

        # If the cond_net parameters are not froozen, apply auxiliary losses
        # to the output of the cond_net (RRDB) to refine the features

        # The conditioning features should be cropped from the
        # center and upsampled to match the resolution of the
        # HR ground truth before loss computations

        h, w = labels_sr.shape[-2:]
        transform = T.CenterCrop((10, 10))
        cond_net_out = transform(cond_net_out)
        cond_net_out = F.interpolate(cond_net_out, size=(h, w), mode="bilinear")

        if not hparams["fix_cond_net_parms"]:
            ret["ace_loss"] = self.criterion_sat(cond_net_out, labels_sr)

        # return ret: loss dict (noise diffusion loss + auxiliary losses)
        # predicted SR image (x_tp1) at  currenttimestep t
        # intermediate noisy versions (x_t_gt, x_t) at current timestep t
        # x_0: predicted denoised image (without noise) at the current timestep
        return ret, (x_tp1, x_t_gt, x_t), t, x_0

    def p_losses(
        self, img_hr, t, cond, img_lr, img_lr_up, closest_idx, dates, noise=None
    ):
        # The auxiliary loss function is introduced here because
        # this is where the denoising process is performed and the
        # model already reconstructs denoised image x_0 from the
        # predicted noise

        # 1. Compute the residual image (x_start) and the closest LR upsampled image
        # Here we are using the same residual image for all time steps
        # Alternatively, we can compute the residual image for each time step

        x_res = [
            self.img2res(img_hr, img_lr_up[:, i]) for i in range(img_lr_up.shape[1])
        ]
        x_start = torch.stack(x_res, dim=1)

        # Select the closest residual image to generate the noise
        # shared across all timesteps
        x_start_res = self.closest_lr_sits_aer(x_start, closest_idx)

        # 2. Compute the noise (if not provided)
        # Make sure noise has the same shape as x_start and is shared across time
        noise = default(noise, lambda: torch.randn_like(x_start_res))

        x_tp1_gt_seq = []
        x_t_gt_seq = []
        x_t_pred_seq = []
        x0_pred_seq = []
        noise_pred_seq = []

        # 3. Iterate over the time steps and compute the denoised
        #  image (x0_pred) and the predicted noise for each time step

        for tps in range(img_lr_up.shape[1]):
            x_tp1_gt = self.q_sample(x_start=x_start[:, tps, :, :], t=t, noise=noise)
            x_t_gt = self.q_sample(x_start=x_start[:, tps, :, :], t=t - 1, noise=noise)
            noise_pred = self.denoise_fn(x_tp1_gt, t, [feat[:, tps] for feat in cond])
            x_t_pred, x0_pred = self.p_sample(
                x_tp1_gt,
                t,
                [feat[:, tps] for feat in cond],
                img_lr_up[:, tps, :, :],
                noise_pred=noise_pred,
            )

            # the predicted residual x0_pred is the denoised image
            # and should be converted back to the orginal image for loss
            # computation x_t_pred is the predicted noisy image at timestep t

            x0_pred_seq.append(self.res2img(x0_pred, img_lr_up[:, tps, :, :]))
            x_t_pred_seq.append(self.res2img(x_t_pred, img_lr_up[:, tps, :, :]))
            x_tp1_gt_seq.append(self.res2img(x_tp1_gt, img_lr_up[:, tps, :, :]))
            x_t_gt_seq.append(self.res2img(x_t_gt, img_lr_up[:, tps, :, :]))
            noise_pred_seq.append(noise_pred)

        # Stack the outputs of the temporal dimensions
        x0_pred = torch.stack(x0_pred_seq, 1)  # torch.Size([4, 6, 64, 10, 10])
        x_t_pred = torch.stack(x_t_pred_seq, 1)  # torch.Size([4, 6, 64, 10, 10])
        x_tp1_gt = torch.stack(x_tp1_gt_seq, 1)  # torch.Size([4, 6, 64, 10, 10])
        x_t_gt = torch.stack(x_t_gt_seq, 1)  # torch.Size([4, 6, 64, 10, 10])
        noise_pred = torch.stack(noise_pred_seq, 1)  # torch.Size([4, 6, 64, 10, 10])

        # 4. Compute the loss
        # Compute the loss based on the predicted noise and the ground truth noise

        # The loss is computed based on the difference between the predicted noise
        # and the actual noise added to the original image

        # If the closest index is provided, use it to select the closest
        # residual image for the predicted noise
        noise_pred = self.closest_lr_sits_aer(noise_pred, closest_idx)

        if self.loss_type == "l1":
            loss = (noise - noise_pred).abs().mean()
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise, noise_pred)
        elif self.loss_type == "ssim":
            loss = (noise - noise_pred).abs().mean()
            loss = loss + (1 - self.ssim_loss(noise, noise_pred))
        elif self.loss_type == "l1_shift":
            loss = self.shift_l1_loss(noise, noise_pred)
        else:
            raise NotImplementedError()

        # 5. Compute the auxiliary loss
        if not hparams["use_aux_loss"]:
            return loss, x_tp1_gt, noise_pred, x_t_pred, x_t_gt, x0_pred
        else:
            aux_losses = {
                "pxl_loss": hparams["px_loss_weight"]
                * pixel_wise_closest_sr_sits_aer_loss(x0_pred, img_hr, closest_idx),
                "grd_loss": hparams["grad_px_loss_weight"]
                * grad_pixel_wise_closest_sr_sits_aer_loss(
                    x0_pred, img_hr, closest_idx
                ),
                "tgm_loss": hparams["temp_grad_mag_loss_weight"]
                * temp_gradient_magnitude_consistency_loss(x0_pred),
                "gvc_loss": hparams["gray_value_px_loss_weight"]
                * gray_value_consistency_loss(x0_pred, img_lr),
                "nzp_loss": hparams["main_loss_weight"] * loss,
            }

            # aux_loss = (
            #     hparams["px_loss_weight"]
            #     * pixel_wise_closest_sr_sits_aer_loss(x0_pred, img_hr, closest_idx)
            #     + hparams["grad_px_loss_weight"]
            #     * grad_pixel_wise_closest_sr_sits_aer_loss(x0_pred, img_hr, closest_idx)
            #     + hparams["temp_grad_mag_loss_weight"]
            #     * temp_gradient_magnitude_consistency_loss(x0_pred)
            #     + hparams["gray_value_px_loss_weight"]
            #     * gray_value_consistency_loss(x0_pred, img_lr)
            # )
            # final_loss = hparams["main_loss_weight"] * loss + aux_loss
        return aux_losses, x_tp1_gt, noise_pred, x_t_pred, x_t_gt, x0_pred

    def shift_l1_loss(self, y_true, y_pred, border=3):
        """
        Modified l1 loss to take into account pixel shifts
        """
        max_pixels_shifts = 2 * border
        size_image = y_true.shape[-1]
        patch_pred = y_pred[
            ..., border : size_image - border, border : size_image - border
        ]

        X = []
        for i in range(max_pixels_shifts + 1):
            for j in range(max_pixels_shifts + 1):
                patch_true = y_true[
                    ...,
                    i : i + (size_image - max_pixels_shifts),
                    j : j + (size_image - max_pixels_shifts),
                ]
                l1_loss = (patch_true - patch_pred).abs().mean()
                X.append(l1_loss)
        X = torch.stack(X)
        min_l1 = X.min(dim=0).values
        return min_l1

    # Simulates the forward diffusion process by adding noise
    # to the original clean x_0 according to a predefined noise schedule
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        t_cond = (t[:, None, None, None] >= 0).float()
        t = t.clamp_min(0)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        ) * t_cond + x_start * (1 - t_cond)

    # Simulates the reverse diffusion process by sampling the next less noisy image
    # at timestep t-1 based on the predicted noise and other parameters.

    # p_sample performs one step of the reverse diffusion process:

    @torch.no_grad()
    def p_sample(
        self,
        x,  # noisy image at the current timestep t
        t,  # current timestep from 500 to 0
        cond,  # conditioning information (features from low-resolution image) to guide densoing process
        img_lr_up,  # upsampled low-resolution image used to predict the noisy image if not given
        noise_pred=None,  # The predicted noise at the current timestep. If not provided, it is computed using the denoise_fn.
        clip_denoised=True,  # A flag to clip the denoised image values to a valid range (e.g., [-1, 1]) to ensure stability.
        repeat_noise=False,  # A flag to control whether the noise should be repeated across the batch.
    ):
        # 1. Predict noise
        if noise_pred is None:
            noise_pred = self.denoise_fn(x, t, cond=cond)
        b, *_, device = *x.shape, x.device

        # 2. Estiamete the clean image and Compute Model Mean and Variance:
        # The function calls self.p_mean_variance to compute:
        # model_mean: The predicted mean of the denoised image.
        # model_log_variance: The log variance of the predicted noise.
        # x0_pred: The predicted denoised image (without noise).

        # These values are used to parameterize the Gaussian distribution
        # from which the next less noisy image is sampled.

        model_mean, _, model_log_variance, x0_pred = self.p_mean_variance(
            x=x, t=t, noise_pred=noise_pred, clip_denoised=clip_denoised
        )

        # 3. Generate Noise
        # The function generates new noise using the noise_like utility function.
        # This noise is added to the predicted mean to sample the next image in
        # the reverse diffusion process
        # If repeat_noise is True, the same noise is repeated across the batch.

        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        # 4. Handle the Final Timestep

        # At the final timestep (t == 0), no additional noise is added.
        # This is controlled by the nonzero_mask,
        #  which is 0 when t == 0 and 1 otherwise

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        # 5. Sample the Next Less Noisy Image

        next_noisy_image = (
            model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        )
        #
        # 5.1 The next less noisy image is sampled using the predicted mean (model_mean)
        # and log variance (model_log_variance), scaled by the nonzero_mask and the generated noise.

        # 5.2. The predicted denoised image (without noise: x0_pred) at the current timestep.
        # This is often used for visualization or intermediate results.
        # Return the Next Noisy Image and Denoised Image
        return next_noisy_image, x0_pred

    # INFERENCE/SAMPLING STEP

    # This function returs:
    # Return	Meaning	Purpose
    # 1. x_{t-1} -- New sample after denoising and resampling
    # Used to proceed to next step in reverse process
    # 2.  x_0 -- Denoised image at current step	For visualization
    # or intermediate loss computation

    # Ensures that no gradients are computed during
    # the execution the sample function
    @torch.no_grad()
    def sample(
        self,
        img_lr,  # low-resolution image used for conditioning
        img_lr_up,  # upsampled LR image used for upsampling the noisy image
        shape,  # shape of the output image
        save_intermediate=False,
        # dates=None,
        # config=None,
        # alphas=None,
    ):
        device = self.betas.device
        b = shape[0]

        # 1. The image is initialized using the forward diffusion process (q)
        # by default res = True, so the image is initialized as randomly distributed noise

        if not hparams["res"]:
            t = torch.full(
                (b,), self.num_timesteps - 1, device=device, dtype=torch.long
            )
            # Upsamples a noisy image to its original size using the q_sample method.
            # perfoming the forward pass process allows us to gradually add noise to the
            # image until we reach the desired level of randomness.
            # The resulting image can then be used as an initial guess for the reverse diffusion process.
            img = self.q_sample(img_lr_up, t)
        else:
            # torch.manual_seed(1234)
            # the image is initialized as a random noise
            img = torch.randn(shape, device=device)

        # 2. Conditioning on the low resolution image
        # We encode the low-resolution image ONLY once
        # before the reverse diffusion process starts
        # and apply it in every iteration.

        # cond_net_out, cond = self.cond_net(img_lr, dates, config, True)
        cond_net_out, cond = self.cond_net.backbone(img_lr)

        # 3. Iterates over the time steps from the reverse diffusion process
        it = reversed(range(0, self.num_timesteps))  # num_timesteps: 500
        if self.sample_tqdm:
            it = tqdm(it, desc="sampling loop time step", total=self.num_timesteps)

        # img_sr_ts_outs: stores the super-resolved images at each time step
        img_sr_ts_outs = []

        for i in range(img_lr_up.shape[1]):
            # Each timestep in the reverse diffusion process is conditioned
            # the corresponding temporal features from the LR image

            images = []

            # For each timestep j, the p_sample method is called
            # to refine the image (img)
            # and generate a reconstruction (x_recon).
            # starting from T, t-1, t-2,...,1: 500 -> 0
            # At each iteration, the generated image is stored
            # in the list 'images'

            for j in it:
                # Performs the reverse diffusion process, refining the noisy image
                # at timestep j to generate a less noisy image.

                # Computes x_t (img), the noisy image at timestep j,
                # Why do we use img as final output?:
                # Because img is the noisy image at timestep j,
                # which is used to generate the next less noisy image
                # and x_recon, the reconstructed image at timestep j.

                # This mirrors how models like DDPM or Stable Diffusion generate output images:
                # they sample from a learned posterior rather than take the deterministic estimate.
                # In diffusion models, you don’t just want the network's prediction of x_0
                # you want to sample from the model’s learned distribution p(x_0).

                img, x_recon = self.p_sample(
                    img,  # x_t
                    torch.full((b,), j, device=device, dtype=torch.long),  # t
                    [feat[:, i] for feat in cond],  # x_e
                    img_lr_up[:, i, :, :],
                )
                if save_intermediate:
                    img_ = self.res2img(img, img_lr_up[:, i, :, :])
                    x_recon_ = self.res2img(x_recon, img_lr_up[:, i, :, :])
                    images.append((img_.cpu(), x_recon_.cpu()))

            # 4. The final image is reconstructed using the res2img method
            img_sr = self.res2img(img, img_lr_up[:, i, :, :])
            img_sr_ts_outs.append(img_sr)

        img_sr = torch.stack(img_sr_ts_outs, 1)

        # If you want a robust prediction, aggregate time-series predictions
        # Average logits over time (assuming model refines over time)

        if save_intermediate:
            return img_sr, cond_net_out, images
        else:
            return img_sr, cond_net_out

    # Convert a residual image back to the original scale
    # by adding back an upsampled low-resolution image
    # This is a common technique in super-resolution models
    # where the network predicts the residual (difference)
    # instead of the full high-resolution image.

    def res2img(self, img_, img_lr_up, clip_input=None):
        img_ = img_[:, :4, :, :]
        if clip_input is None:
            clip_input = hparams["clip_input"]
        if hparams["res"]:
            if clip_input:
                img_ = img_.clamp(-1, 1)
            # rescales the residual image before adding it to the upsampled LR image
            # with a scaling factor defined by hparams['res_rescale'] for stability.
            img_ = img_ / hparams["res_rescale"] + img_lr_up
        return img_

    def img2res(self, x, img_lr_up, clip_input=None):
        # Compute Residual Between HR and LR-upsampled image
        # Subtract the upsampled LR image from the HR image
        # to obtain the residual.

        # x: torch.Size([5, 3, 40, 40])
        # img_lr_up: torch.Size([5, 2, 3, 40, 40])
        x = x[:, :4, :, :]
        if clip_input is None:
            clip_input = hparams["clip_input"]
        if hparams["res"]:
            x = (x - img_lr_up) * hparams["res_rescale"]
            if clip_input:
                x = x.clamp(-1, 1)
        return x


#  ace_loss=2.26, aer=1.71, grd_loss=3.12, gvc_loss=0.0503, nzp_loss=0.4, pxl_loss=0.474, sr=4.92, tgm_loss=0.00633]
