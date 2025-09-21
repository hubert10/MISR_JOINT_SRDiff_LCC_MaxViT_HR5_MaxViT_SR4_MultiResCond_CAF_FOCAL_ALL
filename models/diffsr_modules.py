import functools
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from utils.hparams import hparams
from .module_util import make_layer, initialize_weights
from .commons import Mish, SinusoidalPosEmb, RRDB, Residual, Rezero, LinearAttention
from .commons import ResnetBlock, Upsample, Block, Downsample

class Unet(nn.Module):
    """
    ε_theta = U-Net(x_t, t, cond=low_res_input)
    Takes the noisy version of the HR image at t time step
    Together with conditioning information from (LR) image
    And outputs the predicted noisy and This noise is subtracted
    in the denoising step to reconstruct the high-resolution image.

    ✔ Time conditioning: Uses sinusoidal embeddings for diffusion timesteps.
    ✔ Multi-scale U-Net: Progressive downsampling and upsampling for hierarchical
    feature extraction.
    ✔ Conditional inputs: Uses external features (e.g., RRDB, LTAE) for super-
    resolution guidance.
    ✔ Self-attention (optional): Helps capture long-range dependencies.
    ✔ Residual learning: Predicts high-frequency details, improving stability.
    ✔ Weight normalization handling: Can remove weight_norm for faster inference.

    dim_mults=(1,2,4,8): Defines feature map sizes across layers.
    dims: Stores the channel dimensions at each stage.
    in_out: Stores input-output dimension pairs for each layer.

    """

    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), cond_dim=32):
        super().__init__()
        dims = [hparams["num_channels_sat"], *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0

        # 1.3 Time Embedding (time_pos_emb)
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # 1.4 Encoder (Downsampling Path)
        # Uses ResNet blocks with time embeddings.
        # Each layer halves the spatial resolution (except the last)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # 1.5 Bottleneck (Middle Blocks)
        # ResNet blocks refine the feature maps.
        # Optional self-attention if use_attn=True.

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        if hparams["use_attn"]:
            self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        # 1.6 Decoder (Upsampling Path)
        # Reverses the downsampling process.
        # Concatenates skip connections to retain fine details.

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_out * 2, dim_in, time_emb_dim=dim, groups=groups
                        ),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
        # 1.7 Final Output Layer
        # Uses 1x1 convolution to map to the required output dimension.

        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups), nn.Conv2d(dim, out_dim, 1)
        )

        if hparams["res"] and hparams["up_input"]:
            self.up_proj = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(3, dim, 3),
            )
        if hparams["use_wn"]:
            self.apply_weight_norm()
        if hparams["weight_init"]:
            self.apply(initialize_weights)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                # print(f"| Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def upsample_to_target(self, tensor, target_size):
        """
        Upsample tensor spatially to match target_size (H, W).
        """
        return F.interpolate(
            tensor, size=target_size, mode="bilinear", align_corners=False
        )

    def crop_upsample_to_target(self, cond):
        # Cropping the center of the cond tensor which
        # represents the  super-patten of the Sat LR image
        transform = T.CenterCrop((10, 10))

        processed = []

        for feats in cond:
            H, W = feats.shape[-2:]  # original spatial size
            cropped = transform(feats)
            upscaled = self.upsample_to_target(cropped, target_size=(H, W))
            processed.append(upscaled)
        return processed

    def forward(self, x, time, cond):
        # x: noisy image at current diffusion step.
        # time: timestep scalar used for embedding.
        # cond: low-res image features or sequence of features.
        # img_lr_up: upsampled LR image (used if res and up_input are true).

        t = self.time_pos_emb(time)
        t = self.mlp(t)
        h = []

        # The conditioning features should be cropped from the
        # center and upsampled to match the resolution of the
        # HR extracted features
        cond = self.crop_upsample_to_target(cond)
        # Run through down path, injecting conditioning
        # at the first level

        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x, t)
            x = resnet2(x, t)
            x = x.add_(cond[i])
            h.append(x)
            x = downsample(x)

        # Apply mid blocks (+ optional attention).
        x = self.mid_block1(x, t)
        if hparams["use_attn"]:
            x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Run through up path, using skip connections.
        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)
        # Output the refined super-resolved image
        x = self.final_conv(x)
        return x

    # 3. Speed Optimization (make_generation_fast_)
    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(remove_weight_norm)
