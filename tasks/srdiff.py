import os.path
import json
import torch
import numpy as np
import torch.nn as nn
from trainer import Trainer
import torch.nn.functional as F
from utils.hparams import hparams
from utils.utils import load_ckpt
from models.diffsr_modules import Unet
from models.diffusion import GaussianDiffusion
from models.sits_aerial_seg_model import SITSAerialSegmenter
from losses.focal_smooth import FocalLossWithSmoothing
from models.sits_branch import SITSSegmenter


class SRDiffTrainer(Trainer):
    def build_model(self):
        hidden_size = hparams["hidden_size"]
        dim_mults = hparams["unet_dim_mults"]
        dim_mults = [int(x) for x in dim_mults.split("|")]

        self.criterion_aer = FocalLossWithSmoothing(
            hparams["num_classes"], gamma=2, alpha=1, lb_smooth=0.2
        )
        self.criterion_sat = FocalLossWithSmoothing(
            hparams["num_classes"], gamma=2, alpha=1, lb_smooth=0.2
        )

        self.loss_aux_sat_weight = hparams["loss_aux_sat_weight"]
        self.loss_main_sat_weight = hparams["loss_main_sat_weight"]

        denoise_fn = Unet(
            hidden_size,
            out_dim=hparams["num_channels_sat"],
            cond_dim=hparams["rrdb_num_feat"],
            dim_mults=dim_mults,
        )

        # Define the diffusion model for training
        cond_net = SITSSegmenter(hparams)
        # load_ckpt(cond_net, hparams["cond_net_ckpt"])

        gaussian = GaussianDiffusion(
            denoise_fn=denoise_fn,
            cond_net=cond_net,
            timesteps=hparams["timesteps"],
            loss_type=hparams["loss_type"],
        )

        self.model = SITSAerialSegmenter(gaussian=gaussian, config=hparams)

        if hparams["infer"]:
            if hparams["cond_net_ckpt"] != "" and os.path.exists(
                hparams["cond_net_ckpt"]
            ):
                load_ckpt(self.model, hparams["cond_net_ckpt"])

        # what is used for?
        self.global_step = 0
        return self.model

    def training_step(self, batch):
        img = batch["img"]  # torch.Size([4, 5, 512, 512])
        img_hr = batch["img_hr"]  # torch.Size([4, 5, 512, 512])
        img_lr = batch["img_lr"]  # torch.Size([4, 2, 3, 40, 40])
        img_lr_up = batch["img_lr_up"]  # torch.Size([4, 2, 3, 160, 160])
        labels = batch["labels"]  # torch.Size([4, 2, 3, 160, 160])
        labels_sr = batch["labels_sr"]  # torch.Size([4, 2, 3, 160, 160])
        dates = batch["dates_encoding"]
        closest_idx = batch["closest_idx"]  # torch.Size([4, 2, 3, 160, 160])
        sc_img_hr = img_hr[:, :4, :, :]

        # call gaussian diffusion model for SR-prediction this should also
        # return the SR-SITS images alongside the diffusion losses
        losses, _, _, img_sr = self.model.gaussian(
            sc_img_hr,
            img_lr,
            img_lr_up,
            labels_sr,
            dates=dates,
            closest_idx=closest_idx,
            # config=hparams,
        )

        # for classification branches
        sits_logits, _, aer_outputs = self.model(
            img, img_sr, labels, dates, hparams
        )

        sits_logits = F.interpolate(sits_logits, size=img.shape[2:], mode="bilinear")
        loss_main_sat = self.criterion_sat(sits_logits, labels)

        total_loss = {} # aer-loss + sat-loss (sr-loss, aux-loss, ce-sits-loss)
        
        loss_aer = self.criterion_aer(aer_outputs, labels.long())

        # The CE loss for the SITS classification branch is done at 1.6m GSD
        total_loss["sat_loss"] = hparams["loss_weights_aer_sat"][1] * (sum(losses.values()) + loss_main_sat)
        
        # The CE loss for the AER classification branch is done at 20cm GSD
        total_loss["aer_loss"] = hparams["loss_weights_aer_sat"][0] * loss_aer

        losses["sr_loss"] = loss_main_sat
        losses["aer_loss"] = total_loss["aer_loss"]

        total_loss = sum(total_loss.values())
        return losses, total_loss

    def sample_and_test(self, sample):
        # Sample images and calculate evaluation metrics
        # Used for inference mode
        ret = {k: [] for k in self.metric_keys}
        ret["n_samples"] = 0
        img = sample["img"]
        img_hr = sample["img_hr"]
        img_lr = sample["img_lr"]
        img_lr_up = sample["img_lr_up"]
        labels = sample["labels"]
        dates = sample["dates_encoding"]
        closest_idx = sample["closest_idx"]  # torch.Size([4, 2, 3, 160, 160])
        sc_img_hr = img_hr[:, :4, :, :]

        img_sr, rrdb_out = self.model.gaussian.sample(
            img_lr,
            img_lr_up,
            sc_img_hr.shape,  # 4-channel HR image shape
            # dates=dates,
            # config=hparams,
        )
        # during sampling, only the aer branch is used
        _, _, aer_outputs = self.model(img, img_sr, labels, dates, hparams)
        proba = torch.softmax(aer_outputs, dim=1)
        preds = torch.argmax(proba, dim=1)

        # Loop over batch
        for b in range(img_sr.shape[0]):
            s = self.measure.measure(
                img_sr[b][int(closest_idx[b].item()), :, :, :],  # SR image at t
                sc_img_hr[b],  # reference HR image
                img_lr[b][int(closest_idx[b].item()), :, :, :],  # LR input at t
                preds[b],
                labels[b],
            )
            ret["psnr"].append(s["psnr"])
            ret["ssim"].append(s["ssim"])
            ret["lpips"].append(s["lpips"])
            ret["mae"].append(s["mae"])
            ret["mse"].append(s["mse"])
            ret["shift_mae"].append(s["shift_mae"])
            ret["miou"].append(s["miou"])

            ret["n_samples"] += 1
        return img_sr, preds, rrdb_out, ret, ret

    def build_optimizer(self, model):
        params = list(model.named_parameters())

        # Filter out cond_net parameters that are not trainable
        if hparams["fix_cond_net_parms"]:
            params = [p[1] for p in params if "cond_net" not in p[0]]
        else:
            params = [p[1] for p in params]
        optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["lr"])
        return optimizer

    def build_scheduler(self, optimizer):
        # 1. Scheduler type
        # It uses torch.optim.lr_scheduler.MultiStepLR, which reduces the learning
        # rate (LR) by a factor (gamma) at specific training steps (called milestones)

        # 2. Milestones
        # This means the LR will drop twice:
        # First at 50% of decay_steps
        # Then at 90% of decay_steps
        # Example: if decay_steps = 100000, milestones = [50000, 90000].

        # 3. Gamma factor
        # At each milestone, the LR is multiplied by 0.1 (reduced by 10×).
        # Example: if LR starts at 0.001,
        # at step 50k → LR becomes 0.0001
        # at step 90k → LR becomes 0.00001.

        # 4. Effect
        # This creates a piecewise-constant decay schedule:
        # LR stays constant in between milestones.
        # At the milestone steps, LR suddenly drops.

        scheduler_param = {
            "milestones": [
                np.floor(hparams["decay_steps"] * 0.5),
                np.floor(hparams["decay_steps"] * 0.9),
            ],
            "gamma": 0.1,
        }
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_param)
