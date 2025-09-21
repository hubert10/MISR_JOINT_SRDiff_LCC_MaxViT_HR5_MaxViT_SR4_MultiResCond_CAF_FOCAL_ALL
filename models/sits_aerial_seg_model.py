import torch
import timm
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from models.sits_branch import SITSSegmenter
from utils.hparams import hparams
from timm.layers import create_conv2d, create_pool2d
from models.decoders.aer_decoder import UNetFormerDecoder
from models.fusion_module.aer_cross_sat_atts import FFCA


class SITSAerialSegmenter(nn.Module):
    def __init__(self, gaussian, config):
        super().__init__()
        self.gaussian = gaussian
        self.config = config
        self.embed_dim = config["embed_dim"]
        self.decoder_channels = config["decoder_channels"]
        self.num_classes = config["num_classes"]
        self.dropout = config["dropout"]
        self.window_size = config["window_size"]

        # 1. SITS Encoder
        self.sits_net = SITSSegmenter(config)

        # # 2. Aerial Network
        self.aerial_net = timm.create_model(
            "maxvit_tiny_tf_512.in1k",
            pretrained=True,
            features_only=True,
            num_classes=config["num_classes"],
        )

        # Get first conv layer (usually called 'stem.conv' in MaxViT)
        conv1 = (
            self.aerial_net.stem.conv1
        )  # <-- sometimes it's model.stem.conv or model.conv_stem, check print(model)

        # Create new conv with 5 input channels instead of 3
        new_conv = create_conv2d(
            in_channels=config["num_channels_aer"],  # Use num_channels from config
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=1,  # original padding was None, but we set it to 1 for compatibility
            bias=conv1.bias is not None,
        )

        # Initialize the first 3 channels with pretrained weights
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = conv1.weight  # copy RGB weights
            # Initialize the extra channels randomly (e.g., Kaiming normal)
            nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :])

        # Replace the old conv with the new one
        self.aerial_net.stem.conv1 = new_conv

        encoder_channels = [
            self.embed_dim,
            self.embed_dim * 2,
            self.embed_dim * 4,
            self.embed_dim * 8,
        ]

        # 3. Decoder from U-Net Former paper
        self.decoder = UNetFormerDecoder(
            encoder_channels,
            self.decoder_channels,
            self.dropout,
            self.window_size,
            self.num_classes,
        )
        self.fusion_module = FFCA(
            aer_channels_list=[128, 256, 512],
            sits_channels_list=[64, 128, 256],
            num_heads=8,
        )

    def forward(
        self,
        aerial: torch.FloatTensor,
        img_sr: torch.FloatTensor,
        labels: torch.FloatTensor,
        dates: torch.FloatTensor,
        config,
    ):
        # aerial:  torch.Size([4, 5, 512, 512])
        # aerial:  torch.Size([4, 5, 512, 512])
        h, w = aerial.size()[-2:]
        # Aerial branch
        res0, res1, res2, res3, res4 = self.aerial_net(aerial)

        # SITS branch
        sits_logits, cls_sits, multi_lvls_outs = self.sits_net(img_sr, dates)

        # Fusion FFCA
        res2, res3, res4 = self.fusion_module([res2, res3, res4], multi_lvls_outs)

        # Decoder
        logits = self.decoder(res0, res1, res2, res3, res4, h, w)
        return sits_logits, cls_sits, logits


# Description of GRID attention introduced in MaxViT

#  Imagine you have a 6×6 image, and you want each pixel to "see" other pixels globally.

# Step 1: Split into grid

# Divide the 6×6 image into 2×2 grids, so you have 9 grids in total. Each grid has 2×2 pixels.

# Step 2: Grid-attention with dilation

# Instead of computing attention for all 36 pixels (which is expensive), you:
# First compute local attention within each grid (2×2 → small and fast).
# Then compute attention across grids, but using dilated connections (e.g., only attend to every 2nd grid in each direction).
# This way, even distant pixels can influence each other, without doing full 36×36 attention.
