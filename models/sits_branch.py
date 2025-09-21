import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from models.encoders.maxvit import MaxViT
from models.decoders.sits_decoder import UNetDecoder

# II. Swin Like Architecture (Encoder + Decoder)
# An encoder is implemented here;
# 1. ConvFormerSits(For timeseries) (Swintime)
# A decoder is implemented here;
# 1. UPerHead
# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
# Description: It uses shifted window approach for computing self-attention
# Adapated from https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
# Paper associated to it https://ieeexplore.ieee.org/document/9710580


class SITSSegmenter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config["embed_dim"]
        self.sat_patch_size = config["sat_patch_size"]
        self.decoder_channels = config["decoder_channels"]
        self.num_classes = config["num_classes"]
        self.dropout = config["dropout"]
        self.window_size = config["window_size"]
        self.depths = config["depths"]
        self.dropout_ratio = config["dropout"]
        self.backbone_dims = [self.embed_dim * 2**i for i in range(len(self.depths))]

        self.backbone = MaxViT(
            input_size=(self.sat_patch_size, self.sat_patch_size),
            stem_channels=64,
            block_channels=[128, 256, 512],  # [64, 128, 256, 512]
            block_layers=[2, 2, 5],  # [2, 2, 5, 2]
            head_dim=32,
            stochastic_depth_prob=0.2,
            partition_size=8,
        )

        self.encoder_channels = [
            self.embed_dim,
            self.embed_dim * 2,
            self.embed_dim * 4,
            self.embed_dim * 8,
        ]

        self.decode_head = UNetDecoder(
            self.encoder_channels,
            self.decoder_channels,
            self.dropout_ratio,
            self.window_size,
            self.num_classes,
        )

    def forward(self, x, batch_positions=None):
        # print("Swin Segmentation inputs:", x.shape)
        # x_enc = self.backbone(x, batch_positions)
        h, w = x.size()[-2:]

        # temp_pooled_feats: feats used for sits decoding
        # temp_feats: feats used for conditioning
        temp_pooled_feats, _ = self.backbone(x)

        res0, res1, res2, res3 = temp_pooled_feats
        sits_logits, multi_lvls_cls, enc_features = self.decode_head(
            res0, res1, res2, res3, h, w
        )
        return sits_logits, multi_lvls_cls, enc_features
