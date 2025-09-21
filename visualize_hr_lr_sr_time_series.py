#!/usr/bin/env python
# coding: utf-8

import os
import random
import matplotlib.pyplot as plt
from glob import glob
import cv2
import re
import torch
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot as plot
import torchvision.transforms as T
import rasterio
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import rasterio
from data.data_display import convert_to_color, lut_colors


def load_hr_raster(img_id, rel_path, img_root):
    """
    Load a raster mask from a given image ID and relative path.

    Parameters:
    - img_id: str, e.g. "036385"
    - rel_path: str, e.g. "D055_2018/Z3_UF"
    - msk_root: str, base folder where masks are stored

    Returns:
    - mask: numpy array of the mask
    - profile: rasterio profile (metadata)
    - path: full path to the mask
    """
    img_filename = f"IMG_{img_id}.tif"
    img_path = os.path.join(img_root, rel_path, "img", img_filename)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"IMG file not found: {img_path}")

    with rasterio.open(img_path, "r") as f:
        img = f.read([1, 2, 3]).swapaxes(0, 2).swapaxes(0, 1)
    print("img:", img.shape)
    return img


def load_mask_raster(img_id, rel_path, msk_root):
    """
    Load a raster mask from a given image ID and relative path.

    Parameters:
    - img_id: str, e.g. "036385"
    - rel_path: str, e.g. "D055_2018/Z3_UF"
    - msk_root: str, base folder where masks are stored

    Returns:
    - mask: numpy array of the mask
    - profile: rasterio profile (metadata)
    - path: full path to the mask
    """
    msk_filename = f"MSK_{img_id}.tif"
    msk_path = os.path.join(msk_root, rel_path, "msk", msk_filename)

    if not os.path.exists(msk_path):
        raise FileNotFoundError(f"Mask file not found: {msk_path}")

    with rasterio.open(msk_path, "r") as f:
        mk = f.read([1])
        print("mk:", mk.shape)

    mask = torch.as_tensor(mk, dtype=torch.int32)
    print("mask:", mask.shape)
    return mask


def downsample_majority_vote_with_crop(
    labels, original_size=512, cropped_size=500, output_size=10
):
    """
    Downsamples multi-class label maps using majority vote after cropping.

    Args:
        labels (torch.Tensor): Input label maps of shape [N, 512, 512].
        original_size (int): Original spatial size (assumed square). Default is 512.
        cropped_size (int): Desired spatial size after cropping (assumed square). Default is 500.
        output_size (int): Desired output spatial size (assumed square). Default is 10.

    Returns:
        torch.Tensor: Downsampled label maps of shape [N, 10, 10].
    """
    N, H, W = labels.shape
    assert (
        H == original_size and W == original_size
    ), f"Input label maps must be of shape [N, {original_size}, {original_size}]"
    assert (
        cropped_size % output_size == 0
    ), f"cropped_size must be divisible by output_size. Got cropped_size={cropped_size}, output_size={output_size}"

    # Step 1: Crop the label maps to [N, 500, 500]
    # Assuming center crop: remove 6 pixels from each side
    crop_margin = (original_size - cropped_size) // 2  # 6 pixels
    labels_cropped = labels[
        :,
        crop_margin : crop_margin + cropped_size,
        crop_margin : crop_margin + cropped_size,
    ]

    # Step 2: Reshape to [N, output_size, block_size, output_size, block_size]
    block_size = cropped_size // output_size  # 50
    labels_reshaped = labels_cropped.view(
        N, output_size, block_size, output_size, block_size
    )

    # Step 3: Permute to [N, output_size, output_size, block_size, block_size]
    labels_permuted = labels_reshaped.permute(0, 1, 3, 2, 4)

    # Step 4: Flatten the block pixels to [N, output_size, output_size, block_size * block_size]
    labels_flat = labels_permuted.reshape(
        N, output_size, output_size, block_size * block_size
    )

    # Step 5: Compute mode along the last dimension (majority vote)
    mode, _ = torch.mode(labels_flat, dim=-1)

    return mode  # [N, 10, 10]


def plot_random_hr_lr_sr(hr_root, lr_root, sr_root):
    # Step 1: Randomly select an HR image
    hr_images = glob(os.path.join(hr_root, "**", "img", "*.png"), recursive=True)
    if not hr_images:
        print("No HR images found.")
        return

    hr_img_path = random.choice(hr_images)
    print(f"Selected HR image: {hr_img_path}")

    # Step 2: Extract ID and corresponding LR/SR folders
    base_name = os.path.basename(hr_img_path)  # e.g., IMG_077413.png
    img_id = base_name.split("_")[-1].split(".")[0]  # '077413'

    # Get the relative path after HR root (e.g., D015_2020/Z1_AA)
    rel_path = os.path.relpath(os.path.dirname(os.path.dirname(hr_img_path)), hr_root)

    lr_folder = os.path.join(lr_root, rel_path, "sen", img_id)
    sr_folder = os.path.join(sr_root, rel_path, "sen", img_id)

    # Step 3: Load images
    hr_img = load_hr_raster(img_id, rel_path, img_root_folder)

    down_hr_img = cv2.imread(hr_img_path)
    down_hr_img = (
        cv2.cvtColor(down_hr_img, cv2.COLOR_BGR2RGB)
        if down_hr_img is not None
        else None
    )

    # Load masks
    msk = load_mask_raster(img_id, rel_path, msk_root_folder)
    down_msk = downsample_majority_vote_with_crop(msk)

    msk_color = convert_to_color(msk[0], palette=lut_colors)
    down_msk_color = convert_to_color(down_msk[0], palette=lut_colors)

    print("HR image:", hr_img.shape)
    print("HR mask:", msk.shape)
    print("Downsampled HR image:", down_hr_img.shape)
    print("Downsampled mask:", down_msk_color.shape)

    # Load LR and SR images
    lr_imgs = sorted(glob(os.path.join(lr_folder, "*.png")))
    sr_imgs = sorted(glob(os.path.join(sr_folder, "*.png")))
    lr_images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in lr_imgs]
    sr_images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in sr_imgs]

    num_timesteps = max(len(lr_images), len(sr_images))
    fig_rows = 3
    fig_cols = max(num_timesteps, 4)  # make space for 4 items in first row

    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=(4 * fig_cols, 10))

    # Row 0: HR, HR MSK, downsampled HR, downsampled MSK
    first_row_imgs = [hr_img, msk_color, down_hr_img, down_msk_color]
    titles = ["HR Image", "HR MSK", "Downsampled HR", "Downsampled MSK"]
    for i in range(4):
        axs[0, i].imshow(first_row_imgs[i])
        axs[0, i].axis("off")
        axs[0, i].set_title(titles[i])
    for i in range(4, fig_cols):  # blank out remaining cells
        axs[0, i].axis("off")

    # Row 1: LR
    for i, img in enumerate(lr_images):
        axs[1, i].imshow(img)
        axs[1, i].axis("off")
        axs[1, i].set_title(f"LR {i}")
    for i in range(len(lr_images), fig_cols):
        axs[1, i].axis("off")

    # Row 2: SR
    for i, img in enumerate(sr_images):
        axs[2, i].imshow(img)
        axs[2, i].axis("off")
        axs[2, i].set_title(f"SR {i}")
    for i in range(len(sr_images), fig_cols):
        axs[2, i].axis("off")

    plt.tight_layout()
    plt.show()


root_folder = "D:\\kanyamahanga\\Datasets\\MISR_JOINT_SRDiff_LCC_MaxViT_HR5_MaxViT_SR4_MultiResCond_CAF_FOCAL_ALL_EOLAB\\"
img_root_folder = "D:\\kanyamahanga\\Datasets\\FLAIR\\flair_aerial_test\\"
msk_root_folder = "D:\\kanyamahanga\\Datasets\\FLAIR\\flair_labels_test\\"

plot_random_hr_lr_sr(
    hr_root=root_folder + "HR", lr_root=root_folder + "LR", sr_root=root_folder + "SR"
)
