import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as cT
from utils.utils_dataset import downsample_sr_image

# 1. Pixelwise L1 Loss Function
# This loss function computes the pixel-wise L1 loss between and the
# HR reference image and the closest predicted SR image acquisition
# Notes: Pixel-wise loss often lacks high-frequency structure
# information and may produce perceptually blurry images


def pixel_wise_closest_sr_sits_aer_loss(sat_ts_batch, aerial_batch, closest_indices):
    """
    Compute L1 loss between each aerial image and the closest SITS image aquisition.

    Args:
        sat_ts_batch (Tensor): Satellite image time series of shape (B, T, C, H, W)
        aerial_batch (Tensor): Aerial images of shape (B, C, H, W)
        closest_indices (Tensor or list): Indices (length B) of closest
        time step per sample

    Returns:
        Scalar tensor: Mean L1 loss over the batch
    """

    B, T, C, H, W = sat_ts_batch.shape
    closest_indices = closest_indices.to(torch.long)

    # Ensure closest_indices is a tensor
    if not torch.is_tensor(closest_indices):
        closest_indices = torch.tensor(closest_indices, device=sat_ts_batch.device)

    # Gather closest satellite images
    closest_sat_image = sat_ts_batch[
        torch.arange(B), closest_indices
    ]  # shape (B, C, H, W)

    loss = F.l1_loss(closest_sat_image, aerial_batch, reduction="mean")
    return loss


# 2. Pixel-wise Gradient Loss Function
# Satellite images have sharp edges (e.g., building boundaries, field borders, roads).
# The gradient loss function is designed to capture these high-frequency
# structures in the super-resolved images.
# This loss function computes the pixel-wise gradient loss between
# the HR reference image and the closest SR images.
# Notes: This loss encourages spatial continuity in the output
# image, we hope to recover the image gradients in order
# to capture the high frequency structure information
# Reference: https://arxiv.org/pdf/1809.07099


def compute_gradient(img):
    """
    Computes image gradients using Sobel filters.

    Args:
        img (Tensor): Image tensor of shape (B, C, H, W)

    Returns:
        grad_x, grad_y (Tensor): Gradients along x and y axes with same shape as input
    """
    # Define Sobel kernels (for a single channel)
    sobel_x = (
        torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=img.device,
            dtype=img.dtype,
        ).reshape(1, 1, 3, 3)
        / 8.0
    )

    sobel_y = (
        torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=img.device,
            dtype=img.dtype,
        ).reshape(1, 1, 3, 3)
        / 8.0
    )

    B, C, H, W = img.shape

    # Apply Sobel filter to each channel separately using group convolution
    grad_x = F.conv2d(img, sobel_x.repeat(C, 1, 1, 1), groups=C, padding=1)
    grad_y = F.conv2d(img, sobel_y.repeat(C, 1, 1, 1), groups=C, padding=1)

    return grad_x, grad_y


def grad_pixel_wise_closest_sr_sits_aer_loss(
    cond_net_out, img_hr, closest_indices, loss_type="l1"
):
    """
    Computes a dimensionless gradient-based loss between the closest
    SR image and the high-resolution aerial reference image.
    Normalized by average HR gradient magnitude to become unitless.
    """
    B, T, C, H, W = cond_net_out.shape
    closest_indices = closest_indices.to(torch.long)

    # Select closest super-resolved image per batch element
    closest_sr = cond_net_out[torch.arange(B), closest_indices]  # (B, C, H, W)

    # Compute gradients
    sr_grad_x, sr_grad_y = compute_gradient(closest_sr)
    hr_grad_x, hr_grad_y = compute_gradient(img_hr)

    # Compute raw loss
    if loss_type == "l1":
        grad_loss = F.l1_loss(sr_grad_x, hr_grad_x) + F.l1_loss(sr_grad_y, hr_grad_y)
    elif loss_type == "l2":
        grad_loss = F.mse_loss(sr_grad_x, hr_grad_x) + F.mse_loss(sr_grad_y, hr_grad_y)
    else:
        raise ValueError("loss_type must be 'l1' or 'l2'")

    # Compute average gradient magnitude of HR image for normalization
    hr_grad_magnitude = torch.sqrt(hr_grad_x**2 + hr_grad_y**2)
    mean_hr_grad_mag = hr_grad_magnitude.mean().clamp(min=1e-6)  # Avoid division by 0

    # Normalize the gradient loss
    loss_normalized = grad_loss / mean_hr_grad_mag
    return loss_normalized


# 3. Temporal Color Consistency Loss Function
# Compute the gradient magnitude of the SR images
# using finite differences to capture the strongest
# spatial changes in the super-resolved images.

# In temporal consistency, we're interested in how much
# the image structure changes over time—not necessarily
# in which direction it changes


def compute_gradient_magnitude(image):
    """
    Compute the gradient magnitude of an image using Sobel filters.
    Args:
        image (torch.Tensor): Input tensor (B, C, H, W), expects float.
    Returns:
        torch.Tensor: Gradient magnitude tensor (B, C, H, W).
    """
    B, C, H, W = image.shape

    # Define Sobel kernels
    sobel_x = (
        torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(
            1, 1, 3, 3
        )
        / 8.0
    )

    sobel_y = (
        torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(
            1, 1, 3, 3
        )
        / 8.0
    )

    # Repeat kernels for depthwise conv (1 per channel)
    sobel_x = sobel_x.repeat(C, 1, 1, 1).to(image.device)
    sobel_y = sobel_y.repeat(C, 1, 1, 1).to(image.device)

    # Apply padding to preserve size
    grad_x = F.conv2d(image, sobel_x, padding=1, groups=C)
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=C)

    # Compute gradient magnitude
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    return grad_magnitude


def temp_gradient_magnitude_consistency_loss(img_sr):
    """
    Goal: Compute dimensionless temporal gradient magnitude consistency loss.

    1. Compute squared differences of gradient magnitudes between consecutive frames
        normalized by the number of temporal steps (T-1)
        advantage - prevents longer sequences from blowing up the loss
    2. Normalize by the mean gradient magnitude across all frames
        idea: make the loss dimensionless by relating it to the “average
            spatial variation strength” in the sequence

        a. If gradients are strong (lots of texture/edges), you tolerate larger differences
        b. If gradients are weak (smooth regions), you penalize differences more

    3. Is this the best normalization?
        1. Normalize by the number of temporal steps (T) → you already do this
           Ensures comparability across sequences of different lengths.
        2. Normalize by gradient energy per frame (‖∇I‖² mean) rather than mean magnitude:
           Using squared energy keeps it consistent with the squared difference form
           Right now you divide by mean magnitude (‖∇I‖), which may under- or over-normalize
           when edges are sharp.
           Normalize by variance of gradients across time:

        3. Instead of mean gradient magnitude, normalize by the temporal variance of gradients
            This makes the loss relative to the natural variability of the sequence, not just its overall sharpness
            Advantage: avoids overly penalizing dynamic sequences with strong natural changes
            (e.g., vegetation, water bodies)

    4. Recommendations:
        1. Instead of dividing by mean gradient magnitude, divide by mean squared gradient magnitude
            (avg_grad_energy = (grad_mags**2).mean()).
            This keeps it consistent dimensionally (since you're comparing squared differences)
            Prevents instability when gradients are tiny (almost flat images)

    Args:
        img_sr (torch.Tensor): (B, T, C, H, W), super-resolved time series.

    Returns:
        torch.Tensor: Scalar normalized loss.
    """
    B, T, C, H, W = img_sr.shape
    total_loss = 0.0
    grad_mags = []

    # Compute gradient magnitudes for all timesteps
    for t in range(T):
        grad = compute_gradient_magnitude(img_sr[:, t])  # (B, C, H, W)
        grad_mags.append(grad)

    grad_mags = torch.stack(grad_mags, dim=1)  # (B, T, C, H, W)

    # Compute temporal gradient difference
    for t in range(T - 1):
        diff = grad_mags[:, t + 1] - grad_mags[:, t]  # (B, C, H, W)
        total_loss += torch.mean(diff**2)

    loss_raw = total_loss / (T - 1)  # Raw temporal gradient loss

    # Compute normalization factor: mean gradient magnitude of all frames
    # avg_grad_magnitude = grad_mags.mean().clamp(min=1e-6)

    avg_grad_energy = (grad_mags**2).mean().clamp(min=1e-6)

    # Normalize loss
    loss_normalized = loss_raw / avg_grad_energy  # avg_grad_magnitude
    return loss_normalized


# 4. Define the Gray Value Consistency Loss Function
# between the SR and LR SITS image time at each time step.
# Does it make sense to use a trained network to do this?


def gray_value_consistency_loss(
    img_sr: torch.Tensor, img_lr: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Gray Value Consistency Loss between the super-resolved
    and low-resolution image time series.
    Assumes input shapes are (B, T, C, H, W).

    Args:
        img_sr (torch.Tensor): Super-resolved image sequence of shape
          (B, T, C, H, W): (B, T, C, 64, 64)
        img_lr (torch.Tensor): Low-resolution image sequence of shape
        (B, T, C, h, w): (B, T, C, 40, 40)
          to (default: 64x64)

    Returns:
        torch.Tensor: Scalar MSE loss averaged over batch, time, channels,
          and spatial dimensions.
    """

    # Crop the central 10×10 region for fusion with aerial imagery
    crop = cT.CenterCrop((10, 10))
    img_lr_cropped = crop(img_lr)

    # Downsample the SR-SITS from 1m to 10m before loss computation
    downsample_factor = 10.0 / 1.6
    downsampled_sr = [
        downsample_sr_image(img_sr[:, i], downsample_factor)
        for i in range(img_sr.shape[1])
    ]
    img_sr = torch.stack(downsampled_sr, dim=1)

    # Ensure shapes match
    assert (
        img_sr.shape == img_lr_cropped.shape
    ), "Shape mismatch between SR and upsampled LR images"

    # Compute mean squared error loss

    return F.l1_loss(img_sr, img_lr_cropped, reduction="mean")


def cross_entropy_loss(class_logits, class_labels):
    """
    Cross Entropy Loss computation between SR-SITS
     logits and HR ground truth labels for each
     time step to regularize predictions over time
     and enforce temporal consistency in classificaion
    """
    ce_loss = nn.CrossEntropyLoss()

    if class_logits.dim() == 5:
        _, T, _, _, _ = class_logits.shape
        total_ce_loss = 0
        for t in range(T):
            sr_t = class_logits[:, t]  # shape: [B, C, H, W]
            ce_loss_t = ce_loss(sr_t, class_labels)
            total_ce_loss += ce_loss_t
        # To keep cross-entropy comparable across time steps
        # (and with other losses), normalize by the number of steps:
        # class_labels shape [B, H, W]
        return total_ce_loss / T
    else:
        ce_loss = ce_loss(class_logits, class_labels)
        return ce_loss
