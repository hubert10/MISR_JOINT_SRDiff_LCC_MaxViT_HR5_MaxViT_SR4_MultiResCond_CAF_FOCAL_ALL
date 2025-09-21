import os
import torch
import rasterio
import numpy as np
from sklearn.metrics import confusion_matrix


def downsample_single_label_map_majority_vote(label: torch.Tensor):
    """
    Downsamples a single multi-class label map from 20 cm GSD to 1.6 m GSD
    using majority vote on non-overlapping 8x8 blocks.

    Args:
        label (torch.Tensor): Input label map of shape [H, W] at 20 cm resolution.
                              H and W must be divisible by 8.

    Returns:
        torch.Tensor: Downsampled label map at 1.6 m resolution,
                      shape [H//8, W//8].
    """
    scale_factor = 8
    H, W = label.shape
    assert (
        H % scale_factor == 0 and W % scale_factor == 0
    ), "Input size must be divisible by scale factor (8)."

    output_size_H = H // scale_factor
    output_size_W = W // scale_factor

    # Reshape into blocks
    label_reshaped = label.view(
        output_size_H, scale_factor, output_size_W, scale_factor
    )

    # Permute to [output_size_H, output_size_W, scale_factor, scale_factor]
    label_permuted = label_reshaped.permute(0, 2, 1, 3)

    # Flatten block pixels → [output_size_H, output_size_W, scale_factor * scale_factor]
    label_flat = label_permuted.reshape(
        output_size_H, output_size_W, scale_factor * scale_factor
    )

    # Majority vote along last dimension
    mode, _ = torch.mode(label_flat, dim=-1)
    return mode  # shape: [output_size_H, output_size_W]


def compute_confusion_matrix_folder(hr_folder, num_classes):
    """
    Computes confusion matrix between HR labels and downsampled labels at 1.6 m GSD.

    Args:
        hr_folder (str): Folder containing HR raster label maps.
        num_classes (int): Number of label classes.

    Returns:
        np.ndarray: Confusion matrix of shape [num_classes, num_classes]
    """
    all_hr_labels = []
    all_lr_labels = []

    for filename in os.listdir(hr_folder):
        if filename.lower().endswith((".tif", ".tiff")):
            hr_path = os.path.join(hr_folder, filename)
            with rasterio.open(hr_path) as src:
                hr_label = src.read(1)  # read first band
                hr_label_tensor = torch.from_numpy(hr_label).long()

                # Downsample using majority vote
                lr_label_tensor = downsample_single_label_map_majority_vote(
                    hr_label_tensor
                )

                # Flatten to 1D arrays for confusion matrix computation
                all_hr_labels.append(hr_label_tensor.view(-1))
                all_lr_labels.append(lr_label_tensor.view(-1))

    # Concatenate all maps
    all_hr_labels = torch.cat(all_hr_labels).numpy()
    all_lr_labels = torch.cat(all_lr_labels).numpy()

    # Compute confusion matrix
    cm = confusion_matrix(all_hr_labels, all_lr_labels, labels=list(range(num_classes)))
    return cm


# Example usage:
hr_folder = "/path/to/hr_labels"
num_classes = 5
cm = compute_confusion_matrix_folder(hr_folder, num_classes)
print(cm)
