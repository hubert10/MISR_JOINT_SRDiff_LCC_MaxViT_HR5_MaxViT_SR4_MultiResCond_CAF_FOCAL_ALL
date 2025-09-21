import os
import random
import cv2
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt


def extract_date_from_filename(filename):
    # e.g. "3_2021-10-15-D022_2021-Z1_AA.png" → "2021-10-15"
    parts = filename.split("_")[1]  # "2021-10-15-D022_2021-Z1_AA.png"
    date_str = parts.split("-D")[0]  # → "2021-10-15"
    return datetime.strptime(date_str, "%Y-%m-%d")


def plot_hr_lr_sr_time_series(root_path, subfolder="D022_2021/Z1_AA"):
    hr_path = os.path.join(root_path, "HR", "IMG_077413.tif")
    lr_folder = os.path.join(root_path, "LR", subfolder, "sen")
    sr_folder = os.path.join(root_path, "SR", subfolder, "sen")

    # --- Load HR Image ---
    hr_img = cv2.imread(hr_path, cv2.IMREAD_UNCHANGED)
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB) / 255.0

    # --- Load LR & SR Filenames and Sort by Date ---
    lr_images = sorted(glob(os.path.join(lr_folder, "*.png")))
    sr_images = sorted(glob(os.path.join(sr_folder, "*.png")))

    if not lr_images or not sr_images:
        print("❌ LR or SR image folders are empty.")
        return

    lr_basenames = [os.path.basename(f) for f in lr_images]
    sr_basenames = [os.path.basename(f) for f in sr_images]
    common_files = sorted(set(lr_basenames).intersection(sr_basenames))

    if not common_files:
        print("❌ No matching LR and SR files found.")
        return

    # Sort by date extracted from filename: e.g., '3_2021-10-15-D022_2021-Z1_AA.png'
    common_files = sorted(common_files, key=extract_date_from_filename)

    # Limit to 5 images (optional)
    num_dates = min(5, len(common_files))
    selected_files = common_files[:num_dates]

    # === Plot ===
    fig, axs = plt.subplots(3, num_dates, figsize=(4 * num_dates, 8))
    fig.suptitle(f"HR Image and Time Series of LR & SR\n{subfolder}", fontsize=16)

    # First row: HR image centered
    for i in range(num_dates):
        if i == num_dates // 2:
            axs[0, i].imshow(hr_img)
            axs[0, i].set_title("HR (Aerial)", fontsize=12)
        axs[0, i].axis("off")

    # Second & third rows: LR and SR time series
    for i, fname in enumerate(selected_files):
        date_str = fname.split("_")[1]

        # --- LR ---
        lr_img = cv2.imread(os.path.join(lr_folder, fname), cv2.IMREAD_UNCHANGED)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB) / 255.0
        axs[1, i].imshow(lr_img)
        axs[1, i].set_title(f"LR: {date_str}", fontsize=10)
        axs[1, i].axis("off")

        # --- SR ---
        sr_img = cv2.imread(os.path.join(sr_folder, fname), cv2.IMREAD_UNCHANGED)
        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB) / 255.0
        axs[2, i].imshow(sr_img)
        axs[2, i].set_title(f"SR: {date_str}", fontsize=10)
        axs[2, i].axis("off")

    plt.subplots_adjust(wspace=0.001, hspace=0.2)
    plt.show()


plot_hr_lr_sr_time_series(
    root_path="D:\\kanyamahanga\\Datasets\\200000\\MISR_S2_Aer_X625_JOINT_SRDiff_LCC_SegFormer_HR5_ConvFormer_SR4_SAF_Gray_Value"
)
