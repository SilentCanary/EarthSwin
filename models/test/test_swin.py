
import os
import json
import numpy as np
import pandas as pd
import torch

from swin import SwinFeatureExtractor


# ============================================================
# PATHS
# ============================================================

PATCH_DIR = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/multiscale_patches"
)

TEMPORAL_DIR = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/temporal_dataset"
)

STATS_PATH = os.path.join(
    TEMPORAL_DIR,
    "normalization_stats.json"
)

TRAIN_PATH = os.path.join(
    TEMPORAL_DIR,
    "train_balanced.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(TRAIN_PATH)

sample = df.iloc[0]

anchor_id = int(sample["anchor_id"])

week_1 = int(sample["week_1"])
week_2 = int(sample["week_2"])
week_3 = int(sample["week_3"])

print("Anchor:", anchor_id)
print("Weeks:", week_1, week_2, week_3)
print("Target week:", sample["target_week"])
print("Label:", sample["label"])
print("Augmentation:", sample["augmentation"])


# ============================================================
# LOAD PATCH
# ============================================================

patch_path = os.path.join(
    PATCH_DIR,
    f"sample_{anchor_id:06d}.npz"
)

data = np.load(patch_path)

patch_256 = data["patch_256"]

print("\nOriginal patch shape:", patch_256.shape)


# ============================================================
# SELECT THREE TEMPORAL WEEKS
# ============================================================

weeks = [week_1 - 1, week_2 - 1, week_3 - 1]

patch_256 = patch_256[weeks]

print("Selected patch shape:", patch_256.shape)


# ============================================================
# LOAD NORMALIZATION STATS
# ============================================================

with open(STATS_PATH, "r") as f:
    stats = json.load(f)


# ============================================================
# CREATE MISSINGNESS MASK
# ============================================================

missing_mask = (
    ~np.isfinite(patch_256)
).astype(np.float32)


# ============================================================
# NORMALIZE + REPLACE MISSING VALUES
# ============================================================

normalized = patch_256.astype(np.float32).copy()

channel_names = [
    "VV",
    "rainfall",
    "slope"
]

for channel in range(3):

    channel_data = normalized[:, channel]

    valid = np.isfinite(channel_data)

    channel_name = channel_names[channel]

    mean = stats[channel_name]["mean"]
    std = stats[channel_name]["std"]

    channel_data[valid] = (
        channel_data[valid] - mean
    ) / std

    channel_data[~valid] = 0.0


# ============================================================
# CONVERT TO CHANNELS
# ============================================================

# [3 weeks, 3 variables, H, W]
# ->
# [9, H, W]

data_channels = normalized.reshape(
    9,
    256,
    256
)

mask_channels = missing_mask.reshape(
    9,
    256,
    256
)


# ============================================================
# COMBINE DATA + MISSINGNESS
# ============================================================

earthsentinel_input = np.concatenate(
    [
        data_channels,
        mask_channels
    ],
    axis=0
)

print("\nFinal EarthSentinel input:")
print(earthsentinel_input.shape)


# ============================================================
# CONVERT TO PYTORCH
# ============================================================

x = torch.from_numpy(
    earthsentinel_input
).unsqueeze(0)

print("PyTorch input:", x.shape)


# ============================================================
# CREATE SWIN
# ============================================================

model = SwinFeatureExtractor(
    in_channels=18
)

model.eval()


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    features = model(x)


# ============================================================
# CHECK OUTPUT
# ============================================================

print("\nSwin feature shape:", features.shape)

print(
    "Expected shape:",
    torch.Size([1, 768])
)

print(
    "NaNs in features:",
    torch.isnan(features).sum().item()
)

print(
    "Infs in features:",
    torch.isinf(features).sum().item()
)


# ============================================================
# FINAL TEST
# ============================================================

assert features.shape == (1, 768)

assert not torch.isnan(features).any()

assert not torch.isinf(features).any()

print("\nEARTHSENTINEL SWIN FEATURES TEST PASSED")
