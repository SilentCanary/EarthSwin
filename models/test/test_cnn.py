import os
import json
import sys
import numpy as np
import pandas as pd
import torch


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = "C:/Users/advit/Documents/major project"

PATCH_DIR = (
    PROJECT_DIR
    + "/dataset/EarthSentinel_2016/multiscale_patches"
)

TEMPORAL_DIR = (
    PROJECT_DIR
    + "/dataset/EarthSentinel_2016/temporal_dataset"
)

CSV_PATH = TEMPORAL_DIR + "/train_balanced.csv"

STATS_PATH = TEMPORAL_DIR + "/normalization_stats.json"

# Allow Python to find cnn.py
sys.path.append(PROJECT_DIR + "/models")

from cnn import CNNFeatureExtractor


# ============================================================
# START
# ============================================================

print("=" * 60)
print("EARTHSENTINEL + CNN TEST")
print("=" * 60)


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

with open(STATS_PATH, "r") as f:
    stats = json.load(f)


# ============================================================
# LOAD TRAINING MANIFEST
# ============================================================

print("\nLoading training manifest...")

df = pd.read_csv(CSV_PATH)

print("Dataset rows:", len(df))


# ============================================================
# SELECT ONE SAMPLE
# ============================================================

row = df.iloc[0]

anchor_id = int(row["anchor_id"])

week_1 = int(row["week_1"])
week_2 = int(row["week_2"])
week_3 = int(row["week_3"])

label = int(row["label"])
augmentation = row["augmentation"]

print("\nSelected sample:")
print("Anchor ID:", anchor_id)
print("Weeks:", week_1, week_2, week_3)
print("Target week:", int(row["target_week"]))
print("Label:", label)
print("Augmentation:", augmentation)


# ============================================================
# LOAD 256x256 PATCH
# ============================================================

patch_path = os.path.join(
    PATCH_DIR,
    f"sample_{anchor_id:06d}.npz"
)

print("\nLoading:")
print(patch_path)

with np.load(patch_path) as data:
    patch_256 = data["patch_256"].copy()

print("\nOriginal patch shape:")
print(patch_256.shape)

# Expected:
# (14, 3, 256, 256)


# ============================================================
# SELECT THREE WEEKS
# ============================================================

weeks = [
    week_1 - 1,
    week_2 - 1,
    week_3 - 1
]

patch = patch_256[weeks]

print("\nAfter selecting 3 weeks:")
print(patch.shape)

# Expected:
# (3, 3, 256, 256)


# ============================================================
# CREATE MISSINGNESS MASK
# ============================================================

missing_mask = (
    ~np.isfinite(patch)
).astype(np.float32)

print("\nMissing mask shape:")
print(missing_mask.shape)

print(
    "Missing values:",
    np.sum(missing_mask)
)

print(
    "Valid values:",
    np.sum(missing_mask == 0)
)


# ============================================================
# NORMALIZE VALID VALUES
# ============================================================

patch = patch.astype(np.float32)

band_names = [
    "VV",
    "rainfall",
    "slope"
]

for channel_idx, band_name in enumerate(band_names):

    mean = stats[band_name]["mean"]
    std = stats[band_name]["std"]

    channel_data = patch[:, channel_idx, :, :]

    valid = np.isfinite(channel_data)

    channel_data[valid] = (
        channel_data[valid] - mean
    ) / std

    channel_data[~valid] = 0.0

    patch[:, channel_idx, :, :] = channel_data


# ============================================================
# CONVERT TO CHANNELS
# ============================================================

# Data:
# (3 weeks, 3 variables, 256, 256)
#
# becomes:
# (9, 256, 256)

data_channels = patch.reshape(
    9,
    256,
    256
)

# Mask:
# (3 weeks, 3 variables, 256, 256)
#
# becomes:
# (9, 256, 256)

mask_channels = missing_mask.reshape(
    9,
    256,
    256
)


# ============================================================
# COMBINE DATA + MASKS
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

# Expected:
# (18, 256, 256)


# ============================================================
# CHECK FOR NaNs
# ============================================================

print(
    "\nNaNs in input:",
    np.isnan(earthsentinel_input).sum()
)

print(
    "Infs in input:",
    np.isinf(earthsentinel_input).sum()
)


# ============================================================
# CONVERT TO PYTORCH
# ============================================================

x = torch.from_numpy(
    earthsentinel_input
).unsqueeze(0)

print("\nPyTorch input shape:")
print(x.shape)

# Expected:
# torch.Size([1, 18, 256, 256])


# ============================================================
# CREATE CNN
# ============================================================

print("\nCreating CNN...")

model = CNNFeatureExtractor(
    in_channels=18
)

print("CNN created successfully.")


# ============================================================
# FORWARD PASS
# ============================================================

model.eval()

with torch.no_grad():

    features = model(x)


# ============================================================
# RESULTS
# ============================================================

print("\nInput shape:")
print(x.shape)

print("CNN feature shape:")
print(features.shape)

print("\nExpected feature shape:")
print("[1, 256]")

print("\nFirst 10 feature values:")
print(features[0, :10])

print(
    "\nNaNs in features:",
    torch.isnan(features).sum().item()
)

print(
    "Infs in features:",
    torch.isinf(features).sum().item()
)


# ============================================================
# FINAL CHECK
# ============================================================

assert features.shape == (1, 256), (
    f"Unexpected feature shape: {features.shape}"
)

assert not torch.isnan(features).any(), (
    "CNN output contains NaNs"
)

assert not torch.isinf(features).any(), (
    "CNN output contains Infs"
)


print("\n" + "=" * 60)
print("EARTHSENTINEL CNN TEST PASSED")
print("=" * 60)

