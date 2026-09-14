
import os
import json
import numpy as np
import pandas as pd
import torch

from cnn import CNNFeatureExtractor
from swin import SwinFeatureExtractor
from fusion import CNN_SwinFusion


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
# LOAD SAMPLE
# ============================================================

df = pd.read_csv(TRAIN_PATH)

sample = df.iloc[0]

anchor_id = int(sample["anchor_id"])

week_1 = int(sample["week_1"])
week_2 = int(sample["week_2"])
week_3 = int(sample["week_3"])

weeks = [
    week_1 - 1,
    week_2 - 1,
    week_3 - 1
]

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
# LOAD NORMALIZATION STATS
# ============================================================

with open(STATS_PATH, "r") as f:
    stats = json.load(f)


channel_names = [
    "VV",
    "rainfall",
    "slope"
]


# ============================================================
# CREATE MODELS
# ============================================================

cnn = CNNFeatureExtractor(
    in_channels=6
)

swin = SwinFeatureExtractor(
    in_channels=6
)

fusion = CNN_SwinFusion(
    cnn_dim=256,
    swin_dim=768,
    fusion_dim=512
)

cnn.eval()
swin.eval()
fusion.eval()


# ============================================================
# PROCESS EACH WEEK SEPARATELY
# ============================================================

weekly_features = []


with torch.no_grad():

    for week_index in weeks:

        print("\n----------------------------------------")
        print("Processing week:", week_index + 1)

        # ----------------------------------------------------
        # Get one week
        # Shape: [3, 256, 256]
        # ----------------------------------------------------

        week_data = patch_256[week_index].copy()

        print(
            "Original week shape:",
            week_data.shape
        )


        # ----------------------------------------------------
        # Missingness mask
        # ----------------------------------------------------

        missing_mask = (
            ~np.isfinite(week_data)
        ).astype(np.float32)


        # ----------------------------------------------------
        # Normalize valid values
        # ----------------------------------------------------

        normalized = week_data.astype(
            np.float32
        ).copy()

        for channel in range(3):

            channel_data = normalized[channel]

            valid = np.isfinite(channel_data)

            channel_name = channel_names[channel]

            mean = stats[channel_name]["mean"]
            std = stats[channel_name]["std"]

            channel_data[valid] = (
                channel_data[valid] - mean
            ) / std

            channel_data[~valid] = 0.0


        # ----------------------------------------------------
        # Combine:
        #
        # 3 data channels
        # +
        # 3 missingness channels
        #
        # = 6 channels
        # ----------------------------------------------------

        week_input = np.concatenate(
            [
                normalized,
                missing_mask
            ],
            axis=0
        )

        print(
            "Week input shape:",
            week_input.shape
        )


        # ----------------------------------------------------
        # PyTorch tensor
        # ----------------------------------------------------

        x = torch.from_numpy(
            week_input
        ).unsqueeze(0)

        print(
            "PyTorch input:",
            x.shape
        )


        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        cnn_features = cnn(x)

        print(
            "CNN features:",
            cnn_features.shape
        )


        # ----------------------------------------------------
        # Swin
        # ----------------------------------------------------

        swin_features = swin(x)

        print(
            "Swin features:",
            swin_features.shape
        )


        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        fused_features = fusion(
            cnn_features,
            swin_features
        )

        print(
            "Fused features:",
            fused_features.shape
        )


        # ----------------------------------------------------
        # Check numerical stability
        # ----------------------------------------------------

        assert not torch.isnan(
            fused_features
        ).any()

        assert not torch.isinf(
            fused_features
        ).any()


        # ----------------------------------------------------
        # Store this week's feature
        # ----------------------------------------------------

        weekly_features.append(
            fused_features
        )


# ============================================================
# STACK THE THREE WEEKLY FEATURES
# ============================================================

temporal_sequence = torch.stack(
    weekly_features,
    dim=1
)


# ============================================================
# FINAL SHAPE
# ============================================================

print("\n========================================")

print(
    "Temporal sequence shape:",
    temporal_sequence.shape
)

print(
    "Expected shape:",
    torch.Size([1, 3, 512])
)


# ============================================================
# FINAL CHECKS
# ============================================================

assert temporal_sequence.shape == (
    1,
    3,
    512
)

assert not torch.isnan(
    temporal_sequence
).any()

assert not torch.isinf(
    temporal_sequence
).any()


print(
    "\nNaNs:",
    torch.isnan(
        temporal_sequence
    ).sum().item()
)

print(
    "Infs:",
    torch.isinf(
        temporal_sequence
    ).sum().item()
)

print(
    "\nPER-WEEK CNN + SWIN "
    "FEATURE EXTRACTION TEST PASSED"
)
