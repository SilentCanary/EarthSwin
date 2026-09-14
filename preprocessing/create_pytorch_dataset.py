import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PATCH_DIR = "C:/Users/advit/Documents/major project/dataset/EarthSentinel_2016/multiscale_patches"
TEMPORAL_DIR = "C:/Users/advit/Documents/major project/dataset/EarthSentinel_2016/temporal_dataset"
STATS_PATH = os.path.join(TEMPORAL_DIR, "normalization_stats.json")


class EarthSentinelDataset(Dataset):

    def __init__(self, csv_path, normalize=True, stats=None):

        self.df = pd.read_csv(csv_path)
        self.normalize = normalize
        self.stats = stats

        required_columns = [
            "anchor_id",
            "week_1",
            "week_2",
            "week_3",
            "target_week",
            "label",
            "augmentation"
        ]

        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def __len__(self):
        return len(self.df)

    # ---------------------------------------------------------
    # LOAD ONE SPATIAL ANCHOR
    # ---------------------------------------------------------

    def load_anchor(self, anchor_id):

        filename = f"sample_{anchor_id:06d}.npz"
        path = os.path.join(PATCH_DIR, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Patch not found: {path}")

        data = np.load(path)

        patch_64 = data["patch_64"].copy()
        patch_128 = data["patch_128"].copy()
        patch_256 = data["patch_256"].copy()

        return patch_64, patch_128, patch_256

    # ---------------------------------------------------------
    # AUGMENTATION
    # ---------------------------------------------------------

    def augment(self, array, augmentation):

        if augmentation == "none":
            return array

        elif augmentation == "horizontal_flip":
            return np.flip(array, axis=3).copy()

        elif augmentation == "vertical_flip":
            return np.flip(array, axis=2).copy()

        elif augmentation == "rotate_90":
            return np.rot90(array, k=1, axes=(2, 3)).copy()

        elif augmentation == "rotate_180":
            return np.rot90(array, k=2, axes=(2, 3)).copy()

        elif augmentation == "rotate_270":
            return np.rot90(array, k=3, axes=(2, 3)).copy()

        elif augmentation == "horizontal_vertical_flip":
            array = np.flip(array, axis=2)
            array = np.flip(array, axis=3)
            return array.copy()

        elif augmentation == "transpose":
            return np.transpose(array, (0, 1, 3, 2)).copy()

        else:
            raise ValueError(
                f"Unknown augmentation: {augmentation}"
            )

    # ---------------------------------------------------------
    # CREATE MISSING-VALUE MASK
    #
    # 0 = valid measurement
    # 1 = missing measurement
    # ---------------------------------------------------------

    def create_missing_mask(self, array):

        mask = (~np.isfinite(array)).astype(np.float32)

        return mask

    # ---------------------------------------------------------
    # NORMALIZATION + MISSING VALUE HANDLING
    # ---------------------------------------------------------

    def normalize_and_handle_missing(self, array):

        array = array.astype(np.float32, copy=True)

        # Create mask BEFORE replacing NaNs
        mask = self.create_missing_mask(array)

        if self.normalize and self.stats is not None:

            for channel in range(3):

                channel_data = array[:, channel, :, :]

                valid = np.isfinite(channel_data)

                mean = self.stats[channel]["mean"]
                std = self.stats[channel]["std"]

                if std == 0 or not np.isfinite(std):
                    std = 1.0

                # Normalize ONLY valid measurements
                channel_data[valid] = (
                    channel_data[valid] - mean
                ) / std

                # Replace missing / invalid values with zero
                channel_data[~valid] = 0.0

                array[:, channel, :, :] = channel_data

        else:

            # Even without normalization,
            # replace missing values with zero
            array[~np.isfinite(array)] = 0.0

        return array, mask

    # ---------------------------------------------------------
    # CONVERT
    #
    # [3 weeks, 3 variables, H, W]
    #
    # INTO
    #
    # [9 channels, H, W]
    # ---------------------------------------------------------

    def temporal_to_channels(self, array):

        weeks, variables, height, width = array.shape

        return array.reshape(
            weeks * variables,
            height,
            width
        )

    # ---------------------------------------------------------
    # GET ITEM
    # ---------------------------------------------------------

    def __getitem__(self, index):

        row = self.df.iloc[index]

        anchor_id = int(row["anchor_id"])

        week_1 = int(row["week_1"])
        week_2 = int(row["week_2"])
        week_3 = int(row["week_3"])

        target_week = int(row["target_week"])

        label = int(row["label"])

        augmentation = row["augmentation"]

        # -----------------------------------------------------
        # LOAD ORIGINAL 14-WEEK DATA
        # -----------------------------------------------------

        patch_64, patch_128, patch_256 = self.load_anchor(anchor_id)

        # -----------------------------------------------------
        # SELECT THE 3-WEEK TEMPORAL WINDOW
        # -----------------------------------------------------

        patch_64 = patch_64[
            [week_1 - 1, week_2 - 1, week_3 - 1]
        ]

        patch_128 = patch_128[
            [week_1 - 1, week_2 - 1, week_3 - 1]
        ]

        patch_256 = patch_256[
            [week_1 - 1, week_2 - 1, week_3 - 1]
        ]

        # -----------------------------------------------------
        # CREATE MASKS BEFORE AUGMENTATION
        # -----------------------------------------------------

        mask_64 = self.create_missing_mask(patch_64)
        mask_128 = self.create_missing_mask(patch_128)
        mask_256 = self.create_missing_mask(patch_256)

        # -----------------------------------------------------
        # APPLY THE SAME AUGMENTATION TO DATA AND MASK
        # -----------------------------------------------------

        patch_64 = self.augment(patch_64, augmentation)
        patch_128 = self.augment(patch_128, augmentation)
        patch_256 = self.augment(patch_256, augmentation)

        mask_64 = self.augment(mask_64, augmentation)
        mask_128 = self.augment(mask_128, augmentation)
        mask_256 = self.augment(mask_256, augmentation)

        # -----------------------------------------------------
        # NORMALIZE DATA + REPLACE MISSING VALUES
        # -----------------------------------------------------

        patch_64, _ = self.normalize_and_handle_missing(
            patch_64
        )

        patch_128, _ = self.normalize_and_handle_missing(
            patch_128
        )

        patch_256, _ = self.normalize_and_handle_missing(
            patch_256
        )

        # -----------------------------------------------------
        # CONVERT TEMPORAL DIMENSION INTO CHANNELS
        #
        # 3 weeks × 3 variables = 9 data channels
        # -----------------------------------------------------

        data_64 = self.temporal_to_channels(patch_64)
        data_128 = self.temporal_to_channels(patch_128)
        data_256 = self.temporal_to_channels(patch_256)

        # -----------------------------------------------------
        # CONVERT MASKS INTO 9 CHANNELS
        # -----------------------------------------------------

        mask_64 = self.temporal_to_channels(mask_64)
        mask_128 = self.temporal_to_channels(mask_128)
        mask_256 = self.temporal_to_channels(mask_256)

        # -----------------------------------------------------
        # CONCATENATE
        #
        # First 9 channels = normalized data
        # Last 9 channels  = missing-value masks
        #
        # Total = 18 channels
        # -----------------------------------------------------

        patch_64 = np.concatenate(
            [data_64, mask_64],
            axis=0
        )

        patch_128 = np.concatenate(
            [data_128, mask_128],
            axis=0
        )

        patch_256 = np.concatenate(
            [data_256, mask_256],
            axis=0
        )

        # -----------------------------------------------------
        # CONVERT TO PYTORCH TENSORS
        # -----------------------------------------------------

        patch_64 = torch.tensor(
            patch_64,
            dtype=torch.float32
        )

        patch_128 = torch.tensor(
            patch_128,
            dtype=torch.float32
        )

        patch_256 = torch.tensor(
            patch_256,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return {
            "patch_64": patch_64,
            "patch_128": patch_128,
            "patch_256": patch_256,
            "label": label,
            "anchor_id": anchor_id,
            "target_week": target_week,
            "augmentation": augmentation
        }


# =============================================================
# LOAD NORMALIZATION STATISTICS
# =============================================================

def load_normalization_stats():

    with open(STATS_PATH, "r") as f:
        stats = json.load(f)

    ordered_stats = [
        stats["VV"],
        stats["rainfall"],
        stats["slope"]
    ]

    return ordered_stats


# =============================================================
# BASIC TEST
# =============================================================

if __name__ == "__main__":

    train_csv = os.path.join(
        TEMPORAL_DIR,
        "train_balanced.csv"
    )

    stats = load_normalization_stats()

    dataset = EarthSentinelDataset(
        train_csv,
        normalize=True,
        stats=stats
    )

    print("Dataset size:", len(dataset))

    for index in [0, 1, 10]:

        sample = dataset[index]

        print("\nSample:", index)

        print(
            "Anchor:",
            sample["anchor_id"]
        )

        print(
            "Target week:",
            sample["target_week"]
        )

        print(
            "Augmentation:",
            sample["augmentation"]
        )

        print(
            "Label:",
            sample["label"].item()
        )

        print(
            "patch_64 shape:",
            tuple(sample["patch_64"].shape)
        )

        print(
            "patch_128 shape:",
            tuple(sample["patch_128"].shape)
        )

        print(
            "patch_256 shape:",
            tuple(sample["patch_256"].shape)
        )

        print(
            "patch_64 NaNs:",
            torch.isnan(sample["patch_64"]).any().item()
        )

        print(
            "patch_128 NaNs:",
            torch.isnan(sample["patch_128"]).any().item()
        )

        print(
            "patch_256 NaNs:",
            torch.isnan(sample["patch_256"]).any().item()
        )

        # Check mask values
        for name in ["patch_64", "patch_128", "patch_256"]:

            tensor = sample[name]

            mask = tensor[9:]

            unique_values = torch.unique(mask)

            print(
                name,
                "mask unique values:",
                unique_values.tolist()
            )

    print("\nDATASET TEST PASSED")