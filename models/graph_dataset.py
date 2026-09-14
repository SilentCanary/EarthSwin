import numpy as np
import torch
import pandas as pd

import sys

sys.path.append(
    "C:/Users/advit/Documents/major project/preprocessing"
)

from create_pytorch_dataset import EarthSentinelDataset, load_normalization_stats

TEMPORAL_DIR = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/temporal_dataset"
)


class GraphTrainingDataset:
    """
    Graph-aware loader built on top of the existing EarthSentinelDataset.

    EarthSentinelDataset handles:
        - patch loading
        - temporal selection
        - augmentation
        - normalization
        - missing-value masks
        - 18-channel construction

    This class handles:
        - split anchor selection
        - grouping balanced training samples
        - loading all spatial anchors for a graph pass
        - separating the 18 channels into 3 weekly inputs
        - mapping anchor IDs to graph node positions
    """

    def __init__(
        self,
        csv_path,
        split_anchor_csv,
        normalize=True
    ):
        self.dataset = EarthSentinelDataset(
            csv_path=csv_path,
            normalize=normalize,
            stats=load_normalization_stats()
        )

        split_df = pd.read_csv(split_anchor_csv)

        self.anchor_ids = sorted(
            split_df["anchor_id"]
            .astype(int)
            .unique()
            .tolist()
        )

        self.anchor_id_to_position = {
            anchor_id: position
            for position, anchor_id in enumerate(self.anchor_ids)
        }

        self.df = self.dataset.df

    def split_into_weeks(self, patch):
        """
        Convert:

            [N, 18, H, W]

        into:

            week_1 [N, 6, H, W]
            week_2 [N, 6, H, W]
            week_3 [N, 6, H, W]

        Existing channel layout:

            0:3    -> week 1 data
            3:6    -> week 2 data
            6:9    -> week 3 data

            9:12  -> week 1 missing mask
            12:15 -> week 2 missing mask
            15:18 -> week 3 missing mask
        """

        week_1 = torch.cat(
            [
                patch[:, 0:3],
                patch[:, 9:12]
            ],
            dim=1
        )

        week_2 = torch.cat(
            [
                patch[:, 3:6],
                patch[:, 12:15]
            ],
            dim=1
        )

        week_3 = torch.cat(
            [
                patch[:, 6:9],
                patch[:, 15:18]
            ],
            dim=1
        )

        return week_1, week_2, week_3

    def load_anchor(
        self,
        anchor_id,
        week_1,
        week_2,
        week_3,
        augmentation
    ):
        """
        Load and preprocess one anchor for one temporal window.

        Returns:

            patch_64  -> [18, 64, 64]
            patch_128 -> [18, 128, 128]
            patch_256 -> [18, 256, 256]
        """

        patch_64, patch_128, patch_256 = (
            self.dataset.load_anchor(anchor_id)
        )

        week_indices = [
            week_1 - 1,
            week_2 - 1,
            week_3 - 1
        ]

        patch_64 = patch_64[week_indices]
        patch_128 = patch_128[week_indices]
        patch_256 = patch_256[week_indices]

        # Create missing-value masks before replacing missing values.
        mask_64 = self.dataset.create_missing_mask(patch_64)
        mask_128 = self.dataset.create_missing_mask(patch_128)
        mask_256 = self.dataset.create_missing_mask(patch_256)

        # Apply exactly the same spatial transformation
        # to data and masks.
        patch_64 = self.dataset.augment(
            patch_64,
            augmentation
        )

        patch_128 = self.dataset.augment(
            patch_128,
            augmentation
        )

        patch_256 = self.dataset.augment(
            patch_256,
            augmentation
        )

        mask_64 = self.dataset.augment(
            mask_64,
            augmentation
        )

        mask_128 = self.dataset.augment(
            mask_128,
            augmentation
        )

        mask_256 = self.dataset.augment(
            mask_256,
            augmentation
        )

        # Normalize valid pixels and replace missing values with 0.
        patch_64, _ = self.dataset.normalize_and_handle_missing(
            patch_64
        )

        patch_128, _ = self.dataset.normalize_and_handle_missing(
            patch_128
        )

        patch_256, _ = self.dataset.normalize_and_handle_missing(
            patch_256
        )

        # Convert:
        # [3 weeks, 3 variables, H, W]
        #
        # into:
        # [9, H, W]
        data_64 = self.dataset.temporal_to_channels(patch_64)
        data_128 = self.dataset.temporal_to_channels(patch_128)
        data_256 = self.dataset.temporal_to_channels(patch_256)

        # Convert masks:
        # [3 weeks, 3 variables, H, W]
        #
        # into:
        # [9, H, W]
        mask_64 = self.dataset.temporal_to_channels(mask_64)
        mask_128 = self.dataset.temporal_to_channels(mask_128)
        mask_256 = self.dataset.temporal_to_channels(mask_256)

        # Final format:
        #
        # [data channels, mask channels]
        #
        # = [18, H, W]
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

        return (
            torch.from_numpy(patch_64).float(),
            torch.from_numpy(patch_128).float(),
            torch.from_numpy(patch_256).float()
        )

    def load_graph_inputs(
        self,
        week_1,
        week_2,
        week_3,
        augmentation="none"
    ):
        """
        Load every anchor in this split for one temporal window.

        IMPORTANT:
        All anchors are loaded because GAT needs the complete
        spatial graph.

        The returned tensors contain ALL graph nodes.

        Returns:

            week_1:
                patch_64  [N, 6, 64, 64]
                patch_128 [N, 6, 128, 128]
                patch_256 [N, 6, 256, 256]

            week_2:
                same shapes

            week_3:
                same shapes
        """

        patch_64_list = []
        patch_128_list = []
        patch_256_list = []

        total_anchors = len(self.anchor_ids)

        for position, anchor_id in enumerate(self.anchor_ids):

            patch_64, patch_128, patch_256 = self.load_anchor(
                anchor_id=anchor_id,
                week_1=week_1,
                week_2=week_2,
                week_3=week_3,
                augmentation=augmentation
            )

            patch_64_list.append(patch_64)
            patch_128_list.append(patch_128)
            patch_256_list.append(patch_256)

            if (position + 1) % 100 == 0:
                print(
                    f"Loaded {position + 1}/{total_anchors} "
                    f"anchors"
                )

        # [N, 18, H, W]
        patch_64 = torch.stack(
            patch_64_list,
            dim=0
        )

        patch_128 = torch.stack(
            patch_128_list,
            dim=0
        )

        patch_256 = torch.stack(
            patch_256_list,
            dim=0
        )

        # Split the 18 channels into three 6-channel weeks.
        week_1_64, week_2_64, week_3_64 = (
            self.split_into_weeks(patch_64)
        )

        week_1_128, week_2_128, week_3_128 = (
            self.split_into_weeks(patch_128)
        )

        week_1_256, week_2_256, week_3_256 = (
            self.split_into_weeks(patch_256)
        )

        week_1_inputs = {
            "patch_64": week_1_64,
            "patch_128": week_1_128,
            "patch_256": week_1_256
        }

        week_2_inputs = {
            "patch_64": week_2_64,
            "patch_128": week_2_128,
            "patch_256": week_2_256
        }

        week_3_inputs = {
            "patch_64": week_3_64,
            "patch_128": week_3_128,
            "patch_256": week_3_256
        }

        return (
            week_1_inputs,
            week_2_inputs,
            week_3_inputs
        )

    def get_training_groups(self):
        """
        Group the supervision CSV by:

            week_1
            week_2
            week_3
            target_week
            augmentation

        Each group represents one graph/model forward pass.

        The graph still contains ALL anchors in the split.

        Only the anchor_ids returned here are used to calculate
        the training loss.
        """

        group_columns = [
            "week_1",
            "week_2",
            "week_3",
            "target_week",
            "augmentation"
        ]

        groups = []

        grouped = self.df.groupby(
            group_columns,
            sort=True
        )

        for group_key, group_df in grouped:

            week_1 = int(group_key[0])
            week_2 = int(group_key[1])
            week_3 = int(group_key[2])
            target_week = int(group_key[3])
            augmentation = group_key[4]

            anchor_ids = (
                group_df["anchor_id"]
                .astype(int)
                .tolist()
            )

            labels = (
                group_df["label"]
                .astype(int)
                .tolist()
            )

            groups.append(
                {
                    "week_1": week_1,
                    "week_2": week_2,
                    "week_3": week_3,
                    "target_week": target_week,
                    "augmentation": augmentation,
                    "anchor_ids": anchor_ids,
                    "labels": labels
                }
            )

        return groups

    def get_node_positions(self, anchor_ids):
        """
        Convert anchor IDs into positions in the graph tensors.

        Example:

            graph anchor IDs:
                [2, 17, 41, 105, ...]

            supervised anchor:
                41

            returns:
                position = 2
        """

        positions = []

        for anchor_id in anchor_ids:

            anchor_id = int(anchor_id)

            if anchor_id not in self.anchor_id_to_position:
                raise ValueError(
                    f"Anchor ID {anchor_id} is not present "
                    f"in this split."
                )

            positions.append(
                self.anchor_id_to_position[anchor_id]
            )

        return positions