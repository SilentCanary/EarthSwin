import os
import torch
from torch.utils.data import DataLoader

from create_pytorch_dataset import (
    EarthSentinelDataset,
    TEMPORAL_DIR,
    load_normalization_stats,
)


if __name__ == "__main__":

    train_csv = os.path.join(
        TEMPORAL_DIR,
        "train_balanced.csv"
    )

    # Load the training-only normalization statistics
    stats = load_normalization_stats()

    # Create dataset
    dataset = EarthSentinelDataset(
        csv_path=train_csv,
        normalize=True,
        stats=stats,
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    print()
    print("=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    print(
        f"Dataset size : {len(dataset)}"
    )

    print(
        "Batch size   : 4"
    )

    # ---------------------------------------------------------
    # GET ONE BATCH
    # ---------------------------------------------------------

    batch = next(iter(dataloader))

    # ---------------------------------------------------------
    # BATCH SHAPES
    # ---------------------------------------------------------

    print()
    print("BATCH SHAPES")
    print("-" * 60)

    print(
        f"patch_64  : "
        f"{tuple(batch['patch_64'].shape)}"
    )

    print(
        f"patch_128 : "
        f"{tuple(batch['patch_128'].shape)}"
    )

    print(
        f"patch_256 : "
        f"{tuple(batch['patch_256'].shape)}"
    )

    print(
        f"labels    : "
        f"{tuple(batch['label'].shape)}"
    )

    # ---------------------------------------------------------
    # BATCH DETAILS
    # ---------------------------------------------------------

    print()
    print("BATCH DETAILS")
    print("-" * 60)

    print(
        f"Labels    : "
        f"{batch['label'].tolist()}"
    )

    print(
        f"Anchors   : "
        f"{batch['anchor_id'].tolist()}"
    )

    print(
        f"Weeks     : "
        f"{batch['target_week'].tolist()}"
    )

    print(
        f"64x64 dtype   : "
        f"{batch['patch_64'].dtype}"
    )

    print(
        f"128x128 dtype : "
        f"{batch['patch_128'].dtype}"
    )

    print(
        f"256x256 dtype : "
        f"{batch['patch_256'].dtype}"
    )

    # ---------------------------------------------------------
    # CHECKS
    # ---------------------------------------------------------

    print()
    print("CHECKS")
    print("-" * 60)

    checks = {

        # -----------------------------------------------------
        # BATCH SHAPES
        # -----------------------------------------------------

        "64x64 shape":
            batch["patch_64"].shape
            == (4, 18, 64, 64),

        "128x128 shape":
            batch["patch_128"].shape
            == (4, 18, 128, 128),

        "256x256 shape":
            batch["patch_256"].shape
            == (4, 18, 256, 256),

        "label shape":
            batch["label"].shape
            == (4,),

        # -----------------------------------------------------
        # NO NaNs
        # -----------------------------------------------------

        "64x64 no NaNs":
            not torch.isnan(
                batch["patch_64"]
            ).any(),

        "128x128 no NaNs":
            not torch.isnan(
                batch["patch_128"]
            ).any(),

        "256x256 no NaNs":
            not torch.isnan(
                batch["patch_256"]
            ).any(),

        # -----------------------------------------------------
        # DATA CHANNELS
        #
        # First 9 channels = actual data
        # -----------------------------------------------------

        "64x64 data channels":
            batch["patch_64"][:, :9].shape[1] == 9,

        "128x128 data channels":
            batch["patch_128"][:, :9].shape[1] == 9,

        "256x256 data channels":
            batch["patch_256"][:, :9].shape[1] == 9,

        # -----------------------------------------------------
        # MASK CHANNELS
        #
        # Last 9 channels = missing-value masks
        # -----------------------------------------------------

        "64x64 mask channels":
            batch["patch_64"][:, 9:].shape[1] == 9,

        "128x128 mask channels":
            batch["patch_128"][:, 9:].shape[1] == 9,

        "256x256 mask channels":
            batch["patch_256"][:, 9:].shape[1] == 9,

        # -----------------------------------------------------
        # MASK VALUES
        #
        # 0 = valid
        # 1 = missing
        # -----------------------------------------------------

        "64x64 mask valid":
            torch.all(
                (batch["patch_64"][:, 9:] == 0)
                | (batch["patch_64"][:, 9:] == 1)
            ),

        "128x128 mask valid":
            torch.all(
                (batch["patch_128"][:, 9:] == 0)
                | (batch["patch_128"][:, 9:] == 1)
            ),

        "256x256 mask valid":
            torch.all(
                (batch["patch_256"][:, 9:] == 0)
                | (batch["patch_256"][:, 9:] == 1)
            ),
    }

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    all_passed = True

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(
            f"{name:<25} : {status}"
        )

        if not passed:
            all_passed = False

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    print()

    if all_passed:

        print("=" * 60)
        print("DATALOADER TEST PASSED")
        print("=" * 60)

    else:

        print("=" * 60)
        print("DATALOADER TEST FAILED")
        print("=" * 60)