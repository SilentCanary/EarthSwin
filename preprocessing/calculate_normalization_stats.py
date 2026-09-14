import os
import json
import numpy as np
import pandas as pd

PATCH_DIR = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/multiscale_patches"
)

TEMPORAL_DIR = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/temporal_dataset"
)

OUTPUT_PATH = os.path.join(
    TEMPORAL_DIR,
    "normalization_stats.json"
)


def calculate_stats():

    train_csv = os.path.join(
        TEMPORAL_DIR,
        "train_original.csv"
    )

    df = pd.read_csv(train_csv)

    print("=" * 60)
    print("CALCULATING TRAINING NORMALIZATION STATISTICS")
    print("=" * 60)

    print(f"Training samples : {len(df)}")

    # We only need unique anchors.
    # train_original.csv contains 12 temporal windows
    # per anchor, so avoid loading the same anchor 12 times.
    anchor_ids = sorted(
        df["anchor_id"].astype(int).unique()
    )

    print(f"Unique train anchors : {len(anchor_ids)}")

    # Channel order:
    # 0 = VV
    # 1 = rainfall
    # 2 = slope

    sums = np.zeros(3, dtype=np.float64)
    squared_sums = np.zeros(3, dtype=np.float64)
    counts = np.zeros(3, dtype=np.int64)

    for i, anchor_id in enumerate(anchor_ids):

        filename = (
            f"sample_{anchor_id:06d}.npz"
        )

        path = os.path.join(
            PATCH_DIR,
            filename
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Patch file not found:\n{path}"
            )

        with np.load(path) as data:

            # Use 256x256 only.
            patch = data["patch_256"]

        # Shape:
        # [14 weeks, 3 channels, 256, 256]

        patch = patch.astype(
            np.float64,
            copy=False
        )

        for channel in range(3):

            values = patch[:, channel, :, :]

            valid = np.isfinite(values)

            valid_values = values[valid]

            if len(valid_values) == 0:
                continue

            sums[channel] += np.sum(
                valid_values
            )

            squared_sums[channel] += np.sum(
                valid_values ** 2
            )

            counts[channel] += len(
                valid_values
            )

        if (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1}/"
                f"{len(anchor_ids)} anchors"
            )

    means = sums / counts

    variances = (
        squared_sums / counts
        - means ** 2
    )

    # Floating-point errors can occasionally
    # produce something like -1e-15.
    variances = np.maximum(
        variances,
        0.0
    )

    stds = np.sqrt(variances)

    stats = {
        "VV": {
            "mean": float(means[0]),
            "std": float(stds[0]),
            "count": int(counts[0]),
        },
        "rainfall": {
            "mean": float(means[1]),
            "std": float(stds[1]),
            "count": int(counts[1]),
        },
        "slope": {
            "mean": float(means[2]),
            "std": float(stds[2]),
            "count": int(counts[2]),
        },
    }

    with open(
        OUTPUT_PATH,
        "w"
    ) as f:

        json.dump(
            stats,
            f,
            indent=4
        )

    print()
    print("=" * 60)
    print("NORMALIZATION STATISTICS")
    print("=" * 60)

    for name in [
        "VV",
        "rainfall",
        "slope"
    ]:

        print()
        print(name)

        print(
            f"Mean  : "
            f"{stats[name]['mean']:.6f}"
        )

        print(
            f"Std   : "
            f"{stats[name]['std']:.6f}"
        )

        print(
            f"Count : "
            f"{stats[name]['count']:,}"
        )

    print()
    print(
        f"Saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    calculate_stats()