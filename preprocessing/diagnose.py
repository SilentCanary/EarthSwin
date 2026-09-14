import os
import numpy as np
import pandas as pd


PATCH_DIR = "C:/Users/advit/Documents/major project/dataset/EarthSentinel_2016/multiscale_patches"
TEMPORAL_DIR = "C:/Users/advit/Documents/major project/dataset/EarthSentinel_2016/temporal_dataset"


def analyze_csv(csv_name):

    csv_path = os.path.join(TEMPORAL_DIR, csv_name)
    df = pd.read_csv(csv_path)

    print("\n" + "=" * 70)
    print(csv_name)
    print("Rows:", len(df))
    print("=" * 70)

    scales = ["patch_64", "patch_128", "patch_256"]

    total_rows = len(df)

    completely_missing = {
        scale: 0
        for scale in scales
    }

    very_high_missing = {
        scale: 0
        for scale in scales
    }

    average_missing = {
        scale: []
        for scale in scales
    }

    for _, row in df.iterrows():

        anchor_id = int(row["anchor_id"])

        weeks = [
            int(row["week_1"]),
            int(row["week_2"]),
            int(row["week_3"])
        ]

        path = os.path.join(
            PATCH_DIR,
            f"sample_{anchor_id:06d}.npz"
        )

        data = np.load(path)

        for scale in scales:

            patch = data[scale]

            selected = patch[
                [w - 1 for w in weeks]
            ]

            missing_fraction = (
                np.sum(~np.isfinite(selected))
                / selected.size
            )

            average_missing[scale].append(
                missing_fraction
            )

            if missing_fraction == 1.0:
                completely_missing[scale] += 1

            if missing_fraction >= 0.95:
                very_high_missing[scale] += 1

    for scale in scales:

        fractions = np.array(
            average_missing[scale]
        )

        print(f"\n{scale}")

        print(
            "  Completely missing:",
            completely_missing[scale],
            f"({100 * completely_missing[scale] / total_rows:.2f}%)"
        )

        print(
            "  >=95% missing:",
            very_high_missing[scale],
            f"({100 * very_high_missing[scale] / total_rows:.2f}%)"
        )

        print(
            "  Average missing:",
            f"{100 * fractions.mean():.2f}%"
        )

        print(
            "  Median missing:",
            f"{100 * np.median(fractions):.2f}%"
        )

        print(
            "  Minimum missing:",
            f"{100 * fractions.min():.2f}%"
        )

        print(
            "  Maximum missing:",
            f"{100 * fractions.max():.2f}%"
        )


# ------------------------------------------------------------
# ANALYZE ALL SPLITS
# ------------------------------------------------------------

analyze_csv("train_balanced.csv")
analyze_csv("val.csv")
analyze_csv("test.csv")

print("\nDONE")