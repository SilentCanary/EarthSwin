import os
import glob
import json
import random
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_DIR = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/patch_chunks"
)

LABELS_PATH = "labels.npy"

OUTPUT_DIR = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/temporal_dataset"
)

NUM_WEEKS = 14
WINDOW_SIZE = 3

# Training balance
# Approximately 1 positive : 3 negatives
NEGATIVE_TO_POSITIVE_RATIO = 3

# Number of augmentation variants for every
# genuine positive training sequence.
#
# 0 = original
# 1 = horizontal flip
# 2 = vertical flip
# 3 = rotate 90
# 4 = rotate 180
# 5 = rotate 270
# 6 = horizontal + vertical flip
# 7 = transpose
AUGMENTATIONS = [
    "none",
    "horizontal_flip",
    "vertical_flip",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "horizontal_vertical_flip",
    "transpose",
]

RANDOM_SEED = 42


# ============================================================
# POSITIVE EVENTS
# ============================================================

# These are the six genuine landslide anchor-week events
# already verified against the retained 2631 anchors.

POSITIVE_EVENTS = {
    (2401, 9),
    (2565, 13),
    (2719, 3),
    (2720, 7),
    (3658, 5),
    (3962, 8),
}


# ============================================================
# HELPER
# ============================================================

def get_target_date(week_number):
    """
    Week 1 starts on 2016-06-01.
    """
    return (
        pd.Timestamp("2016-06-01")
        + pd.Timedelta(days=7 * (week_number - 1))
    ).strftime("%Y-%m-%d")


# ============================================================
# LOAD RETAINED ANCHORS FROM CHUNKS
# ============================================================

def load_anchor_index():
    """
    Reads only anchor_id from each chunk.

    We do NOT load the huge patch arrays here.

    Returns:
        anchor_index:
            {
                anchor_id: {
                    "chunk_file": "...",
                    "row": row_index
                }
            }
    """

    chunk_files = sorted(
        glob.glob(os.path.join(CHUNK_DIR, "*.npz"))
    )

    if len(chunk_files) == 0:
        raise FileNotFoundError(
            f"No chunk files found in:\n{CHUNK_DIR}"
        )

    print()
    print("=" * 60)
    print("BUILDING ANCHOR INDEX")
    print("=" * 60)

    anchor_index = {}

    for chunk_file in chunk_files:

        with np.load(chunk_file) as data:
            anchor_ids = data["anchor_id"]

        for row, anchor_id in enumerate(anchor_ids):

            anchor_id = int(anchor_id)

            if anchor_id in anchor_index:
                raise ValueError(
                    f"Duplicate anchor ID found: {anchor_id}"
                )

            anchor_index[anchor_id] = {
                "chunk_file": chunk_file,
                "row": row,
            }

    print(f"Chunks found       : {len(chunk_files)}")
    print(f"Retained anchors   : {len(anchor_index)}")

    return anchor_index


# ============================================================
# CREATE ALL TEMPORAL SEQUENCES
# ============================================================

def create_temporal_manifest(anchor_index, labels):
    """
    Creates one row for every:

        anchor + target week

    using:

        [target_week - 2,
         target_week - 1,
         target_week]

    as the temporal input.

    Target weeks therefore range from 3 to 14.
    """

    records = []

    for anchor_id in sorted(anchor_index.keys()):

        if anchor_id >= labels.shape[0]:
            raise ValueError(
                f"Anchor ID {anchor_id} is outside labels array."
            )

        for target_week in range(
            WINDOW_SIZE,
            NUM_WEEKS + 1
        ):

            target_index = target_week - 1

            label = int(
                labels[anchor_id, target_index]
            )

            input_weeks = [
                target_week - 2,
                target_week - 1,
                target_week,
            ]

            records.append({
                "anchor_id": anchor_id,
                "chunk_file": anchor_index[anchor_id]["chunk_file"],
                "chunk_row": anchor_index[anchor_id]["row"],

                "week_1": input_weeks[0],
                "week_2": input_weeks[1],
                "week_3": input_weeks[2],

                "target_week": target_week,
                "target_date": get_target_date(target_week),

                "label": label,

                "is_genuine_positive": (
                    (anchor_id, target_week)
                    in POSITIVE_EVENTS
                ),

                "augmentation": "none",
            })

    return pd.DataFrame(records)


# ============================================================
# SPLIT ANCHORS
# ============================================================

def create_splits(anchor_ids):
    """
    Split at ANCHOR level.

    All temporal windows belonging to one anchor
    remain in the same split.

    The two adjacent positive anchors 2719 and 2720
    are deliberately kept together.
    """

    rng = random.Random(RANDOM_SEED)

    positive_anchor_ids = sorted(
        {
            anchor_id
            for anchor_id, week in POSITIVE_EVENTS
        }
    )

    # 2719 and 2720 are adjacent spatial anchors
    # and therefore should not be separated.
    positive_groups = [
        [2719, 2720],
        [2401],
        [2565],
        [3658],
        [3962],
    ]

    print()
    print("=" * 60)
    print("POSITIVE EVENT GROUPS")
    print("=" * 60)

    for group in positive_groups:
        print(group)

    # Deliberately assign:
    #
    # TRAIN = 3 positive groups
    # VAL   = 1 positive group
    # TEST  = 1 positive group
    #
    # This gives training access to 4 positive events
    # because the first group contains two events.

    rng.shuffle(positive_groups)

    train_groups = positive_groups[:3]
    val_groups = positive_groups[3:4]
    test_groups = positive_groups[4:]

    train_positive = [
        anchor
        for group in train_groups
        for anchor in group
    ]

    val_positive = [
        anchor
        for group in val_groups
        for anchor in group
    ]

    test_positive = [
        anchor
        for group in test_groups
        for anchor in group
    ]

    remaining = [
        anchor
        for anchor in anchor_ids
        if anchor not in positive_anchor_ids
    ]

    rng.shuffle(remaining)

    # Approximately 80 / 10 / 10
    n = len(remaining)

    n_train = int(0.80 * n)
    n_val = int(0.10 * n)

    train_ids = set(
        train_positive + remaining[:n_train]
    )

    val_ids = set(
        val_positive
        + remaining[n_train:n_train + n_val]
    )

    test_ids = set(
        test_positive
        + remaining[n_train + n_val:]
    )

    # Safety check
    if train_ids & val_ids:
        raise ValueError("Train/validation overlap detected.")

    if train_ids & test_ids:
        raise ValueError("Train/test overlap detected.")

    if val_ids & test_ids:
        raise ValueError("Validation/test overlap detected.")

    if len(
        train_ids | val_ids | test_ids
    ) != len(anchor_ids):

        raise ValueError(
            "Some anchors were not assigned to a split."
        )

    return train_ids, val_ids, test_ids


# ============================================================
# APPLY SPLITS
# ============================================================

def assign_splits(manifest, train_ids, val_ids, test_ids):

    def get_split(anchor_id):

        if anchor_id in train_ids:
            return "train"

        if anchor_id in val_ids:
            return "val"

        if anchor_id in test_ids:
            return "test"

        raise ValueError(
            f"Anchor {anchor_id} has no split."
        )

    manifest["split"] = manifest[
        "anchor_id"
    ].apply(get_split)

    return manifest


# ============================================================
# CREATE BALANCED TRAINING MANIFEST
# ============================================================

def create_balanced_training_manifest(train_manifest):

    positive = train_manifest[
        train_manifest["label"] == 1
    ].copy()

    negative = train_manifest[
        train_manifest["label"] == 0
    ].copy()

    print()
    print("=" * 60)
    print("TRAINING DATA BEFORE BALANCING")
    print("=" * 60)

    print(f"Positive sequences : {len(positive)}")
    print(f"Negative sequences : {len(negative)}")

    if len(positive) == 0:
        raise ValueError(
            "No positive training sequences found."
        )

    rng = np.random.default_rng(RANDOM_SEED)

    # --------------------------------------------------------
    # Positive augmentation
    # --------------------------------------------------------

    positive_augmented = []

    for _, row in positive.iterrows():

        for augmentation in AUGMENTATIONS:

            new_row = row.copy()

            new_row["augmentation"] = augmentation

            positive_augmented.append(new_row)

    positive_augmented = pd.DataFrame(
        positive_augmented
    )

    # --------------------------------------------------------
    # Negative sampling
    # --------------------------------------------------------

    desired_negative_count = (
        len(positive_augmented)
        * NEGATIVE_TO_POSITIVE_RATIO
    )

    desired_negative_count = min(
        desired_negative_count,
        len(negative)
    )

    selected_indices = rng.choice(
        len(negative),
        size=desired_negative_count,
        replace=False,
    )

    negative_selected = negative.iloc[
        selected_indices
    ].copy()

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    balanced = pd.concat(
        [
            positive_augmented,
            negative_selected,
        ],
        ignore_index=True,
    )

    # Shuffle final training manifest
    balanced = balanced.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    return balanced


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    manifest,
    train_manifest,
    val_manifest,
    test_manifest,
    balanced_train,
):

    print()
    print("=" * 60)
    print("FINAL TEMPORAL DATASET")
    print("=" * 60)

    print(
        f"Total sequences : {len(manifest)}"
    )

    print(
        f"Positive        : "
        f"{(manifest['label'] == 1).sum()}"
    )

    print(
        f"Negative        : "
        f"{(manifest['label'] == 0).sum()}"
    )

    print()
    print("ANCHOR SPLITS")
    print("-" * 60)

    print(
        f"Train anchors : "
        f"{train_manifest['anchor_id'].nunique()}"
    )

    print(
        f"Val anchors   : "
        f"{val_manifest['anchor_id'].nunique()}"
    )

    print(
        f"Test anchors  : "
        f"{test_manifest['anchor_id'].nunique()}"
    )

    print()
    print("ORIGINAL SEQUENCES BY SPLIT")
    print("-" * 60)

    for name, df in [
        ("TRAIN", train_manifest),
        ("VAL", val_manifest),
        ("TEST", test_manifest),
    ]:

        positives = int(
            (df["label"] == 1).sum()
        )

        negatives = int(
            (df["label"] == 0).sum()
        )

        print(
            f"{name:5s} | "
            f"total={len(df):5d} | "
            f"positive={positives:2d} | "
            f"negative={negatives:5d}"
        )

    print()
    print("BALANCED TRAINING MANIFEST")
    print("-" * 60)

    positives = int(
        (balanced_train["label"] == 1).sum()
    )

    negatives = int(
        (balanced_train["label"] == 0).sum()
    )

    print(
        f"Total      : {len(balanced_train)}"
    )

    print(
        f"Positive   : {positives}"
    )

    print(
        f"Negative   : {negatives}"
    )

    print(
        f"Ratio      : 1 : "
        f"{negatives / positives:.2f}"
    )

    print()
    print("Positive events by split:")
    print("-" * 60)

    for name, df in [
        ("TRAIN", train_manifest),
        ("VAL", val_manifest),
        ("TEST", test_manifest),
    ]:

        positive_rows = df[
            df["label"] == 1
        ][
            [
                "anchor_id",
                "target_week",
                "target_date",
            ]
        ]

        print()
        print(name)

        if len(positive_rows) == 0:
            print("  NONE")
        else:
            for _, row in positive_rows.iterrows():
                print(
                    f"  Anchor {int(row['anchor_id']):4d} | "
                    f"Week {int(row['target_week']):2d} | "
                    f"{row['target_date']}"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 60)
    print("EARTHSENTINEL TEMPORAL DATASET PREPARATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    labels = np.load(LABELS_PATH)

    print()
    print(f"Labels shape : {labels.shape}")

    if labels.shape != (4440, 14):
        raise ValueError(
            f"Unexpected labels shape: {labels.shape}"
        )

    # --------------------------------------------------------
    # Load retained anchor index
    # --------------------------------------------------------

    anchor_index = load_anchor_index()

    retained_anchor_ids = sorted(
        anchor_index.keys()
    )

    # --------------------------------------------------------
    # Verify positive events
    # --------------------------------------------------------

    retained_positive_events = []

    for anchor_id, week in POSITIVE_EVENTS:

        if anchor_id not in anchor_index:
            raise ValueError(
                f"Positive anchor {anchor_id} "
                f"is not in retained dataset."
            )

        if labels[anchor_id, week - 1] != 1:
            raise ValueError(
                f"Expected positive label missing: "
                f"anchor {anchor_id}, week {week}"
            )

        retained_positive_events.append(
            (anchor_id, week)
        )

    print()
    print("=" * 60)
    print("POSITIVE LABEL VERIFICATION")
    print("=" * 60)

    print(
        f"Positive events verified : "
        f"{len(retained_positive_events)}"
    )

    for anchor_id, week in retained_positive_events:
        print(
            f"  Anchor {anchor_id:4d} | "
            f"Week {week:2d}"
        )

    # --------------------------------------------------------
    # Create temporal sequences
    # --------------------------------------------------------

    manifest = create_temporal_manifest(
        anchor_index,
        labels,
    )

    expected_sequences = (
        len(retained_anchor_ids)
        * (NUM_WEEKS - WINDOW_SIZE + 1)
    )

    if len(manifest) != expected_sequences:
        raise ValueError(
            f"Expected {expected_sequences} sequences, "
            f"got {len(manifest)}"
        )

    # --------------------------------------------------------
    # Split anchors
    # --------------------------------------------------------

    train_ids, val_ids, test_ids = create_splits(
        retained_anchor_ids
    )

    manifest = assign_splits(
        manifest,
        train_ids,
        val_ids,
        test_ids,
    )

    # --------------------------------------------------------
    # Create split manifests
    # --------------------------------------------------------

    train_manifest = manifest[
        manifest["split"] == "train"
    ].copy()

    val_manifest = manifest[
        manifest["split"] == "val"
    ].copy()

    test_manifest = manifest[
        manifest["split"] == "test"
    ].copy()

    # --------------------------------------------------------
    # Balance training set
    # --------------------------------------------------------

    balanced_train = (
        create_balanced_training_manifest(
            train_manifest
        )
    )

    # --------------------------------------------------------
    # Save manifests
    # --------------------------------------------------------

    manifest.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "all_temporal_sequences.csv",
        ),
        index=False,
    )

    train_manifest.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "train_original.csv",
        ),
        index=False,
    )

    val_manifest.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "val.csv",
        ),
        index=False,
    )

    test_manifest.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test.csv",
        ),
        index=False,
    )

    balanced_train.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "train_balanced.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # Save anchor split information
    # --------------------------------------------------------

    split_info = {
        "random_seed": RANDOM_SEED,

        "num_retained_anchors": len(
            retained_anchor_ids
        ),

        "num_temporal_sequences": len(
            manifest
        ),

        "window_size": WINDOW_SIZE,

        "weeks": NUM_WEEKS,

        "positive_events": [
            {
                "anchor_id": int(anchor),
                "week": int(week),
            }
            for anchor, week in POSITIVE_EVENTS
        ],

        "train_anchors": sorted(
            int(x) for x in train_ids
        ),

        "val_anchors": sorted(
            int(x) for x in val_ids
        ),

        "test_anchors": sorted(
            int(x) for x in test_ids
        ),

        "augmentation_types": AUGMENTATIONS,

        "negative_to_positive_ratio":
            NEGATIVE_TO_POSITIVE_RATIO,
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "split_info.json",
        ),
        "w",
    ) as f:

        json.dump(
            split_info,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_summary(
        manifest,
        train_manifest,
        val_manifest,
        test_manifest,
        balanced_train,
    )

    print()
    print("=" * 60)
    print("FILES CREATED")
    print("=" * 60)

    print(
        "all_temporal_sequences.csv"
    )

    print(
        "train_original.csv"
    )

    print(
        "train_balanced.csv"
    )

    print(
        "val.csv"
    )

    print(
        "test.csv"
    )

    print(
        "split_info.json"
    )

    print()
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()