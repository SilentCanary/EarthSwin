import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from earthswin_model import EarthSwin
from graph import DynamicGraphConstructor


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_PATH = "C:/Users/advit/Documents/major project/dataset/EarthSentinel_2016"

CHUNK_PATH = os.path.join(
    BASE_PATH,
    "patch_chunks"
)

TEMPORAL_PATH = os.path.join(
    BASE_PATH,
    "temporal_dataset"
)

NORMALIZATION_PATH = os.path.join(
    TEMPORAL_PATH,
    "normalization_stats.json"
)


# --------------------------------------------------
# Settings
# --------------------------------------------------

NUM_ANCHORS = 9
K = 8

device = torch.device("cpu")

print("Device:", device)


# --------------------------------------------------
# Load temporal CSV
# --------------------------------------------------

train_csv = os.path.join(
    TEMPORAL_PATH,
    "train_original.csv"
)

df = pd.read_csv(train_csv)

print("Total training rows:", len(df))


# We only use original, non-augmented samples
df = df[
    df["augmentation"].isna() |
    (df["augmentation"] == "none")
].copy()


# Pick one target week that has enough rows
target_week = 9

window_df = df[
    df["target_week"] == target_week
].copy()


# Get 9 unique anchors
anchor_ids = window_df["anchor_id"].unique()[:NUM_ANCHORS]

window_df = window_df[
    window_df["anchor_id"].isin(anchor_ids)
].copy()

print("Selected anchors:", anchor_ids)

print("\nSelected temporal rows:")
print(
    window_df[
        [
            "anchor_id",
            "week_1",
            "week_2",
            "week_3",
            "target_week",
            "label",
            "augmentation"
        ]
    ]
)


# --------------------------------------------------
# Load normalization statistics
# --------------------------------------------------

import json

with open(NORMALIZATION_PATH, "r") as f:
    normalization_stats = json.load(f)

print("\nNormalization statistics loaded.")


# --------------------------------------------------
# Find chunk containing an anchor
# --------------------------------------------------

chunk_files = sorted(
    [
        f
        for f in os.listdir(CHUNK_PATH)
        if f.endswith(".npz")
    ]
)

print("Number of chunk files:", len(chunk_files))


# --------------------------------------------------
# Load all required anchors
# --------------------------------------------------

required_anchors = set(
    int(anchor_id)
    for anchor_id in anchor_ids
)

anchor_data = {}


for chunk_file in chunk_files:

    chunk_file_path = os.path.join(
        CHUNK_PATH,
        chunk_file
    )

    chunk = np.load(
        chunk_file_path
    )

    # Check available arrays
    chunk_anchor_ids = chunk["anchor_id"]

    for local_index, anchor_id in enumerate(chunk_anchor_ids):

        anchor_id = int(anchor_id)

        if anchor_id not in required_anchors:
            continue

        anchor_data[anchor_id] = {
            "patch_64": chunk["patch_64"][local_index],
            "patch_128": chunk["patch_128"][local_index],
            "patch_256": chunk["patch_256"][local_index],
            "center_row": chunk["center_row"][local_index],
            "center_col": chunk["center_col"][local_index]
        }

    if len(anchor_data) == len(required_anchors):
        break


print(
    "\nLoaded anchors:",
    len(anchor_data),
    "/",
    len(required_anchors)
)


if len(anchor_data) != len(required_anchors):
    missing = required_anchors - set(anchor_data.keys())

    raise RuntimeError(
        f"Could not load anchors: {missing}"
    )


# --------------------------------------------------
# Normalization helper
# --------------------------------------------------

channel_names = [
    "VV",
    "rainfall",
    "slope"
]


def get_mean_std(channel_name):

    mean = normalization_stats[channel_name]["mean"]
    std = normalization_stats[channel_name]["std"]

    return mean, std


# --------------------------------------------------
# Convert raw 3-channel patch → 6-channel patch
#
# Channels:
#
# 0 = VV
# 1 = rainfall
# 2 = slope
# 3 = VV mask
# 4 = rainfall mask
# 5 = slope mask
# --------------------------------------------------

def prepare_patch(raw_patch):

    raw_patch = raw_patch.astype(
        np.float32
    )

    output = np.zeros(
        (
            6,
            raw_patch.shape[1],
            raw_patch.shape[2]
        ),
        dtype=np.float32
    )

    for channel in range(3):

        values = raw_patch[channel]

        valid_mask = np.isfinite(values)

        output[channel + 3] = valid_mask.astype(
            np.float32
        )

        channel_name = channel_names[channel]

        mean, std = get_mean_std(
            channel_name
        )

        normalized = np.zeros_like(
            values,
            dtype=np.float32
        )

        normalized[valid_mask] = (
            values[valid_mask] - mean
        ) / std

        output[channel] = normalized

    return output


# --------------------------------------------------
# Prepare one week for all anchors
# --------------------------------------------------

def prepare_week(anchor_ids, week_number):

    patches_64 = []
    patches_128 = []
    patches_256 = []

    environmental_features = []
    coordinates = []

    for anchor_id in anchor_ids:

        data = anchor_data[int(anchor_id)]

        raw_64 = data["patch_64"][
            week_number - 1
        ]

        raw_128 = data["patch_128"][
            week_number - 1
        ]

        raw_256 = data["patch_256"][
            week_number - 1
        ]

        patch_64 = prepare_patch(
            raw_64
        )

        patch_128 = prepare_patch(
            raw_128
        )

        patch_256 = prepare_patch(
            raw_256
        )

        patches_64.append(
            patch_64
        )

        patches_128.append(
            patch_128
        )

        patches_256.append(
            patch_256
        )

        # ------------------------------------------
        # Graph environmental features
        # ------------------------------------------

        raw_environment = raw_256

        environment_values = []

        for channel in range(3):

            values = raw_environment[channel]

            finite_values = values[
                np.isfinite(values)
            ]

            if len(finite_values) == 0:
                value = 0.0
            else:
                value = float(
                    np.mean(finite_values)
                )

            environment_values.append(
                value
            )

        environmental_features.append(
            environment_values
        )

        # ------------------------------------------
        # Anchor coordinates
        # ------------------------------------------

        coordinates.append(
            [
                float(data["center_row"]),
                float(data["center_col"])
            ]
        )

    return {
        "patch_64": torch.tensor(
            np.stack(patches_64),
            dtype=torch.float32,
            device=device
        ),

        "patch_128": torch.tensor(
            np.stack(patches_128),
            dtype=torch.float32,
            device=device
        ),

        "patch_256": torch.tensor(
            np.stack(patches_256),
            dtype=torch.float32,
            device=device
        ),

        "environmental_features": torch.tensor(
            np.array(environmental_features),
            dtype=torch.float32,
            device=device
        ),

        "coordinates": torch.tensor(
            np.array(coordinates),
            dtype=torch.float32,
            device=device
        )
    }


# --------------------------------------------------
# Get the three weeks
# --------------------------------------------------

first_row = window_df.iloc[0]

week_1 = int(first_row["week_1"])
week_2 = int(first_row["week_2"])
week_3 = int(first_row["week_3"])

print(
    "\nTemporal window:",
    week_1,
    "→",
    week_2,
    "→",
    week_3,
    "→ target",
    target_week
)


week_1_data = prepare_week(
    anchor_ids,
    week_1
)

week_2_data = prepare_week(
    anchor_ids,
    week_2
)

week_3_data = prepare_week(
    anchor_ids,
    week_3
)


# --------------------------------------------------
# Build graphs for each week
# --------------------------------------------------

graph_constructor = DynamicGraphConstructor(
    k=K
)


def build_graph(week_data):

    edge_index, edge_weights = graph_constructor(
        week_data["coordinates"],
        week_data["environmental_features"]
    )

    return edge_index, edge_weights


edge_index_1, edge_weights_1 = build_graph(
    week_1_data
)

edge_index_2, edge_weights_2 = build_graph(
    week_2_data
)

edge_index_3, edge_weights_3 = build_graph(
    week_3_data
)


print("\nGraph shapes:")

print(
    "Week 1 edge_index:",
    edge_index_1.shape
)

print(
    "Week 1 edge_weights:",
    edge_weights_1.shape
)

print(
    "Week 2 edge_index:",
    edge_index_2.shape
)

print(
    "Week 3 edge_index:",
    edge_index_3.shape
)


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = EarthSwin(
    in_channels=6,
    feature_dim=512,
    num_heads=8,
    temporal_layers=2,
    temporal_feedforward_dim=1024,
    dropout=0.1
).to(device)


print("\nEarthSwin model created.")


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

model.train()

print("\nStarting forward pass...")
print("This may take some time on CPU.")


logits, probabilities = model(
    {
        "patch_64": week_1_data["patch_64"],
        "patch_128": week_1_data["patch_128"],
        "patch_256": week_1_data["patch_256"]
    },

    {
        "patch_64": week_2_data["patch_64"],
        "patch_128": week_2_data["patch_128"],
        "patch_256": week_2_data["patch_256"]
    },

    {
        "patch_64": week_3_data["patch_64"],
        "patch_128": week_3_data["patch_128"],
        "patch_256": week_3_data["patch_256"]
    },

    edge_index_1,
    edge_weights_1,

    edge_index_2,
    edge_weights_2,

    edge_index_3,
    edge_weights_3
)


# --------------------------------------------------
# Check output
# --------------------------------------------------

print("\n========== MODEL OUTPUT ==========")

print(
    "Logits shape:",
    logits.shape
)

print(
    "Probability shape:",
    probabilities.shape
)

print(
    "Probabilities:\n",
    probabilities.detach().cpu().numpy().flatten()
)

print(
    "NaNs:",
    torch.isnan(probabilities).sum().item()
)

print(
    "Infs:",
    torch.isinf(probabilities).sum().item()
)

print(
    "Minimum probability:",
    probabilities.min().item()
)

print(
    "Maximum probability:",
    probabilities.max().item()
)


# --------------------------------------------------
# Check labels
# --------------------------------------------------

labels = []

for anchor_id in anchor_ids:

    row = window_df[
        window_df["anchor_id"] == anchor_id
    ].iloc[0]

    labels.append(
        float(row["label"])
    )


labels = torch.tensor(
    labels,
    dtype=torch.float32,
    device=device
).unsqueeze(1)


print(
    "\nLabels:",
    labels.detach().cpu().numpy().flatten()
)


# --------------------------------------------------
# Loss
# --------------------------------------------------

criterion = nn.BCEWithLogitsLoss()

loss = criterion(
    logits,
    labels
)


print(
    "\nLoss:",
    loss.item()
)


# --------------------------------------------------
# Backward pass
# --------------------------------------------------

model.zero_grad()

loss.backward()


print(
    "\nBackward pass successful."
)


# --------------------------------------------------
# Check gradients
# --------------------------------------------------

gradient_found = False

for name, parameter in model.named_parameters():

    if parameter.grad is not None:

        if torch.isfinite(
            parameter.grad
        ).all():

            gradient_found = True
            break


print(
    "Gradient flow:",
    gradient_found
)


print("\n===================================")
print("EARTHSWIN END-TO-END TEST PASSED")
print("===================================")