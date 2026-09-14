import os
import glob
import numpy as np
import pandas as pd
import torch

from graph import DynamicGraphConstructor
from gat import GAT


print("Starting Actual GAT test...")
print("=" * 60)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

TEMPORAL_DIR = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/temporal_dataset"
)

CHUNK_DIR = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/patch_chunks"
)


# ---------------------------------------------------------
# Load retained anchor IDs
# ---------------------------------------------------------

temporal_csv = os.path.join(
    TEMPORAL_DIR,
    "all_temporal_sequences.csv"
)

df = pd.read_csv(temporal_csv)

retained_anchor_ids = sorted(
    df["anchor_id"].unique()
)

print(f"Retained anchors: {len(retained_anchor_ids)}")


# ---------------------------------------------------------
# Load coordinates and environmental features
# ---------------------------------------------------------

retained_anchor_ids = set(retained_anchor_ids)

coordinates = []
environmental_features = []
loaded_anchor_ids = []


chunk_files = sorted(
    glob.glob(
        os.path.join(
            CHUNK_DIR,
            "*.npz"
        )
    )
)

print(f"Number of chunks: {len(chunk_files)}")


TARGET_WEEK = 12


for chunk_file in chunk_files:

    chunk = np.load(
        chunk_file,
        allow_pickle=True
    )

    anchor_ids = chunk["anchor_id"]

    patch_256 = chunk["patch_256"]
    center_rows = chunk["center_row"]
    center_cols = chunk["center_col"]

    for i, anchor_id in enumerate(anchor_ids):

        anchor_id = int(anchor_id)

        if anchor_id not in retained_anchor_ids:
            continue

        patch = patch_256[
            i,
            TARGET_WEEK - 1
        ]

        # -------------------------------------------------
        # Environmental features
        # -------------------------------------------------

        vv = patch[0]
        rainfall = patch[1]
        slope = patch[2]

        valid_vv = vv[np.isfinite(vv)]
        valid_rainfall = rainfall[np.isfinite(rainfall)]
        valid_slope = slope[np.isfinite(slope)]

        if len(valid_vv) > 0:
            vv_value = np.median(valid_vv)
        else:
            vv_value = 0.0

        if len(valid_rainfall) > 0:
            rainfall_value = np.mean(valid_rainfall)
        else:
            rainfall_value = 0.0

        if len(valid_slope) > 0:
            slope_value = np.mean(valid_slope)
        else:
            slope_value = 0.0

        environmental_features.append([
            vv_value,
            rainfall_value,
            slope_value
        ])

        # -------------------------------------------------
        # Spatial coordinates
        # -------------------------------------------------

        coordinates.append([
            float(center_rows[i]),
            float(center_cols[i])
        ])

        loaded_anchor_ids.append(anchor_id)


# ---------------------------------------------------------
# Convert to tensors
# ---------------------------------------------------------

coordinates = torch.tensor(
    coordinates,
    dtype=torch.float32
)

environmental_features = torch.tensor(
    environmental_features,
    dtype=torch.float32
)

print(f"Loaded anchors: {len(loaded_anchor_ids)}")
print(
    f"Coordinates shape: {coordinates.shape}"
)
print(
    f"Environmental features shape: "
    f"{environmental_features.shape}"
)


# ---------------------------------------------------------
# Validate anchor count
# ---------------------------------------------------------

assert len(loaded_anchor_ids) == 2631
assert coordinates.shape == (2631, 2)
assert environmental_features.shape == (2631, 3)


# ---------------------------------------------------------
# Build graph
# ---------------------------------------------------------

print()
print("Building graph...")

graph_constructor = DynamicGraphConstructor(
    k=8
)

edge_index, edge_weights = graph_constructor(
    coordinates,
    environmental_features
)

print(
    f"Number of nodes: {coordinates.shape[0]}"
)

print(
    f"Number of edges: {edge_index.shape[1]}"
)

print(
    f"Expected edges: {2631 * 8}"
)

print(
    f"Edge weights shape: {edge_weights.shape}"
)


# ---------------------------------------------------------
# Validate graph
# ---------------------------------------------------------

assert edge_index.shape == (
    2,
    2631 * 8
)

assert edge_weights.shape == (
    2631 * 8,
)

assert torch.all(
    torch.isfinite(edge_weights)
)

assert torch.all(
    edge_weights > 0
)


# ---------------------------------------------------------
# Create 512-D node features
# ---------------------------------------------------------

print()
print("Creating 512-D node features...")

node_features = torch.randn(
    2631,
    512,
    dtype=torch.float32,
    requires_grad=True
)

print(
    f"Node features shape: "
    f"{node_features.shape}"
)


# ---------------------------------------------------------
# Create GAT
# ---------------------------------------------------------

print()
print("Creating GAT model...")

gat = GAT(
    input_dim=512,
    hidden_dim=512,
    output_dim=512,
    dropout=0.1
)

gat.train()


# ---------------------------------------------------------
# Forward pass
# ---------------------------------------------------------

print()
print("Running GAT forward pass...")

output_features = gat(
    node_features,
    edge_index,
    edge_weights
)

print(
    f"GAT output shape: "
    f"{output_features.shape}"
)


# ---------------------------------------------------------
# Output validation
# ---------------------------------------------------------

assert output_features.shape == (
    2631,
    512
)

assert torch.all(
    torch.isfinite(output_features)
)


print(
    f"Output NaNs: "
    f"{torch.isnan(output_features).sum().item()}"
)

print(
    f"Output Infs: "
    f"{torch.isinf(output_features).sum().item()}"
)


# ---------------------------------------------------------
# Gradient test
# ---------------------------------------------------------

print()
print("Testing backward pass...")

loss = output_features.mean()

loss.backward()

assert node_features.grad is not None

assert torch.all(
    torch.isfinite(node_features.grad)
)

print("Gradient test passed.")


# ---------------------------------------------------------
# Final validation
# ---------------------------------------------------------

print()
print("=" * 60)
print("ACTUAL GAT TEST PASSED")
print("=" * 60)

print()
print("Final shapes:")
print(
    f"Node features : {node_features.shape}"
)
print(
    f"Edge index    : {edge_index.shape}"
)
print(
    f"Edge weights  : {edge_weights.shape}"
)
print(
    f"GAT output    : {output_features.shape}"
)