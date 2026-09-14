import os
import glob
import numpy as np
import torch

from graph import DynamicGraphConstructor
from gat import GAT
from temporal import TemporalTransformer


# ============================================================
# Configuration
# ============================================================

BASE_DIR = "C:/Users/advit/Documents/major project/dataset/EarthSentinel_2016"

PATCH_CHUNKS_DIR = os.path.join(
    BASE_DIR,
    "patch_chunks"
)

TEMPORAL_DIR = os.path.join(
    BASE_DIR,
    "temporal_dataset"
)

ALL_SEQUENCES_FILE = os.path.join(
    TEMPORAL_DIR,
    "all_temporal_sequences.csv"
)

TARGET_WEEK = 12
K = 8

NUM_NODES = 2631
FEATURE_DIM = 512
SEQUENCE_LENGTH = 3


print("Starting GAT + Temporal Transformer test...")
print("=" * 70)


# ============================================================
# 1. Load retained anchor IDs
# ============================================================

print("\nLoading retained anchors...")

import pandas as pd

df = pd.read_csv(
    ALL_SEQUENCES_FILE
)

retained_anchor_ids = set(
    df["anchor_id"].unique()
)

print(
    "Retained anchors:",
    len(retained_anchor_ids)
)

assert len(retained_anchor_ids) == NUM_NODES


# ============================================================
# 2. Load coordinates and environmental features
# ============================================================

print("\nLoading actual anchor information...")

chunk_files = sorted(
    glob.glob(
        os.path.join(
            PATCH_CHUNKS_DIR,
            "*.npz"
        )
    )
)

coordinates = []
environmental_features = []
loaded_anchor_ids = []

for chunk_file in chunk_files:

    chunk = np.load(
        chunk_file
    )

    anchor_ids = chunk["anchor_id"]

    patch_256 = chunk["patch_256"]

    center_rows = chunk["center_row"]
    center_cols = chunk["center_col"]

    for i, anchor_id in enumerate(anchor_ids):

        anchor_id = int(anchor_id)

        if anchor_id not in retained_anchor_ids:
            continue

        patch = patch_256[i]

        # TARGET_WEEK is 1-indexed
        week_index = TARGET_WEEK - 1

        week_patch = patch[week_index]

        vv = week_patch[0]
        rainfall = week_patch[1]
        slope = week_patch[2]

        # --------------------------------------------------
        # Calculate environmental summary
        # --------------------------------------------------

        vv_valid = vv[np.isfinite(vv)]
        rainfall_valid = rainfall[np.isfinite(rainfall)]
        slope_valid = slope[np.isfinite(slope)]

        if len(vv_valid) > 0:
            vv_value = np.median(vv_valid)
        else:
            vv_value = 0.0

        if len(rainfall_valid) > 0:
            rainfall_value = np.mean(rainfall_valid)
        else:
            rainfall_value = 0.0

        if len(slope_valid) > 0:
            slope_value = np.mean(slope_valid)
        else:
            slope_value = 0.0

        coordinates.append([
            center_rows[i],
            center_cols[i]
        ])

        environmental_features.append([
            vv_value,
            rainfall_value,
            slope_value
        ])

        loaded_anchor_ids.append(
            anchor_id
        )


# ============================================================
# 3. Convert to tensors
# ============================================================

coordinates = torch.tensor(
    coordinates,
    dtype=torch.float32
)

environmental_features = torch.tensor(
    environmental_features,
    dtype=torch.float32
)

print(
    "Loaded anchors:",
    coordinates.shape[0]
)

print(
    "Coordinates shape:",
    coordinates.shape
)

print(
    "Environmental features shape:",
    environmental_features.shape
)

assert coordinates.shape == (
    NUM_NODES,
    2
)

assert environmental_features.shape == (
    NUM_NODES,
    3
)

assert torch.isfinite(
    coordinates
).all()

assert torch.isfinite(
    environmental_features
).all()


# ============================================================
# 4. Build graph
# ============================================================

print("\nBuilding graph...")

graph_constructor = DynamicGraphConstructor(
    k=K
)

edge_index, edge_weights = graph_constructor(
    coordinates,
    environmental_features
)

expected_edges = NUM_NODES * K

print(
    "Number of nodes:",
    NUM_NODES
)

print(
    "Number of edges:",
    edge_index.shape[1]
)

print(
    "Expected edges:",
    expected_edges
)

print(
    "Edge weights shape:",
    edge_weights.shape
)

assert edge_index.shape == (
    2,
    expected_edges
)

assert edge_weights.shape == (
    expected_edges,
)

assert torch.isfinite(
    edge_weights
).all()

assert (edge_weights > 0).all()


# ============================================================
# 5. Create GAT
# ============================================================

print("\nCreating GAT model...")

gat = GAT(
    input_dim=FEATURE_DIM,
    hidden_dim=FEATURE_DIM,
    output_dim=FEATURE_DIM,
    dropout=0.1
)


# ============================================================
# 6. Create Temporal Transformer
# ============================================================

print("\nCreating Temporal Transformer...")

temporal_transformer = TemporalTransformer(
    input_dim=FEATURE_DIM,
    num_heads=8,
    num_layers=2,
    feedforward_dim=1024,
    dropout=0.1,
    sequence_length=SEQUENCE_LENGTH
)


# ============================================================
# 7. Create dummy weekly PatchPyramid features
# ============================================================

print("\nCreating dummy weekly features...")

weekly_features = []

for week in range(SEQUENCE_LENGTH):

    features = torch.randn(
        NUM_NODES,
        FEATURE_DIM,
        requires_grad=True
    )

    weekly_features.append(
        features
    )

    print(
        f"Week {week + 1} feature shape:",
        features.shape
    )


# ============================================================
# 8. Run GAT for each week
# ============================================================

print("\nRunning GAT for 3 weeks...")

gat_outputs = []

for week in range(SEQUENCE_LENGTH):

    print(
        f"Processing week {week + 1}..."
    )

    gat_output = gat(
        weekly_features[week],
        edge_index,
        edge_weights
    )

    print(
        f"GAT output week {week + 1}:",
        gat_output.shape
    )

    assert gat_output.shape == (
        NUM_NODES,
        FEATURE_DIM
    )

    assert torch.isfinite(
        gat_output
    ).all()

    gat_outputs.append(
        gat_output
    )


# ============================================================
# 9. Stack temporal sequence
# ============================================================

print("\nStacking temporal sequence...")

temporal_features = torch.stack(
    gat_outputs,
    dim=1
)

print(
    "Temporal input shape:",
    temporal_features.shape
)

assert temporal_features.shape == (
    NUM_NODES,
    SEQUENCE_LENGTH,
    FEATURE_DIM
)

assert torch.isfinite(
    temporal_features
).all()


# ============================================================
# 10. Run Temporal Transformer
# ============================================================

print("\nRunning Temporal Transformer...")

temporal_output = temporal_transformer(
    temporal_features
)

print(
    "Temporal output shape:",
    temporal_output.shape
)

assert temporal_output.shape == (
    NUM_NODES,
    FEATURE_DIM
)

print(
    "Temporal output NaNs:",
    torch.isnan(
        temporal_output
    ).sum().item()
)

print(
    "Temporal output Infs:",
    torch.isinf(
        temporal_output
    ).sum().item()
)

assert torch.isfinite(
    temporal_output
).all()


# ============================================================
# 11. Test backward pass
# ============================================================

print("\nTesting complete backward pass...")

loss = temporal_output.mean()

loss.backward()

for week in range(SEQUENCE_LENGTH):

    assert weekly_features[week].grad is not None

    assert torch.isfinite(
        weekly_features[week].grad
    ).all()

print(
    "Gradient flow through GAT + Temporal Transformer passed."
)


# ============================================================
# 12. Final results
# ============================================================

print("\n" + "=" * 70)
print("GAT + TEMPORAL TRANSFORMER TEST PASSED")
print("=" * 70)

print("\nFinal shapes:")

print(
    "Nodes              :",
    NUM_NODES
)

print(
    "Edges              :",
    edge_index.shape
)

print(
    "Edge weights       :",
    edge_weights.shape
)

print(
    "Weekly GAT output  :",
    gat_outputs[0].shape
)

print(
    "Temporal input     :",
    temporal_features.shape
)

print(
    "Temporal output    :",
    temporal_output.shape
)

print("\nPipeline:")

print(
    "[2631, 512] × 3 weeks"
)

print(
    "        ↓"
)

print(
    "GAT on each week"
)

print(
    "        ↓"
)

print(
    "[2631, 3, 512]"
)

print(
    "        ↓"
)

print(
    "Temporal Transformer"
)

print(
    "        ↓"
)

print(
    "[2631, 512]"
)

print("=" * 70)