import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from graph import DynamicGraphConstructor


# ============================================================
# PATHS
# ============================================================

CHUNK_DIR = (
    "C:/Users/advit/Documents/major project/dataset/"
    "EarthSentinel_2016/patch_chunks"
)

TEMPORAL_CSV = (
    "C:/Users/advit/Documents/major project/dataset/"
    "EarthSentinel_2016/temporal_dataset/all_temporal_sequences.csv"
)

OUTPUT_DIR = "C:/Users/advit/Documents/major project/models"


# ============================================================
# SETTINGS
# ============================================================

K = 8

# Use week 12 for visualization because VV coverage is much
# better than the extremely sparse week 3.
TARGET_WEEK = 12

# Number of nodes to display in the visualized region.
# The graph itself still contains all 2631 anchors.
DISPLAY_NODES = 300


# ============================================================
# START
# ============================================================

print("Starting Actual Anchor Graph visualization...\n")


# ============================================================
# LOAD RETAINED ANCHOR IDS
# ============================================================

df = pd.read_csv(TEMPORAL_CSV)

retained_anchor_ids = set(
    df["anchor_id"].unique()
)

print(
    "Retained anchors:",
    len(retained_anchor_ids)
)


# ============================================================
# LOAD ANCHORS FROM CHUNKS
# ============================================================

all_anchor_ids = []
all_coordinates = []
all_environmental_features = []

chunk_files = sorted(
    [
        os.path.join(CHUNK_DIR, f)
        for f in os.listdir(CHUNK_DIR)
        if f.endswith(".npz")
    ]
)

print(
    "Number of chunks:",
    len(chunk_files)
)


for chunk_file in chunk_files:

    data = np.load(chunk_file)

    anchor_ids = data["anchor_id"]
    center_rows = data["center_row"]
    center_cols = data["center_col"]
    patch_256 = data["patch_256"]

    for i, anchor_id in enumerate(anchor_ids):

        anchor_id = int(anchor_id)

        # Only keep the 2631 retained anchors
        if anchor_id not in retained_anchor_ids:
            continue

        # --------------------------------------------------------
        # Coordinates
        # --------------------------------------------------------

        center_row = float(
            center_rows[i]
        )

        center_col = float(
            center_cols[i]
        )

        all_coordinates.append(
            [
                center_row,
                center_col
            ]
        )

        # --------------------------------------------------------
        # Environmental features
        # --------------------------------------------------------

        week_index = TARGET_WEEK - 1

        vv = patch_256[
            i,
            week_index,
            0
        ]

        rainfall = patch_256[
            i,
            week_index,
            1
        ]

        slope = patch_256[
            i,
            week_index,
            2
        ]

        vv_valid = vv[
            np.isfinite(vv)
        ]

        rainfall_valid = rainfall[
            np.isfinite(rainfall)
        ]

        slope_valid = slope[
            np.isfinite(slope)
        ]

        # --------------------------------------------------------
        # Patch-level environmental summaries
        # --------------------------------------------------------

        if len(vv_valid) > 0:
            vv_value = float(
                np.median(vv_valid)
            )
        else:
            vv_value = 0.0

        if len(rainfall_valid) > 0:
            rainfall_value = float(
                np.mean(rainfall_valid)
            )
        else:
            rainfall_value = 0.0

        if len(slope_valid) > 0:
            slope_value = float(
                np.mean(slope_valid)
            )
        else:
            slope_value = 0.0

        all_environmental_features.append(
            [
                vv_value,
                rainfall_value,
                slope_value
            ]
        )

        all_anchor_ids.append(
            anchor_id
        )


# ============================================================
# CHECK DATA
# ============================================================

assert len(all_anchor_ids) == len(
    retained_anchor_ids
), (
    f"Expected {len(retained_anchor_ids)} anchors, "
    f"loaded {len(all_anchor_ids)}"
)

assert len(set(all_anchor_ids)) == len(
    retained_anchor_ids
), "Duplicate anchors detected."


print(
    "Loaded anchors:",
    len(all_anchor_ids)
)


# ============================================================
# CONVERT TO TENSORS
# ============================================================

coordinates = torch.tensor(
    all_coordinates,
    dtype=torch.float32
)

environmental_features = torch.tensor(
    all_environmental_features,
    dtype=torch.float32
)


print(
    "Coordinates shape:",
    coordinates.shape
)

print(
    "Environmental features shape:",
    environmental_features.shape
)


# ============================================================
# BUILD ACTUAL GRAPH
# ============================================================

print("\nBuilding graph...")

graph_constructor = DynamicGraphConstructor(
    k=K
)

edge_index, edge_weights = graph_constructor(
    coordinates,
    environmental_features
)


num_nodes = len(all_anchor_ids)

num_edges = edge_weights.shape[0]


print(
    "Number of nodes:",
    num_nodes
)

print(
    "Number of edges:",
    num_edges
)

print(
    "Expected edges:",
    num_nodes * K
)

print(
    "Edge weight range:",
    edge_weights.min().item(),
    "to",
    edge_weights.max().item()
)


# ============================================================
# GRAPH VALIDATION
# ============================================================

assert edge_index.shape[0] == 2

assert edge_index.shape[1] == num_nodes * K

assert edge_weights.shape[0] == num_nodes * K

assert not torch.isnan(
    edge_weights
).any()

assert not torch.isinf(
    edge_weights
).any()

assert not torch.any(
    edge_index[0] == edge_index[1]
)


print("\nGraph validation passed.")


# ============================================================
# SAVE COMPLETE GRAPH DATA
# ============================================================

graph_summary = {

    "project": "EarthSentinel",

    "graph_type": "Dynamic Spatial Graph",

    "target_week": TARGET_WEEK,

    "num_nodes": num_nodes,

    "num_retained_anchors": len(
        retained_anchor_ids
    ),

    "k_neighbors": K,

    "num_directed_edges": num_edges,

    "neighbor_selection": {
        "method": "Euclidean distance",
        "description": (
            "Each anchor is connected to its "
            "8 nearest spatial anchors."
        )
    },

    "edge_strength": {
        "features": [
            "VV similarity",
            "Rainfall similarity",
            "Slope similarity"
        ],
        "description": (
            "Edge strength combines normalized "
            "spatial distance and environmental "
            "feature differences."
        )
    },

    "environmental_features": {
        "VV": {
            "min": float(
                environmental_features[:, 0].min()
            ),
            "max": float(
                environmental_features[:, 0].max()
            )
        },
        "rainfall": {
            "min": float(
                environmental_features[:, 1].min()
            ),
            "max": float(
                environmental_features[:, 1].max()
            )
        },
        "slope": {
            "min": float(
                environmental_features[:, 2].min()
            ),
            "max": float(
                environmental_features[:, 2].max()
            )
        }
    },

    "edge_weights": {
        "min": float(
            edge_weights.min()
        ),
        "max": float(
            edge_weights.max()
        ),
        "mean": float(
            edge_weights.mean()
        )
    },

    "validation": {
        "nan_edges": int(
            torch.isnan(edge_weights).sum()
        ),
        "inf_edges": int(
            torch.isinf(edge_weights).sum()
        ),
        "self_loops": int(
            torch.sum(
                edge_index[0] == edge_index[1]
            )
        )
    },

    # --------------------------------------------------------
    # Complete mapping between graph node index
    # and actual anchor ID
    # --------------------------------------------------------

    "node_to_anchor_id": {
        str(i): int(anchor_id)
        for i, anchor_id
        in enumerate(all_anchor_ids)
    },

    # --------------------------------------------------------
    # Complete graph connectivity
    # --------------------------------------------------------

    "edges": [
        {
            "source_node": int(
                edge_index[0, i]
            ),
            "source_anchor_id": int(
                all_anchor_ids[
                    edge_index[0, i].item()
                ]
            ),
            "target_node": int(
                edge_index[1, i]
            ),
            "target_anchor_id": int(
                all_anchor_ids[
                    edge_index[1, i].item()
                ]
            ),
            "weight": float(
                edge_weights[i]
            )
        }
        for i in range(num_edges)
    ]
}


json_path = os.path.join(
    OUTPUT_DIR,
    "graph_summary.json"
)

with open(
    json_path,
    "w"
) as f:

    json.dump(
        graph_summary,
        f,
        indent=4
    )


print(
    "\nSaved graph summary:"
)

print(
    json_path
)


# ============================================================
# SELECT REPRESENTATIVE REGION
# ============================================================

# We don't plot all 2631 nodes because the resulting image
# would be extremely cluttered.
#
# Instead, select a spatially compact region around the
# median coordinate.

coords_np = coordinates.numpy()

median_row = np.median(
    coords_np[:, 0]
)

median_col = np.median(
    coords_np[:, 1]
)

distance_from_center = np.sqrt(
    (
        coords_np[:, 0]
        - median_row
    ) ** 2
    +
    (
        coords_np[:, 1]
        - median_col
    ) ** 2
)

display_nodes = np.argsort(
    distance_from_center
)[
    :min(DISPLAY_NODES, num_nodes)
]

display_nodes = set(
    display_nodes.tolist()
)


# ============================================================
# CREATE GRAPH VISUALIZATION
# ============================================================

print(
    "\nCreating graph visualization..."
)


fig, ax = plt.subplots(
    figsize=(12, 10)
)


# ------------------------------------------------------------
# DRAW EDGES
# ------------------------------------------------------------

edges_drawn = 0

for i in range(num_edges):

    source = edge_index[
        0,
        i
    ].item()

    target = edge_index[
        1,
        i
    ].item()

    # Only show edges where both nodes
    # belong to the selected region.
    if (
        source not in display_nodes
        or target not in display_nodes
    ):
        continue

    source_x = coords_np[
        source,
        1
    ]

    source_y = coords_np[
        source,
        0
    ]

    target_x = coords_np[
        target,
        1
    ]

    target_y = coords_np[
        target,
        0
    ]

    weight = edge_weights[
        i
    ].item()

    # Use transparency based on edge weight.
    # Stronger edges are easier to see.
    alpha = 0.15 + (
        0.65 * weight
    )

    ax.plot(
        [source_x, target_x],
        [source_y, target_y],
        linewidth=0.7,
        alpha=alpha
    )

    edges_drawn += 1


# ------------------------------------------------------------
# DRAW NODES
# ------------------------------------------------------------

display_node_list = list(
    display_nodes
)

display_x = [
    coords_np[i, 1]
    for i in display_node_list
]

display_y = [
    coords_np[i, 0]
    for i in display_node_list
]


ax.scatter(
    display_x,
    display_y,
    s=18,
    alpha=0.9
)


# ------------------------------------------------------------
# LABEL A FEW NODES
# ------------------------------------------------------------

# Label only a handful so the diagram remains readable.

label_count = min(
    10,
    len(display_node_list)
)

for node in display_node_list[
    :label_count
]:

    x = coords_np[
        node,
        1
    ]

    y = coords_np[
        node,
        0
    ]

    anchor_id = all_anchor_ids[
        node
    ]

    ax.text(
        x,
        y,
        str(anchor_id),
        fontsize=7
    )


# ============================================================
# TITLES
# ============================================================

ax.set_title(
    "EarthSentinel Dynamic Spatial Graph",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(
    "Spatial coordinate (column)"
)

ax.set_ylabel(
    "Spatial coordinate (row)"
)


ax.text(
    0.02,
    0.02,
    (
        f"Actual graph: {num_nodes} anchors | "
        f"K={K} nearest neighbors | "
        f"{num_edges:,} directed edges\n"
        f"Displayed region: {len(display_nodes)} anchors | "
        f"Edge strength: distance + VV + rainfall + slope"
    ),
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="bottom"
)


# ------------------------------------------------------------
# GRID
# ------------------------------------------------------------

ax.grid(
    alpha=0.2
)


# ============================================================
# SAVE IMAGE
# ============================================================

png_path = os.path.join(
    OUTPUT_DIR,
    "actual_anchor_graph.png"
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved graph diagram:"
)

print(
    png_path
)


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "ACTUAL ANCHOR GRAPH TEST + VISUALIZATION PASSED"
)

print(
    "=" * 60
)

print(
    "\nFiles generated:"
)

print(
    "1.",
    json_path
)

print(
    "2.",
    png_path
)