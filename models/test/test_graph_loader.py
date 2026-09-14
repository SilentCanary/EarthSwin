import os
import torch

from graph_dataset import GraphTrainingDataset


TEMPORAL_DIR = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/temporal_dataset"
)

GRAPH_DIR = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/graphs/train"
)


TRAIN_BALANCED_CSV = os.path.join(
    TEMPORAL_DIR,
    "train_balanced.csv"
)

TRAIN_ORIGINAL_CSV = os.path.join(
    TEMPORAL_DIR,
    "train_original.csv"
)


print("=" * 60)
print("GRAPH LOADER TEST")
print("=" * 60)


# ---------------------------------------------------------
# 1. Create graph-aware loader
# ---------------------------------------------------------

loader = GraphTrainingDataset(
    csv_path=TRAIN_BALANCED_CSV,
    split_anchor_csv=TRAIN_ORIGINAL_CSV,
    normalize=True
)

print()
print("Loader created successfully.")

print(
    f"Number of train graph anchors: "
    f"{len(loader.anchor_ids)}"
)


# ---------------------------------------------------------
# 2. Get actual training groups
# ---------------------------------------------------------

groups = loader.get_training_groups()

print()
print(f"Number of training groups: {len(groups)}")

if len(groups) == 0:
    raise RuntimeError("No training groups found.")


# Use the first actual group
group = groups[0]

print()
print("First training group:")
print(f"  week_1:       {group['week_1']}")
print(f"  week_2:       {group['week_2']}")
print(f"  week_3:       {group['week_3']}")
print(f"  target_week:  {group['target_week']}")
print(f"  augmentation: {group['augmentation']}")
print(f"  supervised samples: {len(group['anchor_ids'])}")


# ---------------------------------------------------------
# 3. Load ALL graph inputs
# ---------------------------------------------------------

print()
print("Loading all graph inputs...")

week_1, week_2, week_3 = loader.load_graph_inputs(
    week_1=group["week_1"],
    week_2=group["week_2"],
    week_3=group["week_3"],
    augmentation=group["augmentation"]
)

print()
print("Graph inputs loaded successfully.")


# ---------------------------------------------------------
# 4. Check tensor shapes
# ---------------------------------------------------------

expected_nodes = len(loader.anchor_ids)

expected_shapes = {
    "week_1 patch_64": (expected_nodes, 6, 64, 64),
    "week_1 patch_128": (expected_nodes, 6, 128, 128),
    "week_1 patch_256": (expected_nodes, 6, 256, 256),

    "week_2 patch_64": (expected_nodes, 6, 64, 64),
    "week_2 patch_128": (expected_nodes, 6, 128, 128),
    "week_2 patch_256": (expected_nodes, 6, 256, 256),

    "week_3 patch_64": (expected_nodes, 6, 64, 64),
    "week_3 patch_128": (expected_nodes, 6, 128, 128),
    "week_3 patch_256": (expected_nodes, 6, 256, 256),
}

actual_tensors = {
    "week_1 patch_64": week_1["patch_64"],
    "week_1 patch_128": week_1["patch_128"],
    "week_1 patch_256": week_1["patch_256"],

    "week_2 patch_64": week_2["patch_64"],
    "week_2 patch_128": week_2["patch_128"],
    "week_2 patch_256": week_2["patch_256"],

    "week_3 patch_64": week_3["patch_64"],
    "week_3 patch_128": week_3["patch_128"],
    "week_3 patch_256": week_3["patch_256"],
}


print()
print("Tensor shapes:")

for name, tensor in actual_tensors.items():

    expected = expected_shapes[name]
    actual = tuple(tensor.shape)

    print(
        f"  {name}: "
        f"{actual}"
    )

    if actual != expected:
        raise RuntimeError(
            f"Wrong shape for {name}. "
            f"Expected {expected}, got {actual}"
        )


# ---------------------------------------------------------
# 5. Check graph file
# ---------------------------------------------------------

target_week = group["target_week"]

graph_path = os.path.join(
    GRAPH_DIR,
    f"graph_week_{target_week:02d}.pt"
)

print()
print(f"Loading graph:")
print(f"  {graph_path}")

if not os.path.exists(graph_path):
    raise FileNotFoundError(
        f"Graph file not found: {graph_path}"
    )

graph = torch.load(
    graph_path,
    map_location="cpu",
    weights_only=False
)

print("Graph loaded successfully.")


# ---------------------------------------------------------
# 6. Check graph structure
# ---------------------------------------------------------

graph_anchor_ids = graph["anchor_ids"]

coordinates = graph["coordinates"]
environmental_features = graph["environmental_features"]
edge_index = graph["edge_index"]
edge_weights = graph["edge_weights"]

print()
print("Graph information:")
print(f"  nodes:       {len(graph_anchor_ids)}")
print(f"  coordinates: {tuple(coordinates.shape)}")
print(
    f"  environment: "
    f"{tuple(environmental_features.shape)}"
)
print(f"  edge_index:  {tuple(edge_index.shape)}")
print(f"  edge_weights:{tuple(edge_weights.shape)}")


# ---------------------------------------------------------
# 7. Verify loader and graph contain same anchors
# ---------------------------------------------------------

loader_anchor_ids = loader.anchor_ids

graph_anchor_ids = [
    int(x)
    for x in graph_anchor_ids
]

if loader_anchor_ids != graph_anchor_ids:
    raise RuntimeError(
        "Loader anchor ordering does not match "
        "graph anchor ordering."
    )

print()
print("Anchor ordering matches.")


# ---------------------------------------------------------
# 8. Verify graph dimensions
# ---------------------------------------------------------

if coordinates.shape[0] != expected_nodes:
    raise RuntimeError(
        "Graph node count does not match loader."
    )

if edge_index.shape[1] != expected_nodes * 8:
    raise RuntimeError(
        "Unexpected number of graph edges."
    )

if edge_weights.shape[0] != edge_index.shape[1]:
    raise RuntimeError(
        "Edge weights do not match edge count."
    )

print("Graph dimensions are correct.")


# ---------------------------------------------------------
# 9. Map supervised anchor IDs to graph positions
# ---------------------------------------------------------

supervised_anchor_ids = group["anchor_ids"]

positions = loader.get_node_positions(
    supervised_anchor_ids
)

print()
print("Supervised anchor mapping:")

for anchor_id, position, label in zip(
    supervised_anchor_ids,
    positions,
    group["labels"]
):
    print(
        f"  anchor {anchor_id} "
        f"-> graph node {position} "
        f"-> label {label}"
    )


# ---------------------------------------------------------
# 10. Verify positions are valid
# ---------------------------------------------------------

for position in positions:

    if position < 0 or position >= expected_nodes:
        raise RuntimeError(
            f"Invalid graph position: {position}"
        )


# ---------------------------------------------------------
# 11. Verify supervised labels
# ---------------------------------------------------------

labels = torch.tensor(
    group["labels"],
    dtype=torch.float32
)

print()
print(
    f"Supervised labels: "
    f"{labels.tolist()}"
)

print(
    f"Positive samples: "
    f"{int(labels.sum().item())}"
)

print(
    f"Negative samples: "
    f"{int((labels == 0).sum().item())}"
)


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print()
print("=" * 60)
print("GRAPH LOADER TEST PASSED")
print("=" * 60)