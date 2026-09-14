import os
import json
import numpy as np
import pandas as pd
import torch

from graph import DynamicGraphConstructor


PATCH_DIR = "C:/Users/advit/Documents/major project/dataset/EarthSentinel_2016/multiscale_patches"
TEMPORAL_DIR = "C:/Users/advit/Documents/major project/dataset/EarthSentinel_2016/temporal_dataset"
OUTPUT_DIR = "C:/Users/advit/Documents/major project/dataset/EarthSentinel_2016/graphs"


SPLITS = {
    "train": "train_original.csv",
    "val": "val.csv",
    "test": "test.csv"
}


def load_split_anchor_ids(csv_path):
    df = pd.read_csv(csv_path)

    anchor_ids = sorted(
        df["anchor_id"].astype(int).unique()
    )

    return anchor_ids


def load_anchor_data(anchor_id):
    filename = f"sample_{anchor_id:06d}.npz"
    path = os.path.join(PATCH_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Patch not found: {path}"
        )

    data = np.load(path)

    patch_256 = data["patch_256"].copy()

    center_row = int(data["center_row"])
    center_col = int(data["center_col"])

    return patch_256, center_row, center_col


def compute_environmental_features(patch_256, week):
    """
    Calculate anchor-level environmental features
    for one week.

    Channels:
        0 = VV
        1 = rainfall
        2 = slope

    Uses the mean of valid pixels in the 256x256 patch.
    """

    week_data = patch_256[week - 1]

    features = []

    for channel in range(3):

        channel_data = week_data[channel]

        valid = np.isfinite(channel_data)

        if np.any(valid):
            mean_value = np.mean(
                channel_data[valid]
            )
        else:
            mean_value = 0.0

        features.append(mean_value)

    return features


def build_weekly_graph(
    split_name,
    week,
    anchor_ids
):
    print(
        f"\n[{split_name.upper()}] "
        f"Building graph for week {week}..."
    )

    coordinates = []
    environmental_features = []

    for i, anchor_id in enumerate(anchor_ids):

        patch_256, center_row, center_col = \
            load_anchor_data(anchor_id)

        coordinates.append([
            center_row,
            center_col
        ])

        features = compute_environmental_features(
            patch_256,
            week
        )

        environmental_features.append(
            features
        )

        if (i + 1) % 500 == 0:
            print(
                f"  Loaded "
                f"{i + 1}/{len(anchor_ids)} anchors"
            )

    coordinates = torch.tensor(
        coordinates,
        dtype=torch.float32
    )

    environmental_features = torch.tensor(
        environmental_features,
        dtype=torch.float32
    )

    print(
        f"  Coordinates: "
        f"{coordinates.shape}"
    )

    print(
        f"  Environmental features: "
        f"{environmental_features.shape}"
    )

    graph_constructor = DynamicGraphConstructor(
        k=8
    )

    edge_index, edge_weights = graph_constructor(
        coordinates,
        environmental_features
    )

    print(
        f"  Edge index: "
        f"{edge_index.shape}"
    )

    print(
        f"  Edge weights: "
        f"{edge_weights.shape}"
    )

    print(
        f"  Edge weight range: "
        f"{float(edge_weights.min()):.6f} "
        f"to "
        f"{float(edge_weights.max()):.6f}"
    )

    return {
        "anchor_ids": anchor_ids,
        "coordinates": coordinates,
        "environmental_features": environmental_features,
        "edge_index": edge_index,
        "edge_weights": edge_weights
    }


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    summary = {}

    for split_name, csv_filename in SPLITS.items():

        csv_path = os.path.join(
            TEMPORAL_DIR,
            csv_filename
        )

        anchor_ids = load_split_anchor_ids(
            csv_path
        )

        print("\n===================================")
        print(
            f"{split_name.upper()} SPLIT"
        )
        print(
            f"Anchors: {len(anchor_ids)}"
        )
        print("===================================")

        summary[split_name] = {}

        for week in range(1, 15):

            graph = build_weekly_graph(
                split_name,
                week,
                anchor_ids
            )

            split_output_dir = os.path.join(
                OUTPUT_DIR,
                split_name
            )

            os.makedirs(
                split_output_dir,
                exist_ok=True
            )

            output_path = os.path.join(
                split_output_dir,
                f"graph_week_{week:02d}.pt"
            )

            torch.save(
                graph,
                output_path
            )

            summary[split_name][
                f"week_{week}"
            ] = {
                "num_nodes": len(anchor_ids),
                "num_edges": int(
                    graph["edge_index"].shape[1]
                ),
                "edge_weight_min": float(
                    graph["edge_weights"].min()
                ),
                "edge_weight_max": float(
                    graph["edge_weights"].max()
                )
            }

            print(
                f"  Saved: {output_path}"
            )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "graph_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    print("\n===================================")
    print("ALL SPLIT-SPECIFIC GRAPHS BUILT")
    print("===================================")

    print(
        f"Train graphs: "
        f"{len(SPLITS) - 2 if False else 14}"
    )
    print("Validation graphs: 14")
    print("Test graphs: 14")
    print(
        f"Saved to: {OUTPUT_DIR}"
    )
    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()