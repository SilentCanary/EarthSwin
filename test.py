import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"E:\major\EarthSwin"

MODELS_DIR = os.path.join(PROJECT_DIR, "models")
TEMPORAL_DIR = os.path.join(
    PROJECT_DIR, "dataset", "EarthSentinel_2016", "temporal_dataset"
)

BALANCED_CSV = os.path.join(TEMPORAL_DIR, "train_balanced.csv")
TRAIN_ORIGINAL_CSV = os.path.join(TEMPORAL_DIR, "train_original.csv")
VAL_ORIGINAL_CSV = os.path.join(TEMPORAL_DIR, "val.csv")
TEST_ORIGINAL_CSV = os.path.join(TEMPORAL_DIR, "test.csv")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "earthswin_best.pt")

OUTPUT_DIR = os.path.join(MODELS_DIR, "balanced_diagnostic_results_96")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPECTED_ROWS = 96
EXPECTED_POSITIVES = 24
EXPECTED_NEGATIVES = 72

sys.path.insert(0, MODELS_DIR)

from earthswin_model import EarthSwin
from graph_dataset import GraphTrainingDataset
from train_earthswin import forward_group, DEVICE


# ============================================================
# VALIDATE DIAGNOSTIC CSV
# ============================================================

KEY_COLUMNS = [
    "split",
    "anchor_id",
    "week_1",
    "week_2",
    "week_3",
    "target_week",
    "augmentation",
]

REQUIRED_COLUMNS = KEY_COLUMNS + ["label"]


def validate_csv(df):

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected exactly 96 entries, found {len(df)}. "
            "Evaluation stopped."
        )

    df = df.copy()

    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["augmentation"] = df["augmentation"].astype(str).str.strip()

    numeric_columns = [
        "anchor_id", "week_1", "week_2",
        "week_3", "target_week", "label"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    if not set(df["split"].unique()).issubset({"train", "val", "test"}):
        raise ValueError("Split must contain only train, val, or test.")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1.")

    positives = int((df["label"] == 1).sum())
    negatives = int((df["label"] == 0).sum())

    if positives != EXPECTED_POSITIVES or negatives != EXPECTED_NEGATIVES:
        raise ValueError(
            f"Expected 24 positives and 72 negatives. "
            f"Found {positives} positives and {negatives} negatives."
        )

    if df.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            "Duplicate diagnostic entries found. "
            "Each temporal window and augmentation must be unique."
        )

    return df


def make_key(row):

    return (
        int(row["anchor_id"]),
        int(row["week_1"]),
        int(row["week_2"]),
        int(row["week_3"]),
        int(row["target_week"]),
        str(row["augmentation"]),
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = EarthSwin(
        in_channels=6,
        feature_dim=512,
        num_heads=8,
        temporal_layers=2,
        temporal_feedforward_dim=1024,
        dropout=0.1,
    )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print("Model loaded:", BEST_MODEL_PATH)
    print("Device:", DEVICE)

    return model


# ============================================================
# INFERENCE FOR ONE SPLIT
# ============================================================

@torch.no_grad()
def collect_predictions(
    model,
    diagnostic_df,
    split_name,
    original_csv_path,
):

    split_df = diagnostic_df[
        diagnostic_df["split"] == split_name
    ].copy().reset_index(drop=True)

    if split_df.empty:
        print(f"No entries for {split_name}")
        return pd.DataFrame()

    print("\n======================================")
    print("INFERENCE:", split_name.upper())
    print("======================================")

    print("Requested entries:", len(split_df))

    # IMPORTANT:
    # Create a split-specific CSV.
    # Do not give the loader a combined val + test CSV.

    split_csv = os.path.join(
        OUTPUT_DIR,
        f"diagnostic_{split_name}_input.csv",
    )

    split_df.to_csv(split_csv, index=False)

    loader = GraphTrainingDataset(
        csv_path=split_csv,
        split_anchor_csv=original_csv_path,
        normalize=True,
    )

    groups = loader.get_training_groups()

    # Map every diagnostic entry to its exact row.
    row_lookup = {}

    for row_index, row in split_df.iterrows():

        key = make_key(row)

        if key in row_lookup:
            raise RuntimeError(f"Duplicate sample key: {key}")

        row_lookup[key] = row_index

    prediction_count = np.zeros(len(split_df), dtype=int)

    probabilities_output = np.full(
        len(split_df),
        np.nan,
        dtype=float,
    )

    # --------------------------------------------------------
    # PROCESS TEMPORAL GROUPS
    # --------------------------------------------------------

    for group_number, group in enumerate(groups, start=1):

        week_1 = int(group["week_1"])
        week_2 = int(group["week_2"])
        week_3 = int(group["week_3"])
        target_week = int(group["target_week"])

        augmentation = str(group["augmentation"])

        group_anchor_ids = [
            int(x) for x in group["anchor_ids"]
        ]

        group_labels = [
            int(x) for x in group["labels"]
        ]

        if len(group_anchor_ids) != len(group_labels):
            raise RuntimeError(
                "Group anchor and label counts do not match."
            )

        matched_indices = []

        # Exact matching using all temporal fields.
        for i, anchor_id in enumerate(group_anchor_ids):

            key = (
                anchor_id,
                week_1,
                week_2,
                week_3,
                target_week,
                augmentation,
            )

            row_index = row_lookup.get(key)

            if row_index is None:
                continue

            csv_label = int(split_df.loc[row_index, "label"])

            if csv_label != group_labels[i]:
                raise RuntimeError(
                    f"Label mismatch for sample {key}"
                )

            matched_indices.append((i, row_index))

        if not matched_indices:
            continue

        # Existing EarthSwin forward pass.
        logits = forward_group(
            model,
            loader,
            graph_split=split_name,
            group=group,
        )

        # Get the positions of the matched graph nodes.
        matched_anchor_ids = [
            group_anchor_ids[i] for i, _ in matched_indices
        ]

        positions = loader.get_node_positions(
            matched_anchor_ids
        )

        supervised_logits = logits[positions].reshape(-1)

        probabilities = torch.sigmoid(
            supervised_logits
        ).detach().cpu().numpy()

        if len(probabilities) != len(matched_indices):
            raise RuntimeError(
                "Number of probabilities does not match matched rows."
            )

        for (_, row_index), probability in zip(
            matched_indices, probabilities
        ):

            prediction_count[row_index] += 1
            probabilities_output[row_index] = float(probability)

        print(
            f"Group {group_number}/{len(groups)} | "
            f"Weeks: {week_1},{week_2},{week_3} | "
            f"Target: {target_week} | "
            f"Augmentation: {augmentation} | "
            f"Matched: {len(matched_indices)}"
        )

    # --------------------------------------------------------
    # STRICT ONE-PREDICTION-PER-ENTRY VALIDATION
    # --------------------------------------------------------

    missing = np.where(prediction_count == 0)[0]
    repeated = np.where(prediction_count > 1)[0]

    if len(missing) > 0:

        raise RuntimeError(
            "Some diagnostic rows received no prediction:\n"
            + split_df.loc[missing, KEY_COLUMNS].to_string(index=False)
        )

    if len(repeated) > 0:

        raise RuntimeError(
            "Some diagnostic rows received multiple predictions:\n"
            + split_df.loc[repeated, KEY_COLUMNS].to_string(index=False)
        )

    if not np.all(np.isfinite(probabilities_output)):
        raise RuntimeError("Missing or invalid probabilities detected.")

    result = split_df.copy()

    result["probability"] = probabilities_output
    result["prediction_count"] = prediction_count

    print(
        f"\n{split_name.upper()} predictions: "
        f"{len(result)}"
    )

    print("Every entry predicted exactly once.")

    return result


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(df, threshold):

    if df.empty:
        return None

    y_true = df["label"].astype(int).to_numpy()

    y_prob = df["probability"].astype(float).to_numpy()

    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    ap = (
        average_precision_score(y_true, y_prob)
        if np.sum(y_true) > 0
        else None
    )

    return {
        "entries": int(len(df)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "average_precision": (
            None if ap is None else float(ap)
        ),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def threshold_sweep(val_df):

    thresholds = np.linspace(0.01, 0.99, 99)

    results = []

    for threshold in thresholds:

        results.append(
            calculate_metrics(val_df, threshold)
        )

    return pd.DataFrame(results)


def print_metrics(title, metrics):

    print("\n======================================")
    print(title)
    print("======================================")

    if metrics is None:
        print("No entries.")
        return

    print("Entries   :", metrics["entries"])
    print("Threshold :", round(metrics["threshold"], 3))
    print("Accuracy  :", round(metrics["accuracy"], 4))
    print("Precision :", round(metrics["precision"], 4))
    print("Recall    :", round(metrics["recall"], 4))
    print("F1        :", round(metrics["f1"], 4))
    print("AP        :", metrics["average_precision"])

    print(
        f"TN={metrics['tn']} | FP={metrics['fp']} | "
        f"FN={metrics['fn']} | TP={metrics['tp']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nLoading diagnostic CSV:", BALANCED_CSV)

    diagnostic_df = pd.read_csv(BALANCED_CSV)

    diagnostic_df = validate_csv(diagnostic_df)

    print("\nValidated diagnostic entries:", len(diagnostic_df))
    print("Positive:", (diagnostic_df["label"] == 1).sum())
    print("Negative:", (diagnostic_df["label"] == 0).sum())

    print("\nSplit counts:")
    print(diagnostic_df.groupby(["split", "label"]).size())

    model = load_model()

    original_csv_by_split = {
        "train": TRAIN_ORIGINAL_CSV,
        "val": VAL_ORIGINAL_CSV,
        "test": TEST_ORIGINAL_CSV,
    }

    predictions_by_split = {}

    for split_name in diagnostic_df["split"].unique():
        predictions_by_split[split_name] = collect_predictions(
            model,
            diagnostic_df,
            split_name,
            original_csv_by_split[split_name],
        )

    # --------------------------------------------------------
    # VERIFY EXACTLY 96 PREDICTIONS
    # --------------------------------------------------------

    all_predictions = pd.concat(
        predictions_by_split.values(),
        ignore_index=True,
    )

    if len(all_predictions) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected 96 predictions, got {len(all_predictions)}."
        )

    if all_predictions.duplicated(KEY_COLUMNS).any():
        raise RuntimeError("Duplicate predictions detected.")

    # --------------------------------------------------------
    # THRESHOLD SELECTION ON VALIDATION WHEN AVAILABLE.
    # --------------------------------------------------------

    threshold_df = predictions_by_split.get(
        "val",
        predictions_by_split.get("train", pd.DataFrame()),
    )

    if threshold_df.empty:
        raise RuntimeError("No split is available for threshold selection.")

    if threshold_df["label"].nunique() < 2:

        warnings.warn(
            "Validation has only one class. "
            "Using threshold 0.50."
        )

        best_threshold = 0.50
        sweep_df = pd.DataFrame()

    else:

        sweep_df = threshold_sweep(threshold_df)

        best_row = sweep_df.loc[
            sweep_df["f1"].idxmax()
        ]

        best_threshold = float(best_row["threshold"])

        sweep_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "validation_threshold_sweep.csv",
            ),
            index=False,
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    combined_metrics = calculate_metrics(
        all_predictions,
        best_threshold,
    )

    for df in list(predictions_by_split.values()) + [all_predictions]:
        df["predicted_label"] = (
            df["probability"] >= best_threshold
        ).astype(int)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    for split_name, predictions in predictions_by_split.items():
        predictions.to_csv(
            os.path.join(OUTPUT_DIR, f"{split_name}_predictions.csv"),
            index=False,
        )

    all_predictions.to_csv(
        os.path.join(OUTPUT_DIR, "all_diagnostic_predictions.csv"),
        index=False,
    )

    metrics_output = {
        "threshold_selected_on": (
            "validation" if "val" in predictions_by_split else "train"
        ),
        "threshold": best_threshold,
        "expected_entries": EXPECTED_ROWS,
        "actual_entries": len(all_predictions),
        "splits": {
            split_name: calculate_metrics(predictions, best_threshold)
            for split_name, predictions in predictions_by_split.items()
        },
        "combined_diagnostic_summary": combined_metrics,
    }

    with open(
        os.path.join(OUTPUT_DIR, "diagnostic_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(metrics_output, f, indent=4)

    for split_name, predictions in predictions_by_split.items():
        print_metrics(
            f"{split_name.upper()} METRICS",
            calculate_metrics(predictions, best_threshold),
        )

    print_metrics("COMBINED DIAGNOSTIC SUMMARY", combined_metrics)

    print("\n======================================")
    print("SUCCESS: ALL DIAGNOSTIC ENTRIES EVALUATED")
    print("======================================")

    print("Total predictions:", len(all_predictions))
    print(
        "Unique entries:",
        all_predictions[KEY_COLUMNS].drop_duplicates().shape[0],
    )

    print("Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()