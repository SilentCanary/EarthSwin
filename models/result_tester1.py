import os
import sys
import json

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"E:\major\EarthSwin"

MODELS_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

TEMPORAL_DIR = os.path.join(
    PROJECT_DIR,
    "dataset/EarthSentinel_2016/temporal_dataset"
)

BALANCED_CSV = os.path.join(
    TEMPORAL_DIR,
    "balanced_diagnostic.csv"
)

VAL_ORIGINAL_CSV = os.path.join(
    TEMPORAL_DIR,
    "val.csv"
)

TEST_ORIGINAL_CSV = os.path.join(
    TEMPORAL_DIR,
    "test.csv"
)

BEST_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "earthswin_best.pt"
)

OUTPUT_DIR = os.path.join(
    MODELS_DIR,
    "balanced_diagnostic_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# IMPORT PROJECT CODE
# ============================================================

sys.path.insert(
    0,
    MODELS_DIR
)

from earthswin_model import EarthSwin
from graph_dataset import GraphTrainingDataset

from train_earthswin import (
    forward_group,
    DEVICE,
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading model...")

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
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print(
        "Model loaded:",
        BEST_MODEL_PATH
    )

    return model


# ============================================================
# RUN INFERENCE FOR ONE SPLIT
# ============================================================

@torch.no_grad()
def collect_predictions(
    model,
    diagnostic_df,
    split_name,
    original_csv_path,
):
    """
    Run inference for either validation or test.

    The diagnostic CSV contains only the samples we want
    to evaluate.

    original_csv_path is still used as split_anchor_csv
    so the existing graph node ordering remains valid.
    """

    split_df = diagnostic_df[
        diagnostic_df["split"] == split_name
    ].copy()

    if len(split_df) == 0:

        print(
            f"\nNo diagnostic samples for {split_name}."
        )

        return pd.DataFrame()

    print("\n========================================")
    print(f"INFERENCE: {split_name.upper()}")
    print("========================================")

    print(
        "Diagnostic rows:",
        len(split_df)
    )

    print(
        "Positive:",
        int((split_df["label"] == 1).sum())
    )

    print(
        "Negative:",
        int((split_df["label"] == 0).sum())
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # The diagnostic CSV is used for the supervised rows,
    # while the ORIGINAL split CSV preserves the graph's
    # expected anchor/node ordering.
    # --------------------------------------------------------

    evaluation_csv_path = (
        BALANCED_CSV
        if os.path.exists(BALANCED_CSV)
        else original_csv_path
    )

    loader = GraphTrainingDataset(
        csv_path=evaluation_csv_path,
        split_anchor_csv=original_csv_path,
        normalize=True,
    )

    groups = loader.get_training_groups()

    prediction_rows = []

    requested_anchor_ids = set(
        split_df["anchor_id"].unique()
    )

    # --------------------------------------------------------
    # PROCESS EACH AUGMENTATION / TEMPORAL GROUP
    # --------------------------------------------------------

    for group in groups:

        group_anchor_ids = set(
            group["anchor_ids"]
        )

        # Ignore groups that have no samples from this split.
        if not group_anchor_ids.intersection(
            requested_anchor_ids
        ):
            continue

        augmentation = group.get(
            "augmentation",
            "none"
        )

        print(
            f"Processing group | "
            f"augmentation={augmentation}"
        )

        # ----------------------------------------------------
        # Original EarthSwin forward pass
        # ----------------------------------------------------

        logits = forward_group(
            model,
            loader,
            graph_split=split_name,
            group=group,
        )

        # ----------------------------------------------------
        # Convert graph-wide logits into supervised
        # anchor predictions.
        # ----------------------------------------------------

        positions = loader.get_node_positions(
            group["anchor_ids"]
        )

        supervised_logits = logits[
            positions
        ]

        probabilities = torch.sigmoid(
            supervised_logits
        ).detach().cpu().numpy()

        labels = np.asarray(
            group["labels"]
        )

        # ----------------------------------------------------
        # Save prediction metadata
        # ----------------------------------------------------

        for i, anchor_id in enumerate(
            group["anchor_ids"]
        ):

            matching_rows = split_df[
                split_df["anchor_id"] == anchor_id
            ].copy()

            # Match the exact augmentation group.
            matching_rows = matching_rows[
                matching_rows[
                    "augmentation"
                ].astype(str)
                ==
                str(augmentation)
            ]

            for _, row in matching_rows.iterrows():

                prediction_rows.append({

                    "split": split_name,

                    "anchor_id": anchor_id,

                    "week_1": row["week_1"],

                    "week_2": row["week_2"],

                    "week_3": row["week_3"],

                    "target_week": row["target_week"],

                    "label": int(row["label"]),

                    "augmentation": augmentation,

                    "probability": float(
                        probabilities[i]
                    ),
                })

    predictions = pd.DataFrame(
        prediction_rows
    )

    print(
        f"\nPredictions generated: "
        f"{len(predictions)}"
    )

    return predictions


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    predictions,
    threshold,
):

    y_true = predictions[
        "label"
    ].astype(int).to_numpy()

    y_prob = predictions[
        "probability"
    ].astype(float).to_numpy()

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    average_precision = (
        average_precision_score(
            y_true,
            y_prob,
        )
        if np.sum(y_true) > 0
        else None
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    return {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "average_precision": (
            None
            if average_precision is None
            else float(average_precision)
        ),

        "tn": int(cm[0, 0]),

        "fp": int(cm[0, 1]),

        "fn": int(cm[1, 0]),

        "tp": int(cm[1, 1]),
    }


# ============================================================
# THRESHOLD SWEEP
# ============================================================

def threshold_sweep(
    predictions
):

    thresholds = np.linspace(
        0.01,
        0.99,
        99,
    )

    results = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            predictions,
            threshold,
        )

        results.append(
            metrics
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD DIAGNOSTIC CSV
    # ========================================================

    print(
        "\nLoading:",
        BALANCED_CSV
    )

    if os.path.exists(BALANCED_CSV):
        diagnostic_df = pd.read_csv(BALANCED_CSV)
    else:
        print(
            "\nDiagnostic CSV not found; "
            "using the repository val.csv and test.csv files."
        )
        diagnostic_df = pd.concat(
            [
                pd.read_csv(VAL_ORIGINAL_CSV),
                pd.read_csv(TEST_ORIGINAL_CSV),
            ],
            ignore_index=True,
        )

    print(
        "\nDiagnostic dataset:"
    )

    print(
        "Total:",
        len(diagnostic_df)
    )

    print(
        "Positive:",
        int(
            (diagnostic_df["label"] == 1).sum()
        )
    )

    print(
        "Negative:",
        int(
            (diagnostic_df["label"] == 0).sum()
        )
    )

    print(
        "\nBy split:"
    )

    print(
        diagnostic_df.groupby(
            ["split", "label"]
        ).size()
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # VAL INFERENCE
    # ========================================================

    val_predictions = collect_predictions(
        model=model,
        diagnostic_df=diagnostic_df,
        split_name="val",
        original_csv_path=VAL_ORIGINAL_CSV,
    )

    # ========================================================
    # TEST INFERENCE
    # ========================================================

    test_predictions = collect_predictions(
        model=model,
        diagnostic_df=diagnostic_df,
        split_name="test",
        original_csv_path=TEST_ORIGINAL_CSV,
    )

    # ========================================================
    # SAVE SPLIT-SPECIFIC PREDICTIONS
    # ========================================================

    if len(val_predictions) > 0:

        val_predictions.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "val_predictions.csv"
            ),
            index=False
        )

    if len(test_predictions) > 0:

        test_predictions.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "test_predictions.csv"
            ),
            index=False
        )

    # ========================================================
    # COMBINE PREDICTIONS
    #
    # THIS IS WHERE WE DO THE THING YOU WERE TALKING ABOUT.
    #
    # Inference is separate.
    # Metrics are combined.
    # ========================================================

    all_predictions = pd.concat(
        [
            val_predictions,
            test_predictions,
        ],
        ignore_index=True
    )

    print("\n========================================")
    print("COMBINED DIAGNOSTIC PREDICTIONS")
    print("========================================")

    print(
        "Total predictions:",
        len(all_predictions)
    )

    print(
        "Positive:",
        int(
            (all_predictions["label"] == 1).sum()
        )
    )

    print(
        "Negative:",
        int(
            (all_predictions["label"] == 0).sum()
        )
    )

    # ========================================================
    # SAVE ALL RAW PREDICTIONS
    # ========================================================

    all_predictions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "all_predictions.csv"
        ),
        index=False
    )

    # ========================================================
    # ONE THRESHOLD SWEEP ON ALL 96 PREDICTIONS
    # ========================================================

    threshold_results = threshold_sweep(
        all_predictions
    )

    threshold_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "threshold_sweep.csv"
        ),
        index=False
    )

    # ========================================================
    # FIND BEST F1 THRESHOLD
    # ========================================================

    best_row = threshold_results.loc[
        threshold_results["f1"].idxmax()
    ]

    best_threshold = float(
        best_row["threshold"]
    )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    final_metrics = calculate_metrics(
        all_predictions,
        best_threshold,
    )

    # ========================================================
    # ADD PREDICTED LABEL TO EACH ROW
    # ========================================================

    all_predictions[
        "predicted_label"
    ] = (
        all_predictions[
            "probability"
        ]
        >= best_threshold
    ).astype(int)

    all_predictions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "all_predictions_with_labels.csv"
        ),
        index=False
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        os.path.join(
            OUTPUT_DIR,
            "diagnostic_metrics.json"
        ),
        "w"
    ) as f:

        json.dump(
            final_metrics,
            f,
            indent=4
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n========================================")
    print("FINAL COMBINED DIAGNOSTIC METRICS")
    print("========================================")

    print(
        f"Threshold : {best_threshold:.3f}"
    )

    print(
        f"Accuracy  : "
        f"{final_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{final_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{final_metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{final_metrics['f1']:.4f}"
    )

    print(
        f"AP        : "
        f"{final_metrics['average_precision']:.4f}"
    )

    print(
        f"\nTN = {final_metrics['tn']}"
    )

    print(
        f"FP = {final_metrics['fp']}"
    )

    print(
        f"FN = {final_metrics['fn']}"
    )

    print(
        f"TP = {final_metrics['tp']}"
    )

    print("\n========================================")
    print("OUTPUT FILES")
    print("========================================")

    print(
        "val_predictions.csv"
    )

    print(
        "test_predictions.csv"
    )

    print(
        "all_predictions.csv"
    )

    print(
        "all_predictions_with_labels.csv"
    )

    print(
        "threshold_sweep.csv"
    )

    print(
        "diagnostic_metrics.json"
    )

    print(
        "\nSaved to:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()