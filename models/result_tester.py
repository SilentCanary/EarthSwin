import os
import json
import numpy as np
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from earthswin_model import EarthSwin
from graph_dataset import GraphTrainingDataset
from train_earthswin import forward_group


MODEL_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BASE_DIR = os.path.dirname(
    MODEL_DIR
)


DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "EarthSentinel_2016"
)


TEMPORAL_DIR = os.path.join(
    DATASET_DIR,
    "temporal_dataset"
)


GRAPH_DIR = os.path.join(
    DATASET_DIR,
    "graphs"
)


CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "earthswin_best.pt"
)


VAL_CSV = os.path.join(
    TEMPORAL_DIR,
    "val.csv"
)


TEST_CSV = os.path.join(
    TEMPORAL_DIR,
    "test.csv"
)


RESULTS_PATH = os.path.join(
    MODEL_DIR,
    "checkpoint_evaluation_results.json"
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def load_model():

    if not os.path.exists(
        CHECKPOINT_PATH
    ):
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    print(
        "\nLoading checkpoint..."
    )

    model = EarthSwin()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Train loss: "
        f"{checkpoint.get('train_loss', 'unknown')}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint.get('val_loss', 'unknown')}"
    )

    return model, checkpoint


def collect_predictions(
    model,
    split_name,
    csv_path
):

    split_anchor_csv = csv_path

    if not os.path.exists(
        csv_path
    ):
        raise FileNotFoundError(
            f"\nCSV file not found:\n"
            f"{csv_path}"
        )


    graph_split_dir = os.path.join(
        GRAPH_DIR,
        split_name
    )

    if not os.path.exists(
        graph_split_dir
    ):
        raise FileNotFoundError(
            f"\nGraph directory not found:\n"
            f"{graph_split_dir}"
        )


    print(
        f"\nUsing {split_name} CSV:"
    )

    print(
        csv_path
    )


    print(
        f"Using {split_name} anchor CSV:"
    )

    print(
        split_anchor_csv
    )


    print(
        f"Using {split_name} graphs:"
    )

    print(
        graph_split_dir
    )


    dataset = GraphTrainingDataset(
        csv_path=csv_path,
        split_anchor_csv=split_anchor_csv,
        normalize=True
    )


    print(
        f"{split_name.capitalize()} graph anchors: "
        f"{len(dataset.anchor_ids)}"
    )


    groups = dataset.get_training_groups()


    print(
        f"{split_name.capitalize()} groups: "
        f"{len(groups)}"
    )


    labels = []
    scores = []


    model.eval()


    with torch.no_grad():

        for group_index, group in enumerate(
            groups,
            start=1
        ):

            print(
                f"\nProcessing "
                f"{split_name} group "
                f"{group_index}/{len(groups)}"
            )


            logits = forward_group(
                model,
                dataset,
                split_name,
                group
            )


            if logits.ndim == 1:

                probabilities = torch.sigmoid(
                    logits
                )

            elif (
                logits.ndim == 2
                and logits.shape[1] == 1
            ):

                probabilities = torch.sigmoid(
                    logits.squeeze(1)
                )

            elif (
                logits.ndim == 2
                and logits.shape[1] == 2
            ):

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )[:, 1]

            else:

                raise ValueError(
                    f"Unexpected logits shape: "
                    f"{tuple(logits.shape)}"
                )


            probabilities = (
                probabilities
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )


            group_labels = np.asarray(
                group["labels"]
            ).reshape(-1)


            if len(group_labels) != len(
                probabilities
            ):

                raise ValueError(
                    "\nLabel/prediction size mismatch "
                    f"in {split_name} group "
                    f"{group_index}.\n"
                    f"Labels: {len(group_labels)}\n"
                    f"Predictions: {len(probabilities)}"
                )


            labels.extend(
                group_labels.tolist()
            )

            scores.extend(
                probabilities.tolist()
            )


    labels = np.asarray(
        labels,
        dtype=np.int64
    )


    scores = np.asarray(
        scores,
        dtype=np.float64
    )


    print(
        f"\nFinished {split_name} evaluation."
    )

    print(
        f"Total samples: "
        f"{len(labels)}"
    )

    print(
        f"Positive samples: "
        f"{labels.sum()}"
    )

    print(
        f"Negative samples: "
        f"{len(labels) - labels.sum()}"
    )


    return labels, scores


def select_threshold(
    labels,
    scores
):

    print(
        "\nSelecting threshold using "
        "validation set..."
    )


    thresholds = np.linspace(
        0.05,
        0.95,
        181
    )


    best_threshold = 0.5
    best_f1 = -1.0


    for threshold in thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)


        f1 = f1_score(
            labels,
            predictions,
            zero_division=0
        )


        if f1 > best_f1:

            best_f1 = f1

            best_threshold = threshold


    print(
        f"Best validation threshold: "
        f"{best_threshold:.4f}"
    )


    print(
        f"Validation F1 at threshold: "
        f"{best_f1:.4f}"
    )


    return best_threshold


def calculate_metrics(
    labels,
    scores,
    threshold
):

    predictions = (
        scores >= threshold
    ).astype(int)


    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )


    if len(
        np.unique(labels)
    ) > 1:

        average_precision = (
            average_precision_score(
                labels,
                scores
            )
        )

    else:

        average_precision = float(
            "nan"
        )


    cm = confusion_matrix(
        labels,
        predictions
    )


    report = classification_report(
        labels,
        predictions,
        zero_division=0
    )


    return {

        "threshold":
            float(threshold),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "average_precision":
            float(average_precision),

        "confusion_matrix":
            cm.tolist(),

        "classification_report":
            report
    }


def print_results(
    name,
    metrics
):

    print(
        "\n" + "=" * 60
    )

    print(
        f"{name.upper()} RESULTS"
    )

    print(
        "=" * 60
    )


    print(
        f"Threshold:         "
        f"{metrics['threshold']:.4f}"
    )


    print(
        f"Precision:         "
        f"{metrics['precision']:.4f}"
    )


    print(
        f"Recall:            "
        f"{metrics['recall']:.4f}"
    )


    print(
        f"F1:                "
        f"{metrics['f1']:.4f}"
    )


    print(
        f"Average Precision: "
        f"{metrics['average_precision']:.4f}"
    )


    print(
        "\nConfusion Matrix:"
    )


    print(
        np.asarray(
            metrics[
                "confusion_matrix"
            ]
        )
    )


    print(
        "\nClassification Report:"
    )


    print(
        metrics[
            "classification_report"
        ]
    )


def main():

    print(
        "=" * 60
    )

    print(
        "EARTHSWIN CHECKPOINT EVALUATION"
    )

    print(
        "=" * 60
    )


    print(
        f"\nRepository: "
        f"{BASE_DIR}"
    )

    print(
        f"Models:     "
        f"{MODEL_DIR}"
    )

    print(
        f"Dataset:    "
        f"{DATASET_DIR}"
    )

    print(
        f"Device:     "
        f"{DEVICE}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )


    print(
        "\nChecking paths..."
    )


    paths = {

        "Dataset":
            DATASET_DIR,

        "Temporal dataset":
            TEMPORAL_DIR,

        "Graphs":
            GRAPH_DIR,

        "Checkpoint":
            CHECKPOINT_PATH,

        "Validation CSV":
            VAL_CSV,

        "Test CSV":
            TEST_CSV,

        "Validation graphs":
            os.path.join(
                GRAPH_DIR,
                "val"
            ),

        "Test graphs":
            os.path.join(
                GRAPH_DIR,
                "test"
            )
    }


    missing_paths = []


    for name, path in paths.items():

        if os.path.exists(path):

            print(
                f"[OK] {name}: {path}"
            )

        else:

            print(
                f"[MISSING] "
                f"{name}: {path}"
            )

            missing_paths.append(
                path
            )


    if missing_paths:

        raise FileNotFoundError(
            "\nRequired paths are missing. "
            "Fix the paths above before evaluation."
        )


    model, checkpoint = load_model()


    print(
        "\n"
        + "=" * 60
    )

    print(
        "COLLECTING VALIDATION PREDICTIONS"
    )

    print(
        "=" * 60
    )


    val_labels, val_scores = (
        collect_predictions(
            model,
            "val",
            VAL_CSV
        )
    )


    print(
        f"\nValidation samples: "
        f"{len(val_labels)}"
    )


    print(
        f"Validation positives: "
        f"{val_labels.sum()}"
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "COLLECTING TEST PREDICTIONS"
    )

    print(
        "=" * 60
    )


    test_labels, test_scores = (
        collect_predictions(
            model,
            "test",
            TEST_CSV
        )
    )


    print(
        f"\nTest samples: "
        f"{len(test_labels)}"
    )


    print(
        f"Test positives: "
        f"{test_labels.sum()}"
    )


    threshold = select_threshold(
        val_labels,
        val_scores
    )


    val_metrics = calculate_metrics(
        val_labels,
        val_scores,
        threshold
    )


    test_metrics = calculate_metrics(
        test_labels,
        test_scores,
        threshold
    )


    print_results(
        "Validation",
        val_metrics
    )


    print_results(
        "Test",
        test_metrics
    )


    results = {

        "checkpoint":
            CHECKPOINT_PATH,

        "checkpoint_epoch":
            checkpoint.get(
                "epoch",
                None
            ),

        "train_loss":
            checkpoint.get(
                "train_loss",
                None
            ),

        "validation_loss":
            checkpoint.get(
                "val_loss",
                None
            ),

        "device":
            str(DEVICE),

        "threshold_selected_on":
            "validation",

        "threshold":
            float(threshold),

        "validation":
            val_metrics,

        "test":
            test_metrics
    }


    with open(
        RESULTS_PATH,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 60
    )


    print(
        f"\nResults saved to:"
    )

    print(
        RESULTS_PATH
    )


if __name__ == "__main__":

    main()