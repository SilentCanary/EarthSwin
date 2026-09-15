import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.append(
    "C:/Users/advit/Documents/major project/models"
)

from earthswin_model import EarthSwin
from graph_dataset import GraphTrainingDataset


def compute_pr_metrics(scores, labels, precision_floor=0.3):

    order = np.argsort(-scores)
    scores, labels = scores[order], labels[order]

    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    total_pos = labels.sum()

    if total_pos == 0:
        return {"pr_auc": float("nan"), "recall_at_precision": float("nan")}

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos

    trapezoid_fn = getattr(np, "trapezoid", None) or np.trapz
    pr_auc = trapezoid_fn(precision, recall)

    valid = precision >= precision_floor
    recall_at_precision = recall[valid].max() if valid.any() else 0.0

    return {
        "pr_auc": float(pr_auc),
        "recall_at_precision": float(recall_at_precision),
    }


PROJECT_DIR = "C:/Users/advit/Documents/major project"

TEMPORAL_DIR = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/temporal_dataset"
)

GRAPH_DIR = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/graphs"
)

MODEL_DIR = (
    "C:/Users/advit/Documents/major project/"
    "models"
)

TRAIN_BALANCED_CSV = os.path.join(
    TEMPORAL_DIR,
    "train_balanced.csv"
)

TRAIN_ORIGINAL_CSV = os.path.join(
    TEMPORAL_DIR,
    "train_original.csv"
)

VAL_CSV = os.path.join(
    TEMPORAL_DIR,
    "val.csv"
)

VAL_ORIGINAL_CSV = os.path.join(
    TEMPORAL_DIR,
    "val.csv"
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "earthswin_best.pt"
)

# TRAINING SETTINGS

EPOCHS = 15

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

BATCH_SIZE = 8

GRADIENT_CLIP = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# LOAD GRAPH


def load_graph(split, week):

    path = os.path.join(
        GRAPH_DIR,
        split,
        f"graph_week_{week:02d}.pt"
    )

    graph = torch.load(
        path,
        map_location="cpu"
    )

    return graph


# LOAD A BATCH OF ANCHORS

def load_anchor_batch(
    loader,
    anchor_ids,
    week_1,
    week_2,
    week_3,
    augmentation
):

    patch_64_list = []
    patch_128_list = []
    patch_256_list = []

    for anchor_id in anchor_ids:

        patch_64, patch_128, patch_256 = loader.load_anchor(
            anchor_id=anchor_id,
            week_1=week_1,
            week_2=week_2,
            week_3=week_3,
            augmentation=augmentation
        )

        patch_64_list.append(patch_64)
        patch_128_list.append(patch_128)
        patch_256_list.append(patch_256)

    patch_64 = torch.stack(
        patch_64_list,
        dim=0
    )

    patch_128 = torch.stack(
        patch_128_list,
        dim=0
    )

    patch_256 = torch.stack(
        patch_256_list,
        dim=0
    )

    week_1_64, week_2_64, week_3_64 = loader.split_into_weeks(
        patch_64
    )

    week_1_128, week_2_128, week_3_128 = loader.split_into_weeks(
        patch_128
    )

    week_1_256, week_2_256, week_3_256 = loader.split_into_weeks(
        patch_256
    )

    week_1_inputs = {
        "patch_64": week_1_64,
        "patch_128": week_1_128,
        "patch_256": week_1_256
    }

    week_2_inputs = {
        "patch_64": week_2_64,
        "patch_128": week_2_128,
        "patch_256": week_2_256
    }

    week_3_inputs = {
        "patch_64": week_3_64,
        "patch_128": week_3_128,
        "patch_256": week_3_256
    }

    return (
        week_1_inputs,
        week_2_inputs,
        week_3_inputs
    )


# COMPUTE PATCH PYRAMID FEATURES FOR ALL NODES


def compute_pyramid_features(
    model,
    loader,
    week_1,
    week_2,
    week_3,
    augmentation
):

    all_week_1_features = []
    all_week_2_features = []
    all_week_3_features = []

    total_anchors = len(loader.anchor_ids)

    for start in range(
        0,
        total_anchors,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total_anchors
        )

        batch_anchor_ids = loader.anchor_ids[
            start:end
        ]

        (
            week_1_inputs,
            week_2_inputs,
            week_3_inputs
        ) = load_anchor_batch(
            loader=loader,
            anchor_ids=batch_anchor_ids,
            week_1=week_1,
            week_2=week_2,
            week_3=week_3,
            augmentation=augmentation
        )

        week_1_inputs = {
            key: value.to(DEVICE)
            for key, value in week_1_inputs.items()
        }

        week_2_inputs = {
            key: value.to(DEVICE)
            for key, value in week_2_inputs.items()
        }

        week_3_inputs = {
            key: value.to(DEVICE)
            for key, value in week_3_inputs.items()
        }

        features_1 = model.patch_pyramid(
            week_1_inputs["patch_64"],
            week_1_inputs["patch_128"],
            week_1_inputs["patch_256"]
        )

        features_2 = model.patch_pyramid(
            week_2_inputs["patch_64"],
            week_2_inputs["patch_128"],
            week_2_inputs["patch_256"]
        )

        features_3 = model.patch_pyramid(
            week_3_inputs["patch_64"],
            week_3_inputs["patch_128"],
            week_3_inputs["patch_256"]
        )

        all_week_1_features.append(
            features_1
        )

        all_week_2_features.append(
            features_2
        )

        all_week_3_features.append(
            features_3
        )

        print(
            f"  Encoded anchors "
            f"{end}/{total_anchors}",
            end="\r"
        )

    print()

    week_1_features = torch.cat(
        all_week_1_features,
        dim=0
    )

    week_2_features = torch.cat(
        all_week_2_features,
        dim=0
    )

    week_3_features = torch.cat(
        all_week_3_features,
        dim=0
    )

    return (
        week_1_features,
        week_2_features,
        week_3_features
    )


# FORWARD ONE GROUP


def forward_group(
    model,
    loader,
    graph_split,
    group
):

    week_1 = group["week_1"]
    week_2 = group["week_2"]
    week_3 = group["week_3"]

    augmentation = group["augmentation"]

    print(
        f"  Window: "
        f"{week_1} -> {week_2} -> {week_3} "
        f"(target {group['target_week']})"
    )

    print(
        f"  Augmentation: {augmentation}"
    )

    (
        week_1_features,
        week_2_features,
        week_3_features
    ) = compute_pyramid_features(
        model=model,
        loader=loader,
        week_1=week_1,
        week_2=week_2,
        week_3=week_3,
        augmentation=augmentation
    )

    graph_1 = load_graph(
        graph_split,
        week_1
    )

    graph_2 = load_graph(
        graph_split,
        week_2
    )

    graph_3 = load_graph(
        graph_split,
        week_3
    )

    edge_index_1 = graph_1["edge_index"].to(
        DEVICE
    )

    edge_weights_1 = graph_1["edge_weights"].to(
        DEVICE
    )

    edge_index_2 = graph_2["edge_index"].to(
        DEVICE
    )

    edge_weights_2 = graph_2["edge_weights"].to(
        DEVICE
    )

    edge_index_3 = graph_3["edge_index"].to(
        DEVICE
    )

    edge_weights_3 = graph_3["edge_weights"].to(
        DEVICE
    )

    spatial_features_1 = model.gat(
        week_1_features,
        edge_index_1,
        edge_weights_1
    )

    spatial_features_2 = model.gat(
        week_2_features,
        edge_index_2,
        edge_weights_2
    )

    spatial_features_3 = model.gat(
        week_3_features,
        edge_index_3,
        edge_weights_3
    )

    temporal_features = torch.stack(
        [
            spatial_features_1,
            spatial_features_2,
            spatial_features_3
        ],
        dim=1
    )

    temporal_output = model.temporal_transformer(
        temporal_features
    )

    logits = model.classifier(
        temporal_output
    )

    return logits


# TRAIN ONE EPOCH


def train_one_epoch(
    model,
    loader,
    groups,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    np.random.shuffle(groups)

    for group_number, group in enumerate(
        groups,
        start=1
    ):

        print(
            f"\nTraining group "
            f"{group_number}/{len(groups)}"
        )

        optimizer.zero_grad()

        logits = forward_group(
            model=model,
            loader=loader,
            graph_split="train",
            group=group
        )

        supervised_anchor_ids = group[
            "anchor_ids"
        ]

        positions = loader.get_node_positions(
            supervised_anchor_ids
        )

        positions = torch.tensor(
            positions,
            dtype=torch.long,
            device=DEVICE
        )

        labels = torch.tensor(
            group["labels"],
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(1)

        supervised_logits = logits[
            positions
        ]

        loss = criterion(
            supervised_logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )

        optimizer.step()

        batch_size = labels.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        print(
            f"  Supervised samples: "
            f"{batch_size}"
        )

        print(
            f"  Positive samples: "
            f"{int(labels.sum().item())}"
        )

        print(
            f"  Loss: {loss.item():.6f}"
        )

    average_loss = (
        total_loss / total_samples
    )

    return average_loss


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    groups,
    criterion
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    # NEW: collected across all validation groups so PR-AUC /
    # recall-at-precision can be computed over the whole val split,
    # not per-group (a single group may have 0 or 1 positives, which
    # isn't enough to compute a meaningful precision/recall curve on
    # its own).
    all_scores = []
    all_labels = []

    with torch.no_grad():

        for group_number, group in enumerate(
            groups,
            start=1
        ):

            print(
                f"\nValidation group "
                f"{group_number}/{len(groups)}"
            )

            logits = forward_group(
                model=model,
                loader=loader,
                graph_split="val",
                group=group
            )

            supervised_anchor_ids = group[
                "anchor_ids"
            ]

            positions = loader.get_node_positions(
                supervised_anchor_ids
            )

            positions = torch.tensor(
                positions,
                dtype=torch.long,
                device=DEVICE
            )

            labels = torch.tensor(
                group["labels"],
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(1)

            supervised_logits = logits[
                positions
            ]

            loss = criterion(
                supervised_logits,
                labels
            )

            batch_size = labels.shape[0]

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            print(
                f"  Loss: {loss.item():.6f}"
            )

            # NEW: stash this group's scores/labels for the PR metrics
            scores = torch.sigmoid(
                supervised_logits
            ).squeeze(1).detach().cpu().numpy()

            all_scores.append(scores)
            all_labels.append(
                labels.squeeze(1).cpu().numpy()
            )

    average_loss = (
        total_loss / total_samples
    )

    # NEW
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    val_metrics = compute_pr_metrics(all_scores, all_labels)

    return average_loss, val_metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("EARTHSWIN TRAINING")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    print("\nCreating training loader...")

    train_loader = GraphTrainingDataset(
        csv_path=TRAIN_BALANCED_CSV,
        split_anchor_csv=TRAIN_ORIGINAL_CSV,
        normalize=True
    )

    print(
        f"Train graph anchors: "
        f"{len(train_loader.anchor_ids)}"
    )

    train_groups = train_loader.get_training_groups()

    print(
        f"Training groups: "
        f"{len(train_groups)}"
    )

    print("\nCreating validation loader...")

    val_loader = GraphTrainingDataset(
        csv_path=VAL_CSV,
        split_anchor_csv=VAL_ORIGINAL_CSV,
        normalize=True
    )

    print(
        f"Validation graph anchors: "
        f"{len(val_loader.anchor_ids)}"
    )

    val_groups = val_loader.get_training_groups()

    print(
        f"Validation groups: "
        f"{len(val_groups)}"
    )


    print("\nCreating EarthSwin model...")

    model = EarthSwin(
        in_channels=6,
        feature_dim=512,
        num_heads=8,
        temporal_layers=2,
        temporal_feedforward_dim=1024,
        dropout=0.1
    )

    model = model.to(DEVICE)

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # LOSS

    criterion = nn.BCEWithLogitsLoss()


    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    # TRAINING LOOP
  

    best_val_loss = float("inf")
    best_val_score = -1.0  # NEW: recall-at-precision / PR-AUC, used for selection

    history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print("\n")
        print("=" * 60)
        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )
        print("=" * 60)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            groups=train_groups,
            criterion=criterion,
            optimizer=optimizer
        )

        print("\n")
        print(
            f"TRAIN LOSS: "
            f"{train_loss:.6f}"
        )

        val_loss, val_metrics = validate(
            model=model,
            loader=val_loader,
            groups=val_groups,
            criterion=criterion
        )

        print("\n")
        print(
            f"VALIDATION LOSS: "
            f"{val_loss:.6f}"
        )

        # NEW
        print(
            f"VALIDATION PR-AUC: "
            f"{val_metrics['pr_auc']:.4f}"
        )

        print(
            f"VALIDATION RECALL@P0.3: "
            f"{val_metrics['recall_at_precision']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pr_auc": val_metrics["pr_auc"],
                "val_recall_at_precision": val_metrics["recall_at_precision"]
            }
        )

        score = val_metrics["recall_at_precision"]

        if np.isnan(score):
            score = val_metrics["pr_auc"]

        if np.isnan(score):
            score = -val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if score > best_val_score:

            best_val_score = score

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_pr_auc": val_metrics["pr_auc"],
                    "val_recall_at_precision": val_metrics["recall_at_precision"]
                },
                BEST_MODEL_PATH
            )

            print(
                "\n*** NEW BEST MODEL SAVED ***"
            )

            print(
                f"Path: {BEST_MODEL_PATH}"
            )

    history_path = os.path.join(
        MODEL_DIR,
        "training_history.json"
    )

    with open(
        history_path,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation loss (for reference only, not used for selection): "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best validation score (recall@precision / PR-AUC, used for selection): "
        f"{best_val_score:.4f}"
    )

    print(
        f"Best model: "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"History: "
        f"{history_path}"
    )


if __name__ == "__main__":
    main()
