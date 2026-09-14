import os
import glob
import numpy as np

CHUNK_DIR = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/patch_chunks"
)

LABELS_PATH = "labels.npy"

labels = np.load(LABELS_PATH)

chunk_files = sorted(
    glob.glob(os.path.join(CHUNK_DIR, "*.npz"))
)

retained_anchor_ids = []

for path in chunk_files:
    with np.load(path) as data:
        retained_anchor_ids.extend(
            data["anchor_id"].tolist()
        )

retained_anchor_ids = np.array(
    retained_anchor_ids,
    dtype=np.int64
)

retained_anchor_ids = np.unique(retained_anchor_ids)

positive_locations = np.argwhere(labels == 1)

retained_positive_locations = [
    (int(anchor_id), int(week_idx + 1))
    for anchor_id, week_idx in positive_locations
    if anchor_id in set(retained_anchor_ids)
]

print("=" * 60)
print("FINAL DATASET LABEL CHECK")
print("=" * 60)

print(f"Original anchors       : {labels.shape[0]}")
print(f"Retained anchors       : {len(retained_anchor_ids)}")
print(f"Weeks                  : {labels.shape[1]}")

print()
print(f"Original positives     : {len(positive_locations)}")
print(f"Retained positives     : {len(retained_positive_locations)}")
print(
    f"Removed positives      : "
    f"{len(positive_locations) - len(retained_positive_locations)}"
)

print()
print("Retained positive events:")

for anchor_id, week in retained_positive_locations:
    print(
        f"  Anchor {anchor_id:4d} | Week {week:2d}"
    )

print()
print("=" * 60)

final_samples = len(retained_anchor_ids) * 14
final_positives = len(retained_positive_locations)
final_negatives = final_samples - final_positives

print("FINAL SPATIAL DATASET")
print("=" * 60)

print(f"Total anchor-week samples : {final_samples}")
print(f"Positive samples          : {final_positives}")
print(f"Negative samples          : {final_negatives}")

temporal_samples = len(retained_anchor_ids) * 12
temporal_negatives = temporal_samples - final_positives

print()
print("3-WEEK TEMPORAL DATASET")
print("=" * 60)

print(f"Total sequences            : {temporal_samples}")
print(f"Positive sequences         : {final_positives}")
print(f"Negative sequences         : {temporal_negatives}")