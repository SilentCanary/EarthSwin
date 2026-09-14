import numpy as np

LABELS_PATH = "labels.npy"

labels = np.load(LABELS_PATH)

print("=" * 70)
print("LABELS.NPY DIAGNOSTIC")
print("=" * 70)

print(f"Shape : {labels.shape}")
print(f"Dtype : {labels.dtype}")
print(f"Min   : {labels.min()}")
print(f"Max   : {labels.max()}")
print()

unique, counts = np.unique(labels, return_counts=True)

print("Unique values:")
for value, count in zip(unique, counts):
    print(f"  {value}: {count}")

print()

positive_locations = np.argwhere(labels == 1)

print(f"Total positive entries: {len(positive_locations)}")
print()

print("Positive locations:")
for anchor_id, week_idx in positive_locations:
    print(
        f"  Anchor {anchor_id:4d} | "
        f"Week {week_idx + 1:2d}"
    )

print()

print("=" * 70)
print("POSITIVES BY WEEK")
print("=" * 70)

for week_idx in range(labels.shape[1]):
    count = np.sum(labels[:, week_idx] == 1)
    print(
        f"Week {week_idx + 1:2d}: "
        f"{count} positive anchors"
    )

print()

print("=" * 70)
print("POSITIVES BY ANCHOR")
print("=" * 70)

positive_anchor_ids = np.unique(positive_locations[:, 0])

for anchor_id in positive_anchor_ids:
    weeks = np.where(labels[anchor_id] == 1)[0] + 1
    print(
        f"Anchor {anchor_id:4d}: "
        f"Weeks {weeks.tolist()}"
    )

print()

print("=" * 70)
print("DONE")
print("=" * 70)