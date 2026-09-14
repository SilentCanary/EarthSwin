import os
import numpy as np

patch_dir = "C:/Users/advit/Documents/major project/dataset/EarthSentinel_2016/multiscale_patches"

files = sorted(
    f for f in os.listdir(patch_dir)
    if f.endswith(".npz")
)

completely_empty = 0
partially_valid = 0
fully_valid = 0

valid_fractions = []

for i, file_name in enumerate(files):

    file_path = os.path.join(
        patch_dir,
        file_name
    )

    with np.load(file_path) as data:

        patch = data["patch_256"]

        # Valid if at least one band has a finite value
        valid = np.isfinite(patch).any(axis=1)

        valid_fraction = valid.mean()

        valid_fractions.append(valid_fraction)

        if valid_fraction == 0:
            completely_empty += 1

        elif valid_fraction == 1:
            fully_valid += 1

        else:
            partially_valid += 1

print("=" * 60)
print("PATCH VALIDITY CHECK")
print("=" * 60)

print(f"Total patches       : {len(files)}")
print(f"Completely empty    : {completely_empty}")
print(f"Partially valid     : {partially_valid}")
print(f"Fully valid         : {fully_valid}")

print("\nValid-pixel fraction:")

print(
    f"Minimum             : {min(valid_fractions):.4f}"
)

print(
    f"Maximum             : {max(valid_fractions):.4f}"
)

print(
    f"Mean                : {np.mean(valid_fractions):.4f}"
)

print(
    f"Median              : {np.median(valid_fractions):.4f}"
)

print("=" * 60)