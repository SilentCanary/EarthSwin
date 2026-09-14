import sys
import torch
import numpy as np

sys.path.append(
    "C:/Users/advit/Documents/major project/models"
)

from patch_pyramid import PatchPyramid


# --------------------------------------------------
# Load one real sample
# --------------------------------------------------

sample_path = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/multiscale_patches/"
    "sample_000796.npz"
)

sample = np.load(sample_path)

patch_64 = sample["patch_64"]
patch_128 = sample["patch_128"]
patch_256 = sample["patch_256"]


# --------------------------------------------------
# Select one week
# --------------------------------------------------

week_index = 9

patch_64 = patch_64[week_index]
patch_128 = patch_128[week_index]
patch_256 = patch_256[week_index]


# --------------------------------------------------
# Create 6 channels
# 3 data + 3 missingness masks
# --------------------------------------------------

def prepare_patch(patch):

    data = patch.astype(np.float32)

    missing_mask = (
        ~np.isfinite(data)
    ).astype(np.float32)

    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Normalize roughly for architecture testing.
    # Exact training normalization will be handled
    # by the actual Dataset.
    data = torch.from_numpy(data)
    missing_mask = torch.from_numpy(missing_mask)

    data = torch.cat(
        [data, missing_mask],
        dim=0
    )

    return data.unsqueeze(0)


patch_64 = prepare_patch(patch_64)
patch_128 = prepare_patch(patch_128)
patch_256 = prepare_patch(patch_256)


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = PatchPyramid(
    in_channels=6,
    cnn_dim=256,
    swin_dim=768,
    scale_feature_dim=512,
    output_dim=512
)

model.eval()


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

with torch.no_grad():

    output = model(
        patch_64,
        patch_128,
        patch_256
    )


# --------------------------------------------------
# Check results
# --------------------------------------------------

print("Patch 64 shape:", patch_64.shape)
print("Patch 128 shape:", patch_128.shape)
print("Patch 256 shape:", patch_256.shape)

print("Pyramid output shape:", output.shape)

print("NaNs:", torch.isnan(output).sum().item())
print("Infs:", torch.isinf(output).sum().item())


# --------------------------------------------------
# Expected result
# --------------------------------------------------

expected_shape = (1,512)

assert output.shape == expected_shape
assert not torch.isnan(output).any()
assert not torch.isinf(output).any()

print()
print("PATCH PYRAMID TEST PASSED")