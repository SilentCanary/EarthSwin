"""
Create multi-scale spatial patches for the landslide project.

For every geographic anchor, extract:
    - 64 x 64
    - 128 x 128
    - 256 x 256

All three patches are centered on the same geographic location.

Missing values (NaNs) are preserved.
They will be handled later during dataset preparation.
"""

import os
import rasterio
import numpy as np


def load_weekly_files(data_dir):
    files = sorted(
        (
            f for f in os.listdir(data_dir)
            if f.lower().endswith(".tif")
            and f.startswith("HP_week_")
        ),
        key=lambda f: int(
            f.split("_week_")[1].split(".")[0]
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No weekly .tif files found in {data_dir}"
        )

    return files


def get_dataset_info(data_dir, files):
    first_file = os.path.join(
        data_dir,
        files[0]
    )

    with rasterio.open(first_file) as src:
        bands = src.count
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

    return bands, height, width, transform, crs


def extract_multiscale_patch(
    datasets,
    center_row,
    center_col,
    patch_sizes=(64, 128, 256),
):
    """
    Extract multiple spatial contexts around one geographic center.

    Returns
    -------
    patches : dict
        {
            64:  [weeks, bands, 64, 64],
            128: [weeks, bands, 128, 128],
            256: [weeks, bands, 256, 256]
        }

    NaN values are preserved.
    """

    bands = datasets[0].count
    patches = {}

    for patch_size in patch_sizes:

        half = patch_size // 2

        row_start = center_row - half
        col_start = center_col - half

        patch_data = np.empty(
            (
                len(datasets),
                bands,
                patch_size,
                patch_size,
            ),
            dtype=np.float32,
        )

        for week_idx, src in enumerate(datasets):

            window = rasterio.windows.Window(
                col_start,
                row_start,
                patch_size,
                patch_size,
            )

            patch = src.read(
                window=window
            ).astype(np.float32)

            patch_data[week_idx] = patch

        patches[patch_size] = patch_data

    return patches


def generate_anchor_locations(
    height,
    width,
    anchor_size=256,
    stride=256,
):
    """
    Generate centers for the largest spatial context.
    """

    half = anchor_size // 2

    anchors = []

    for row in range(
        half,
        height - half + 1,
        stride,
    ):
        for col in range(
            half,
            width - half + 1,
            stride,
        ):
            anchors.append((row, col))

    return anchors


def create_multiscale_patches(
    data_dir,
    output_dir,
    patch_sizes=(64, 128, 256),
    anchor_size=256,
    stride=256,
):
    """
    Create multi-scale samples for all geographic anchors.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------
    # LOAD WEEKLY FILES
    # --------------------------------------------------

    files = load_weekly_files(data_dir)

    bands, height, width, transform, crs = (
        get_dataset_info(
            data_dir,
            files
        )
    )

    # --------------------------------------------------
    # VERIFY EXPECTED DATASET
    # --------------------------------------------------

    if len(files) != 14:
        raise RuntimeError(
            f"Expected 14 weekly files, "
            f"but found {len(files)}"
        )

    if bands != 3:
        raise RuntimeError(
            f"Expected 3 bands, but found {bands}"
        )

    # --------------------------------------------------
    # GENERATE ANCHORS
    # --------------------------------------------------

    anchors = generate_anchor_locations(
        height=height,
        width=width,
        anchor_size=anchor_size,
        stride=stride,
    )

    # --------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------

    metadata = {
        "patch_sizes": patch_sizes,
        "anchor_size": anchor_size,
        "stride": stride,
        "bands": bands,
        "num_weeks": len(files),
        "height": height,
        "width": width,
        "num_anchors": len(anchors),
        "weekly_files": files,
        "transform": transform,
        "crs": str(crs),
    }

    np.save(
        os.path.join(
            output_dir,
            "metadata.npy"
        ),
        metadata,
        allow_pickle=True,
    )

    # --------------------------------------------------
    # PRINT DATASET INFORMATION
    # --------------------------------------------------

    print(f"Found {len(files)} weekly files.")
    print(f"Raster size: {height} x {width}")
    print(f"Bands: {bands}")
    print(f"Number of anchors: {len(anchors)}")

    print("\nWeekly files:")

    for file_name in files:
        print(" ", file_name)

    # --------------------------------------------------
    # OPEN ALL WEEKLY DATASETS
    # --------------------------------------------------

    datasets = [
        rasterio.open(
            os.path.join(
                data_dir,
                file_name
            )
        )
        for file_name in files
    ]

    try:

        # --------------------------------------------------
        # CREATE PATCHES
        # --------------------------------------------------

        for sample_idx, (row, col) in enumerate(
            anchors
        ):

            patches = extract_multiscale_patch(
                datasets=datasets,
                center_row=row,
                center_col=col,
                patch_sizes=patch_sizes,
            )

            if len(patches) != len(patch_sizes):
                continue

            output_file = os.path.join(
                output_dir,
                f"sample_{sample_idx:06d}.npz",
            )

            np.savez_compressed(
                output_file,
                patch_64=patches[64],
                patch_128=patches[128],
                patch_256=patches[256],
                center_row=row,
                center_col=col,
            )

            if (sample_idx + 1) % 50 == 0:

                print(
                    f"Processed {sample_idx + 1}/"
                    f"{len(anchors)} anchors"
                )

    finally:

        for src in datasets:
            src.close()

    print("\n" + "=" * 70)
    print("PATCH GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total anchors: {len(anchors)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":

    create_multiscale_patches(

        data_dir=(
            "C:/Users/advit/Documents/"
            "major project/dataset/"
            "EarthSentinel_2016/merged"
        ),

        output_dir=(
            "C:/Users/advit/Documents/"
            "major project/dataset/"
            "EarthSentinel_2016/multiscale_patches"
        ),

        patch_sizes=(64, 128, 256),

        anchor_size=256,

        stride=256,
    )