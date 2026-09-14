
import os
import numpy as np


def is_spatially_valid(patch_256):
    """
    Check whether a spatial patch contains any valid geographic data.

    Rainfall and slope share the static geographic footprint.
    We use these bands instead of VV because VV can be missing
    for an entire week when Sentinel-1 data is unavailable.

    patch_256 shape:
        [weeks, bands, height, width]

    Band order:
        0 = VV
        1 = rainfall
        2 = slope

    Returns
    -------
    bool
        True if at least one spatial pixel belongs to the
        valid geographic region.
    """

    rainfall = patch_256[:, 1, :, :]
    slope = patch_256[:, 2, :, :]

    valid_rainfall = np.isfinite(rainfall)
    valid_slope = np.isfinite(slope)

    valid_spatial = (
        valid_rainfall | valid_slope
    )

    return np.any(valid_spatial)


def create_chunks(
    input_dir,
    output_dir,
    patches_per_chunk=100,
):
    """
    Group valid multi-scale .npz samples into storage chunks.

    Completely empty spatial patches are skipped.

    Partially valid boundary patches are retained.

    Original sample/anchor IDs are preserved.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No .npz samples found in: {input_dir}"
        )

    print(f"Found {len(files)} patch files.")

    # --------------------------------------------------
    # FIRST PASS
    # Determine which patches contain valid geography.
    # --------------------------------------------------

    valid_files = []
    skipped_files = []

    print("\nChecking spatial validity...")

    for idx, file_name in enumerate(files):

        file_path = os.path.join(
            input_dir,
            file_name,
        )

        with np.load(file_path) as data:

            patch_256 = data["patch_256"]

            if is_spatially_valid(patch_256):
                valid_files.append(file_name)
            else:
                skipped_files.append(file_name)

        if (idx + 1) % 500 == 0:
            print(
                f"Checked {idx + 1}/{len(files)} patches"
            )

    print("\n" + "=" * 60)
    print("SPATIAL FILTERING")
    print("=" * 60)

    print(
        f"Total patch files : {len(files)}"
    )

    print(
        f"Valid patches     : {len(valid_files)}"
    )

    print(
        f"Skipped empty     : {len(skipped_files)}"
    )

    print("=" * 60)

    if not valid_files:
        raise RuntimeError(
            "No valid patches found."
        )

    # --------------------------------------------------
    # CREATE CHUNKS
    # --------------------------------------------------

    chunk_idx = 0

    for start in range(
        0,
        len(valid_files),
        patches_per_chunk
    ):

        chunk_files = valid_files[
            start:start + patches_per_chunk
        ]

        patch_64_list = []
        patch_128_list = []
        patch_256_list = []

        rows = []
        cols = []
        anchor_ids = []

        for file_name in chunk_files:

            file_path = os.path.join(
                input_dir,
                file_name,
            )

            with np.load(file_path) as data:

                patch_64_list.append(
                    data["patch_64"]
                )

                patch_128_list.append(
                    data["patch_128"]
                )

                patch_256_list.append(
                    data["patch_256"]
                )

                rows.append(
                    data["center_row"]
                )

                cols.append(
                    data["center_col"]
                )

                # Preserve the original anchor ID.
                #
                # sample_002401.npz -> 2401
                anchor_id = int(
                    os.path.splitext(file_name)[0]
                    .split("_")[1]
                )

                anchor_ids.append(anchor_id)

        chunk = {
            "patch_64": np.stack(
                patch_64_list
            ),

            "patch_128": np.stack(
                patch_128_list
            ),

            "patch_256": np.stack(
                patch_256_list
            ),

            "center_row": np.asarray(
                rows
            ),

            "center_col": np.asarray(
                cols
            ),

            "anchor_id": np.asarray(
                anchor_ids
            ),
        }

        output_file = os.path.join(
            output_dir,
            f"chunk_{chunk_idx:03d}.npz",
        )

        np.savez_compressed(
            output_file,
            **chunk,
        )

        print(
            f"Saved {output_file} "
            f"with {len(chunk_files)} samples."
        )

        chunk_idx += 1

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)

    print(
        f"Input patches     : {len(files)}"
    )

    print(
        f"Valid patches      : {len(valid_files)}"
    )

    print(
        f"Skipped empty      : {len(skipped_files)}"
    )

    print(
        f"Chunks created     : {chunk_idx}"
    )

    print("=" * 60)


if __name__ == "__main__":

    create_chunks(

        input_dir=(
            "C:/Users/advit/Documents/"
            "major project/dataset/"
            "EarthSentinel_2016/"
            "multiscale_patches"
        ),

        output_dir=(
            "C:/Users/advit/Documents/"
            "major project/dataset/"
            "EarthSentinel_2016/"
            "patch_chunks"
        ),

        patches_per_chunk=100,
    )
