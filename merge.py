import os
import glob
import rasterio
from rasterio.merge import merge

input_dir = r"C:\\Users\\advit\\Documents\\major project\\dataset\\EarthSentinel_2016"
output_dir = os.path.join(input_dir, "merged")

os.makedirs(output_dir, exist_ok=True)

weeks_to_merge = [1, 2, 3, 4, 5, 13, 14]

for week in weeks_to_merge:
    print("\n" + "=" * 70)
    print(f"WEEK {week}")
    print("=" * 70)

    pattern = os.path.join(
        input_dir,
        f"HP_week_{week}-*.tif"
    )

    files = sorted(glob.glob(pattern))

    if len(files) != 2:
        raise RuntimeError(
            f"Week {week}: expected exactly 2 files, "
            f"but found {len(files)}"
        )

    print("Input files:")

    for file in files:
        print(" ", os.path.basename(file))

    # --------------------------------------------------
    # Open both tiles
    # --------------------------------------------------

    src1 = rasterio.open(files[0])
    src2 = rasterio.open(files[1])

    try:

        # --------------------------------------------------
        # VERIFY THEY ARE COMPATIBLE
        # --------------------------------------------------

        if src1.crs != src2.crs:
            raise RuntimeError(
                f"Week {week}: CRS mismatch"
            )

        if src1.count != src2.count:
            raise RuntimeError(
                f"Week {week}: band count mismatch"
            )

        if src1.height != src2.height:
            raise RuntimeError(
                f"Week {week}: height mismatch"
            )

        if src1.dtypes != src2.dtypes:
            raise RuntimeError(
                f"Week {week}: dtype mismatch"
            )

        if src1.descriptions != src2.descriptions:
            raise RuntimeError(
                f"Week {week}: band name mismatch"
            )

        # Check pixel sizes
        if src1.res != src2.res:
            raise RuntimeError(
                f"Week {week}: pixel resolution mismatch"
            )

        # --------------------------------------------------
        # PRINT INPUT GEOMETRY
        # --------------------------------------------------

        print("\nTile 1:")
        print("  Size:", src1.width, "x", src1.height)
        print("  Bounds:", src1.bounds)

        print("\nTile 2:")
        print("  Size:", src2.width, "x", src2.height)
        print("  Bounds:", src2.bounds)

        # --------------------------------------------------
        # MERGE
        # --------------------------------------------------

        print("\nMerging...")

        mosaic, transform = merge(
            [src1, src2]
        )

        # --------------------------------------------------
        # BUILD OUTPUT PROFILE
        # --------------------------------------------------

        profile = src1.profile.copy()

        profile.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": mosaic.shape[0],
                "compress": "deflate",
                "BIGTIFF": "YES",
            }
        )

        output_file = os.path.join(
            output_dir,
            f"HP_week_{week}.tif"
        )

        # --------------------------------------------------
        # WRITE MERGED FILE
        # --------------------------------------------------

        with rasterio.open(
            output_file,
            "w",
            **profile
        ) as dst:

            dst.write(mosaic)

            # Preserve band names
            for band_index, name in enumerate(
                src1.descriptions,
                start=1
            ):
                if name:
                    dst.set_band_description(
                        band_index,
                        name
                    )

        # --------------------------------------------------
        # VERIFY OUTPUT
        # --------------------------------------------------

        with rasterio.open(output_file) as check:

            expected_width = src1.width + src2.width
            expected_height = src1.height

            if check.width != expected_width:
                raise RuntimeError(
                    f"Week {week}: WRONG OUTPUT WIDTH. "
                    f"Expected {expected_width}, "
                    f"got {check.width}"
                )

            if check.height != expected_height:
                raise RuntimeError(
                    f"Week {week}: WRONG OUTPUT HEIGHT. "
                    f"Expected {expected_height}, "
                    f"got {check.height}"
                )

            if check.count != 3:
                raise RuntimeError(
                    f"Week {week}: WRONG BAND COUNT. "
                    f"Expected 3, got {check.count}"
                )

            if check.crs != src1.crs:
                raise RuntimeError(
                    f"Week {week}: CRS changed"
                )

            print("\nOUTPUT VERIFIED:")
            print("  File:", output_file)
            print(
                "  Size:",
                check.width,
                "x",
                check.height
            )
            print("  Bands:", check.count)
            print("  CRS:", check.crs)
            print("  Band names:", check.descriptions)
            print("  Bounds:", check.bounds)

            print("\n✓ Week", week, "merged successfully.")

    finally:
        src1.close()
        src2.close()


print("\n" + "=" * 70)
print("ALL 14 WEEKS MERGED AND VERIFIED")
print("=" * 70)