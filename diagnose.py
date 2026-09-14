import os
import numpy as np
import rasterio

input_dir = r"C:/Users/advit/Documents/major project/dataset/EarthSentinel_2016/merged"

weeks = range(1, 15)

reference = None

print("\n" + "=" * 80)
print("EARTHSENTINEL - FINAL MERGED DATASET SANITY CHECK")
print("=" * 80)

for week in weeks:

    print("\n" + "-" * 80)
    print(f"WEEK {week}")
    print("-" * 80)

    file_path = os.path.join(
        input_dir,
        f"HP_week_{week}.tif"
    )

    # --------------------------------------------------
    # FILE CHECK
    # --------------------------------------------------

    if not os.path.exists(file_path):
        print("❌ FILE NOT FOUND")
        continue

    problems = []

    # --------------------------------------------------
    # OPEN
    # --------------------------------------------------

    with rasterio.open(file_path) as src:

        print("File:", os.path.basename(file_path))
        print("Size:", src.width, "x", src.height)
        print("Bands:", src.count)
        print("Dtypes:", src.dtypes)
        print("CRS:", src.crs)
        print("Resolution:", src.res)
        print("Band names:", src.descriptions)
        print("Nodata:", src.nodata)
        print("Bounds:", src.bounds)

        # --------------------------------------------------
        # BASIC STRUCTURE
        # --------------------------------------------------

        if src.width != 19020:
            problems.append(
                f"wrong width: {src.width}"
            )

        if src.height != 15516:
            problems.append(
                f"wrong height: {src.height}"
            )

        if src.count != 3:
            problems.append(
                f"wrong band count: {src.count}"
            )

        if src.dtypes != (
            "float32",
            "float32",
            "float32"
        ):
            problems.append(
                f"unexpected dtypes: {src.dtypes}"
            )

        if src.descriptions != (
            "VV",
            "rainfall",
            "slope"
        ):
            problems.append(
                f"unexpected band names: {src.descriptions}"
            )

        # --------------------------------------------------
        # COMPARE GEOMETRY TO WEEK 1
        # --------------------------------------------------

        current_reference = (
            src.crs,
            src.transform,
            src.width,
            src.height,
            src.count,
            src.dtypes
        )

        if reference is None:

            reference = current_reference

        else:

            if src.crs != reference[0]:
                problems.append("CRS differs from Week 1")

            if src.transform != reference[1]:
                problems.append(
                    "transform differs from Week 1"
                )

            if src.width != reference[2]:
                problems.append(
                    "width differs from Week 1"
                )

            if src.height != reference[3]:
                problems.append(
                    "height differs from Week 1"
                )

            if src.count != reference[4]:
                problems.append(
                    "band count differs from Week 1"
                )

            if src.dtypes != reference[5]:
                problems.append(
                    "dtype differs from Week 1"
                )

        # --------------------------------------------------
        # READ AND CHECK EACH BAND
        # --------------------------------------------------

        for band in range(1, 4):

            data = src.read(band)

            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()

            finite = data[np.isfinite(data)]

            total_pixels = data.size

            print(
                f"\nBand {band} - "
                f"{src.descriptions[band - 1]}"
            )

            print(
                "  Total pixels:",
                total_pixels
            )

            print(
                "  Finite pixels:",
                finite.size
            )

            print(
                "  NaN pixels:",
                nan_count
            )

            print(
                "  Inf pixels:",
                inf_count
            )

            if inf_count > 0:
                problems.append(
                    f"Band {band} contains Inf"
                )

            if finite.size == 0:

                print("  ⚠️ No finite pixels")

            else:

                print(
                    "  Min:",
                    float(np.min(finite))
                )

                print(
                    "  P5:",
                    float(np.percentile(finite, 5))
                )

                print(
                    "  Median:",
                    float(np.median(finite))
                )

                print(
                    "  Mean:",
                    float(np.mean(finite))
                )

                print(
                    "  P95:",
                    float(np.percentile(finite, 95))
                )

                print(
                    "  Max:",
                    float(np.max(finite))
                )

        # --------------------------------------------------
        # WEEK 4 SPECIAL CHECK
        # --------------------------------------------------

        if week == 4:

            vv = src.read(1)

            finite_vv = vv[np.isfinite(vv)]

            print("\nWEEK 4 VV SPECIAL CHECK")
            print(
                "  Finite VV pixels:",
                finite_vv.size
            )

            print(
                "  NaN VV pixels:",
                np.isnan(vv).sum()
            )

            if finite_vv.size != 0:

                problems.append(
                    "Week 4 VV has finite pixels; "
                    "expected fully missing VV"
                )

                print(
                    "  ❌ Week 4 VV is NOT fully missing"
                )

            else:

                print(
                    "  ✓ Week 4 VV is fully NaN/missing"
                )

        # --------------------------------------------------
        # FINAL RESULT FOR THIS WEEK
        # --------------------------------------------------

        if problems:

            print("\n❌ PROBLEMS FOUND:")

            for problem in problems:
                print("  -", problem)

        else:

            print(
                f"\n✓ WEEK {week} PASSED"
            )


print("\n" + "=" * 80)
print("SANITY CHECK COMPLETE")
print("=" * 80)