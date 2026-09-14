import os
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box, Point


CSV_PATH = "C:/Users/advit/Documents/convolution neural network/satellite/Global_Landslide_Catalog_Export.csv"

RASTER_PATH = (
    "C:/Users/advit/Documents/major project/"
    "dataset/EarthSentinel_2016/merged/HP_week_1.tif"
)

PATCH_SIZE = 256
STRIDE = 256


# ---------------------------------------------------------
# Load events
# ---------------------------------------------------------

df = pd.read_csv(CSV_PATH)

df["event_date_parsed"] = pd.to_datetime(
    df["event_date"],
    errors="coerce"
)

hp_2016 = df[
    (df["event_date_parsed"].dt.year == 2016) &
    (
        df["admin_division_name"]
        .astype(str)
        .str.contains(
            "Himachal",
            case=False,
            na=False
        )
    )
].copy()

print("=" * 80)
print("2016 HIMACHAL EVENTS")
print("=" * 80)

print(
    hp_2016[
        [
            "event_id",
            "event_date",
            "event_title",
            "longitude",
            "latitude"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# Raster information
# ---------------------------------------------------------

with rasterio.open(RASTER_PATH) as src:

    transform = src.transform
    crs = src.crs
    H = src.height
    W = src.width

print()
print("=" * 80)
print("RASTER")
print("=" * 80)

print("Shape :", H, W)
print("CRS   :", crs)
print("Transform:", transform)

num_patches_h = (H - PATCH_SIZE) // STRIDE + 1
num_patches_w = (W - PATCH_SIZE) // STRIDE + 1

print("Patch grid:", num_patches_h, "x", num_patches_w)
print("Total patches:", num_patches_h * num_patches_w)


# ---------------------------------------------------------
# Known current labels
# ---------------------------------------------------------

known_labels = {
    (2016, 6, 17): 2719,
    (2016, 7, 2): 3658,
    (2016, 7, 17): 2720,
    (2016, 7, 26): 3962,
    (2016, 7, 31): 2401,
    (2016, 8, 29): 2565,
}


# ---------------------------------------------------------
# Match each event to patch
# ---------------------------------------------------------

print()
print("=" * 80)
print("EVENT → PATCH MAPPING")
print("=" * 80)

for _, event in hp_2016.iterrows():

    date = event["event_date_parsed"]

    lon = float(event["longitude"])
    lat = float(event["latitude"])

    # Convert geographic coordinate to raster row/column
    row, col = rasterio.transform.rowcol(
        transform,
        lon,
        lat
    )

    row = int(row)
    col = int(col)

    # Determine patch containing pixel
    patch_row = row // STRIDE
    patch_col = col // STRIDE

    i_start = patch_row * STRIDE
    j_start = patch_col * STRIDE

    if (
        i_start + PATCH_SIZE > H
        or
        j_start + PATCH_SIZE > W
    ):
        anchor_id = None
    else:
        anchor_id = (
            patch_row * num_patches_w
            + patch_col
        )

    key = (
        date.year,
        date.month,
        date.day
    )

    expected_anchor = known_labels.get(key)

    print()
    print(
        f"Event {int(event['event_id'])}"
    )
    print(
        f"Date       : {date}"
    )
    print(
        f"Location   : ({lat:.6f}, {lon:.6f})"
    )
    print(
        f"Raster px  : row={row}, col={col}"
    )
    print(
        f"Patch grid : row={patch_row}, col={patch_col}"
    )
    print(
        f"Anchor     : {anchor_id}"
    )
    print(
        f"Expected   : {expected_anchor}"
    )

    if expected_anchor is not None:

        if anchor_id == expected_anchor:
            print("MATCH      : YES ✓")
        else:
            print("MATCH      : NO ✗")

print()
print("=" * 80)
print("DONE")
print("=" * 80)