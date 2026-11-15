"""The on-disk chip format. Real pipeline (chips.py) and synthetic generator (synth.py) both write this.

One chip = one .npz. All arrays are *physical units*, float16 unless noted (the model normalises with DEM_STATS / S2 scaling).

  s2        [2, 6, H, W]   reflectance 0..1 (pre, post); bands B2 B3 B4 B8 B11 B12; invalid pixels = 0
  s2_valid  [2, 1, H, W]   uint8 1 = clear
  s1        [T, 2, H, W]   VV, VH in dB (about -25..0); invalid = 0
  s1_valid  [T, 1, H, W]   uint8 (no-data, layover/shadow = 0)
  s1_dt     [T]            float32 days since 1 May of the event year (acquisition date)
  dem       [8, H, W]      DEM_CHANNELS below, physical units
  lith      [H, W]         uint8 class id 0..n_lith-1 (lithology if available, else soil-texture class)
  water     [H, W]         uint8 1 = permanent water / river channel
  rain      [4]            float32 mm: max 1-day, max 3-day, sum 7-day, sum 30-day before the post date
  ctx_s2    [S-1, 2, 6, h, w]   scale 2..S crops (chip*2^(s-1) px) resampled to h=w=ctx
  ctx_valid [S-1, 2, 1, h, w]
  ctx_dem   [S-1, 8, h, w]
  label     [H, W]         uint8: 0 background, 1 landslide, 255 ignore
  meta      float32 [10]   META_FIELDS below (chip-level attributes used for graph edges)
"""
from __future__ import annotations

import numpy as np

S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
DEM_CHANNELS = ["elev", "slope", "asp_x", "asp_y", "plan_curv", "prof_curv", "twi", "relief"]
# aspect convention: downslope direction a measured clockwise from north; asp_x = sin a (east), asp_y = cos a (north)
IDX_ELEV, IDX_SLOPE, IDX_ASPX, IDX_ASPY, IDX_PLAN, IDX_PROF, IDX_TWI, IDX_RELIEF = range(8)

# (mean, std) used by the model to normalise DEM channels; elev/slope/twi etc. are physical on disk.
DEM_STATS = {
    "elev": (2500.0, 1200.0), "slope": (25.0, 15.0), "asp_x": (0.0, 0.7), "asp_y": (0.0, 0.7),
    "plan_curv": (0.0, 0.05), "prof_curv": (0.0, 0.05), "twi": (8.0, 3.0), "relief": (150.0, 150.0),
}

META_FIELDS = ["x", "y", "z", "nx", "ny", "nz", "river_d", "road_d", "block", "rain_c"]
M_X, M_Y, M_Z, M_NX, M_NY, M_NZ, M_RIVER, M_ROAD, M_BLOCK, M_RAIN = range(10)

INDEX_COLUMNS = ["chip_id", "path", "x", "y", "region", "block", "n_pos_px", "n_valid_px", "has_s1", "has_s2"]

CHIP_KEYS = ["s2", "s2_valid", "s1", "s1_valid", "s1_dt", "dem", "lith", "water", "rain",
             "ctx_s2", "ctx_valid", "ctx_dem", "label", "meta"]


def rain_features(daily: np.ndarray) -> np.ndarray:
    """daily mm (oldest -> newest, ending the day before the post image) -> [max 1d, max 3d, sum 7d, sum 30d]."""
    daily = np.asarray(daily, np.float32)
    return np.array([daily.max(), np.convolve(daily, np.ones(3), "valid").max(), daily[-7:].sum(), daily[-30:].sum()], np.float32)


def save_chip(path, chip: dict, compress: bool = False) -> None:
    out = {}
    for k in CHIP_KEYS:
        a = np.asarray(chip[k])
        if k in ("s2", "s1", "dem", "ctx_s2", "ctx_dem"):
            a = a.astype(np.float16)
        elif k in ("s1_dt", "rain", "meta"):
            a = a.astype(np.float32)
        elif k in ("s2_valid", "s1_valid", "ctx_valid", "lith", "water", "label"):
            a = a.astype(np.uint8)
        out[k] = a
    (np.savez_compressed if compress else np.savez)(path, **out)


def load_chip(path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in CHIP_KEYS}


def write_index(path, rows: list[dict]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in INDEX_COLUMNS})


def read_index(path) -> list[dict]:
    import csv
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("x", "y"):
                r[k] = float(r[k])
            for k in ("block", "n_pos_px", "n_valid_px", "has_s1", "has_s2"):
                r[k] = int(float(r[k]))
            rows.append(r)
    return rows
