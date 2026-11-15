"""Landslide4Sense (Zenodo 10463239) -> chips in the PMGST format, so the pipeline can be trained on REAL labelled landslides today.

  pip install h5py
  python -m pmgst.data.l4s --src data/l4s/TrainData --inspect                     # look at keys / shapes / value ranges first
  python -m pmgst.data.l4s --src data/l4s/TrainData --out data/l4s_chips --compress
  python -m pmgst.train --config configs/l4s.yaml

What this data is: one post-event Sentinel-2 composite (12 bands) + slope + DEM per 128x128 patch at ~10 m, pixel labels, four
non-Himachal sites. There is no pre-event image, no Sentinel-1, no rain and no coordinates, so those inputs are written as
"missing" (valid = 0) and configs/l4s.yaml switches the graph / pyramid / S1 branches off. Patches get fake positions on a line in
index order (8 patches per spatial block), so folds are contiguous index ranges: a proxy for spatial blocking that only holds if
neighbouring patch ids come from the same event. Scores are therefore optimistic. Use it to validate the training code on real
scars and to compare architectures, never to claim Himachal accuracy.

Band assumptions (override with flags if --inspect shows otherwise): channels 0-11 = S2 B1..B12 without B8A, so B2,B3,B4,B8,B11,B12 are
channels 1,2,3,7,10,11; channel 12 = slope, 13 = DEM (swapped automatically when the means say otherwise).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from .formats import rain_features, save_chip, write_index
from .splits import block_ids
from .terrain import area_resample, terrain_channels

S2_PICK = [1, 2, 3, 7, 10, 11]


def _h5():
    try:
        import h5py
        return h5py
    except ImportError as e:
        raise SystemExit("h5py is required: pip install h5py") from e


def read_h5(path, key=None):
    h5py = _h5()
    with h5py.File(path, "r") as f:
        k = key if (key and key in f) else next(iter(f.keys()))
        return np.array(f[k]), k


def _hwc(a, n_ch):
    """Return [H,W,C] whatever the stored layout is."""
    if a.ndim == 2:
        return a[..., None]
    if a.shape[0] == n_ch and a.shape[-1] != n_ch:
        return np.moveaxis(a, 0, -1)
    return a


def list_pairs(src: Path):
    imgs = {int(re.search(r"(\d+)", f.stem).group(1)): f for f in (src / "img").glob("*.h5")}
    masks = {int(re.search(r"(\d+)", f.stem).group(1)): f for f in (src / "mask").glob("*.h5")}
    ids = sorted(set(imgs) & set(masks))
    return [(i, imgs[i], masks[i]) for i in ids]


def inspect(src: Path, n: int = 5):
    pairs = list_pairs(src)
    print(f"{len(pairs)} image/mask pairs in {src}")
    for i, fi, fm in pairs[:n]:
        a, ka = read_h5(fi); m, km = read_h5(fm)
        a = _hwc(a, 14).astype(np.float64)
        print(f"  #{i}: img key '{ka}' shape {a.shape}, mask key '{km}' shape {m.shape} values {np.unique(m).tolist()}")
        print("     per-channel mean:", np.round(a.reshape(-1, a.shape[-1]).mean(0), 3).tolist())
        print("     per-channel p99 :", np.round(np.percentile(a.reshape(-1, a.shape[-1]), 99, axis=0), 3).tolist())


def convert_patch(img, mask, idx, s2_scale, swap_dem_slope=None):
    """img [H,W,14] float, mask [H,W] {0,1} -> chip dict (see formats.py)."""
    a = _hwc(img, 14).astype(np.float32); H, W, _ = a.shape
    bands = np.moveaxis(a[..., S2_PICK], -1, 0) * s2_scale                      # [6,H,W] reflectance
    bands = np.clip(bands, 0, 1)
    slope_ch, dem_ch = a[..., 12], a[..., 13]
    swap = (slope_ch.mean() > dem_ch.mean()) if swap_dem_slope is None else swap_dem_slope
    if swap:
        slope_ch, dem_ch = dem_ch, slope_ch
    valid2 = (bands.sum(0) > 0).astype(np.uint8)
    s2 = np.zeros((2, 6, H, W), np.float32); s2[1] = bands * valid2
    s2v = np.zeros((2, 1, H, W), np.uint8); s2v[1, 0] = valid2
    dem = terrain_channels(dem_ch.astype(np.float64), 10.0)
    dem[1] = slope_ch
    lab = (_hwc(mask, 1)[..., 0] > 0).astype(np.uint8)
    ctx = 64 if H == 128 else H // 2
    ctx_dem = np.stack([area_resample(dem, ctx)] * 2)
    meta = np.array([idx * 1280.0, 0.0, dem[0].mean(), 0, 0, 1, 5000.0, 5000.0, 0.0, 0.0], np.float32)
    return dict(
        s2=s2, s2_valid=s2v, s1=np.zeros((1, 2, H, W), np.float32), s1_valid=np.zeros((1, 1, H, W), np.uint8), s1_dt=np.zeros(1, np.float32),
        dem=dem, lith=np.zeros((H, W), np.uint8), water=np.zeros((H, W), np.uint8), rain=rain_features(np.zeros(30)),
        ctx_s2=np.zeros((2, 2, 6, ctx, ctx), np.float32), ctx_valid=np.zeros((2, 2, 1, ctx, ctx), np.uint8), ctx_dem=ctx_dem,
        label=lab, meta=meta)


def convert(src: Path, out: Path, max_n: int = 0, compress: bool = True, s2_scale: float | None = None, swap=None):
    pairs = list_pairs(src)
    if max_n:
        pairs = pairs[:max_n]
    if not pairs:
        raise SystemExit(f"no img/mask .h5 pairs under {src} (expected {src}/img/image_N.h5 and {src}/mask/mask_N.h5)")
    if s2_scale is None:                                              # raw digital numbers (~0..10000) or already reflectance?
        p99 = [np.percentile(_hwc(read_h5(f)[0], 14)[..., S2_PICK], 99) for _, f, _ in pairs[:30]]
        s2_scale = 1e-4 if np.median(p99) > 5 else 1.0
        print(f"[l4s] S2 p99 median {np.median(p99):.3g} -> scale {s2_scale}")
    (out / "chips").mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (i, fi, fm) in enumerate(pairs):
        img, _ = read_h5(fi); msk, _ = read_h5(fm)
        c = convert_patch(img, msk, k, s2_scale, swap)
        cid = f"l4s_{i:05d}"
        save_chip(out / "chips" / f"{cid}.npz", c, compress=compress)
        H, W = c["label"].shape
        rows.append(dict(chip_id=cid, path=f"chips/{cid}.npz", x=float(c["meta"][0]), y=0.0, region="l4s", block=0,
                         n_pos_px=int((c["label"] == 1).sum()), n_valid_px=H * W, has_s1=0, has_s2=1))
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(pairs)}")
    blk = block_ids([r["x"] for r in rows], [r["y"] for r in rows], 10_240.0)     # 8 consecutive patches per block
    for r, b in zip(rows, blk):
        r["block"] = int(b)
    write_index(out / "index.csv", rows)
    npos = sum(r["n_pos_px"] for r in rows)
    print(f"[l4s] {len(rows)} chips, {sum(r['n_pos_px'] > 0 for r in rows)} with landslides, positive px {100 * npos / (len(rows) * 128 * 128):.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing img/ and mask/ (e.g. data/l4s/TrainData)")
    ap.add_argument("--out", default="data/l4s_chips")
    ap.add_argument("--inspect", action="store_true"); ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--compress", action="store_true"); ap.add_argument("--s2-scale", type=float, default=None)
    ap.add_argument("--swap-dem-slope", action="store_true", help="force channel 12=DEM, 13=slope")
    a = ap.parse_args()
    if a.inspect:
        inspect(Path(a.src)); return
    convert(Path(a.src), Path(a.out), a.max, a.compress, a.s2_scale, True if a.swap_dem_slope else None)


if __name__ == "__main__":
    main()
