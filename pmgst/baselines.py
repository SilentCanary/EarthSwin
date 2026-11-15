"""Cheap baselines that PMGST-Net-D must beat (docs/07 B0-B2). Per-pixel, no deep learning.

  B0  slope threshold         (does terrain alone explain it?)
  B1  S2 change / S1 change   (single-signal change thresholds)
  B2  gradient-boosted trees on per-pixel features (sklearn HistGradientBoosting)

  python -m pmgst.baselines --config configs/rtx3060.yaml
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)       # nanmedian on all-invalid S1 pixels

from .config import load_config
from .data.dataset import ChipDataset
from .data.formats import DEM_CHANNELS, IDX_SLOPE, load_chip
from .metrics import PixelMetrics


def pixel_features(c: dict, stride: int = 1):
    """Per-pixel feature matrix [N,F] and label [N] from one chip (landslide-agnostic engineered features)."""
    H, W = c["label"].shape
    s2, v2 = c["s2"].astype(np.float32), c["s2_valid"][:, 0].astype(bool)
    ok2 = v2.all(0)
    ndvi = lambda x: (x[3] - x[2]) / (x[3] + x[2] + 1e-6)
    d_ndvi = np.where(ok2, ndvi(s2[1]) - ndvi(s2[0]), 0.0)
    d_bright = np.where(ok2, s2[1, :4].mean(0) - s2[0, :4].mean(0), 0.0)
    d_swir = np.where(ok2, s2[1, 4] - s2[0, 4], 0.0)
    s1, v1 = c["s1"].astype(np.float32), c["s1_valid"][:, 0].astype(bool)
    dt = c["s1_dt"]
    feats_s1 = []
    for ch in range(2):
        x = np.where(v1, s1[:, ch], np.nan)
        n = x.shape[0]
        a, b = x[: n // 2], x[n // 2:]
        with np.errstate(all="ignore"):
            d = np.nanmedian(b, 0) - np.nanmedian(a, 0)
            sd = np.nanstd(x, 0)
        feats_s1 += [np.nan_to_num(d), np.nan_to_num(sd)]
    dem = c["dem"].astype(np.float32)
    F_ = np.stack([d_ndvi, d_bright, d_swir, *feats_s1, *dem, ok2.astype(np.float32)], 0).reshape(-1, H * W).T
    rain = np.broadcast_to(c["rain"][None], (H * W, 4))
    return np.concatenate([F_, rain], 1), c["label"].reshape(-1)


def run(cfg, max_px=400_000, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(seed)
    tr, va = ChipDataset(cfg, "train", augment=False), ChipDataset(cfg, "val", augment=False)
    X, y = [], []
    for r in tr.rows:
        f, l = pixel_features(load_chip(tr.root / r["path"]))
        keep = (l != 255) & ((l == 1) | (rng.random(len(l)) < 0.05))
        X.append(f[keep]); y.append(l[keep] == 1)
    X, y = np.concatenate(X), np.concatenate(y)
    if len(X) > max_px:
        i = rng.choice(len(X), max_px, replace=False); X, y = X[i], y[i]
    w = np.where(y, 1.0, 20.0)                                      # negatives were subsampled 20x
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31).fit(X, y, sample_weight=w)
    res = {}
    mets = {k: PixelMetrics() for k in ("B0_slope", "B1_dNDVI", "B1_S1change", "B2_gbdt")}
    names = DEM_CHANNELS
    for r in va.rows:
        c = load_chip(va.root / r["path"]); f, l = pixel_features(c)
        lab = l.reshape(c["label"].shape)
        sc = lambda x: (1 / (1 + np.exp(-(x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)))).reshape(lab.shape)
        mets["B0_slope"].update(sc(f[:, 7 + IDX_SLOPE]), lab)             # features 0-6 are change stats, 7.. is DEM
        mets["B1_dNDVI"].update(sc(-f[:, 0]), lab)
        mets["B1_S1change"].update(sc(-f[:, 3]), lab)
        mets["B2_gbdt"].update(gb.predict_proba(f)[:, 1].reshape(lab.shape), lab)
    for k, m in mets.items():
        res[k] = m.compute()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=None); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    out = run(load_config(a.config, a.set))
    for k, v in out.items():
        print(f"{k:12s} AP {v['ap']:.3f}  F1 {v['f1']:.3f}  IoU {v['iou']:.3f}")
