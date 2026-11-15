"""Fit the DEM normalisation on the training chips (real terrain differs from the defaults in formats.DEM_STATS).

Aspect channels stay fixed (mean 0): the D4 augmentation rotates aspect vectors, so their mean must stay 0.
The fitted mean/std are written into the model's dem_mu / dem_sd buffers, so they travel inside the checkpoint.
"""
from __future__ import annotations

import numpy as np

from .formats import DEM_CHANNELS, DEM_STATS, IDX_ASPX, IDX_ASPY, load_chip


def fit_dem_stats(ds, max_chips: int = 400, seed: int = 0):
    """ds: ChipDataset (training split). Returns (mu[8], sd[8]) float32."""
    n = len(ds.rows)
    idx = np.random.default_rng(seed).choice(n, min(n, max_chips), replace=False)
    s = np.zeros(len(DEM_CHANNELS)); q = np.zeros(len(DEM_CHANNELS)); cnt = 0
    for i in idx:
        d = load_chip(ds.root / ds.rows[i]["path"])["dem"].astype(np.float64).reshape(len(DEM_CHANNELS), -1)
        s += d.sum(1); q += (d ** 2).sum(1); cnt += d.shape[1]
    mu = s / cnt; sd = np.sqrt(np.maximum(q / cnt - mu ** 2, 1e-12))
    for j in (IDX_ASPX, IDX_ASPY):
        mu[j], sd[j] = DEM_STATS[DEM_CHANNELS[j]]
    sd = np.maximum(sd, 1e-3)
    return mu.astype(np.float32), sd.astype(np.float32)
