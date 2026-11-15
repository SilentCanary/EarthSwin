"""Spatial block K-fold with buffer (docs/04 §8). numpy only."""
from __future__ import annotations

import numpy as np


def block_ids(x, y, block_m: float = 10_000.0):
    """Integer block id per chip from UTM coordinates."""
    bx = np.floor(np.asarray(x) / block_m).astype(np.int64)
    by = np.floor(np.asarray(y) / block_m).astype(np.int64)
    uniq, inv = np.unique(np.stack([bx, by], 1), axis=0, return_inverse=True)
    return inv.reshape(-1)


def assign_folds(block: np.ndarray, n_pos: np.ndarray, k: int = 5, seed: int = 42) -> np.ndarray:
    """Fold per chip. Blocks are stratified by landslide load so each fold gets a similar share."""
    ub = np.unique(block)
    load = np.array([n_pos[block == b].sum() for b in ub], dtype=np.float64)
    order = np.argsort(-load, kind="stable")
    rng = np.random.default_rng(seed)
    fold_of = {}
    # deal sorted blocks round-robin in shuffled groups of k -> balanced load
    for start in range(0, len(order), k):
        grp = order[start:start + k]
        folds = rng.permutation(k)[: len(grp)]
        for g, f in zip(grp, folds):
            fold_of[ub[g]] = int(f)
    return np.array([fold_of[b] for b in block], dtype=np.int64)


def make_split(x, y, fold, test_fold: int, val_fold: int, buffer_m: float, chip_m: float):
    """Returns boolean masks (train, val, test). Train chips closer than buffer_m (edge to edge) to any
    val/test chip are dropped to limit spatial-autocorrelation leakage."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    test = fold == test_fold
    val = (fold == val_fold) & ~test
    train = ~(test | val)
    held = np.where(test | val)[0]
    if len(held) and buffer_m >= 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.stack([x[held], y[held]], 1))
        # Chebyshev distance between centres must exceed chip + buffer
        d, _ = tree.query(np.stack([x, y], 1), k=1, p=np.inf)
        train &= d >= (chip_m + buffer_m)
    return train, val, test
