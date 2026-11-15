"""ChipDataset + spatial-cluster batch sampler.

A *batch* is one spatial cluster of chips (a seed chip + its nearest neighbours). The model builds a graph over the
cluster (blocks 6-7), so neighbouring chips must be in the same batch. Labels of neighbours are never used as inputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..config import Config
from .formats import (IDX_ASPX, IDX_ASPY, IDX_SLOPE, M_BLOCK, load_chip, read_index)
from .splits import assign_folds, make_split


# ---------------------------------------------------------------- geometry augmentation (D4)
def d4_apply(chip: dict, k: int, flip: bool) -> dict:
    """Flip (optional) then rotate by k*90 deg CCW, consistently for every raster; aspect vector rotated accordingly."""
    spatial2 = ("lith", "water", "label")                   # [H,W]
    spatial_n = ("s2", "s2_valid", "s1", "s1_valid", "dem", "ctx_s2", "ctx_valid", "ctx_dem")  # [...,H,W]
    out = dict(chip)

    def tf(a):
        if flip:
            a = a[..., ::-1]
        if k:
            a = np.rot90(a, k, axes=(-2, -1))
        return np.ascontiguousarray(a)

    for key in spatial2 + spatial_n:
        out[key] = tf(chip[key])
    for key in ("dem", "ctx_dem"):
        d = out[key].copy()
        ax, ay = IDX_ASPX, IDX_ASPY
        dim = d.ndim - 3                                     # channel axis position
        sl = [slice(None)] * d.ndim
        def ch(i):
            s = list(sl); s[dim] = i; return tuple(s)
        if flip:
            d[ch(ax)] = -d[ch(ax)]
        for _ in range(k):                                   # (x, y) -> (-y, x)
            x_, y_ = d[ch(ax)].copy(), d[ch(ay)].copy()
            d[ch(ax)], d[ch(ay)] = -y_, x_
        out[key] = d
    return out


# ---------------------------------------------------------------- dataset
class ChipDataset(Dataset):
    def __init__(self, cfg: Config, split: str = "train", rows: list[dict] | None = None, augment: bool | None = None,
                 drop: str | None = None):
        """drop: None | "s2" | "s1" - switch a modality off at evaluation time (cloud-robust / S2-free metrics)."""
        self.cfg = cfg
        self.drop = drop
        self.root = Path(cfg.data.root)
        self.split = split
        rows = rows if rows is not None else self.rows_for_split(cfg, split)
        self.rows = rows
        self.augment = (split == "train" and cfg.data.aug) if augment is None else augment
        self.xy = np.array([[r["x"], r["y"]] for r in rows], dtype=np.float64).reshape(-1, 2)
        self.n_pos = np.array([r["n_pos_px"] for r in rows], dtype=np.int64)
        self.rng = np.random.default_rng(cfg.train.seed)

    # -- splitting ------------------------------------------------------------
    @staticmethod
    def all_rows(cfg: Config) -> list[dict]:
        return read_index(Path(cfg.data.root) / "index.csv")

    @staticmethod
    def rows_for_split(cfg: Config, split: str) -> list[dict]:
        rows = ChipDataset.all_rows(cfg)
        x = np.array([r["x"] for r in rows]); y = np.array([r["y"] for r in rows])
        block = np.array([r["block"] for r in rows]); npos = np.array([r["n_pos_px"] for r in rows])
        hold = cfg.data.holdout_region
        if hold:                                                # unseen-district test: whole region = test fold
            reg = np.array([r["region"] for r in rows])
            ish = reg == hold
            if not ish.any() or ish.all():
                raise ValueError(f"holdout_region={hold!r} must match some, but not all, chips (regions: {sorted(set(reg))})")
            fold = np.full(len(rows), cfg.data.test_fold, np.int64)
            f = assign_folds(block[~ish], npos[~ish], cfg.data.folds - 1, cfg.train.seed)
            fold[~ish] = np.where(f >= cfg.data.test_fold, f + 1, f)      # skip the test slot
        else:
            fold = assign_folds(block, npos, cfg.data.folds, cfg.train.seed)
        tr, va, te = make_split(x, y, fold, cfg.data.test_fold, cfg.data.val_fold,
                                cfg.data.buffer_m, cfg.data.chip * 10.0)
        m = {"train": tr, "val": va, "test": te, "all": np.ones(len(rows), bool)}[split]
        return [r for r, keep in zip(rows, m) if keep]

    def __len__(self):
        return len(self.rows)

    # -- loading ---------------------------------------------------------------
    def _load(self, i: int) -> dict:
        r = self.rows[i]
        return load_chip(self.root / r["path"])

    def __getitem__(self, i: int) -> dict:
        c = self._load(i)
        d = self.cfg.data
        if self.augment:
            c = d4_apply(c, int(self.rng.integers(4)), bool(self.rng.integers(2)))
        c = self._pick_dates(c, train=self.augment)
        if self.drop == "s2":
            c["s2_valid"] = np.zeros_like(c["s2_valid"]); c["ctx_valid"] = np.zeros_like(c["ctx_valid"])
        elif self.drop == "s1":
            c["s1_pad"] = np.ones_like(c["s1_pad"])
        if self.augment:                                       # modality dropout (block 5.3)
            if self.rng.random() < d.p_drop_s2:
                c["s2_valid"] = np.zeros_like(c["s2_valid"]); c["ctx_valid"] = np.zeros_like(c["ctx_valid"])
            keep = self.rng.random(c["s1_pad"].shape) >= d.p_drop_s1
            c["s1_pad"] = c["s1_pad"] | (~keep)
        return self._to_tensors(c, i)

    def _pick_dates(self, c: dict, train: bool) -> dict:
        T = self.cfg.data.t_max
        n_have = c["s1"].shape[0]
        real = np.where(c["s1_valid"].reshape(n_have, -1).any(1))[0] if n_have else np.array([], int)
        if len(real) > T:
            idx = np.sort(self.rng.choice(real, T, replace=False)) if train else \
                  real[np.linspace(0, len(real) - 1, T).round().astype(int)]
        else:
            idx = real
        n = len(idx)
        H, W = c["label"].shape
        s1 = np.zeros((T, 2, H, W), np.float16); v = np.zeros((T, 1, H, W), np.uint8)
        dt = np.zeros(T, np.float32); pad = np.ones(T, bool)
        if n:
            s1[:n] = c["s1"][idx]; v[:n] = c["s1_valid"][idx]; dt[:n] = c["s1_dt"][idx]; pad[:n] = False
        c = dict(c); c.update(s1=s1, s1_valid=v, s1_dt=dt, s1_pad=pad)
        return c

    def _to_tensors(self, c: dict, i: int) -> dict:
        t = {
            "s2": torch.from_numpy(c["s2"].astype(np.float32)),
            "s2_valid": torch.from_numpy(c["s2_valid"].astype(np.float32)),
            "s1": torch.from_numpy(c["s1"].astype(np.float32)),
            "s1_valid": torch.from_numpy(c["s1_valid"].astype(np.float32)),
            "s1_dt": torch.from_numpy(c["s1_dt"]),
            "s1_pad": torch.from_numpy(c["s1_pad"]),
            "dem": torch.from_numpy(c["dem"].astype(np.float32)),
            "lith": torch.from_numpy(c["lith"].astype(np.int64)),
            "water": torch.from_numpy(c["water"].astype(np.float32))[None],
            "rain": torch.from_numpy(c["rain"].astype(np.float32)),
            "ctx_s2": torch.from_numpy(c["ctx_s2"].astype(np.float32)),
            "ctx_valid": torch.from_numpy(c["ctx_valid"].astype(np.float32)),
            "ctx_dem": torch.from_numpy(c["ctx_dem"].astype(np.float32)),
            "label": torch.from_numpy(c["label"].astype(np.int64)),
            "meta": torch.from_numpy(c["meta"].astype(np.float32)),
            "idx": torch.tensor(i),
        }
        return t


def collate(items: list[dict]) -> dict:
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# ---------------------------------------------------------------- spatial-cluster batches
def hard_negatives(ds: "ChipDataset", slope_deg: float = 15.0, min_frac: float = 0.2) -> np.ndarray:
    """Indices of landslide-free chips that are mostly steep: where false positives live (scree, river beds, cut
    slopes). Cached per data root. Only dem is read from each .npz, so this is cheap."""
    import json
    cache = ds.root / f"hard_neg_{int(slope_deg)}_{int(min_frac * 100)}.json"
    table = json.loads(cache.read_text()) if cache.exists() else {}
    changed = False
    for r in ds.rows:
        if r["n_pos_px"] == 0 and r["chip_id"] not in table:
            with np.load(ds.root / r["path"]) as z:
                table[r["chip_id"]] = float((z["dem"][IDX_SLOPE] > slope_deg).mean())
            changed = True
    if changed:
        try:
            cache.write_text(json.dumps(table))
        except OSError:
            pass
    return np.array([i for i, r in enumerate(ds.rows) if r["n_pos_px"] == 0 and table[r["chip_id"]] >= min_frac], dtype=np.int64)


class ClusterBatchSampler(Sampler):
    """Yields lists of dataset indices that are spatial neighbours (a seed + its nearest chips).
    Seed mix: pos_frac on landslide chips, hard_frac on hard negatives, the rest uniformly random."""

    def __init__(self, ds: ChipDataset, batch: int, steps: int, pos_frac: float = 0.5, seed: int = 0,
                 hard_frac: float = 0.0):
        from scipy.spatial import cKDTree
        self.ds, self.batch, self.steps, self.pos_frac, self.hard_frac = ds, batch, steps, pos_frac, hard_frac
        self.rng = np.random.default_rng(seed)
        self.tree = cKDTree(ds.xy) if len(ds) else None
        self.pos_idx = np.where(ds.n_pos > 0)[0]
        self.hard_idx = hard_negatives(ds) if (hard_frac > 0 and len(ds)) else np.array([], np.int64)

    def __len__(self):
        return self.steps

    def __iter__(self):
        n = len(self.ds)
        B = min(self.batch, n)
        for _ in range(self.steps):
            u = self.rng.random()
            if len(self.pos_idx) and u < self.pos_frac:
                seed = int(self.rng.choice(self.pos_idx))
            elif len(self.hard_idx) and u < self.pos_frac + self.hard_frac:
                seed = int(self.rng.choice(self.hard_idx))
            else:
                seed = int(self.rng.integers(n))
            k = min(n, max(B, 2 * B))
            _, nb = self.tree.query(self.ds.xy[seed], k=k)
            nb = np.atleast_1d(nb)
            rest = nb[nb != seed]
            pick = self.rng.choice(rest, B - 1, replace=False) if B > 1 else np.array([], int)
            yield [seed] + [int(j) for j in pick]


def sweep_clusters(ds: ChipDataset, batch: int) -> list[list[int]]:
    """Deterministic cover of the split by spatial clusters (every chip exactly once). Used for val/test."""
    from scipy.spatial import cKDTree
    n = len(ds)
    if n == 0:
        return []
    tree = cKDTree(ds.xy)
    left = set(range(n))
    out = []
    order = np.lexsort((ds.xy[:, 1], ds.xy[:, 0]))
    for s in order:
        s = int(s)
        if s not in left:
            continue
        _, nb = tree.query(ds.xy[s], k=min(n, max(batch * 4, batch)))
        grp = [int(j) for j in np.atleast_1d(nb) if int(j) in left][:batch]
        for j in grp:
            left.discard(j)
        out.append(grp)
    return out
