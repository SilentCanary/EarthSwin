"""Evaluation on a held-out split with spatial-block bootstrap CIs (docs/07).

  python -m pmgst.evaluate --ckpt runs/exp/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .calibration import apply_calibration
from .data.dataset import ChipDataset
from .infer import load_model, predict
from .metrics import ObjectMetrics, PixelMetrics
from .train import amp_setup


def _hist(p, lab, bins=500):
    pm = PixelMetrics(bins); pm.update(p, lab)
    return pm.pos.copy(), pm.neg.copy()


def _stats(pos, neg, bins):
    pm = PixelMetrics(bins); pm.pos, pm.neg = pos, neg
    r = pm.compute()
    return r["ap"], r["f1"], r["iou"]


def bootstrap(hist_by_block: dict, bins=500, n=300, seed=0):
    keys = list(hist_by_block); rng = np.random.default_rng(seed); res = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        pos = sum(hist_by_block[keys[i]][0] for i in pick); neg = sum(hist_by_block[keys[i]][1] for i in pick)
        if pos.sum() > 0:
            res.append(_stats(pos, neg, bins))
    a = np.array(res)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return dict(ap_ci=ci(a[:, 0]), f1_ci=ci(a[:, 1]), iou_ci=ci(a[:, 2]), n_boot=len(a))


def evaluate_ckpt(ckpt, split="test", full_graph=True, root=None, bins=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, calib, thr = load_model(ckpt, device)
    if root:
        cfg.data.root = root
    _, dtype, _ = amp_setup(cfg, device)
    ds = ChipDataset(cfg, split, augment=False)
    p, _ = predict(model, ds, cfg, device, dtype, full_graph)
    labels = np.stack([np.load(Path(cfg.data.root) / r["path"])["label"] for r in ds.rows])
    blocks = np.array([r["block"] for r in ds.rows])
    pc = apply_calibration(p, calib)
    res = {}
    for name, pp in (("raw", p), ("calibrated", pc)):
        pm = PixelMetrics(); pm.update(pp, labels)
        res[name] = pm.compute()
    h = {}
    for b in np.unique(blocks):
        m = blocks == b
        h[int(b)] = _hist(p[m], labels[m], bins)
    res["bootstrap"] = bootstrap(h, bins)
    om = ObjectMetrics(); om.update(p >= thr, labels)
    res["objects"] = om.compute()
    res["calib_a_b"], res["thr_val"], res["n_chips"] = list(calib), thr, len(ds)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--root", default=None); ap.add_argument("--cluster", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = evaluate_ckpt(a.ckpt, a.split, not a.cluster, a.root)
    print(json.dumps(r, indent=1, default=float))
    Path(a.out or Path(a.ckpt).parent / f"eval_{a.split}.json").write_text(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    main()
