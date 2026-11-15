"""Evaluation on a held-out split with spatial-block bootstrap CIs (docs/07).

  python -m pmgst.evaluate --ckpt runs/exp/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data.dataset import ChipDataset
from .infer import predict_ensemble
from .metrics import ObjectMetrics, PixelMetrics


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


def evaluate_ckpt(ckpt, split="test", full_graph=True, root=None, bins=500, drop=None):
    """ckpt: one checkpoint or comma-separated checkpoints (ensemble of calibrated probabilities)."""
    E = predict_ensemble(ckpt, split, root, full_graph, drop)
    ds, cfg, p, pc, thr, calib = E["ds"], E["cfg"], E["p_raw"], E["p_cal"], E["thr_raw"], E["calibs"][0]
    labels = np.stack([np.load(Path(cfg.data.root) / r["path"])["label"] for r in ds.rows])
    blocks = np.array([r["block"] for r in ds.rows])
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
    res["calib_a_b"], res["thr_val"], res["n_chips"], res["drop"], res["n_models"] = calib, thr, len(ds), drop, E["n_models"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint, or several separated by commas (ensemble)"); ap.add_argument("--split", default="test")
    ap.add_argument("--root", default=None); ap.add_argument("--cluster", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--drop", choices=["s2", "s1"], default=None, help="switch a modality off (cloud-robust metric: --drop s2)")
    a = ap.parse_args()
    r = evaluate_ckpt(a.ckpt, a.split, not a.cluster, a.root, drop=a.drop)
    print(json.dumps(r, indent=1, default=float))
    name = f"eval_{a.split}" + (f"_no{a.drop}" if a.drop else "") + ".json"
    Path(a.out or Path(a.ckpt.split(",")[0]).parent / (name if "," not in a.ckpt else name.replace(".json", "_ens.json"))).write_text(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    main()
