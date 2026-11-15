"""Multi-seed ablation + baseline table with mean +- sd over seeds (docs/07: a block stays only if it earns its place).

  python -m pmgst.ablate --config configs/rtx3060.yaml --out runs/ablate --seeds 3 --variants full no_graph no_physics b3_unet
  python -m pmgst.ablate --config configs/rtx3060.yaml --out runs/ablate --seeds 3 --variants all --dry-run     # list the plan
Each run trains on the same spatial split (data.test_fold fixed), only the seed changes; metrics come from evaluate on the TEST fold.
Resumable: a run with an existing eval_test.json is skipped. Writes <out>/ablation.md and ablation.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config

VARIANTS = {
    "full": [],
    "no_pyramid": ["model.use_pyramid=false"],
    "no_graph": ["model.use_graph=false"],
    "no_physics": ["model.use_physics=false", "loss.lam_phys=0"],
    "no_contrastive": ["loss.lam_con=0"],
    "s2_only": ["model.use_s1=false"],
    "s1_only": ["model.use_s2=false"],
    "memory": ["model.use_memory=true"],
    "logit_head": ["model.head=logit", "loss.lam_unc=0"],
    "b3_unet": ["model.arch=unet"],
    "b4_siam": ["model.arch=siam"],
    "b5_utae": ["model.arch=utae"],
}
CORE = ["full", "no_pyramid", "no_graph", "no_physics", "s2_only", "s1_only", "b3_unet", "b4_siam", "b5_utae"]
METRICS = [("ap", "AP"), ("f1", "F1"), ("iou", "IoU"), ("ece", "ECE")]


def plan(names, seeds):
    return [(n, s) for n in names for s in range(seeds)]


def run_one(config, base_overrides, name, seed, out, epochs=None):
    from .evaluate import evaluate_ckpt
    from .train import train
    d = Path(out) / f"{name}_s{seed}"
    res_f = d / "eval_test.json"
    if res_f.exists():
        return json.loads(res_f.read_text())
    ov = list(base_overrides) + VARIANTS[name] + [f"train.out_dir={d}", f"train.seed={seed}"]
    if epochs:
        ov.append(f"train.epochs={epochs}")
    cfg = load_config(config, ov)
    train(cfg)
    r = evaluate_ckpt(d / "best.pt", "test")
    res_f.write_text(json.dumps(r, indent=1, default=float))
    return r


def summarise(results: dict) -> str:
    """results[name] = list of eval dicts (one per seed). Calibrated metrics for AP/F1/IoU/ECE, mean +- sd."""
    lines = ["| variant | seeds | " + " | ".join(m for _, m in METRICS) + " | obj F1 |", "|---|---|" + "---|" * (len(METRICS) + 1)]
    for name, rs in results.items():
        cells = []
        for k, _ in METRICS:
            v = np.array([r["calibrated"][k] for r in rs], float)
            cells.append(f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:.3f}")
        o = np.array([r["objects"]["obj_f1"] for r in rs], float)
        lines.append(f"| {name} | {len(rs)} | " + " | ".join(cells) + f" | {o.mean():.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--out", default="runs/ablate")
    ap.add_argument("--variants", nargs="+", default=["core"], help=f"names from {sorted(VARIANTS)}, 'core' or 'all'")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="overrides applied to every run, e.g. data.root=data/chips")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    names = CORE if a.variants == ["core"] else list(VARIANTS) if a.variants == ["all"] else a.variants
    bad = [n for n in names if n not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}; choose from {sorted(VARIANTS)}")
    runs = plan(names, a.seeds)
    print(f"{len(runs)} runs: " + ", ".join(f"{n}/s{s}" for n, s in runs))
    if a.dry_run:
        return
    results = {n: [] for n in names}
    for n, s in runs:
        print(f"=== {n} seed {s} ===")
        results[n].append(run_one(a.config, a.set, n, s, a.out, a.epochs or None))
    md = summarise(results)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "ablation.md").write_text(md + "\n"); (Path(a.out) / "ablation.json").write_text(json.dumps(results, indent=1, default=float))
    print(md)


if __name__ == "__main__":
    main()
