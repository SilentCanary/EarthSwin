"""Chip QA: run this on real chips BEFORE the first training run. Most real-data failures are data bugs, not model bugs.

  python -m pmgst.data.qa --root data/chips                    # report + qa_report.json, exit code 1 if an ERROR was found
  python -m pmgst.data.qa --root data/chips --png qa.png       # + montage of landslide chips (look at it: do outlines sit on scars?)

Checks: NaN/inf, value ranges per modality, label values, valid-pixel fractions, class balance, positives per fold/block,
DEM statistics vs the model's normalisation, and a label-alignment test (landslides must be steeper than background and
must show a post-event NDVI drop; if not, the labels are shifted / in the wrong CRS / from another season).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .formats import DEM_CHANNELS, DEM_STATS, IDX_SLOPE, load_chip, read_index
from .splits import assign_folds


def _ndvi(s2):                                   # s2 [..., 6, H, W]: B2 B3 B4 B8 B11 B12
    b4, b8 = s2[..., 2, :, :], s2[..., 3, :, :]
    return (b8 - b4) / (b8 + b4 + 1e-6)


def run_qa(root, max_chips: int = 0, folds: int = 5, seed: int = 42, png: str | None = None) -> dict:
    root = Path(root)
    rows = read_index(root / "index.csv")
    err, warn = [], []
    if not rows:
        return dict(errors=["index.csv is empty"], warnings=[], n_chips=0)
    ids = [r["chip_id"] for r in rows]
    if len(set(ids)) != len(ids):
        err.append("duplicate chip_id in index.csv")
    xy = {(round(r["x"]), round(r["y"])) for r in rows}
    if len(xy) != len(rows):
        warn.append(f"{len(rows) - len(xy)} chips share a centre with another chip (overlapping tiles?)")
    sel = list(range(len(rows)))
    if max_chips and len(sel) > max_chips:
        sel = sorted(np.random.default_rng(0).choice(len(rows), max_chips, replace=False).tolist())

    dem_sum = np.zeros(len(DEM_CHANNELS)); dem_sq = np.zeros(len(DEM_CHANNELS)); dem_n = 0
    s2_valid, s1_valid_t, bad_finite, bad_range, bad_label, no_s2_post, miss_file = [], [], [], [], [], [], []
    sl_p, sl_n, dn_p, dn_n = [], [], [], []
    rain_bad = 0
    for i in sel:
        r = rows[i]
        f = root / r["path"]
        if not f.exists():
            miss_file.append(r["chip_id"]); continue
        c = load_chip(f)
        for k, a in c.items():
            if a.dtype.kind == "f" and not np.isfinite(a.astype(np.float32)).all():
                bad_finite.append((r["chip_id"], k)); break
        s2 = c["s2"].astype(np.float32); v2 = c["s2_valid"].astype(bool)
        if s2.min() < -0.01 or s2.max() > 1.5:
            bad_range.append((r["chip_id"], "s2", float(s2.min()), float(s2.max())))
        s1 = c["s1"].astype(np.float32); v1 = c["s1_valid"].astype(bool)
        if v1.any():
            s1v = s1[np.repeat(v1, 2, 1)]
            if s1v.min() < -50 or s1v.max() > 20:
                bad_range.append((r["chip_id"], "s1(dB)", float(s1v.min()), float(s1v.max())))
        lab = c["label"]
        if not set(np.unique(lab).tolist()) <= {0, 1, 255}:
            bad_label.append(r["chip_id"])
        if (c["rain"] < 0).any():
            rain_bad += 1
        s2_valid.append(float(v2[1].mean())); s1_valid_t.append(v1.reshape(v1.shape[0], -1).mean(1))
        if not v2[1].any():
            no_s2_post.append(r["chip_id"])
        d = c["dem"].astype(np.float32)
        dem_sum += d.reshape(len(DEM_CHANNELS), -1).sum(1); dem_sq += (d.reshape(len(DEM_CHANNELS), -1) ** 2).sum(1); dem_n += d[0].size
        # alignment diagnostics on pixels with clear pre+post optical
        ok = v2[0, 0] & v2[1, 0] & (lab != 255)
        pos, neg = ok & (lab == 1), ok & (lab == 0)
        if pos.any() and neg.any():
            dn = _ndvi(s2[1]) - _ndvi(s2[0])
            sl_p.append(d[IDX_SLOPE][pos].mean()); sl_n.append(d[IDX_SLOPE][neg].mean())
            dn_p.append(dn[pos].mean()); dn_n.append(dn[neg].mean())

    n = len(sel)
    if miss_file:
        err.append(f"{len(miss_file)} chip files listed in index.csv are missing (first: {miss_file[0]})")
    if bad_finite:
        err.append(f"{len(bad_finite)} chips contain NaN/inf (first: {bad_finite[0]})")
    if bad_label:
        err.append(f"{len(bad_label)} chips have label values outside {{0,1,255}}")
    if bad_range:
        err.append(f"{len(bad_range)} chips have out-of-range values (first: {bad_range[0]}) - scaling bug?")
    if rain_bad:
        warn.append(f"{rain_bad} chips have negative rainfall features")

    npos = np.array([r["n_pos_px"] for r in rows]); nval = np.array([r["n_valid_px"] for r in rows])
    block = np.array([r["block"] for r in rows])
    pos_chips = int((npos > 0).sum())
    frac = float(npos.sum() / max(nval.sum(), 1))
    if pos_chips == 0:
        err.append("no chip contains a landslide pixel")
    elif frac < 1e-4:
        warn.append(f"landslide pixels are {100 * frac:.4f}% of labelled pixels: very sparse, expect low AP")
    fold = assign_folds(block, npos, folds, seed)
    per_fold = {int(k): dict(chips=int((fold == k).sum()), with_landslides=int(((fold == k) & (npos > 0)).sum()),
                             pos_px=int(npos[fold == k].sum())) for k in range(folds)}
    for k, v in per_fold.items():
        if v["with_landslides"] < 5:
            warn.append(f"fold {k} has only {v['with_landslides']} chips with landslides: its val/test metrics will be very noisy")
    regions = {}
    for r in rows:
        g = regions.setdefault(r["region"], dict(chips=0, with_landslides=0))
        g["chips"] += 1; g["with_landslides"] += int(r["n_pos_px"] > 0)

    mean = dem_sum / max(dem_n, 1); sd = np.sqrt(np.maximum(dem_sq / max(dem_n, 1) - mean ** 2, 0))
    dem_report = {}
    for j, ch in enumerate(DEM_CHANNELS):
        m0, s0 = DEM_STATS[ch]
        dem_report[ch] = dict(mean=float(mean[j]), std=float(sd[j]), model_mean=m0, model_std=s0)
        if abs(mean[j] - m0) > 1.0 * s0 or not (0.33 < (sd[j] + 1e-9) / s0 < 3.0):
            warn.append(f"DEM channel {ch}: data mean/std {mean[j]:.3g}/{sd[j]:.3g} far from the model's {m0}/{s0} - use data.fit_norm=true")

    align = None
    if sl_p:
        align = dict(n_chips=len(sl_p), slope_pos_minus_bg_deg=float(np.mean(sl_p) - np.mean(sl_n)),
                     dndvi_pos_minus_bg=float(np.mean(dn_p) - np.mean(dn_n)))
        if align["slope_pos_minus_bg_deg"] <= 0:
            warn.append("landslide pixels are NOT steeper than background: labels may be shifted or in the wrong CRS")
        if align["dndvi_pos_minus_bg"] >= 0:
            warn.append("landslide pixels show no post-event NDVI drop vs background: wrong season/dates, or labels shifted")
    else:
        warn.append("alignment test skipped (no chip with clear pre+post optical and both classes)")
    if no_s2_post:
        warn.append(f"{len(no_s2_post)} chips have no valid post-event optical pixel (cloud): they rely on S1 only")
    s1t = np.mean(np.stack(s1_valid_t), 0) if s1_valid_t else np.zeros(0)
    if s1t.size and (s1t < 0.05).all():
        err.append("S1 is invalid almost everywhere: orbit mismatch, empty export, or layover mask too aggressive")

    rep = dict(n_chips=len(rows), n_checked=n, chips_with_landslides=pos_chips, landslide_px_fraction=frac,
               s2_post_valid_mean=float(np.mean(s2_valid)) if s2_valid else None,
               s1_valid_by_date=[round(float(x), 3) for x in s1t], folds=per_fold, regions=regions, dem=dem_report,
               alignment=align, errors=err, warnings=warn)
    (root / "qa_report.json").write_text(json.dumps(rep, indent=1))
    if png:
        montage(root, rows, png)
    return rep


def montage(root, rows, out_png: str, n: int = 12, seed: int = 0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = [r for r in rows if r["n_pos_px"] > 0]
    if not pos:
        return
    pick = [pos[i] for i in np.random.default_rng(seed).choice(len(pos), min(n, len(pos)), replace=False)]
    cols = 4; rws = int(np.ceil(len(pick) / cols))
    fig, axs = plt.subplots(rws, cols * 2, figsize=(cols * 5, rws * 2.6))
    axs = np.atleast_2d(axs)
    for ax in axs.ravel():
        ax.axis("off")
    for k, r in enumerate(pick):
        c = load_chip(Path(root) / r["path"])
        rgb = c["s2"][1, [2, 1, 0]].astype(np.float32).transpose(1, 2, 0)
        rgb = np.clip(rgb / max(np.percentile(rgb[rgb > 0], 98), 1e-3) if (rgb > 0).any() else rgb, 0, 1)
        a1, a2 = axs[(k // cols), 2 * (k % cols)], axs[(k // cols), 2 * (k % cols) + 1]
        a1.imshow(rgb); a1.contour(c["label"] == 1, levels=[0.5], colors="red", linewidths=0.8)
        a1.set_title(f"{r['chip_id'][-14:]} post S2", fontsize=7)
        a2.imshow(c["dem"][IDX_SLOPE].astype(np.float32), cmap="magma"); a2.contour(c["label"] == 1, levels=[0.5], colors="cyan", linewidths=0.8)
        a2.set_title("slope (deg)", fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--max-chips", type=int, default=0, help="0 = check all")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--png", default=None)
    a = ap.parse_args()
    r = run_qa(a.root, a.max_chips, a.folds, png=a.png)
    print(f"[qa] {r['n_chips']} chips ({r.get('chips_with_landslides', 0)} with landslides, "
          f"{100 * r.get('landslide_px_fraction', 0):.3f}% of labelled px) | S2 post valid {r.get('s2_post_valid_mean')}")
    if r.get("alignment"):
        print(f"[qa] alignment: landslide - background slope {r['alignment']['slope_pos_minus_bg_deg']:+.1f} deg, "
              f"dNDVI {r['alignment']['dndvi_pos_minus_bg']:+.3f} (expect + and -)")
    for k, v in r.get("folds", {}).items():
        print(f"[qa] fold {k}: {v['chips']} chips, {v['with_landslides']} with landslides, {v['pos_px']} px")
    for w in r["warnings"]:
        print("[warn]", w)
    for e in r["errors"]:
        print("[ERROR]", e)
    print("[qa] report ->", Path(a.root) / "qa_report.json")
    raise SystemExit(1 if r["errors"] else 0)


if __name__ == "__main__":
    main()
