"""Block 12: risk prediction & early-warning outputs from a trained checkpoint.

  python -m pmgst.infer --ckpt runs/exp/best.pt --split test --out outputs/exp

Outputs per region: probability + uncertainty maps (GeoTIFF if rasterio is installed, else .npy), landslide polygons
(GeoJSON), per-chip warning table (CSV), alerts.json. Region inference uses ONE graph over all chips (two passes):
pass 1 embeds every chip, the GAT runs once over the whole region, pass 2 decodes with the refined node vectors.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from torch.utils.data import DataLoader

from .calibration import apply_calibration
from .config import from_dict
from .data.dataset import ChipDataset, collate, sweep_clusters, to_device
from .data.formats import M_BLOCK
from .models.build import build_model
from .train import amp_setup, make_loader, pick_device

LEVELS = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]


def load_model(ckpt_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = from_dict(ck["cfg"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, tuple(ck.get("calib", (1.0, 0.0))), float(ck.get("thr", 0.5))


def warning_levels(p_max: float, u_mean: float, taus=(0.25, 0.5, 0.75), u_max: float = 0.5):
    """Level from calibrated max probability; high uncertainty adds a REVIEW flag and never downgrades."""
    lvl = int(p_max >= taus[0]) + int(p_max >= taus[1]) + int(p_max >= taus[2])
    return LEVELS[lvl], bool(u_mean > u_max)


@torch.no_grad()
def predict(model, ds: ChipDataset, cfg, device, dtype, full_graph: bool = True):
    """Returns p [N,H,W], u [N,H,W] in dataset order."""
    P = [None] * len(ds); U = [None] * len(ds)
    def run(loader, node_override_fn=None):
        for batch in loader:
            ids = batch["idx"].tolist()
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                out = model(batch, node_override=node_override_fn(ids) if node_override_fn else None)
            for j, i in enumerate(ids):
                P[i] = out["p"][j].float().cpu().numpy(); U[i] = out["u"][j].float().cpu().numpy()
    if not full_graph:
        run(make_loader(ds, sweep_clusters(ds, cfg.train.batch), cfg, device))
        return np.stack(P), np.stack(U)
    seq = [list(range(i, min(i + cfg.train.batch, len(ds)))) for i in range(0, len(ds), cfg.train.batch)]
    h0, meta, lh = [None] * len(ds), [None] * len(ds), [None] * len(ds)
    for batch in make_loader(ds, seq, cfg, device):
        ids = batch["idx"].tolist(); b = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            h = model.embed_only(b)
        for j, i in enumerate(ids):
            h0[i], meta[i], lh[i] = h[j].float(), b["meta"][j], model.lith_hist(b["lith"])[j]
    H0, M, LH = torch.stack(h0), torch.stack(meta), torch.stack(lh)
    refined, _ = model.refine_nodes(H0, M, LH, M[:, M_BLOCK].long())
    run(make_loader(ds, seq, cfg, device), lambda ids: refined[ids])
    return np.stack(P), np.stack(U)


def predict_ensemble(ckpts, split="test", root=None, full_graph=True, drop=None, device=None):
    """ckpts: path or comma-separated paths. Every model scores the same chips; probabilities are averaged (raw and calibrated
    separately), uncertainty is averaged. Returns dict(ds, cfg, p_raw, p_cal, u, thr_raw, thr_cal, calibs)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = [c.strip() for c in str(ckpt_list(ckpts)).split(",") if c.strip()]
    P, PC, U, thr_r, thr_c, calibs, ds, cfg0 = [], [], [], [], [], [], None, None
    for k, path in enumerate(paths):
        model, cfg, calib, thr = load_model(path, device)
        if root:
            cfg.data.root = root
        if ds is None:
            ds, cfg0 = ChipDataset(cfg, split, augment=False, drop=drop), cfg
        _, dtype, _ = amp_setup(cfg, device)
        p, u = predict(model, ds, cfg, device, dtype, full_graph)
        P.append(p); PC.append(apply_calibration(p, calib)); U.append(u)
        thr_r.append(thr); thr_c.append(float(apply_calibration(np.array([thr]), calib)[0])); calibs.append(list(calib))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return dict(ds=ds, cfg=cfg0, p_raw=np.mean(P, 0), p_cal=np.mean(PC, 0), u=np.mean(U, 0),
                thr_raw=float(np.mean(thr_r)), thr_cal=float(np.mean(thr_c)), calibs=calibs, n_models=len(paths))


def ckpt_list(c):
    return ",".join(c) if isinstance(c, (list, tuple)) else c


def extract_objects(p: np.ndarray, u: np.ndarray, thr: float, min_px: int = 9):
    """Threshold -> 3x3 opening -> connected components -> drop < min_px. Returns label image + attribute list."""
    m = ndi.binary_opening(p >= thr, structure=np.ones((3, 3), bool))
    lab, n = ndi.label(m)
    objs = []
    for i in range(1, n + 1):
        mm = lab == i
        a = int(mm.sum())
        if a < min_px:
            lab[mm] = 0
            continue
        cy, cx = ndi.center_of_mass(mm)
        objs.append(dict(id=i, area_px=a, p_mean=float(p[mm].mean()), p_max=float(p[mm].max()),
                         u_mean=float(u[mm].mean()), cy=float(cy), cx=float(cx)))
    return lab, objs


def mosaic(rows, arrs, chip_px: int, res: float = 10.0, fill=np.nan):
    """Place chips on a canvas by their UTM centres. Returns canvas, (x_min, y_max) of the top-left corner."""
    xs = np.array([r["x"] for r in rows]); ys = np.array([r["y"] for r in rows])
    half = chip_px * res / 2
    x0, y1 = xs.min() - half, ys.max() + half
    W = int(round((xs.max() + half - x0) / res)); H = int(round((y1 - (ys.min() - half)) / res))
    canvas = np.full((H, W), fill, np.float32); cnt = np.zeros((H, W), np.float32)
    acc = np.zeros((H, W), np.float32)
    for r, a in zip(rows, arrs):
        c0 = int(round((r["x"] - half - x0) / res)); r0 = int(round((y1 - (r["y"] + half)) / res))
        acc[r0:r0 + chip_px, c0:c0 + chip_px] += a; cnt[r0:r0 + chip_px, c0:c0 + chip_px] += 1
    ok = cnt > 0
    canvas[ok] = acc[ok] / cnt[ok]
    return canvas, (x0, y1)


def save_raster(path: Path, arr: np.ndarray, origin, res=10.0, crs="EPSG:32643"):
    try:
        import rasterio
        from rasterio.transform import from_origin
        with rasterio.open(path.with_suffix(".tif"), "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                           count=1, dtype="float32", crs=crs, transform=from_origin(origin[0], origin[1], res, res),
                           nodata=float("nan"), compress="deflate") as dst:
            dst.write(arr.astype(np.float32), 1)
    except ImportError:
        np.save(path.with_suffix(".npy"), arr)
        path.with_suffix(".json").write_text(json.dumps(dict(x_min=origin[0], y_max=origin[1], res=res, crs=crs)))


def run_region(ckpt, split="test", out_dir="outputs/run", full_graph=True, taus=(0.25, 0.5, 0.75), u_max=0.5,
               overrides=None, device=None):
    """ckpt may be several comma-separated checkpoints (ensemble: averaged calibrated probability)."""
    E = predict_ensemble(ckpt, split, (overrides or {}).get("root"), full_graph, device=device)
    ds, cfg, p, u, thr_c = E["ds"], E["cfg"], E["p_cal"], E["u"], E["thr_cal"]
    calib = E["calibs"][0]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    csz = cfg.data.chip
    table, feats, alerts = [], [], []
    for i, r in enumerate(ds.rows):
        lab, objs = extract_objects(p[i], u[i], thr_c)
        level, review = warning_levels(float(p[i].max()), float(u[i].mean()), taus, u_max)
        table.append(dict(chip_id=r["chip_id"], x=r["x"], y=r["y"], p_max=round(float(p[i].max()), 4),
                          frac_above_thr=round(float((p[i] >= thr_c).mean()), 5), n_objects=len(objs),
                          u_mean=round(float(u[i].mean()), 4), level=level, review=review))
        for o in objs:
            gx = r["x"] + (o["cx"] - csz / 2) * 10.0; gy = r["y"] - (o["cy"] - csz / 2) * 10.0
            feats.append(dict(type="Feature", geometry=dict(type="Point", coordinates=[gx, gy]),
                              properties=dict(chip_id=r["chip_id"], **{k: o[k] for k in ("area_px", "p_mean", "p_max", "u_mean")},
                                              area_m2=o["area_px"] * 100)))
        if level in ("HIGH", "VERY HIGH"):
            alerts.append(dict(chip_id=r["chip_id"], x=r["x"], y=r["y"], level=level, review=review, p_max=table[-1]["p_max"]))
    with open(out / "warnings.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    (out / "landslides.geojson").write_text(json.dumps(dict(type="FeatureCollection", crs=dict(type="name", properties=dict(name="EPSG:32643")), features=feats)))
    (out / "alerts.json").write_text(json.dumps(alerts, indent=1))
    pm, org = mosaic(ds.rows, p, csz); um, _ = mosaic(ds.rows, u, csz)
    save_raster(out / "probability", pm, org); save_raster(out / "uncertainty", um, org)
    print(f"[infer] {E['n_models']} model(s) | {len(ds)} chips | {len(feats)} objects | alerts {len(alerts)} | calib a={calib[0]:.2f} b={calib[1]:.2f} thr={thr_c:.3f} -> {out}")
    return dict(p=p, u=u, table=table, ds=ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--root", default=None, help="override data.root")
    ap.add_argument("--cluster", action="store_true", help="per-cluster graphs instead of one region graph")
    ap.add_argument("--taus", type=float, nargs=3, default=(0.25, 0.5, 0.75))
    a = ap.parse_args()
    run_region(a.ckpt, a.split, a.out, not a.cluster, tuple(a.taus), overrides={"root": a.root} if a.root else None)


if __name__ == "__main__":
    main()
