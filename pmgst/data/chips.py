"""Exported GEE tiles + label polygons -> training chips (same format as the synthetic generator).

  pip install rasterio geopandas shapely pyproj
  python -m pmgst.data.chips --raw data/raw --out data/chips --labels labels/mandi_kullu_2023.gpkg \
        --label-extent labels/mandi_kullu_extent.gpkg --chip 128 --ctx 64

* Labels: polygons of NEW landslides of the event season (any CRS). Pixels outside --label-extent are set to
  ignore (255): unmapped is not "no landslide". A 1-pixel ring outside each polygon is ignore too.
* Chips whose labelled area is empty are skipped.  --neg-keep < 1 randomly drops chips without landslides to save disk.
* S1: zeros from the export are no-data -> validity mask (audit C2 fix); dates are the 12-day bin centres.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
from scipy import ndimage as ndi

from .formats import rain_features, save_chip, write_index
from .splits import block_ids
from .terrain import area_resample, d8_flow_accumulation, terrain_channels

BIN_DAYS, N_BINS = 12, 15


def _rio():
    try:
        import rasterio
        return rasterio
    except ImportError as e:
        raise SystemExit("rasterio is required for the real-data pipeline: pip install rasterio geopandas shapely pyproj") from e


def read_tif(path):
    rio = _rio()
    with rio.open(path) as s:
        return s.read(), s.transform, s.res[0], s.crs


def find_tiles(raw: Path):
    out = []
    for f in sorted(raw.glob("s2post_x*_y*.tif")):
        m = re.match(r"s2post_x(-?\d+)_y(-?\d+)\.tif", f.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


def rasterize_labels(gdf, transform, shape, crs, extent=None):
    from rasterio import features
    g = gdf.to_crs(crs)
    lbl = features.rasterize([(geom, 1) for geom in g.geometry if geom is not None and not geom.is_empty],
                             out_shape=shape, transform=transform, fill=0, dtype="uint8") if len(g) else np.zeros(shape, np.uint8)
    lbl = lbl.astype(bool)
    ring = ndi.binary_dilation(lbl, iterations=1) & ~lbl
    out = lbl.astype(np.uint8); out[ring] = 255
    if extent is not None:
        e = extent.to_crs(crs)
        inside = features.rasterize([(geom, 1) for geom in e.geometry], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
        out[~inside] = 255
    return out


def build_tile(raw: Path, x0: int, y1: int, labels, extent, roads, chip: int, ctx: int, n_scales: int, buffer_m: float,
               neg_keep: float, rng, out_chips: Path, region: str):
    rio = _rio()
    nm = lambda k: raw / f"{k}_x{x0}_y{y1}.tif"
    post_raw, tf, res, crs = read_tif(nm("s2post")); pre_raw, _, _, _ = read_tif(nm("s2pre"))
    H, W = post_raw.shape[1:]
    to_s2 = lambda a: np.clip(a[:6].astype(np.float32) / 10000.0, 0, 1)
    s2 = np.stack([to_s2(pre_raw), to_s2(post_raw)])                                    # [2,6,H,W]
    s2v = np.stack([(pre_raw[6] > 0) & (pre_raw[:6].sum(0) > 0), (post_raw[6] > 0) & (post_raw[:6].sum(0) > 0)])[:, None].astype(np.uint8)
    s2 = s2 * s2v
    s1_raw, _, _, _ = read_tif(nm("s1"))
    vv, vh = s1_raw[:N_BINS].astype(np.float32) / 100.0, s1_raw[N_BINS:].astype(np.float32) / 100.0
    ok = (s1_raw[:N_BINS] != 0) & (s1_raw[N_BINS:] != 0) & (vv > -35) & (vv < 10) & (vh > -40) & (vh < 5)   # zeros = no data
    s1 = np.stack([vv, vh], 1) * ok[:, None]                                            # [T,2,H,W]
    s1v = ok[:, None].astype(np.uint8)
    s1_dt = (np.arange(N_BINS) * BIN_DAYS + BIN_DAYS / 2).astype(np.float32)
    dem_raw, _, dres, _ = read_tif(nm("dem"))
    z = dem_raw[0].astype(np.float64)
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    ch = terrain_channels(z, dres)
    f = int(round(dres / res))
    dem = ndi.zoom(ch, (1, f, f), order=1, mode="nearest", grid_mode=True) if f > 1 else ch
    assert dem.shape[1:] == (H, W), f"DEM grid {dem.shape} does not match S2 grid {(H, W)}"
    acc = d8_flow_accumulation(z, dres)
    channel = acc * dres * dres >= 1e6                                                   # >= 1 km2 upstream area
    river_d = ndi.distance_transform_edt(~channel) * dres
    road_d = np.full(river_d.shape, 5000.0)
    if roads is not None and len(roads):
        from rasterio import features
        sub = roads.to_crs(crs)
        line = features.rasterize([(g, 1) for g in sub.geometry if g is not None], out_shape=river_d.shape,
                                  transform=tf * tf.scale(f, f), fill=0, dtype="uint8", all_touched=True).astype(bool)
        if line.any():
            road_d = ndi.distance_transform_edt(~line) * dres
    up = lambda a: ndi.zoom(a, f, order=0, mode="nearest", grid_mode=True) if f > 1 else a
    river_d10, road_d10 = up(river_d), up(road_d)
    rain_raw, rtf, rres, _ = read_tif(nm("rain"))
    st, _, _, _ = read_tif(nm("static"))
    water = (st[0] == 80).astype(np.uint8)
    lith = np.clip(st[1], 0, 12).astype(np.uint8)
    label_full = rasterize_labels(labels, tf, (H, W), crs, extent) if labels is not None else np.zeros((H, W), np.uint8)

    px_buf = int(round(buffer_m / res)); csz_m = chip * res
    n_core = int(round((H * res - 2 * buffer_m) / csz_m))
    rows = []
    for gy in range(n_core):
        for gx in range(n_core):
            r0, c0 = px_buf + gy * chip, px_buf + gx * chip
            sl = (slice(r0, r0 + chip), slice(c0, c0 + chip))
            lab = label_full[sl]
            n_valid, n_pos = int((lab != 255).sum()), int((lab == 1).sum())
            if n_valid == 0 or s1v[(slice(None), slice(None)) + sl].sum() == 0:
                continue
            if n_pos == 0 and rng.random() > neg_keep:
                continue
            cr, cc = r0 + chip // 2, c0 + chip // 2
            c_s2, c_v, c_dem = [], [], []
            for s in range(2, n_scales + 1):
                size = chip * 2 ** (s - 1)
                a, b = cr - size // 2, cc - size // 2
                win = (slice(a, a + size), slice(b, b + size))
                v = s2v[(slice(None), slice(None)) + win].astype(np.float32)
                c_s2.append(area_resample(s2[(slice(None), slice(None)) + win], ctx, np.broadcast_to(v, (2, 6, size, size))))
                c_v.append((area_resample(v, ctx) >= 0.5).astype(np.uint8))
                c_dem.append(area_resample(dem[(slice(None),) + win], ctx))
            d = dem[(slice(None),) + sl]
            cx = tf.c + cc * res; cy = tf.f - cr * res                   # geometric chip centre (chip is even-sized)
            rcol = min(int((cx - rtf.c) // rres), rain_raw.shape[2] - 1); rrow = min(int((rtf.f - cy) // rres), rain_raw.shape[1] - 1)
            rain = rain_features(rain_raw[:, rrow, rcol])
            beta = np.radians(d[1].mean()); ax, ay = d[2].mean(), d[3].mean(); nrm = np.hypot(ax, ay) + 1e-9
            meta = np.array([cx, cy, d[0].mean(), np.sin(beta) * ax / nrm, np.sin(beta) * ay / nrm, np.cos(beta),
                             river_d10[sl].mean(), road_d10[sl].mean(), 0.0, rain[2] / 100.0], np.float32)
            cid = f"{region}_x{x0}_y{y1}_{gy:02d}{gx:02d}"
            save_chip(out_chips / f"{cid}.npz", dict(
                s2=s2[(slice(None), slice(None)) + sl], s2_valid=s2v[(slice(None), slice(None)) + sl],
                s1=s1[(slice(None), slice(None)) + sl], s1_valid=s1v[(slice(None), slice(None)) + sl], s1_dt=s1_dt,
                dem=d, lith=lith[sl], water=water[sl], rain=rain, ctx_s2=np.stack(c_s2), ctx_valid=np.stack(c_v),
                ctx_dem=np.stack(c_dem), label=lab, meta=meta))
            rows.append(dict(chip_id=cid, path=f"chips/{cid}.npz", x=cx, y=cy, region=region, block=0, n_pos_px=n_pos,
                             n_valid_px=n_valid, has_s1=1, has_s2=int(s2v[1][(0,) + sl].any())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=None, help="polygon file (gpkg/geojson/shp) of landslides of the event season")
    ap.add_argument("--label-extent", default=None, help="polygon file of the area the inventory covers")
    ap.add_argument("--roads", default=None, help="optional OSM road lines")
    ap.add_argument("--chip", type=int, default=128); ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--n-scales", type=int, default=3)
    ap.add_argument("--buffer", type=float, default=2560.0, help="tile buffer in metres (as exported)")
    ap.add_argument("--neg-keep", type=float, default=1.0)
    ap.add_argument("--region", default="hp"); ap.add_argument("--block-m", type=float, default=10_000.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import geopandas as gpd
    raw, out = Path(a.raw), Path(a.out); (out / "chips").mkdir(parents=True, exist_ok=True)
    labels = gpd.read_file(a.labels) if a.labels else None
    extent = gpd.read_file(a.label_extent) if a.label_extent else None
    roads = gpd.read_file(a.roads) if a.roads else None
    rng = np.random.default_rng(a.seed)
    tiles = find_tiles(raw); print(f"{len(tiles)} tiles")
    rows = []
    for k, (x0, y1) in enumerate(tiles):
        r = build_tile(raw, x0, y1, labels, extent, roads, a.chip, a.ctx, a.n_scales, a.buffer, a.neg_keep, rng, out / "chips", a.region)
        rows += r
        print(f"  tile {k + 1}/{len(tiles)} x{x0} y{y1}: {len(r)} chips ({sum(q['n_pos_px'] > 0 for q in r)} with landslides)")
    blk = block_ids([r["x"] for r in rows], [r["y"] for r in rows], a.block_m)
    for r, b in zip(rows, blk):
        r["block"] = int(b)
    write_index(out / "index.csv", rows)
    npos = sum(r["n_pos_px"] for r in rows); nv = sum(r["n_valid_px"] for r in rows)
    print(f"[chips] {len(rows)} chips, {sum(r['n_pos_px'] > 0 for r in rows)} with landslides, positive px {100 * npos / max(nv, 1):.3f}% of labelled px")


if __name__ == "__main__":
    main()
