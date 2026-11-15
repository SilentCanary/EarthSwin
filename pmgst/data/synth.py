"""Synthetic region generator -> chips in the real on-disk format.

Why: lets every block, the training loop and the evaluation run end to end before real data exists, and gives a
sanity benchmark (the model must learn it, and S2-only / S1-only / flat-terrain ablations must behave sensibly).

What is simulated (all physically motivated):
  * fractal DEM -> slope/aspect/curvature/TWI (same code as real data)
  * landslides: elongated scars along the downslope direction on steep slopes (label)
  * S2: vegetation -> bare-soil change on landslides; harvest-like change on GENTLE slopes (distractor);
        clouds in pre/post (some chips fully clouded in post -> only S1 can see the event)
  * S1: speckled backscatter time series, persistent VV drop after the event date; transient global wet-snow dips
  * rain features, lithology classes, rivers, roads, chip-level graph attributes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .formats import (IDX_ASPX, IDX_ASPY, IDX_ELEV, IDX_SLOPE, rain_features, save_chip, write_index)
from .splits import block_ids
from .terrain import area_resample, terrain_channels

VEG = np.array([0.03, 0.05, 0.03, 0.32, 0.16, 0.08], np.float32)
GRASS = np.array([0.05, 0.08, 0.07, 0.36, 0.24, 0.13], np.float32)
ROCK = np.array([0.15, 0.18, 0.21, 0.27, 0.34, 0.31], np.float32)
SOIL = np.array([0.13, 0.17, 0.20, 0.25, 0.33, 0.29], np.float32)


def _smooth_noise(rng, shape, sigma):
    n = ndi.gaussian_filter(rng.standard_normal(shape), sigma, mode="reflect")
    return (n - n.mean()) / (n.std() + 1e-9)


def _blob(shape, cy, cx, length, width, ang):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    dy, dx = yy - cy, xx - cx
    u = dx * np.cos(ang) + dy * np.sin(ang)
    v = -dx * np.sin(ang) + dy * np.cos(ang)
    return (u / length) ** 2 + (v / width) ** 2 <= 1.0


def make_region(seed: int, chips_x: int, chips_y: int, chip: int, n_dates: int = 14, storm: float = 1.0):
    rng = np.random.default_rng(seed)
    H, W = chips_y * chip, chips_x * chip
    # ---- DEM
    z = 2200 + 450 * _smooth_noise(rng, (H, W), 40) + 150 * _smooth_noise(rng, (H, W), 14) \
        + 40 * _smooth_noise(rng, (H, W), 5) + 6 * _smooth_noise(rng, (H, W), 1.5)
    dem = terrain_channels(z, 10.0)
    slope = dem[IDX_SLOPE]
    # ---- static maps
    lith = np.digitize(_smooth_noise(rng, (H, W), 30), np.linspace(-1.5, 1.5, 12)).astype(np.uint8)  # 0..12
    water = ((dem[6] > np.percentile(dem[6], 98.5)) & (slope < 12)).astype(np.uint8)
    # ---- landslides
    susc = 1 / (1 + np.exp(-(slope - 33) / 3.0)) * (0.4 + 0.6 * (lith > 4)) * (1 - water)
    n_seeds = int(0.00015 * H * W * storm) + 3
    ys, xs = np.nonzero(susc > 0.25)
    lbl = np.zeros((H, W), bool)
    t_event = np.full((H, W), np.inf, np.float32)
    if len(ys):
        w = susc[ys, xs]; w = w / w.sum()
        for k in rng.choice(len(ys), size=n_seeds, p=w):
            cy, cx = ys[k], xs[k]
            ang = np.arctan2(-dem[IDX_ASPY][cy, cx], dem[IDX_ASPX][cy, cx])      # image coords: row grows south
            b = _blob((H, W), cy, cx, rng.uniform(6, 22), rng.uniform(2.5, 6), ang) & (slope > 18)
            lbl |= b
            t_event[b] = np.minimum(t_event[b], rng.uniform(25, 150))
    ring = ndi.binary_dilation(lbl, iterations=1) & ~lbl
    label = lbl.astype(np.uint8); label[ring] = 255
    # distractors: harvest-like changes on gentle slopes
    dis = np.zeros((H, W), bool)
    gentle = np.argwhere((slope < 10) & (water == 0))
    for k in rng.choice(len(gentle), size=min(len(gentle), n_seeds), replace=False) if len(gentle) else []:
        cy, cx = gentle[k]
        dis |= _blob((H, W), cy, cx, rng.uniform(5, 14), rng.uniform(4, 10), rng.uniform(0, np.pi))
    dis &= ~lbl
    # ---- S2
    lc = _smooth_noise(rng, (H, W), 12)
    base = np.where((lc > 0.6)[None], ROCK[:, None, None],
                    np.where((lc > -0.3)[None], VEG[:, None, None], GRASS[:, None, None])).astype(np.float32)
    base = np.where((dem[IDX_ELEV] > 2900)[None], ROCK[:, None, None], base)
    tex = lambda: (0.012 * rng.standard_normal((6, H, W)) + 0.02 * _smooth_noise(rng, (H, W), 3)[None]).astype(np.float32)
    pre = base + tex()
    post = base * (1 + 0.04 * rng.standard_normal()) + tex()
    a = 0.85
    post = np.where(lbl[None], (1 - a) * post + a * SOIL[:, None, None] + 0.01 * rng.standard_normal((6, H, W)), post)
    post[3] = np.where(dis, post[3] * 0.6, post[3]); post[0] = np.where(dis, post[0] + 0.03, post[0])
    cloud_pre = _smooth_noise(rng, (H, W), 25) > 1.3
    cloud_post = _smooth_noise(rng, (H, W), 25) > 1.0
    s2 = np.stack([pre, post]).clip(0, 1).astype(np.float32)               # [2,6,H,W]
    s2_valid = np.stack([~cloud_pre, ~cloud_post])[:, None].astype(np.uint8)  # [2,1,H,W]
    s2 = s2 * s2_valid
    # whole chips cloudy in post (S1-only cases) are added at chip level below
    # ---- S1 time series
    days = np.sort(rng.choice(np.arange(0, 180, 6), size=n_dates, replace=False)).astype(np.float32)
    veg_like = (lc <= 0.6)
    vv0 = np.where(veg_like, -9.5, -7.0) + 1.2 * _smooth_noise(rng, (H, W), 6) + 0.25 * (slope - 25) / 15
    vh0 = vv0 - 6.0 + 0.8 * _smooth_noise(rng, (H, W), 5)
    s1 = np.zeros((n_dates, 2, H, W), np.float32)
    for i, d in enumerate(days):
        off = 1.0 * rng.standard_normal()                                  # per-date global offset (moisture)
        wet_snow = (dem[IDX_ELEV] > 3100) * (-3.0 if rng.random() < 0.15 else 0.0)
        ch = (d >= t_event)
        farm = dis & (d >= 60)
        vv = vv0 + off + wet_snow + np.where(ch, -3.2, 0.0) + np.where(farm, 2.0, 0.0)
        vh = vh0 + off + wet_snow + np.where(ch, -2.4, 0.0) + np.where(farm, 1.5, 0.0)
        for c, m in enumerate((vv, vh)):
            lin = 10 ** (m / 10) * rng.gamma(4.0, 1 / 4.0, size=(H, W))    # 4-look speckle
            s1[i, c] = 10 * np.log10(np.maximum(lin, 1e-4))
    holes = (slope > 55) | (rng.random((H, W)) < 0.01)
    s1_valid = np.broadcast_to((~holes)[None, None], (n_dates, 1, H, W)).astype(np.uint8).copy()
    s1 = s1 * s1_valid
    # ---- rain (region storm, chip multipliers applied later)
    rain_daily = rng.gamma(0.6, 4.0, size=60).astype(np.float32)
    rain_daily[-8:-4] += storm * rng.uniform(40, 90, size=4)
    return dict(dem=dem, lith=lith, water=water, label=label, s2=s2, s2_valid=s2_valid, s1=s1, s1_valid=s1_valid,
                days=days, rain_daily=rain_daily, H=H, W=W)


def write_synthetic(out_dir, seed=0, chips_x=8, chips_y=8, chip=64, ctx=32, n_scales=3, n_dates=14,
                    block_m=None, storm=1.0, log=print):
    out_dir = Path(out_dir); (out_dir / "chips").mkdir(parents=True, exist_ok=True)
    R = make_region(seed, chips_x, chips_y, chip, n_dates, storm)
    H, W = R["H"], R["W"]
    pad = chip * 2 ** (n_scales - 1)                       # margin so the biggest context crop always exists
    P = lambda a: np.pad(a, [(0, 0)] * (a.ndim - 2) + [(pad, pad), (pad, pad)], mode="reflect")
    dem_p, s2_p, s2v_p = P(R["dem"]), P(R["s2"]), P(R["s2_valid"].astype(np.float32))
    water_dist = ndi.distance_transform_edt(R["water"] == 0) * 10.0
    rng = np.random.default_rng(seed + 1)
    x0, y0 = 700_000.0, 3_500_000.0
    roads = [(rng.normal(), rng.normal(), rng.uniform(-300, 300)) for _ in range(2)]
    rows, cxs, cys = [], [], []
    for gy in range(chips_y):
        for gx in range(chips_x):
            sl = (slice(gy * chip, (gy + 1) * chip), slice(gx * chip, (gx + 1) * chip))
            cy, cx = gy * chip + chip // 2, gx * chip + chip // 2
            # context crops (scales 2..S) centred on the chip, area-resampled to ctx
            c_s2, c_v, c_dem = [], [], []
            for s in range(2, n_scales + 1):
                size = chip * 2 ** (s - 1)
                ys_, xs_ = cy + pad - size // 2, cx + pad - size // 2
                win = (slice(ys_, ys_ + size), slice(xs_, xs_ + size))
                v = s2v_p[(slice(None), slice(None)) + win]                      # [2,1,size,size]
                c_s2.append(area_resample(s2_p[(slice(None), slice(None)) + win], ctx, np.broadcast_to(v, (2, 6, size, size))))
                c_v.append((area_resample(v, ctx) >= 0.5).astype(np.uint8))
                c_dem.append(area_resample(dem_p[(slice(None),) + win], ctx))
            s2 = R["s2"][(slice(None), slice(None)) + sl].copy(); s2v = R["s2_valid"][(slice(None), slice(None)) + sl].copy()
            if rng.random() < 0.10:                                              # fully clouded post image
                s2v[1] = 0; s2[1] = 0
            lbl = R["label"][sl]
            dem = R["dem"][(slice(None),) + sl]
            sl_mean, asp = dem[IDX_SLOPE].mean(), np.arctan2(dem[IDX_ASPX].mean(), dem[IDX_ASPY].mean())
            beta = np.radians(sl_mean)
            xx, yy = x0 + cx * 10.0, y0 + (H - cy) * 10.0
            mult = rng.uniform(0.7, 1.3)
            rain = rain_features(R["rain_daily"] * mult)
            road_d = min(abs(a * (xx - x0) + b * (yy - y0) + c) / np.hypot(a, b) for a, b, c in roads)
            meta = np.array([xx, yy, dem[IDX_ELEV].mean(), np.sin(beta) * np.sin(asp), np.sin(beta) * np.cos(asp),
                             np.cos(beta), water_dist[sl].mean(), road_d, 0.0, rain[2] / 100.0], np.float32)
            chip_d = dict(s2=s2, s2_valid=s2v, s1=R["s1"][(slice(None), slice(None)) + sl],
                          s1_valid=R["s1_valid"][(slice(None), slice(None)) + sl], s1_dt=R["days"],
                          dem=dem, lith=R["lith"][sl], water=R["water"][sl], rain=rain,
                          ctx_s2=np.stack(c_s2), ctx_valid=np.stack(c_v), ctx_dem=np.stack(c_dem),
                          label=lbl, meta=meta)
            cid = f"s{seed}_{gy:03d}_{gx:03d}"
            save_chip(out_dir / "chips" / f"{cid}.npz", chip_d)
            rows.append(dict(chip_id=cid, path=f"chips/{cid}.npz", x=xx, y=yy, region=f"synth{seed}", block=0,
                             n_pos_px=int((lbl == 1).sum()), n_valid_px=int((lbl != 255).sum()),
                             has_s1=1, has_s2=int(s2v[1].any())))
            cxs.append(xx); cys.append(yy)
    bm = block_m or chip * 10.0 * 2
    blk = block_ids(cxs, cys, bm)
    for r, b in zip(rows, blk):
        r["block"] = int(b)
    write_index(out_dir / "index.csv", rows)
    npos = sum(r["n_pos_px"] for r in rows)
    log(f"[synth] seed={seed}: {len(rows)} chips, {sum(r['n_pos_px'] > 0 for r in rows)} with landslides, "
        f"{npos} positive px ({100 * npos / (len(rows) * chip * chip):.2f}%)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chips", type=int, default=8, help="chips per side")
    ap.add_argument("--chip", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=32)
    a = ap.parse_args()
    write_synthetic(a.out, a.seed, a.chips, a.chips, a.chip, a.ctx)
