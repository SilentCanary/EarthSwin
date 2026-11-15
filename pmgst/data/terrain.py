"""DEM -> the 8 terrain channels of the chip format (docs/06 §10). numpy/scipy only."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .formats import DEM_CHANNELS


def d8_flow_accumulation(z: np.ndarray, res: float = 10.0) -> np.ndarray:
    """Cells draining through each cell (D8, steepest descent). Pits stop the flow (fine for a TWI proxy)."""
    H, W = z.shape
    zp = np.pad(z.astype(np.float64), 1, constant_values=np.inf)   # edges never drain outward
    best = np.zeros((H, W)); tgt = np.full((H, W), -1, np.int64)
    rr, cc = np.mgrid[0:H, 0:W]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nz = zp[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            drop = (z - nz) / (res * np.hypot(dy, dx))
            m = drop > best
            best = np.where(m, drop, best)
            tgt = np.where(m, (rr + dy) * W + (cc + dx), tgt)
    order = np.argsort(-z.ravel(), kind="stable")
    acc = np.ones(H * W)
    t = tgt.ravel()
    for i in order:
        j = t[i]
        if j >= 0:
            acc[j] += acc[i]
    return acc.reshape(H, W)


def terrain_channels(z: np.ndarray, res: float = 10.0) -> np.ndarray:
    """z [H,W] metres -> [8,H,W] in DEM_CHANNELS order (physical units)."""
    z = z.astype(np.float64)
    zy, zx = np.gradient(z, res)                       # d/drow (south), d/dcol (east)
    slope = np.degrees(np.arctan(np.hypot(zx, zy)))
    de, dn = -zx, zy                                   # downslope components (east, north)
    nrm = np.hypot(de, dn)
    ok = slope > 1.0
    asp_x = np.where(ok, de / (nrm + 1e-12), 0.0)
    asp_y = np.where(ok, dn / (nrm + 1e-12), 0.0)
    r = np.gradient(zx, res, axis=1); t = np.gradient(zy, res, axis=0); s = np.gradient(zx, res, axis=0)
    p, q = zx, zy
    pq = p * p + q * q
    safe = pq > 1e-8
    prof = np.where(safe, -(r * p * p + 2 * s * p * q + t * q * q) / (np.maximum(pq, 1e-8) * (1 + pq) ** 1.5), 0.0)
    plan = np.where(safe, -(t * p * p - 2 * s * p * q + r * q * q) / (np.maximum(pq, 1e-8) ** 1.5), 0.0)
    prof, plan = np.clip(prof, -0.5, 0.5), np.clip(plan, -0.5, 0.5)
    acc = d8_flow_accumulation(z, res)
    tanb = np.tan(np.radians(np.maximum(slope, 1.0)))
    twi = np.log((acc * res) / tanb)                   # specific catchment area / tan(beta)
    relief = ndi.maximum_filter(z, size=15) - ndi.minimum_filter(z, size=15)
    out = np.stack([z, slope, asp_x, asp_y, plan, prof, twi, relief]).astype(np.float32)
    assert out.shape[0] == len(DEM_CHANNELS)
    return out


def area_resample(a: np.ndarray, out: int, w: np.ndarray | None = None) -> np.ndarray:
    """Block-mean resample [..., S, S] -> [..., out, out] (S divisible by out). With weights w (same trailing shape)
    it is a masked mean (value 0 where no weight)."""
    S = a.shape[-1]
    f = S // out
    assert f * out == S, f"{S} not divisible by {out}"
    def blk(x):
        return x.reshape(*x.shape[:-2], out, f, out, f).mean(axis=(-3, -1))
    if w is None:
        return blk(a)
    num, den = blk(a * w), blk(w)
    return np.where(den > 0, num / np.maximum(den, 1e-6), 0.0)


def sar_geometry_mask(slope_deg, asp_x, asp_y, look_az_deg: float, inc_deg: float = 39.0, margin_deg: float = 3.0):
    """True where Sentinel-1 backscatter is unusable because of layover or radar shadow (approximate, geometry only).

    look_az_deg: azimuth the radar looks TOWARD (clockwise from north). Sentinel-1 at ~32 N: ascending ~ 78, descending ~ 282.
    inc_deg: ellipsoid incidence angle (IW swath 29-46, 39 = mid-swath). asp_x/asp_y: downslope direction (E, N), as in the chip format.
    Slope tilt toward the sensor a_r = atan(tan(slope) * cos(angle between downslope and the toward-sensor direction)).
      layover : a_r >= inc - margin           shadow : a_r <= -(90 - inc) + margin
    """
    phi = np.radians(look_az_deg)
    toward = (-np.sin(phi), -np.cos(phi))                       # horizontal unit vector pointing at the sensor (E, N)
    c = asp_x * toward[0] + asp_y * toward[1]                   # + : slope faces the sensor
    a_r = np.degrees(np.arctan(np.tan(np.radians(slope_deg)) * c))
    return (a_r >= inc_deg - margin_deg) | (a_r <= -(90.0 - inc_deg) + margin_deg)
