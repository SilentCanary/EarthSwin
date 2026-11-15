"""Block 13: explainability.

  attention_maps      built-in: modality weights alpha_m(p), scale weights beta_s(p), temporal masks a_t(p), GAT alpha_ij
  gradcam             which pixels/regions drove the landslide logit (deepest fused stage)
  integrated_grads    pixel attributions per input group with a mean-value baseline
  group_importance    Monte-Carlo Shapley values over input groups (S2, S1, terrain, rain, lithology)
  deletion_curve      faithfulness check: remove top-attributed S2 pixels, score should fall fast
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data.formats import DEM_STATS

GROUPS = ("s2", "s1", "terrain", "rain", "lith")


def attention_maps(model, batch) -> dict:
    with torch.no_grad():
        out = model(batch, aux=True)
    mods = ["S2", "S1", "DEM"] if model.use_s2 and model.use_s1 else (["S2", "DEM"] if model.use_s2 else ["S1", "DEM"] if model.use_s1 else ["DEM"])
    return dict(
        modality_names=mods,
        modality_alpha=[a.float().cpu() for a, _ in out["alphas"]],             # per stage [B,M,1,h,w]
        scale_beta=[None if b is None else b.float().cpu() for b in (out.get("betas") or [])],
        temporal_att=None if out.get("att") is None else out["att"].float().cpu(),   # [B,heads,T,h,w]
        gat_alpha=out["naux"].get("gat_alpha"), edge_index=out["naux"].get("edge_index"),
        learned_lambda=out["naux"].get("lam"))


def gradcam(model, batch, stage: int = -1, region: torch.Tensor | None = None) -> torch.Tensor:
    """Grad-CAM for the summed landslide logit over `region` (default: pixels predicted p >= 0.5). Returns [B,H,W] in 0..1."""
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = model(batch, aux=True)
        feats = out["stage_feats"][stage]
        m = (out["p"].detach() >= 0.5) if region is None else region.bool()
        if m.sum() == 0:
            m = out["p"].detach() >= out["p"].detach().flatten(1).quantile(0.99, dim=1).view(-1, 1, 1)
        score = (out["logit"] * m).sum()
        g = torch.autograd.grad(score, feats, allow_unused=True)[0]
    w = g.mean((2, 3), keepdim=True)
    cam = F.relu((w * feats).sum(1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=batch["label"].shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    mx = cam.flatten(1).amax(1).clamp_min(1e-9).view(-1, 1, 1)
    return (cam / mx).cpu()


def _score(model, b):
    out = model(b)
    m = (out["p"].detach() >= 0.5)
    return (out["logit"] * m).sum() if m.any() else out["logit"].amax((1, 2)).sum()


def integrated_gradients(model, batch, steps: int = 16) -> dict:
    """IG for s2, s1, dem, rain. Baseline = dataset mean (0 after normalisation). Returns |attribution| summed over
    channels as maps [B,H,W] (rain: [B,R])."""
    base = {"s2": model.s2_mu.view(1, 1, -1, 1, 1).expand_as(batch["s2"]).clone(),
            "s1": model.s1_mu.view(1, 1, -1, 1, 1).expand_as(batch["s1"]).clone(),
            "dem": model.dem_mu.expand_as(batch["dem"]).clone(), "rain": torch.zeros_like(batch["rain"])}
    attr = {k: torch.zeros_like(v) for k, v in base.items()}
    for i in range(1, steps + 1):
        b = dict(batch)
        for k in base:
            b[k] = (base[k] + (i / steps) * (batch[k] - base[k])).requires_grad_(True)
        with torch.enable_grad():
            s = _score(model, b)
            grads = torch.autograd.grad(s, [b[k] for k in base], allow_unused=True)
        for k, g in zip(base, grads):
            if g is not None:
                attr[k] += g / steps
    out = {}
    for k in base:
        a = (batch[k] - base[k]) * attr[k]
        out[k] = a.abs().sum(tuple(range(1, a.ndim))) if k == "rain" else a.abs().flatten(1, -3).sum(1) if a.ndim == 5 else a.abs().sum(1)
    return {k: v.detach().cpu() for k, v in out.items()}


def _ablate(b: dict, keep: set, model) -> dict:
    o = dict(b)
    if "s2" not in keep:
        o["s2_valid"] = torch.zeros_like(b["s2_valid"]); o["ctx_valid"] = torch.zeros_like(b["ctx_valid"])
    if "s1" not in keep:
        o["s1_valid"] = torch.zeros_like(b["s1_valid"]); o["s1_pad"] = torch.ones_like(b["s1_pad"])
    if "terrain" not in keep:
        o["dem"] = model.dem_mu.expand_as(b["dem"]).clone()
        o["ctx_dem"] = model.dem_mu.unsqueeze(1).expand_as(b["ctx_dem"]).clone()
    if "rain" not in keep:
        o["rain"] = torch.zeros_like(b["rain"])
    if "lith" not in keep:
        o["lith"] = torch.zeros_like(b["lith"])
    return o


@torch.no_grad()
def group_importance(model, batch, n_perm: int = 12, seed: int = 0) -> dict:
    """Shapley value of each input group for v(S) = mean p on landslide pixels - mean p on background pixels."""
    rng = np.random.default_rng(seed)
    y = (batch["label"] == 1); bg = (batch["label"] == 0)
    if not y.any():
        return {g: float("nan") for g in GROUPS}
    cache = {}
    def v(keep: frozenset):
        if keep not in cache:
            p = model(_ablate(batch, set(keep), model))["p"].float()
            cache[keep] = float(p[y].mean() - p[bg].mean())
        return cache[keep]
    phi = {g: 0.0 for g in GROUPS}
    for _ in range(n_perm):
        order = rng.permutation(GROUPS); cur = frozenset()
        for g in order:
            nxt = cur | {g}
            phi[g] += (v(nxt) - v(cur)) / n_perm
            cur = nxt
    return phi


@torch.no_grad()
def deletion_curve(model, batch, attribution: torch.Tensor, fracs=(0, 0.05, 0.1, 0.2, 0.4, 0.6)) -> list:
    """Remove (mark invalid) the top-attributed fraction of S2 pixels; returns mean p on landslide pixels."""
    y = batch["label"] == 1
    order = attribution.flatten(1).argsort(1, descending=True)
    res = []
    for f in fracs:
        b = dict(batch)
        v = batch["s2_valid"].clone()
        k = int(f * attribution[0].numel())
        if k:
            mask = torch.zeros_like(attribution.flatten(1)); mask.scatter_(1, order[:, :k], 1.0)
            v = v * (1 - mask.view_as(attribution))[:, None, None]
        b["s2_valid"] = v
        res.append(float(model(b)["p"].float()[y].mean()) if y.any() else float("nan"))
    return res
