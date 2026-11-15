"""Deep baselines B3-B5 (docs/07 §baselines). Same batch dict and output dict as PMGSTNet, so train / evaluate /
infer / profile_gpu work unchanged. No Swin, no graph, no physics, no pyramid: this is what the extra blocks must beat.

  unet  (B3)  post-event S2 + DEM                              plain CNN pyramid + FPN decoder
  siam  (B4)  pre/post S2 (shared CNN, change features) + S1 mean/std + DEM
  utae  (B5)  siam + the S1 time series through per-date CNN + temporal attention (same L-TAE masks as block 8)

Select with  --set model.arch=unet|siam|utae.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..data.formats import DEM_CHANNELS, DEM_STATS, IDX_SLOPE
from .encoders import CNNBranch, gn
from .heads import FPNDecoder, Head
from .pmgst_net import S1_MU, S1_SD, S2_MU, S2_SD
from .temporal import TemporalEncoder


class DLBaseline(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg, self.variant = cfg, cfg.model.arch
        assert self.variant in ("unet", "siam", "utae"), self.variant
        m, d = cfg.model, cfg.data
        dims = list(m.dims); self.dims = dims
        for n, v in (("dem_mu", [DEM_STATS[c][0] for c in DEM_CHANNELS]), ("dem_sd", [DEM_STATS[c][1] for c in DEM_CHANNELS])):
            self.register_buffer(n, torch.tensor(v).view(1, -1, 1, 1))
        self.register_buffer("s2_mu", torch.tensor(S2_MU).view(1, 1, -1, 1, 1))
        self.register_buffer("s2_sd", torch.tensor(S2_SD).view(1, 1, -1, 1, 1))
        self.register_buffer("s1_mu", torch.tensor(S1_MU).view(1, 1, -1, 1, 1))
        self.register_buffer("s1_sd", torch.tensor(S1_SD).view(1, 1, -1, 1, 1))
        nb = d.s2_bands
        if self.variant == "unet":
            self.enc = nn.ModuleList([CNNBranch(nb + 1 + d.dem_ch, dims)])          # post S2 + valid + DEM
            self.n_branch, raw_ch = 1, nb + 2
        else:
            self.s2_enc = CNNBranch(nb + 1, dims)                                   # shared over pre and post
            self.change = nn.ModuleList(nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), gn(c), nn.GELU()) for c in dims)
            self.static = CNNBranch(4 + 1 + d.dem_ch, dims)                         # S1 mean/std (VV,VH) + presence + DEM
            self.n_branch, raw_ch = 2, nb * 2 + 2
            if self.variant == "utae":
                self.s1_enc = CNNBranch(3, dims)
                self.temporal = TemporalEncoder(dims[-1], m.t_heads, m.t_layers)
                self.n_branch = 3
        self.merge = nn.ModuleList(nn.Sequential(nn.Conv2d(self.n_branch * c, c, 1, bias=False), gn(c), nn.GELU())
                                   for c in dims)
        self.decoder = FPNDecoder(dims, m.d_dec, raw_ch=raw_ch, drop=m.mc_dropout)
        self.head = Head(64, m.head)

    # -- helpers ------------------------------------------------------------------
    def _dem(self, dem):
        return (dem - self.dem_mu) / self.dem_sd

    def _s1_stats(self, b):
        """per-pixel mean/std of VV,VH over valid, non-padded dates + presence flag -> [B,5,H,W]"""
        s1, v = b["s1"], b["s1_valid"] * (~b["s1_pad"])[:, :, None, None, None].float()
        x = ((s1 - self.s1_mu) / self.s1_sd) * v
        n = v.sum(1).clamp_min(1.0)
        mean = x.sum(1) / n
        var = ((x - mean[:, None] * v) ** 2).sum(1) / n
        pres = (v.sum(1) > 0).float()
        return torch.cat([mean, var.sqrt(), pres], 1)

    def _s2_stages(self, b):
        valid = b["s2_valid"]
        x = torch.cat([((b["s2"] - self.s2_mu) / self.s2_sd) * valid, valid], 2)
        B = x.shape[0]
        f = self.s2_enc(x.flatten(0, 1))
        out = []
        for k, a in enumerate(f):
            # zero the features of invalid (cloud / dropped) S2: a GroupNorm on a constant-zero input amplifies gradients
            # ~300x per layer (1e21 over the encoder) and the global grad-clip then freezes every other weight
            pres = (F.interpolate(valid.flatten(0, 1), size=a.shape[-2:], mode="area") >= 0.5).to(a.dtype)
            a = (a * pres).view(B, 2, *a.shape[1:])
            out.append(self.change[k](torch.cat([a[:, 1], (a[:, 1] - a[:, 0]).abs()], 1)))
        return out

    def _stages(self, b):
        dem = self._dem(b["dem"])
        if self.variant == "unet":
            x = torch.cat([((b["s2"][:, 1] - self.s2_mu[:, 0]) / self.s2_sd[:, 0]) * b["s2_valid"][:, 1],
                           b["s2_valid"][:, 1], dem], 1)
            return [self.merge[k](f) for k, f in enumerate(self.enc[0](x))]
        s2 = self._s2_stages(b)
        st = self.static(torch.cat([self._s1_stats(b), dem], 1))
        parts = [s2, st]
        if self.variant == "utae":
            B, T = b["s1"].shape[:2]
            v = b["s1_valid"]
            x = torch.cat([((b["s1"] - self.s1_mu) / self.s1_sd) * v, v], 2)
            f = self.s1_enc(x.flatten(0, 1))
            low = f[-1].view(B, T, *f[-1].shape[1:])
            vlow = nn.functional.interpolate(v.flatten(0, 1), size=low.shape[-2:], mode="area").view(B, T, *low.shape[-2:])
            att = self.temporal.masks(low, b["s1_dt"], b["s1_pad"][:, :, None, None] | (vlow < 0.5))
            parts.append([self.temporal.apply(att, a.view(B, T, *a.shape[1:])) for a in f])
        return [self.merge[k](torch.cat([p[k] for p in parts], 1)) for k in range(len(self.dims))]

    # -- PMGSTNet-compatible interface -------------------------------------------------
    def forward(self, b: dict, node_override=None, aux: bool = False) -> dict:
        feats = self._stages(b)
        valid2 = b["s2_valid"].prod(1)
        post = ((b["s2"][:, 1] - self.s2_mu[:, 0]) / self.s2_sd[:, 0]) * b["s2_valid"][:, 1]
        slope = self._dem(b["dem"])[:, IDX_SLOPE:IDX_SLOPE + 1]
        if self.variant == "unet":
            raw = torch.cat([post, slope, b["s2_valid"][:, 1]], 1)
        else:
            delta = ((b["s2"][:, 1] - b["s2"][:, 0]) / self.s2_sd[:, 0]) * valid2
            raw = torch.cat([post, delta, slope, valid2], 1)
        x, emb = self.decoder(feats, raw)
        out = self.head(x)
        out.update(emb=emb, node=None, node_ref=None, fs_wet=None, maps=None, alphas=[], naux={})
        return out

    @torch.no_grad()
    def embed_only(self, b: dict) -> torch.Tensor:                # region inference: nothing to share across chips
        return b["dem"].new_zeros(b["dem"].shape[0], 1)

    def lith_hist(self, lith):
        return lith.new_zeros(lith.shape[0], 1).float()

    def refine_nodes(self, h0, meta, lh, block):
        return h0, {}

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
