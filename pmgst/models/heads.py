"""Block 11: uncertainty-aware decoder: FPN (UPerNet-lite) + full-resolution refinement + evidential Beta head.

Evidential head:  e+ = softplus(z+), e- = softplus(z-), alpha = e+ + 1, beta = e- + 1, S = alpha + beta
   p = alpha / S,   vacuity u = 2 / S,   Var[p] = alpha beta / (S^2 (S+1))
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ConvBlock, gn


class FPNDecoder(nn.Module):
    def __init__(self, dims, d: int, raw_ch: int, drop: float = 0.0):
        super().__init__()
        self.lat = nn.ModuleList(nn.Conv2d(c, d, 1) for c in dims)
        self.smooth = nn.ModuleList(ConvBlock(d, d) for _ in dims)
        self.fuse = ConvBlock(len(dims) * d, d)
        self.skip = ConvBlock(raw_ch, 16)
        self.refine = nn.Sequential(ConvBlock(d + 16, 64), nn.Dropout2d(drop), ConvBlock(64, 64))

    def forward(self, feats: list[torch.Tensor], raw: torch.Tensor):
        """feats: 4 maps (strides 4..32). raw [B,C,H,W] full-res input stack for boundary detail.
        Returns (full-res features [B,64,H,W], stride-4 embedding [B,d,H/4,W/4])."""
        p = self.lat[-1](feats[-1]); outs = [self.smooth[-1](p)]
        for k in range(len(feats) - 2, -1, -1):
            p = self.lat[k](feats[k]) + F.interpolate(p, size=feats[k].shape[-2:], mode="bilinear", align_corners=False)
            outs.append(self.smooth[k](p))
        size = feats[0].shape[-2:]
        emb = self.fuse(torch.cat([F.interpolate(o, size=size, mode="bilinear", align_corners=False) for o in outs], 1))
        up = F.interpolate(emb, size=raw.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat([up, self.skip(raw)], 1)), emb


class Head(nn.Module):
    def __init__(self, cin=64, kind="evidential"):
        super().__init__()
        self.kind = kind
        self.conv = nn.Conv2d(cin, 2 if kind == "evidential" else 1, 1)

    def forward(self, x):
        z = self.conv(x).float()
        if self.kind == "evidential":
            ev = F.softplus(z)
            alpha, beta = ev[:, 0] + 1.0, ev[:, 1] + 1.0
            S = alpha + beta
            p = alpha / S
            return dict(p=p, u=2.0 / S, var=alpha * beta / (S * S * (S + 1)), alpha=alpha, beta=beta,
                        logit=torch.log(p.clamp(1e-6, 1 - 1e-6) / (1 - p).clamp(1e-6, 1)))
        logit = z[:, 0]
        p = torch.sigmoid(logit)
        return dict(p=p, u=4 * p * (1 - p), var=p * (1 - p), logit=logit)     # u = predictive-entropy proxy
