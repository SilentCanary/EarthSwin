"""Block 8: temporal transformer over the Sentinel-1 series (U-TAE / L-TAE style, pixel-dense).

1. Per-pixel sequence at the lowest-resolution stage (T tokens) goes through pre-norm transformer layers with
   irregular-date sinusoidal encoding and a padding mask (missing dates / invalid pixels).
2. A master-query attention readout gives masks a_t(p) per head; the SAME masks are upsampled and applied to the
   features of every stage:  F^k(p) = sum_t a_t(p) F^k_t(p)   (channels split across heads).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def time_encoding(dt: torch.Tensor, d: int, tau: float = 1000.0) -> torch.Tensor:
    """dt [B,T] days -> [B,T,d] sinusoidal encoding of the real acquisition day."""
    k = torch.arange(d // 2, device=dt.device, dtype=torch.float32)
    div = tau ** (2 * k / d)
    ang = dt.float().unsqueeze(-1) / div
    pe = torch.cat([ang.sin(), ang.cos()], -1)
    return pe if pe.shape[-1] == d else F.pad(pe, (0, d - pe.shape[-1]))


class PreNormLayer(nn.Module):
    def __init__(self, d, heads, drop=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(d); self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))

    def forward(self, z, pad):                                       # pad [N,T] True = ignore
        y = self.n1(z)
        z = z + self.attn(y, y, y, key_padding_mask=pad, need_weights=False)[0]
        return z + self.ffn(self.n2(z))


class TemporalEncoder(nn.Module):
    def __init__(self, d_low: int, heads: int = 4, layers: int = 1, drop: float = 0.1):
        super().__init__()
        assert d_low % heads == 0
        self.heads, self.d = heads, d_low
        self.layers = nn.ModuleList(PreNormLayer(d_low, heads, drop) for _ in range(layers))
        self.kproj = nn.Linear(d_low, d_low)
        self.query = nn.Parameter(torch.randn(heads, d_low // heads) * 0.1)   # L-TAE master query

    def masks(self, low: torch.Tensor, dt: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """low [B,T,D,h,w]; dt [B,T]; pad [B,T,h,w] True = missing. Returns a [B,heads,T,h,w]."""
        B, T, D, h, w = low.shape
        z = low.permute(0, 3, 4, 1, 2).reshape(B * h * w, T, D)
        z = z + time_encoding(dt, D).to(z.dtype).unsqueeze(1).expand(B, h * w, T, D).reshape(B * h * w, T, D)
        p = pad.permute(0, 2, 3, 1).reshape(B * h * w, T)
        allpad = p.all(1, keepdim=True)
        p_safe = p & ~allpad                                          # all-missing sequences: avoid NaN
        for layer in self.layers:
            z = layer(z, p_safe)
        k = self.kproj(z).view(B * h * w, T, self.heads, D // self.heads)
        logit = (k * self.query).sum(-1) / math.sqrt(D // self.heads)  # N,T,heads
        logit = logit.float().masked_fill(p_safe.unsqueeze(-1), float("-inf"))
        a = logit.softmax(1)
        a = torch.where(allpad.unsqueeze(-1), torch.zeros_like(a), a)
        return a.view(B, h, w, T, self.heads).permute(0, 4, 3, 1, 2)

    @staticmethod
    def apply(att: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """att [B,H_,T,h,w]; feats [B,T,D,H,W] -> [B,D,H,W] (masks resized to the stage resolution)."""
        B, T, D, H, W = feats.shape
        heads = att.shape[1]
        a = att.flatten(1, 2)
        if a.shape[-2:] != (H, W):
            a = F.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
        a = a.view(B, heads, T, H, W).permute(0, 2, 1, 3, 4).unsqueeze(3)          # B,T,heads,1,H,W
        f = feats.view(B, T, heads, D // heads, H, W)
        return (a.to(f.dtype) * f).sum(1).reshape(B, D, H, W)
