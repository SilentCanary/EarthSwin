"""Block 4: hybrid CNN + Swin-Transformer encoder (one module, shared across scales and dates), plus the light
CNN used for S1 (many dates) and DEM. Pure PyTorch (no timm), channels-last inside Swin.

Stage strides 4, 8, 16, 32. Feature width per stage = dims[k]. Window size adapts when a map is smaller than the
window (context crops are half-size), so one set of weights works for every scale.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(8, ch), ch)


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), gn(cout), nn.GELU())


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1, bias=False); self.n1 = gn(c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1, bias=False); self.n2 = gn(c)

    def forward(self, x):
        y = F.gelu(self.n1(self.c1(x)))
        return F.gelu(x + self.n2(self.c2(y)))


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__(); self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


# ------------------------------------------------------------------ Swin
@lru_cache(maxsize=None)
def _rel_index(ws_eff: int, ws_max: int) -> torch.Tensor:
    c = torch.stack(torch.meshgrid(torch.arange(ws_eff), torch.arange(ws_eff), indexing="ij")).flatten(1)  # 2,N
    rel = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0) + (ws_max - 1)
    return rel[..., 0] * (2 * ws_max - 1) + rel[..., 1]


@lru_cache(maxsize=None)
def _shift_mask(H: int, W: int, ws: int, shift: int) -> torch.Tensor:
    img = torch.zeros(1, H, W, 1)
    cnt = 0
    for h in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
        for w in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
            img[:, h, w, :] = cnt; cnt += 1
    win = img.view(1, H // ws, ws, W // ws, ws, 1).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws)
    m = win[:, None, :] - win[:, :, None]
    return m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)            # [nW, N, N]


def window_partition(x, ws):
    B, H, W, C = x.shape
    return x.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)


def window_reverse(w, ws, B, H, W):
    C = w.shape[-1]
    return w.view(B, H // ws, W // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, ws_max):
        super().__init__()
        self.heads, self.ws_max = heads, ws_max
        self.qkv = nn.Linear(dim, 3 * dim); self.proj = nn.Linear(dim, dim)
        self.rpb = nn.Parameter(torch.zeros((2 * ws_max - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rpb, std=0.02)

    def forward(self, x, ws_eff, mask=None, B=1):
        Bw, N, C = x.shape
        q, k, v = self.qkv(x).view(Bw, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        idx = _rel_index(ws_eff, self.ws_max).to(x.device).reshape(-1)
        bias = self.rpb[idx].view(N, N, -1).permute(2, 0, 1)                  # h,N,N
        if mask is not None:
            nW = mask.shape[0]
            bias = (bias[None] + mask.to(x.device)[:, None]).repeat(Bw // nW, 1, 1, 1)  # [Bw,h,N,N]
        else:
            bias = bias[None].expand(Bw, -1, -1, -1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.proj(o.transpose(1, 2).reshape(Bw, N, C))


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, ws, shift, dp=0.0):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.n1 = nn.LayerNorm(dim); self.attn = WindowAttention(dim, heads, ws)
        self.n2 = nn.LayerNorm(dim); self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.dp = DropPath(dp)

    def forward(self, x):                                                    # [B,H,W,C]
        B, H, W, C = x.shape
        ws = max(d for d in range(1, min(self.ws, H, W) + 1) if H % d == 0 and W % d == 0)   # largest window that tiles the map
        shift = 0 if (min(H, W) <= ws or self.shift == 0) else ws // 2
        y = self.n1(x)
        if shift:
            y = torch.roll(y, (-shift, -shift), (1, 2))
        y = window_partition(y, ws)
        y = self.attn(y, ws, _shift_mask(H, W, ws, shift) if shift else None, B)
        y = window_reverse(y, ws, B, H, W)
        if shift:
            y = torch.roll(y, (shift, shift), (1, 2))
        x = x + self.dp(y)
        return x + self.dp(self.mlp(self.n2(x)))


class PatchMerging(nn.Module):
    def __init__(self, cin, cout):
        super().__init__(); self.norm = nn.LayerNorm(4 * cin); self.red = nn.Linear(4 * cin, cout, bias=False)

    def forward(self, x):                                                    # [B,H,W,C]
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.red(self.norm(x))


class SwinBranch(nn.Module):
    def __init__(self, cin, dims, depths, heads, ws, dp=0.0, use_ckpt=False):
        super().__init__()
        self.embed = nn.Conv2d(cin, dims[0], 4, 4); self.enorm = nn.LayerNorm(dims[0])
        n = sum(depths); k = 0
        self.stages = nn.ModuleList(); self.merges = nn.ModuleList()
        for i, (d, h) in enumerate(zip(depths, heads)):
            blocks = nn.ModuleList()
            for j in range(d):
                blocks.append(SwinBlock(dims[i], h, ws, 0 if j % 2 == 0 else ws // 2, dp * k / max(n - 1, 1))); k += 1
            self.stages.append(blocks)
            self.merges.append(PatchMerging(dims[i], dims[i + 1]) if i < len(dims) - 1 else nn.Identity())
        self.use_ckpt = use_ckpt

    def _run(self, blocks, x):
        for b in blocks:
            x = b(x)
        return x

    def forward(self, x):
        x = self.enorm(self.embed(x).permute(0, 2, 3, 1))
        outs = []
        for blocks, merge in zip(self.stages, self.merges):
            if self.use_ckpt and self.training and x.requires_grad:
                x = checkpoint(self._run, blocks, x, use_reentrant=False)
            else:
                x = self._run(blocks, x)
            outs.append(x.permute(0, 3, 1, 2))
            x = merge(x)
        return outs


# ------------------------------------------------------------------ CNN branch + hybrid
class CNNBranch(nn.Module):
    def __init__(self, cin, dims, blocks=1):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(cin, max(dims[0] // 2, 8), 2), ConvBlock(max(dims[0] // 2, 8), dims[0], 2))
        self.stages = nn.ModuleList()
        for i, d in enumerate(dims):
            layers = [] if i == 0 else [ConvBlock(dims[i - 1], d, 2)]
            layers += [ResBlock(d) for _ in range(blocks)]
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x):
        x = self.stem(x); outs = []
        for s in self.stages:
            x = s(x); outs.append(x)
        return outs


class HybridEncoder(nn.Module):
    """Block 4. CNN (local) || Swin (context) -> concat -> 1x1 conv at every stage."""

    def __init__(self, cin, dims, depths, heads, window, drop_path=0.0, use_ckpt=False):
        super().__init__()
        self.cnn = CNNBranch(cin, dims)
        self.swin = SwinBranch(cin, dims, depths, heads, window, drop_path, use_ckpt)
        self.fuse = nn.ModuleList(nn.Sequential(nn.Conv2d(2 * d, d, 1, bias=False), gn(d), nn.GELU()) for d in dims)

    def forward(self, x):
        c, s = self.cnn(x), self.swin(x)
        return [f(torch.cat([a, b], 1)) for f, a, b in zip(self.fuse, c, s)]


class LightEncoder(nn.Module):
    """Small CNN pyramid for S1 (applied to every date) and DEM."""

    def __init__(self, cin, dims):
        super().__init__(); self.cnn = CNNBranch(cin, dims, blocks=1)

    def forward(self, x):
        return self.cnn(x)
