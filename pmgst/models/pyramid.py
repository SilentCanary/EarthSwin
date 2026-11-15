"""Block 3: multi-scale pyramid. Context scales (s=2,3) are encoded by the *same* encoder, then aligned to the
scale-1 footprint (centre crop of 1/2^(s-1), resample) and fused with per-pixel scale attention beta_s(p)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def align_to_scale1(f: torch.Tensor, s: int, size: tuple[int, int]) -> torch.Tensor:
    """f: [B,D,h,w] features of scale s (footprint 2^(s-1) x larger). Crop the central 1/2^(s-1), resize to `size`."""
    if s > 1:
        div = 2 ** (s - 1)
        h, w = f.shape[-2:]
        ch, cw = max(1, h // div), max(1, w // div)
        t, l = (h - ch) // 2, (w - cw) // 2
        f = f[..., t:t + ch, l:l + cw]
    return F.interpolate(f, size=size, mode="bilinear", align_corners=False)


class ScaleFusion(nn.Module):
    def __init__(self, d):
        super().__init__(); self.w = nn.Conv2d(d, 1, 1)

    def forward(self, feats: list[torch.Tensor]):
        if len(feats) == 1:
            return feats[0], None
        st = torch.stack(feats, 1)                                   # B,S,D,h,w
        B, S, D, h, w = st.shape
        beta = self.w(st.flatten(0, 1)).view(B, S, 1, h, w).softmax(1)
        return (beta * st).sum(1), beta
