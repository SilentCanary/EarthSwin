"""Block 5: cross-modal attention fusion. Per-location softmax over modalities with a validity mask
(docs/05 §5.1): clouds/no-data get weight 0, DEM is always valid so the denominator is never empty."""
from __future__ import annotations

import torch
import torch.nn as nn


class ModalityFusion(nn.Module):
    def __init__(self, d: int, n_mod: int, hidden: int | None = None):
        super().__init__()
        h = hidden or max(d // 4, 8)
        self.mu = nn.Parameter(torch.zeros(n_mod, d, 1, 1))              # learnable modality embeddings
        self.W = nn.Conv2d(d, h, 1); self.U = nn.Conv2d(d, h, 1, bias=False); self.v = nn.Conv2d(h, 1, 1)

    def forward(self, feats: torch.Tensor, present: torch.Tensor):
        """feats [B,M,D,h,w]; present [B,M,1,h,w] (1 = valid). Returns fused [B,D,h,w], alpha [B,M,1,h,w]."""
        B, M, D, h, w = feats.shape
        pm = present.to(feats.dtype)
        e = (feats + self.mu) * pm
        ebar = e.sum(1) / pm.sum(1).clamp_min(1.0)
        s = self.W(e.flatten(0, 1)).view(B, M, -1, h, w) + self.U(ebar).unsqueeze(1)
        s = self.v(torch.tanh(s).flatten(0, 1)).view(B, M, 1, h, w).float()
        s = s.masked_fill(present < 0.5, float("-inf"))
        # guard: a pixel with no valid modality gets uniform weights (its features are zero anyway)
        none = (present.sum(1, keepdim=True) < 0.5)
        s = torch.where(none, torch.zeros_like(s), s)
        alpha = s.softmax(1).to(feats.dtype)
        return (alpha * e).sum(1), alpha


def alpha_entropy(alpha: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """Mean entropy of modality weights over pixels with >=2 valid modalities (maximise it to avoid collapse)."""
    a = alpha.float().clamp_min(1e-6)
    ent = -(a * a.log()).sum(1)
    m = (present.sum(1) >= 2).float()
    return (ent * m).sum() / m.sum().clamp_min(1.0)
