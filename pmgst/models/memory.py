"""Block 9: adaptive memory of historic events (chip-level retrieval). P2 - enable only if the ablation shows a gain.

Keys are unit-norm chip embeddings of landslide chips (and hard negatives). Read = softmax-weighted top-k values,
gated into the node feature. Leakage control: entries from the same spatial block as the query are masked.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    def __init__(self, d_in: int, dim: int = 128, size: int = 2048, topk: int = 8, tau: float = 0.1):
        super().__init__()
        self.size, self.topk, self.tau = size, topk, tau
        self.q = nn.Linear(d_in, dim); self.v = nn.Linear(d_in, dim)
        self.gate = nn.Linear(d_in + dim, d_in); self.out = nn.Linear(dim, d_in)
        self.register_buffer("keys", torch.zeros(size, dim)); self.register_buffer("vals", torch.zeros(size, dim))
        self.register_buffer("ys", torch.zeros(size)); self.register_buffer("group", torch.full((size,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long)); self.register_buffer("n", torch.zeros((), dtype=torch.long))

    def forward(self, h: torch.Tensor, group: torch.Tensor):
        """h [N,d_in]; group [N] block ids. Returns (h'', y_mem [N] retrieval prior, q [N,dim])."""
        q = F.normalize(self.q(h), dim=-1)
        n = int(self.n)
        if n == 0:
            return h, torch.zeros(h.shape[0], device=h.device), q
        sims = q @ self.keys[:n].T                                          # [N,n]
        sims = sims.masked_fill(group.view(-1, 1) == self.group[:n].view(1, -1), float("-inf"))
        k = min(self.topk, n)
        top, idx = sims.topk(k, dim=1)
        ok = torch.isfinite(top)
        w = (top / self.tau).masked_fill(~ok, float("-inf")).softmax(1)
        w = torch.nan_to_num(w, nan=0.0)
        r = (w.unsqueeze(-1) * self.vals[:n][idx]).sum(1)
        y_mem = (w * self.ys[:n][idx]).sum(1)
        g = torch.sigmoid(self.gate(torch.cat([h, r], -1)))
        return h + g * self.out(r), y_mem, q

    @torch.no_grad()
    def write(self, h: torch.Tensor, y: torch.Tensor, group: torch.Tensor):
        """Insert chip embeddings (detached). y: 1 = landslide chip, 0 = hard negative."""
        q = F.normalize(self.q(h.detach()), dim=-1); v = self.v(h.detach())
        for i in range(h.shape[0]):
            p = int(self.ptr)
            self.keys[p], self.vals[p], self.ys[p], self.group[p] = q[i], v[i], y[i].float(), group[i]
            self.ptr.fill_((p + 1) % self.size); self.n.fill_(min(int(self.n) + 1, self.size))
