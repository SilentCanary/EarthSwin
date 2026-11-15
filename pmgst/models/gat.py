"""Block 7: GATv2 with edge features, learned edge-weight bias, residual + LayerNorm. Pure PyTorch (edge-list,
scatter softmax) so it needs no torch_geometric. Self loops come from DynamicGraph, so a node always keeps its own
evidence (fixes audit C4)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_softmax(logit: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """logit [E,H], index [E] -> softmax over the edges sharing the same destination."""
    H = logit.shape[1]
    idx = index.unsqueeze(1).expand(-1, H)
    mx = torch.full((n, H), float("-inf"), device=logit.device, dtype=logit.dtype).scatter_reduce(0, idx, logit, "amax", include_self=True)
    ex = (logit - mx[index]).exp()
    den = torch.zeros(n, H, device=logit.device, dtype=logit.dtype).index_add_(0, index, ex)
    return ex / den[index].clamp_min(1e-16)


class GATv2Layer(nn.Module):
    def __init__(self, d: int, heads: int, edge_dim: int, drop: float = 0.1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.Wl = nn.Linear(d, d); self.Wr = nn.Linear(d, d); self.We = nn.Linear(edge_dim, d, bias=False)
        self.att = nn.Parameter(torch.empty(heads, self.dh)); nn.init.xavier_uniform_(self.att)
        self.norm = nn.LayerNorm(d); self.drop = nn.Dropout(drop)

    def forward(self, x, edge_index, edge_feat, log_w):
        src, dst = edge_index
        N = x.shape[0]
        xl, xr = self.Wl(x), self.Wr(x)
        e = F.leaky_relu(xl[src] + xr[dst] + self.We(edge_feat), 0.2).view(-1, self.h, self.dh)
        logit = (e * self.att).sum(-1) + log_w.unsqueeze(1)             # [E,H]
        alpha = self.drop(scatter_softmax(logit, dst, N))
        msg = alpha.unsqueeze(-1) * xl[src].view(-1, self.h, self.dh)
        out = torch.zeros(N, self.h, self.dh, device=x.device, dtype=x.dtype).index_add_(0, dst, msg).reshape(N, -1)
        return self.norm(x + F.elu(out)), alpha


class GATStack(nn.Module):
    def __init__(self, d, layers=2, heads=4, edge_dim=7, drop=0.1):
        super().__init__()
        self.layers = nn.ModuleList(GATv2Layer(d, heads, edge_dim, drop) for _ in range(layers))

    def forward(self, x, edge_index, edge_feat, log_w):
        alpha = None
        for l in self.layers:
            x, alpha = l(x, edge_index, edge_feat, log_w)
        return x, alpha
