"""Block 6: dynamic graph over chips. Nodes = chips of a spatial cluster (or a whole region at inference).

Edges: k nearest neighbours in space (+ self loops). Edge feature d_ij = 7 distances (docs/05 §6.2):
  geo, elevation, slope-normal dissimilarity, river, road, geology (Jaccard-like), rain (time-varying).
Learned edge weight  w_ij = exp(-sum_k lambda_k d_ij^k),  lambda_k = softplus(theta_k) > 0.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..data.formats import M_NX, M_NY, M_NZ, M_RAIN, M_RIVER, M_ROAD, M_X, M_Y, M_Z

N_EDGE_FEATS = 7
SIGMA = dict(geo=5000.0, elev=500.0, river=2000.0, road=2000.0)


def knn_edges(xy: torch.Tensor, k: int):
    """xy [N,2] -> (src, dst) directed edges j->i for the k nearest neighbours of every i (no self)."""
    N = xy.shape[0]
    if N <= 1:
        e = torch.zeros(2, 0, dtype=torch.long, device=xy.device)
        return e[0], e[1]
    k = min(k, N - 1)
    if N > 2048:                                                 # region-scale inference: KD-tree on CPU
        from scipy.spatial import cKDTree
        pts = xy.detach().double().cpu().numpy()
        _, nb = cKDTree(pts).query(pts, k=k + 1)
        nbr = torch.from_numpy(nb[:, 1:]).to(xy.device)
    else:
        c = xy.double()
        c = c - c.mean(0)                                        # centre: UTM coords would wreck float precision
        d = torch.cdist(c, c, compute_mode="donot_use_mm_for_euclid_dist")
        d.fill_diagonal_(float("inf"))
        nbr = d.topk(k, largest=False).indices                   # [N,k]
    dst = torch.arange(N, device=xy.device).unsqueeze(1).expand(-1, k).reshape(-1)
    return nbr.reshape(-1), dst


class DynamicGraph(nn.Module):
    def __init__(self, k: int = 8):
        super().__init__()
        self.k = k
        self.theta = nn.Parameter(torch.full((N_EDGE_FEATS,), math.log(math.e - 1)))   # lambda = 1 at init

    def lam(self):
        return nn.functional.softplus(self.theta)

    def forward(self, meta: torch.Tensor, lith_hist: torch.Tensor):
        """meta [N,10] (formats.META_FIELDS); lith_hist [N,K] normalised class histogram.
        Returns edge_index [2,E] (src,dst) incl. self loops, edge_feat [E,7], log_w [E]."""
        N = meta.shape[0]
        src, dst = knn_edges(meta[:, [M_X, M_Y]], self.k)
        loops = torch.arange(N, device=meta.device)
        src, dst = torch.cat([src, loops]), torch.cat([dst, loops])
        m_i, m_j = meta[dst], meta[src]
        d_geo = (m_i[:, [M_X, M_Y]] - m_j[:, [M_X, M_Y]]).norm(dim=1) / SIGMA["geo"]
        d_elev = (m_i[:, M_Z] - m_j[:, M_Z]).abs() / SIGMA["elev"]
        n_i, n_j = m_i[:, [M_NX, M_NY, M_NZ]], m_j[:, [M_NX, M_NY, M_NZ]]
        d_slope = 1.0 - (n_i * n_j).sum(1) / (n_i.norm(dim=1) * n_j.norm(dim=1)).clamp_min(1e-6)
        d_river = (m_i[:, M_RIVER] - m_j[:, M_RIVER]).abs() / SIGMA["river"]
        d_road = (m_i[:, M_ROAD] - m_j[:, M_ROAD]).abs() / SIGMA["road"]
        inter = torch.minimum(lith_hist[dst], lith_hist[src]).sum(1)
        union = torch.maximum(lith_hist[dst], lith_hist[src]).sum(1).clamp_min(1e-6)
        d_geol = 1.0 - inter / union
        d_rain = (m_i[:, M_RAIN] - m_j[:, M_RAIN]).abs()
        ef = torch.stack([d_geo, d_elev, d_slope, d_river, d_road, d_geol, d_rain], 1).float()
        log_w = -(ef * self.lam()).sum(1)
        return torch.stack([src, dst]), ef, log_w
