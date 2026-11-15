"""Block 8b: physics-guided feature fusion.

Differentiable infinite-slope stability (docs/06):
    FS(m) = c' / (gamma z sin(b) cos(b)) + (1 - m gamma_w/gamma) tan(phi') / tan(b)
    m     = min(1, (q/T) * exp(TWI) / cos(b))          (steady-state wetness from TWI; q/T driven by rain)
Cohesion c' and friction angle phi' are *learned per lithology class* inside physical bounds; they start from
generic values, so the module is a physics prior the network can refine, never an unconstrained feature.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.formats import DEM_STATS, IDX_SLOPE, IDX_TWI

GAMMA, GAMMA_W, DEPTH = 18.0, 9.81, 1.5          # kN/m3, kN/m3, m (soil depth)
C_MAX, PHI_MIN, PHI_MAX = 25.0, 15.0, 45.0       # kPa, deg, deg


class PhysicsModule(nn.Module):
    n_maps = 6

    def __init__(self, n_lith: int, n_rain: int = 4):
        super().__init__()
        t = torch.linspace(0, 1, n_lith)
        c0 = 2.0 + 10.0 * torch.sin(t * math.pi)                              # generic spread of cohesion (kPa)
        phi0 = 26.0 + 10.0 * t
        self.c_raw = nn.Parameter(torch.logit((c0 / C_MAX).clamp(0.02, 0.98)))
        self.phi_raw = nn.Parameter(torch.logit(((phi0 - PHI_MIN) / (PHI_MAX - PHI_MIN)).clamp(0.02, 0.98)))
        self.q0 = nn.Parameter(torch.tensor(math.log(math.exp(0.02) - 1)))       # softplus^-1(0.02)
        self.k_rain = nn.Parameter(torch.zeros(n_rain))
        self.register_buffer("rain_scale", torch.tensor([60.0, 120.0, 200.0, 500.0])[:n_rain])

    def forward(self, dem: torch.Tensor, lith: torch.Tensor, rain: torch.Tensor):
        """dem [B,8,H,W] physical units; lith [B,H,W] long; rain [B,R] mm. -> maps [B,6,H,W], FS_wet [B,1,H,W]"""
        slope = dem[:, IDX_SLOPE:IDX_SLOPE + 1].float().clamp(1.0, 85.0)
        twi = dem[:, IDX_TWI:IDX_TWI + 1].float()
        b = torch.deg2rad(slope)
        lith = lith.clamp(0, self.c_raw.numel() - 1)
        c = (torch.sigmoid(self.c_raw) * C_MAX)[lith].unsqueeze(1)
        phi = torch.deg2rad(PHI_MIN + (PHI_MAX - PHI_MIN) * torch.sigmoid(self.phi_raw))[lith].unsqueeze(1)
        wet = F.softplus(self.q0) * (1.0 + F.softplus((rain / self.rain_scale) @ self.k_rain))   # [B]
        m = (wet.view(-1, 1, 1, 1) * torch.exp(twi.clamp(max=15.0)) / torch.cos(b)).clamp(0.0, 1.0)
        cohesive = c / (GAMMA * DEPTH * torch.sin(b) * torch.cos(b))
        fs_dry = cohesive + torch.tan(phi) / torch.tan(b)
        fs_wet = cohesive + (1.0 - m * GAMMA_W / GAMMA) * torch.tan(phi) / torch.tan(b)
        rain_map = (rain / self.rain_scale).mean(1).view(-1, 1, 1, 1).expand_as(m)
        maps = torch.cat([fs_dry.clamp(0, 5) / 5.0, fs_wet.clamp(0, 5) / 5.0, m, slope / 90.0,
                          (twi - DEM_STATS["twi"][0]) / DEM_STATS["twi"][1], rain_map], 1)
        return maps, fs_wet


class PhysicsFiLM(nn.Module):
    """f' = (1 + gamma) * f + beta, with (gamma, beta) from the physics maps pooled to each stage. Zero-init => identity."""

    def __init__(self, dims, n_maps=PhysicsModule.n_maps):
        super().__init__()
        self.proj = nn.ModuleList(nn.Conv2d(n_maps, 2 * d, 1) for d in dims)
        for p in self.proj:
            nn.init.zeros_(p.weight); nn.init.zeros_(p.bias)

    def forward(self, k: int, f: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        u = F.adaptive_avg_pool2d(maps, f.shape[-2:])
        g, b = self.proj[k](u.to(f.dtype)).chunk(2, 1)
        return (1 + g) * f + b
