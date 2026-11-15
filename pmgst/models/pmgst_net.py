"""PMGST-Net-D assembly (detection). Every block of the diagram has a config switch for ablations.

Data flow (B chips = one spatial cluster):
  S2 pre/post --(3: pyramid, 4: shared hybrid encoder, siamese change)--+
  S1 series  --(4: light enc, 8: temporal transformer)------------------+--(5: modality fusion)--(8b: physics FiLM)
  DEM        --(3, 4: light enc)----------------------------------------+        |
                                                                                   v
                              (6,7: chip graph + GATv2) <-- node embedding of the deepest stage
                                        |  (9: memory, optional)
                                        v
                       (11: FPN decoder + evidential head) --> p, uncertainty
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..data.formats import DEM_CHANNELS, DEM_STATS, M_BLOCK, IDX_SLOPE
from .encoders import HybridEncoder, LightEncoder, gn
from .fusion import ModalityFusion
from .gat import GATStack
from .graph import N_EDGE_FEATS, DynamicGraph
from .heads import FPNDecoder, Head
from .memory import MemoryBank
from .physics import PhysicsFiLM, PhysicsModule
from .pyramid import ScaleFusion, align_to_scale1
from .temporal import TemporalEncoder

S2_MU = [0.06, 0.09, 0.09, 0.27, 0.22, 0.14]
S2_SD = [0.04, 0.05, 0.06, 0.09, 0.10, 0.08]
S1_MU = [-10.0, -16.0]
S1_SD = [4.0, 4.0]


def _present(valid: torch.Tensor, size) -> torch.Tensor:
    return (F.interpolate(valid, size=size, mode="area") >= 0.5).float()


class PMGSTNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        m, d = cfg.model, cfg.data
        dims = list(m.dims)
        self.dims = dims
        self.n_scales = d.n_scales if m.use_pyramid else 1
        self.use_s2, self.use_s1 = m.use_s2, m.use_s1
        self.register_buffer("dem_mu", torch.tensor([DEM_STATS[c][0] for c in DEM_CHANNELS]).view(1, -1, 1, 1))
        self.register_buffer("dem_sd", torch.tensor([DEM_STATS[c][1] for c in DEM_CHANNELS]).view(1, -1, 1, 1))
        self.register_buffer("s2_mu", torch.tensor(S2_MU).view(1, 1, -1, 1, 1))
        self.register_buffer("s2_sd", torch.tensor(S2_SD).view(1, 1, -1, 1, 1))
        self.register_buffer("s1_mu", torch.tensor(S1_MU).view(1, 1, -1, 1, 1))
        self.register_buffer("s1_sd", torch.tensor(S1_SD).view(1, 1, -1, 1, 1))

        # blocks 3-4: encoders (one shared S2 encoder for every scale and both dates)
        if self.use_s2:
            self.s2_enc = HybridEncoder(d.s2_bands + 1, dims, m.depths, m.heads, m.window, m.drop_path, m.ckpt)
            self.s2_change = nn.ModuleList(nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), gn(c), nn.GELU()) for c in dims)
            self.sf_s2 = nn.ModuleList(ScaleFusion(c) for c in dims)
        self.dem_enc = LightEncoder(d.dem_ch, dims)
        self.sf_dem = nn.ModuleList(ScaleFusion(c) for c in dims)
        if self.use_s1:
            self.s1_enc = LightEncoder(3, dims)
            self.temporal = TemporalEncoder(dims[-1], m.t_heads, m.t_layers)      # block 8
        # block 5
        n_mod = int(self.use_s2) + int(self.use_s1) + 1
        self.mfuse = nn.ModuleList(ModalityFusion(c, n_mod) for c in dims)
        # block 8b
        if m.use_physics:
            self.physics = PhysicsModule(d.n_lith, d.n_rain)
            self.film = PhysicsFiLM(dims)
        # blocks 6-7 (+9)
        d_in = 2 * dims[-1] + (PhysicsModule.n_maps if m.use_physics else 0)
        self.node_in = nn.Sequential(nn.Linear(d_in, m.d_graph), nn.LayerNorm(m.d_graph), nn.GELU())
        if m.use_graph:
            self.graph = DynamicGraph(m.knn)
            self.gat = GATStack(m.d_graph, m.gat_layers, m.gat_heads, N_EDGE_FEATS)
        if m.use_memory:
            self.memory = MemoryBank(m.d_graph, m.mem_dim, m.mem_size, m.mem_topk)
        self.node_out = nn.Linear(m.d_graph, dims[-1])
        nn.init.zeros_(self.node_out.weight); nn.init.zeros_(self.node_out.bias)
        # block 11
        self.decoder = FPNDecoder(dims, m.d_dec, raw_ch=d.s2_bands * 2 + 2, drop=m.mc_dropout)
        self.head = Head(64, m.head)

    # ------------------------------------------------------------------ helpers
    def _dem(self, dem):
        return (dem - self.dem_mu) / self.dem_sd

    def _s2_feats(self, s2, valid):
        """s2 [B,2,6,h,w] reflectance; valid [B,2,1,h,w] -> 4 stage maps of the change representation."""
        B = s2.shape[0]
        x = torch.cat([((s2 - self.s2_mu) / self.s2_sd) * valid, valid], 2)
        feats = self.s2_enc(x.flatten(0, 1))
        out = []
        for k, f in enumerate(feats):
            f = f.view(B, 2, *f.shape[1:])
            pre, post = f[:, 0], f[:, 1]
            out.append(self.s2_change[k](torch.cat([post, (post - pre).abs()], 1)))
        return out

    # ------------------------------------------------------------------ backbone: blocks 3,4,5,8,8b
    def backbone(self, b: dict, aux: bool = False):
        B = b["dem"].shape[0]
        dem = self._dem(b["dem"])
        S = self.n_scales
        s2_sc, dem_sc = [], []
        for s in range(1, S + 1):
            if s == 1:
                s2, v, dm = b["s2"], b["s2_valid"], dem
            else:
                s2, v, dm = b["ctx_s2"][:, s - 2], b["ctx_valid"][:, s - 2], self._dem(b["ctx_dem"][:, s - 2])
            if self.use_s2:
                s2_sc.append(self._s2_feats(s2, v))
            dem_sc.append(self.dem_enc(dm))
        sizes = [f.shape[-2:] for f in dem_sc[0]]

        s1_feats, s1_present = None, None
        if self.use_s1:
            T = b["s1"].shape[1]
            val = b["s1_valid"]
            x = torch.cat([((b["s1"] - self.s1_mu) / self.s1_sd) * val, val], 2)
            fs = self.s1_enc(x.flatten(0, 1))
            low = fs[-1]; h, w = low.shape[-2:]
            vlow = F.interpolate(val.flatten(0, 1), size=(h, w), mode="area").view(B, T, h, w)
            pad = b["s1_pad"][:, :, None, None] | (vlow < 0.5)
            att = self.temporal.masks(low.view(B, T, *low.shape[1:]), b["s1_dt"], pad)
            s1_feats = [TemporalEncoder.apply(att, f.view(B, T, *f.shape[1:])) for f in fs]
            s1_present = []
            for sz in sizes:
                vk = F.interpolate(val.flatten(0, 1), size=sz, mode="area").view(B, T, *sz) >= 0.5
                s1_present.append((vk & ~b["s1_pad"][:, :, None, None]).any(1, keepdim=True).float())

        maps = fs_wet = None
        if self.cfg.model.use_physics:
            maps, fs_wet = self.physics(b["dem"], b["lith"], b["rain"])

        stages, alphas, betas = [], [], []
        for k, sz in enumerate(sizes):
            mods, pres = [], []
            if self.use_s2:
                f, beta = self.sf_s2[k]([align_to_scale1(s2_sc[i][k], i + 1, sz) for i in range(S)])
                mods.append(f); pres.append(_present(b["s2_valid"].prod(1), sz))
                betas.append(beta)
            if self.use_s1:
                mods.append(s1_feats[k]); pres.append(s1_present[k])
            f, beta = self.sf_dem[k]([align_to_scale1(dem_sc[i][k], i + 1, sz) for i in range(S)])
            mods.append(f); pres.append(torch.ones_like(pres[0]) if pres else torch.ones(B, 1, *sz, device=f.device))
            fused, alpha = self.mfuse[k](torch.stack(mods, 1), torch.stack(pres, 1))
            if maps is not None:
                fused = self.film(k, fused, maps)
            stages.append(fused); alphas.append((alpha, torch.stack(pres, 1)))
        out = dict(stages=stages, maps=maps, fs_wet=fs_wet, alphas=alphas)
        if aux:
            out.update(betas=betas, att=att if self.use_s1 else None)
        return out

    # ------------------------------------------------------------------ nodes: blocks 6,7,9
    def node_embed(self, st: dict) -> torch.Tensor:
        f4 = st["stages"][-1]
        v = [f4.mean((2, 3)), f4.amax((2, 3))]
        if st["maps"] is not None:
            v.append(st["maps"].mean((2, 3)).to(f4.dtype))
        return self.node_in(torch.cat(v, 1))

    def lith_hist(self, lith: torch.Tensor) -> torch.Tensor:
        return F.one_hot(lith.clamp(0, self.cfg.data.n_lith - 1), self.cfg.data.n_lith).float().mean((1, 2))

    def refine_nodes(self, h0: torch.Tensor, meta: torch.Tensor, lith_hist: torch.Tensor, group: torch.Tensor | None = None):
        """Graph attention (+ optional memory) over node embeddings. Runs in fp32."""
        aux = {}
        dev = h0.device.type
        with torch.autocast(device_type=dev, enabled=False):
            h = h0.float()
            if self.cfg.model.use_graph:
                ei, ef, lw = self.graph(meta.float(), lith_hist.float())
                h, a = self.gat(h, ei, ef, lw)
                aux.update(edge_index=ei, gat_alpha=a, lam=self.graph.lam().detach())
            if self.cfg.model.use_memory:
                g = group if group is not None else torch.full((h.shape[0],), -1, device=h.device, dtype=torch.long)
                h, y_mem, q = self.memory(h, g)
                aux.update(y_mem=y_mem, mem_q=q)
        return h, aux

    # ------------------------------------------------------------------ full forward
    def forward(self, b: dict, node_override: torch.Tensor | None = None, aux: bool = False) -> dict:
        st = self.backbone(b, aux)
        h0 = self.node_embed(st)
        if node_override is None:
            h, naux = self.refine_nodes(h0, b["meta"], self.lith_hist(b["lith"]), b["meta"][:, M_BLOCK].long())
        else:
            h, naux = node_override, {}
        feats = list(st["stages"])
        feats[-1] = feats[-1] + self.node_out(h.to(feats[-1].dtype))[:, :, None, None]
        valid2 = b["s2_valid"].prod(1)
        post = ((b["s2"][:, 1] - self.s2_mu[:, 0]) / self.s2_sd[:, 0]) * b["s2_valid"][:, 1]
        delta = ((b["s2"][:, 1] - b["s2"][:, 0]) / self.s2_sd[:, 0]) * valid2
        raw = torch.cat([post, delta, self._dem(b["dem"])[:, IDX_SLOPE:IDX_SLOPE + 1], valid2], 1)
        if not self.use_s2:
            raw = raw * 0
        x, emb = self.decoder(feats, raw)
        out = self.head(x)
        out.update(emb=emb, node=h0, node_ref=h, fs_wet=st["fs_wet"], maps=st["maps"], alphas=st["alphas"], naux=naux)
        if aux:
            out.update(stage_feats=feats, betas=st.get("betas"), att=st.get("att"))
        return out

    @torch.no_grad()
    def embed_only(self, b: dict) -> torch.Tensor:
        """Pass 1 of region inference: node embedding h0 only (no decoder)."""
        return self.node_embed(self.backbone(b))

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
