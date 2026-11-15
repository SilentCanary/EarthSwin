# 03 — Implementation Plan

> From today's EarthSwin v0 to a detection model that works (Phase 1), then early warning (Phase 2) and production (Phase 3).
> Calendar assumes start **Mon 16 Nov 2026** and major-project evaluation around **April 2027**. Adjust dates, not order.

---

## 1. Milestones at a glance

| Milestone | Dates | Outcome | Exit check |
|---|---|---|---|
| **M0 — Repair & setup** | 16 Nov – 29 Nov | Repo restructured, critical bugs fixed, label requests sent, GEE pipeline started | Audit items C1–C10 closed or retired; CI green |
| **M1 — Data v1** | 30 Nov – 20 Dec | HP 2023 S1/S2/DEM/rain stack, chips, first labels, spatial splits; baselines B0–B2 | `04_DATA_SPECIFICATION.md` QA checklist passes |
| **M2 — Baselines & harness** | 21 Dec – 17 Jan | U-Net / U-TAE / foundation-model baselines; evaluation harness with CIs; pretraining on external sets | Baseline table filled (`07_EVALUATION_PROTOCOL.md` §6) |
| **M3 — PMGST-Net-D core** | 18 Jan – 14 Feb | Blocks 3, 4, 5, 8 + segmentation head trained end-to-end | Beats B3 on validation folds |
| **M4 — Full model & ablations** | 15 Feb – 14 Mar | Blocks 6–7, 8b, 11, 14 added; ablation table | Each kept block justified |
| **M5 — Results & demo** | 15 Mar – 4 Apr | XAI, robustness, HP-wide 2023 map, dashboard, report | Phase 1 exit criteria (`02_REQUIREMENTS.md` §7) |
| **M6 — Early-warning prototype** | 5 Apr – May | Phase 2 prototype + final report / paper draft | Phase 2 prototype metrics |
| **Post-May** | Jun 2027 → | Phase 3 production & self-training | `11_EARLY_WARNING_AND_PRODUCTION_ROADMAP.md` |

### Suggested work split (choose owners)
- **Track A — Data & labels:** GEE exports, preprocessing, label acquisition/digitisation, chips, splits, QA.
- **Track B — Models & training:** model code, training loops, pretraining, ablations, inference, dashboard.
- **Shared:** evaluation harness, error analysis, documentation, report.

---

## 2. Phase 0 — Repair (M0)

### 2.1 Restructure the repository

```
EarthSwin/
├── configs/                 # YAML: data/, model/, train/, eval/
├── pmgst/                   # importable package
│   ├── data/                # gee_export.py, chips.py, labels.py, datasets.py, transforms.py, splits.py
│   ├── models/              # encoders.py, pyramid.py, fusion.py, temporal.py, graph.py, gat.py,
│   │                        # physics.py, memory.py, heads.py, pmgst_net.py
│   ├── losses.py  metrics.py  calibration.py  xai.py
│   ├── train.py   evaluate.py  infer.py
├── baselines/               # threshold.py, s1_logratio.py, xgb.py, unet.py, utae.py, gfm.py
├── tests/                   # pytest: shapes, transforms, masks, splits, losses
├── docs/                    # these documents
└── legacy/2016_glc/         # current pipeline, archived unchanged for the report
```

Keep the old code runnable under `legacy/` so the report can show "before vs after".

### 2.2 Fixes (each is a small PR with a test)

**F1. S1 no-data → NaN (audit C2).** In patch creation, before masks or stats:

```python
import numpy as np

S1_BANDS = ("VV", "VH")

def sanitize_s1(arr: np.ndarray, band_names: list[str], nodata: float = 0) -> np.ndarray:
    """arr: [T, C, H, W] raw export. Marks S1 no-data as NaN so masks and stats see it."""
    arr = arr.astype(np.float32, copy=True)
    for c, name in enumerate(band_names):
        if name in S1_BANDS:
            arr[:, c][arr[:, c] == nodata] = np.nan
    return arr
```
Test: after sanitising, `(arr[:, vv] == 0).sum() == 0` and the mask channel is non-zero for week 4.
Units: the 2016 export stores VV as int16 that looks like dB×1000 — confirm in the GEE script and divide accordingly.

**F2. Keep VH.** Change `create_patches.py:189` to keep `(0, 1, 2, 3)` (VV, VH, rain, slope).

**F3. GAT self-loops + residual (C4).**

```python
import torch

def add_self_loops(edge_index, edge_weight, num_nodes):
    loops = torch.arange(num_nodes, device=edge_index.device)
    edge_index = torch.cat([edge_index, torch.stack([loops, loops])], dim=1)
    edge_weight = torch.cat([edge_weight, torch.ones(num_nodes, device=edge_weight.device)])
    return edge_index, edge_weight

class ResidualGAT(torch.nn.Module):
    def __init__(self, dim, layers=2, heads=4, dropout=0.1):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        self.convs = torch.nn.ModuleList(
            GATv2Conv(dim, dim // heads, heads=heads, edge_dim=1,
                      add_self_loops=True, dropout=dropout) for _ in range(layers))
        self.norms = torch.nn.ModuleList(torch.nn.LayerNorm(dim) for _ in range(layers))

    def forward(self, x, edge_index, edge_weight):
        e = edge_weight.unsqueeze(-1)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + torch.nn.functional.gelu(conv(x, edge_index, e)))
        return x
```
Test: a graph with no edges returns features ≈ input (not zeros).

**F4. Replace the degenerate attention (C5)** with modality fusion over spatial tokens (full maths in `05_ARCHITECTURE_PMGST_NET.md`, Block 5):

```python
import torch, torch.nn as nn

class ModalityFusion(nn.Module):
    """Per-location softmax over modalities; missing modalities are masked out."""
    def __init__(self, dim: int, n_modalities: int):
        super().__init__()
        self.mod_embed = nn.Parameter(torch.zeros(n_modalities, dim, 1, 1))
        self.score = nn.Sequential(nn.Conv2d(dim, dim // 4, 1), nn.GELU(), nn.Conv2d(dim // 4, 1, 1))

    def forward(self, feats, present):
        # feats: [B, M, D, H, W]; present: [B, M] bool (DEM is always present, so never all-masked)
        B, M, D, H, W = feats.shape
        s = self.score((feats + self.mod_embed).flatten(0, 1)).view(B, M, 1, H, W)
        s = s.masked_fill(~present.view(B, M, 1, 1, 1), float("-inf"))
        alpha = torch.softmax(s, dim=1)                      # α_m(p)
        return (alpha * feats).sum(dim=1), alpha             # [B, D, H, W], weights for XAI
```

**F5. Unfreeze and share the encoder (C1).** One encoder module reused for every scale and time step; initialise from pretrained weights (see `10_ADVANCED_ARCHITECTURES.md` A2) or from self-supervised pretraining (A3). Delete `FREEZE_PATCH_PYRAMID`.

**F6. One graph per region, spatial-block splits (C6).** See §3.4.

**F7. Hygiene.** Config files (no hard-coded paths), seeds, `torch.load(..., weights_only=True)`, model-only checkpoints, `pytest` + GitHub Actions on CPU, remove misleading prints (`merge.py:203`).

### 2.3 Start long-lead items now (they gate M1)
1. **Request inventories** (template in Appendix A): IIT Mandi–Durham 2023 Mandi/Kullu inventory (Sana et al. 2025); HPSCCC 2023 post-monsoon inventory; GSI Bhukosh download; NRSC atlas data.
2. **Earth Engine** project registration (non-commercial).
3. **Download** Landslide4Sense and Sen12Landslides for pretraining.

---

## 3. Phase 1 — Detection (M1–M5)

### 3.1 Data pipeline (Track A) — spec in `04_DATA_SPECIFICATION.md`
1. AOI = HP boundary (`preprocessing/Himachal_GeoJSON.geojson`), grid EPSG:32643 @ 10 m.
2. Exports (GEE, Python API):
   - S2 L2A pre (1 May – 30 Jun 2023) and post (15 Sep – 31 Oct 2023) cloud-masked median composites.
   - S1 GRD VV/VH every acquisition 1 May – 31 Oct 2023, asc + desc, plus pre/post medians.
   - GLO-30 DEM, derivatives; IMERG daily; WorldCover; HydroRIVERS.
3. Chip into 256 × 256 px (2.56 km) tiles with 64 px overlap; context crops at 512 and 1024 px resampled to 256.
4. Rasterise labels to {1 landslide, 0 background, 255 ignore}.
5. Chip index parquet + spatial-block folds; DVC-track everything.

### 3.2 Baselines first (Track B, M1–M2)
Run B0–B6 from `07_EVALUATION_PROTOCOL.md` on identical folds **before** building the full model. They tell us what "good" looks like for HP and catch data bugs early.

### 3.3 PMGST-Net-D build order (M3–M4)
Add one block at a time; keep it only if the ablation shows a gain (≥ +1 F1, or better calibration/robustness at equal F1).

| Step | Add | Expected value |
|---|---|---|
| 1 | Shared encoder (CNN + Swin) + UPerNet/FPN decoder on post-event S2 + DEM | Strong single-date baseline |
| 2 | Bi-temporal change branch (pre/post, `|f_post − f_pre|`) | Removes static bare-soil false positives |
| 3 | S1 branch + modality fusion (Block 5) | Cloud robustness |
| 4 | Temporal attention over S1 series (Block 8) | Timing + robustness |
| 5 | Multi-scale context pyramid (Block 3) | Large landslides, run-out context |
| 6 | Graph refinement over chips/slope units (Blocks 6–7) | Spatial coherence |
| 7 | Physics fusion (Block 8b) + physics loss | Fewer implausible detections (flat terrain, rivers) |
| 8 | Uncertainty head (Block 11) + calibration | Trustworthy probabilities |
| 9 | Memory module (Block 9) | Only if time allows |

### 3.4 Spatial splits (code sketch)

```python
import numpy as np, geopandas as gpd

def assign_block_folds(chips: gpd.GeoDataFrame, block_m=10_000, k=5, seed=42):
    """chips: one row per chip, projected CRS, column 'n_pos_px'. Returns fold per chip."""
    c = chips.geometry.centroid
    chips = chips.assign(block=(c.x // block_m).astype(int).astype(str) + "_" +
                               (c.y // block_m).astype(int).astype(str))
    load = chips.groupby("block")["n_pos_px"].sum()
    strata = np.digitize(load.rank(pct=True), [0.5, 0.8, 0.95])     # balance landslide load
    rng = np.random.default_rng(seed)
    fold_of = {}
    for s in np.unique(strata):
        blocks = load.index[strata == s].to_numpy()
        rng.shuffle(blocks)
        for i, b in enumerate(blocks):
            fold_of[b] = i % k
    return chips.assign(fold=chips["block"].map(fold_of))

def training_mask(chips, test_fold, buffer_m=2_560):
    """Exclude training chips within buffer of any test-fold chip (spatial autocorrelation)."""
    test_union = chips[chips.fold == test_fold].geometry.unary_union.buffer(buffer_m)
    return (chips.fold != test_fold) & ~chips.geometry.intersects(test_union)
```

### 3.5 Training recipe (starting point)
- Optimiser AdamW, lr 3e-4 head / 5e-5 backbone, weight decay 0.05, cosine schedule, 5-epoch warm-up.
- Batch 8–16 chips (AMP, gradient checkpointing for Swin), 60–100 epochs with early stopping on validation PR-AUC.
- Sampler: 50 % chips containing landslide pixels, 30 % random, 20 % hard negatives (river beds, bare scree, roads, construction, fresh debris fans).
- Loss: focal (γ = 2) + Dice, then add terms per `05_ARCHITECTURE_PMGST_NET.md` Block 14.
- Augmentation: D4 flips/rotations applied consistently (aspect channels transformed, see `04_DATA_SPECIFICATION.md` §10), modality dropout (p = 0.3 for S2), radiometric jitter.

### 3.6 Experiment tracking
- MLflow or Weights & Biases: config, git hash, DVC data hash, metrics with CIs, artefacts (maps, reliability diagrams).
- Naming: `p1/<stage>/<model>-<modalities>-fold<k>-<date>`.

### 3.7 Deliverables at M5
Trained models (all folds), HP 2023 probability + uncertainty COGs, landslide polygons, baseline/ablation/robustness tables, XAI figures, error analysis, dashboard demo, report chapters.

---

## 4. Phase 2 and 3
See `11_EARLY_WARNING_AND_PRODUCTION_ROADMAP.md`. Only start once Phase 1 exit criteria pass.

---

## 5. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| No HP polygon inventory obtained by Jan | Medium | High | Self-digitise ≥ 300–500 polygons (Mandi/Kullu 2023 is visually clear in S2); use Sen12Landslides/L4S for pretraining; evaluate on own polygons |
| Monsoon clouds block S2 | High | Medium | Post-monsoon composites (Sep–Oct); S1 branch; modality dropout |
| GPU memory too small for Swin + temporal | Medium | Medium | Swin-T only, AMP, checkpointing, smaller chips, U-TAE-sized temporal module |
| Overfitting to one district | Medium | High | Spatial CV, unseen-district test, strong augmentation, pretraining |
| Scope creep (memory, contrastive, evidential all at once) | High | High | Build order §3.3; every block must pass ablation |
| Time loss to data engineering | High | Medium | GEE for heavy lifting; freeze data v1 by 20 Dec |

## 6. Definition of done (every PR)
- Tests pass in CI; config-driven; no hard-coded paths.
- If it changes data: DVC version bumped and QA notebook re-run.
- If it changes the model: experiment logged with CIs on ≥ 3 folds.
- Docs updated where behaviour changed.

---

## Appendix A — Inventory request (template)

> Subject: Request for 2023 Himachal landslide inventory for a B.Tech research project
>
> Dear Dr. ___,
>
> We're final-year B.Tech students at JIIT Noida working on landslide detection in Himachal Pradesh with Sentinel-1 and Sentinel-2 data. Your paper "___" describes a landslide inventory from the 2023 monsoon that would let us train and validate our model properly.
>
> Would you be willing to share the polygons (or point locations with dates), for research use only? We'll cite your work in everything we publish, won't redistribute the data, and are happy to share our detection maps and code with you.
>
> Thank you for your time.
> Sobhan & Advitiya, JIIT Noida
