# PMGST-Net — Project Documentation

**Physics-guided Multi-scale Spatio-Temporal Graph Transformer Network for Landslide Detection and Early Warning in Himachal Pradesh**

B.Tech major project · Sobhan & Advitiya · JIIT Noida · docs v1.0 (15 Nov 2026)

---

## The plan in four lines
1. **Detect properly first (Phase 1).** Pixel-level landslide maps from Sentinel-1 + Sentinel-2 + terrain, trained on real polygon inventories, with calibrated uncertainty.
2. **Then warn (Phase 2).** Use the dated detections to train a model that says "be wary of this slope in the next 1–7 days".
3. **Then run and improve itself (Phase 3).** A scheduled pipeline with gated self-training and humans in the loop.
4. **Every block of the architecture diagram must earn its place in an ablation.**

## Read in this order

| # | Document | What it answers | Who |
|---|---|---|---|
| 01 | [Current state audit](01_CURRENT_STATE_AUDIT.md) | What EarthSwin v0 does, what's broken, why it can't learn yet | Both — start here |
| 02 | [Requirements](02_REQUIREMENTS.md) | What we must build; acceptance tests; phase exit criteria | Both |
| 03 | [Implementation plan](03_IMPLEMENTATION_PLAN.md) | Milestones Nov → May, fixes with code, build order, risks | Both |
| 04 | [Data specification](04_DATA_SPECIFICATION.md) | Sources, labels, preprocessing, tensors, splits, QA | Data track |
| 05 | [Architecture (PMGST-Net)](05_ARCHITECTURE_PMGST_NET.md) | Every diagram block 1–15 with maths and shapes | Model track |
| 06 | [Physics & mathematics](06_PHYSICS_AND_MATHEMATICS.md) | Slope stability, hydrology, rain thresholds, SAR/optical maths, calibration | Both |
| 07 | [Evaluation protocol](07_EVALUATION_PROTOCOL.md) | Splits, metrics, baselines, ablations, sanity checks | Both |
| 08 | [Research papers](08_RESEARCH_PAPERS.md) | What published work uses and what we take | Report writing |
| 09 | [NASA, ISRO & institutions](09_NASA_ISRO_AND_INSTITUTIONS.md) | LHASA, GLC, OPERA, NISAR, NRSC atlas, GSI NLFC, IIT Mandi, MIT, … | Report writing |
| 10 | [Advanced architectures](10_ADVANCED_ARCHITECTURES.md) | Improvements beyond the diagram (A1–A14) | Model track |
| 11 | [Early warning & production](11_EARLY_WARNING_AND_PRODUCTION_ROADMAP.md) | Phase 2 and Phase 3 design, self-training safeguards | Later |
| 12 | [References](12_REFERENCES.md) | All citations | Report writing |

## Key decisions (proposed — confirm as a team)

| # | Decision | Rationale | Doc |
|---|---|---|---|
| D1 | Phase 1 task = **pixel-level detection** at 10 m (+ object/slope-unit scores), not 5 km patch classification | v0 had 6 labels on 5 km patches; detection needs pixel truth | 01, 02 |
| D2 | Primary data season = **2023 monsoon, HP**; second season for temporal test | Exceptional, documented event with polygon inventories | 04 |
| D3 | Labels from **polygon inventories** (IIT Mandi–Durham, HPSCCC, GSI, NRSC) + self-digitised; GLC points only as weak labels | Fixes the root cause (audit C3) | 04 |
| D4 | Inputs = **S2 pre/post + S1 VV/VH time series + DEM derivatives** (+ rainfall for physics) | Optical is the main detection signal; S1 for clouds/timing | 04, 05 |
| D5 | Backbone = **change-aware Siamese encoder, pretrained** (TerraMind / Prithvi / SSL4EO), not frozen-random Swin | Audit C1; label scarcity | 05, 10 |
| D6 | Evaluation = **spatial block CV + region/temporal hold-outs**, CIs, baselines B0–B6 | Audit C6, C9 | 07 |
| D7 | Physics first as **features + plausibility loss**; full physics layer in Phase 2 | Detection needs plausibility; warning needs mechanism | 05, 06, 10 |
| D8 | Memory module and contrastive loss are **optional (P1/P2)** | Highest complexity, least evidence | 05, 10 |

## Current status (one paragraph)
EarthSwin v0 implements roughly a third of the diagram (pyramid, CNN+Swin, graph, GAT, temporal transformer) but the encoder is frozen at random weights, S1 no-data is stored as 0, the GAT drops each node's own features, the cross-modal attention is mathematically a no-op, and the dataset holds only 6 landslide labels — so validation/test performance is at chance. Details and measurements: `01_CURRENT_STATE_AUDIT.md`.

## Glossary

| Term | Meaning |
|---|---|
| Anchor (v0) | 256 px (~5 km) patch centre in the 2016 dataset |
| Chip | 256 × 256 px training tile at 10 m (2.56 km) |
| Slope unit | Terrain polygon bounded by ridges and channels; graph node in Blocks 6–7 |
| Detection | Mapping landslides that already happened |
| Nowcast / forecast | Hazard for now / for the next $h$ days |
| S1 GRD | Sentinel-1 ground-range-detected radar backscatter (VV, VH) |
| S2 L2A | Sentinel-2 surface reflectance |
| RTC / γ⁰ | Radiometric terrain-corrected radar backscatter |
| IMERG | NASA GPM multi-satellite precipitation (0.1°, 30 min) |
| GLC / COOLR | NASA's media-based global landslide catalogue / repository |
| PR-AUC | Area under precision–recall curve (main metric under imbalance) |
| ECE | Expected calibration error |
| FS | Factor of safety (slope stability; < 1 = failure) |
| TWI | Topographic wetness index |
| LoRA | Low-rank adaptation for fine-tuning large encoders |

## Conventions
- Requirement IDs `FR-*`, `NFR-*` (doc 02); priorities **P0/P1/P2**.
- Every experiment logs config + git hash + data version; results reported as mean [95 % CI] over spatial folds.
- Numbers marked *(verify)* come from memory of the literature — check before putting in the report.

## Open questions for the team
1. GPU model and VRAM? (sets backbone and batch size)
2. Who owns Track A (data & labels) vs Track B (models & training)?
3. Is a paper submission a goal? (affects rigour and timeline)
4. Earth Engine access set up? Inventory requests sent?
5. Evaluation date for the major project?

---
## Code (added)
Implementation of PMGST-Net-D (all 15 blocks of the diagram) lives in `pmgst/`. How to run it, what is tested and what is not: `13_BUILD_AND_RUN.md`.
