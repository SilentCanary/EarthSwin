# 02 — Requirements Specification (PMGST-Net)

> Status: v1.0 draft · Owners: Sobhan, Advitiya · Last updated: 15 Nov 2026
> Read with: `01_CURRENT_STATE_AUDIT.md` (why), `03_IMPLEMENTATION_PLAN.md` (how/when), `05_ARCHITECTURE_PMGST_NET.md` (model), `07_EVALUATION_PROTOCOL.md` (how we prove it).

---

## 1. Problem statement

Himachal Pradesh (HP) loses lives, roads and homes to monsoon landslides every year. During the July–September 2023 monsoon alone, floods and landslides killed 428 people in HP, with over 40 % of deaths due to slope failures. Authorities need (a) **fast, reliable maps of where landslides have happened** and (b) **advance warning of where they are likely next**.

**PMGST-Net** (Physics-guided Multi-scale Spatio-Temporal Graph Transformer Network) is our system for both, built in three phases:

| Phase | Goal | Question answered | When |
|---|---|---|---|
| **P1 — Detection** (major-project core) | Detect and map landslides that have occurred | "Where did slopes fail in this period?" | Nov 2026 → Mar 2027 |
| **P2 — Early warning** | Estimate near-future landslide likelihood | "Where should we be wary in the next 1–7 days?" | Mar 2027 → May 2027 (prototype) |
| **P3 — Production / self-training** | Run continuously, improve itself with human oversight | "Is the system still right this season?" | After May 2027 |

**Principle:** P2 and P3 depend on P1. Detection outputs become the dated labels that train the warning model, and the confirmed detections feed self-training. **Nothing in P2/P3 starts until P1 meets its exit criteria (§7).**

## 2. Stakeholders and users

| User | Need | Phase |
|---|---|---|
| Project team / examiners | Reproducible, measurable results; clear ablations | P1 |
| Researchers | Open method, maps and metrics comparable to benchmarks | P1–P2 |
| District / state disaster management (HPSDMA) | Post-event landslide maps; daily hazard levels | P1 (maps), P2 (levels) |
| Road agencies (PWD, NHAI, BRO) | Road-corridor alerts | P2–P3 |
| Public | Plain-language alerts (only via authorities) | P3 |

PMGST-Net is a **research decision-support tool**, not an official warning service. Official warnings in India are issued by GSI's National Landslide Forecasting Centre and state authorities.

## 3. Conventions

- IDs: `FR-<area>-NN` functional, `NFR-NN` non-functional, `DR-NN` data, `MR-NN` model, `ER-NN` evaluation.
- Priority: **P0** must-have for the major project · **P1** should-have · **P2** nice-to-have / later phase.
- Each requirement has an **acceptance test**. A requirement is "done" only when its test passes and is recorded in the experiment log.

---

## 4. Functional requirements

### 4.1 Data acquisition and preprocessing (`FR-DATA`)

| ID | Requirement | Pri | Acceptance test |
|---|---|---|---|
| FR-DATA-01 | Acquire Sentinel-2 L2A surface reflectance (B2, B3, B4, B8, B11, B12 + SCL) for the AOI, pre- and post-event windows | P0 | Cloud-masked median composites exported for every labelled region; < 5 % no-data after compositing in post-monsoon window |
| FR-DATA-02 | Acquire Sentinel-1 GRD IW **VV and VH**, ascending and descending, as time series | P0 | Stack covers every 12-day repeat in the study window; per-acquisition metadata (orbit, date, incidence) stored |
| FR-DATA-03 | Convert S1 no-data (`0`) to NaN and produce explicit validity masks before any statistics | P0 | Unit test: no zero-valued VV/VH pixels survive; mask channels non-zero where data is missing |
| FR-DATA-04 | Derive terrain layers from Copernicus GLO-30 DEM: elevation, slope, aspect (sin/cos), plan & profile curvature, TPI, roughness, flow accumulation, TWI | P0 | Layers on the common 10 m grid; slope histogram matches DEM-tool reference within 1° |
| FR-DATA-05 | Acquire rainfall (GPM IMERG V07 daily + CHIRPS fallback) and compute antecedent accumulations (1, 3, 7, 14, 30 days) | P0 | Accumulations per chip/slope unit per date stored; spot-check vs IMD station summaries for 3 dates |
| FR-DATA-06 | Acquire static context: ESA WorldCover, OSM roads, HydroRIVERS/derived drainage, GSI lithology where available | P1 | Rasterised/vectorised on common grid; provenance recorded |
| FR-DATA-07 | Acquire soil moisture (SMAP L4) | P1 (P0 for P2) | Daily 9 km series for study period |
| FR-DATA-08 | Reproject all layers to one grid (EPSG:32643, 10 m) with documented resampling per layer | P0 | Co-registration check: S1/S2/DEM edge offset < 1 px on 5 manual tie points |
| FR-DATA-09 | Mask layover/shadow areas for S1 using local incidence angle | P1 | Mask layer exists; fraction reported per chip |
| FR-DATA-10 | Store data as chunked, versioned arrays (Zarr or COG) with a chip index (parquet) | P0 | `dvc status` clean; index lists chip id, bounds, split, label stats |

### 4.2 Labels (`FR-LBL`)

| ID | Requirement | Pri | Acceptance test |
|---|---|---|---|
| FR-LBL-01 | Obtain at least one **polygon** landslide inventory for HP (priority: IIT Mandi–Durham 2023 Mandi/Kullu inventory; HPSCCC 2023 inventory; GSI Bhukosh; NRSC atlas) | P0 | ≥ 1,000 polygons with event period ≤ one monsoon season, in our grid |
| FR-LBL-02 | Self-digitise ≥ 300 additional polygons in at least one other district using the protocol in `04_DATA_SPECIFICATION.md` §6 | P0 | Polygons + attributes; 10 % double-labelled with inter-annotator IoU ≥ 0.6 |
| FR-LBL-03 | Assign event dates or date ranges to polygons (S1 amplitude time-series dating where unknown) | P1 | ≥ 70 % of polygons have date range ≤ 24 days |
| FR-LBL-04 | Rasterise labels as {landslide, background, ignore}; ignore = clouds, no-data, layover/shadow, 1-px boundary band | P0 | Label rasters align with polygons (visual QA on 20 chips) |
| FR-LBL-05 | Keep GLC/COOLR points only as weak labels (never as pixel truth) | P0 | Code path separates weak labels; no GLC points in pixel loss |
| FR-LBL-06 | Prepare external pretraining sets: Landslide4Sense, Sen12Landslides (optional: CAS, GVLM) | P1 | Loaders return tensors in our channel schema |

### 4.3 Model (`FR-MOD`) — Phase 1 detection

| ID | Requirement | Pri | Acceptance test |
|---|---|---|---|
| FR-MOD-01 | Pixel-level segmentation head producing landslide probability at 10 m | P0 | Output map aligned to input grid |
| FR-MOD-02 | Chip/slope-unit level detection score derived from pixel map | P0 | Score = f(pixel map) documented; object list exported as GeoJSON |
| FR-MOD-03 | Multi-scale pyramid (block 3) as context crops 2.56 / 5.12 / 10.24 km resampled to 256 px | P0 | Ablation shows effect (keep only if ≥ +1 F1 or better calibration) |
| FR-MOD-04 | Hybrid CNN + Swin encoder **shared across scales and time**, initialised from pretrained EO weights or self-supervised pretraining | P0 | No frozen-random weights; pretrained init documented |
| FR-MOD-05 | Cross-modal fusion across S1 / S2 / DEM / rainfall with masked softmax for missing modalities (block 5) | P0 | Model runs with any modality missing; weights α_m logged |
| FR-MOD-06 | Temporal module over pre/post or S1 time series with real day offsets and padding masks (block 8) | P0 | Handles irregular dates; ablation vs pre/post-only |
| FR-MOD-07 | Graph module over chips or slope units with self-loops, residuals and physically meaningful edges (blocks 6–7) | P1 | Ablation vs no-graph |
| FR-MOD-08 | Physics-guided fusion of slope, curvature, TWI, factor-of-safety proxy, rainfall exceedance (block 8b) | P1 | Ablation; physics features logged |
| FR-MOD-09 | Uncertainty output per pixel (evidential or MC-dropout/ensemble) (block 11) | P1 | Reliability diagram; risk–coverage curve |
| FR-MOD-10 | Adaptive memory module (block 9) | P2 | Only if ablation shows gain without leakage |
| FR-MOD-11 | Multi-objective loss: classification (focal + Dice) mandatory; physics, contrastive, uncertainty terms optional, each justified by ablation (block 14) | P0 (cls) / P1 (others) | Loss terms configurable; ablation table |

### 4.4 Training (`FR-TRN`)

| ID | Requirement | Pri | Acceptance test |
|---|---|---|---|
| FR-TRN-01 | All hyperparameters and paths in config files (YAML); no hard-coded paths | P0 | `grep -r "C:/Users"` returns nothing |
| FR-TRN-02 | Deterministic seeds; log git hash, config, data version per run | P0 | Re-running a config reproduces F1 within ±0.5 points |
| FR-TRN-03 | Class-imbalance handling: positive-aware chip sampling, hard-negative mining, focal/Dice | P0 | Sampler stats logged per epoch |
| FR-TRN-04 | Mixed precision, gradient checkpointing option; fits one GPU (assumed ≤ 12 GB) | P0 | Peak memory logged |
| FR-TRN-05 | Pretrain → fine-tune workflow (external datasets → HP) | P1 | Both stages reproducible from config |
| FR-TRN-06 | Early stopping on validation PR-AUC (not loss) | P0 | Best-epoch selection uses PR-AUC |
| FR-TRN-07 | Model-only checkpoints loaded with `weights_only=True` | P0 | Checkpoint < 2× parameter size |

### 4.5 Evaluation (`FR-EVAL`) — details in `07_EVALUATION_PROTOCOL.md`

| ID | Requirement | Pri | Acceptance test |
|---|---|---|---|
| FR-EVAL-01 | Spatial block cross-validation with buffers; region hold-out; temporal hold-out where data allows | P0 | Split files versioned; no chip within buffer of a test block in training |
| FR-EVAL-02 | Pixel metrics (P, R, F1, IoU, PR-AUC), object metrics, calibration (Brier, ECE) with 95 % block-bootstrap CIs | P0 | Metrics script outputs all with CIs |
| FR-EVAL-03 | Baselines B0–B6 run on identical splits | P0 | Baseline table filled |
| FR-EVAL-04 | Ablation of every architecture block and modality | P0 | Ablation table filled |
| FR-EVAL-05 | Robustness: S2-missing (cloud), S1-only, unseen district, unseen year | P1 | Robustness table filled |
| FR-EVAL-06 | Sanity tests: shuffled-label test drops to chance; overfit-one-batch test passes | P0 | Both logged |

### 4.6 Inference and outputs (`FR-INF`)

| ID | Requirement | Pri | Acceptance test |
|---|---|---|---|
| FR-INF-01 | Sliding-window inference over HP with overlap blending | P0 | Seamless mosaic, no tile seams |
| FR-INF-02 | Outputs: probability COG, uncertainty COG, landslide polygons (GeoJSON) with area, confidence, date range | P0 | Opens in QGIS; polygons ≥ minimum mapping unit |
| FR-INF-03 | HP-wide inference for one season in < 2 h on one GPU | P1 | Timed run |
| FR-INF-04 | Simple web map/dashboard to browse outputs | P1 | Demo runs locally |

### 4.7 Explainability (`FR-XAI`) — block 13

| ID | Requirement | Pri | Acceptance test |
|---|---|---|---|
| FR-XAI-01 | Grad-CAM / attention maps for sample detections | P1 | Figures for ≥ 10 TP, 5 FP, 5 FN |
| FR-XAI-02 | Modality and channel-group attribution (SHAP or Integrated Gradients) | P1 | Global attribution bar chart |
| FR-XAI-03 | Error analysis by land cover, slope class, landslide size, cloud fraction | P0 | Error-analysis table |

### 4.8 Early warning (`FR-EW`) — Phase 2 (see `11_EARLY_WARNING_AND_PRODUCTION_ROADMAP.md`)

| ID | Requirement | Pri | Acceptance test |
|---|---|---|---|
| FR-EW-01 | Predict P(landslide in next h days) per slope unit or 1 km cell for h ∈ {1, 3, 7} | P2 | Model outputs daily maps |
| FR-EW-02 | Inputs strictly before forecast time (causal); rainfall forecasts optional | P2 | Leakage test passes |
| FR-EW-03 | Warning levels LOW / MODERATE / HIGH / VERY HIGH with calibrated thresholds | P2 | Thresholds fixed on validation season |
| FR-EW-04 | Beat rainfall-threshold baseline and LHASA v2 on POD/FAR/CSI | P2 | Comparison table |

### 4.9 Production (`FR-PROD`) — Phase 3

| ID | Requirement | Pri | Acceptance test |
|---|---|---|---|
| FR-PROD-01 | Scheduled ingestion → inference → publication pipeline | P2 | Runs unattended for 4 weeks |
| FR-PROD-02 | Self-training with gated pseudo-labels + human verification; champion/challenger promotion | P2 | No model promoted without passing frozen benchmark |
| FR-PROD-03 | Drift monitoring (input, output rate, calibration) with alerts to maintainers | P2 | Dashboard + alert on synthetic drift |

---

## 5. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-01 | **Reproducibility:** one command reproduces any reported number from versioned data + config + git hash |
| NFR-02 | **Code quality:** Python package layout, type hints on public functions, `pytest` unit tests for data transforms and model shapes, CI (GitHub Actions) running tests on CPU |
| NFR-03 | **Compute:** training fits one consumer GPU (assumed 8–12 GB VRAM; confirm); preprocessing runs in Google Earth Engine where possible |
| NFR-04 | **Data licensing:** Copernicus (free, attribution), NASA (open), GSI/NRSC/HPSCCC data used under their terms; inventories obtained from authors cited and not redistributed without permission |
| NFR-05 | **Transparency:** every map carries model version, data dates, and an uncertainty layer |
| NFR-06 | **Safety & ethics:** outputs labelled "research product, not an official warning"; false-alarm and miss rates published with every release |
| NFR-07 | **Documentation:** these docs kept current; each module has a docstring explaining its maths |

## 6. Data and model requirements summary

- **DR-01** Polygon inventory with dates (FR-LBL-01/03). **DR-02** S1 VV+VH time series. **DR-03** S2 L2A composites. **DR-04** DEM derivatives. **DR-05** rainfall + antecedent indices. **DR-06** split files and chip index. Full spec: `04_DATA_SPECIFICATION.md`.
- **MR-01** No frozen random modules. **MR-02** Every block justified by ablation. **MR-03** Graceful degradation under missing modalities. **MR-04** Calibrated probabilities. Full spec: `05_ARCHITECTURE_PMGST_NET.md`.

## 7. Phase exit criteria

### Phase 0 — Repair (gate to start P1 experiments)
- C1–C10 from the audit fixed or explicitly retired (old 2016 pipeline archived).
- Config system, seeds, tests, CI in place.

### Phase 1 — Detection (major-project success criteria)

Targets are provisional; re-baseline after B3/B4 baselines run (`07_EVALUATION_PROTOCOL.md`).

| Metric (HP spatial-block CV, mean ± 95 % CI) | Minimum | Stretch | Reference point |
|---|---|---|---|
| Pixel F1 | ≥ 0.60 | ≥ 0.70 | Landslide4Sense 2022 winner: 74.54 % test F1 (different data) |
| Pixel IoU | ≥ 0.45 | ≥ 0.55 | — |
| Object-level F1 (polygons ≥ 900 m²) | ≥ 0.65 | ≥ 0.75 | — |
| PR-AUC | ≥ 0.65 | ≥ 0.75 | — |
| ECE (calibration) | ≤ 0.05 | ≤ 0.03 | — |
| Cloud-robust mode (no S2), pixel F1 | ≥ 0.40 | ≥ 0.50 | S1 detection is harder; Handwerger et al. 2022 use heatmaps not pixels |
| Gain over best baseline (B3–B6) | ≥ +3 F1 **or** equal F1 with better calibration and cloud robustness | — | — |
| Unseen-district F1 drop | ≤ 10 points | ≤ 5 points | — |

Plus: complete ablation table, error analysis, HP-wide 2023 map, written report.

### Phase 2 — Early warning (prototype)
- Daily/3-day hazard for a held-out monsoon: ROC-AUC ≥ 0.80, better CSI than rainfall-threshold baseline at equal alert rate, calibrated levels.

### Phase 3 — Production
- 4 weeks unattended operation; documented promotion gate; drift alarms tested.

## 8. Out of scope (for now)

- Issuing public warnings directly.
- InSAR deformation time series (consider with NISAR L-band data later).
- Earthquake-triggered landslide modelling (Newmark) beyond documentation.
- Snow avalanches, GLOFs, debris-flow runout simulation.

## 9. Traceability: diagram block → requirements

| Block | Requirements |
|---|---|
| 1 Multi-source data | FR-DATA-01, 02, 04–07 |
| 2 Preprocessing | FR-DATA-03, 08, 09, 10 |
| 3 Pyramid | FR-MOD-03 |
| 4 CNN + Swin | FR-MOD-04 |
| 5 Cross-modal attention | FR-MOD-05 |
| 6 Dynamic graph | FR-MOD-07 |
| 7 GAT | FR-MOD-07 |
| 8 Temporal transformer | FR-MOD-06 |
| 8b Physics fusion | FR-MOD-08 |
| 9 Memory | FR-MOD-10 |
| 11 Uncertainty decoder | FR-MOD-09 |
| 12 Risk & warning | FR-INF-02, FR-EW-01..04 |
| 13 XAI | FR-XAI-01..03 |
| 14 Loss | FR-MOD-11 |
| 15 Deployment | FR-INF-03/04, FR-PROD-01..03 |

## 10. Assumptions and open questions

| # | Assumption / question | Owner | Impact if wrong |
|---|---|---|---|
| A1 | One GPU with 8–12 GB VRAM is available | Team | Smaller backbone / cloud GPU needed |
| A2 | At least one HP polygon inventory can be obtained by Jan 2027 | Team | Rely on self-digitised + external datasets; smaller HP test set |
| A3 | Google Earth Engine account available (non-commercial) | Team | Use Copernicus Data Space / ASF instead |
| A4 | Major-project evaluation ~Apr 2027 | Team | Re-plan milestones |
| Q1 | Is a paper submission a goal? (decides ablation/CI rigour and venue) | Team | — |
| Q2 | Who owns which track (data vs model)? | Team | — |
