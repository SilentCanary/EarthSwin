# 01 — Current State Audit (EarthSwin v0)

> Audit date: 15 Nov 2026 · Repo: `E:\major\EarthSwin` (origin `github.com/SilentCanary/EarthSwin`, last commit `20c976a`)
> Purpose: an honest baseline of what exists today, so every requirement in `02_REQUIREMENTS.md` traces back to a real gap.

---

## 1. Snapshot

| Item | Value |
|---|---|
| Code | ~12k lines of Python (models, preprocessing, tests, testers) |
| Raw imagery (`images/`) | 14 weekly stacks `HP_week{1..14}_stack.tif`, 9.1 GB, **4 bands: VV, VH, rainfall, slope**, int16, EPSG:4326, pixel 0.0001797° (GEE export scale ≈ 20 m) |
| Processed data (`dataset/EarthSentinel_2016/`) | 13 GB: 4,440 multiscale patch files (`.npz`), 45 chunk files, temporal manifests, split-specific graphs |
| Period | 14 weeks, 1 Jun – 31 Aug 2016 (Himachal Pradesh monsoon) |
| Labels | `preprocessing/labels.npy` shape (4440, 14): **6 positives total** from NASA Global Landslide Catalog (GLC) |
| Checkpoint | `models/earthswin_best.pt`, 397 MB (epoch 7, includes optimizer state) |
| Commits | `bb2bf73` initial v1 → `0e38bd9` "overfitted model" → `20c976a` basic code |

## 2. Pipeline as implemented

```
images/*.tif ──merge.py──► merged weekly rasters
   └─ preprocessing/create_patches.py  → 64/128/256 px crops per anchor (bands VV, rain, slope; VH dropped)
   └─ preprocessing/create_chunks.py   → chunked npz, validity filter
   └─ preprocessing/create_labels.py   → maps labels.npy to retained anchors
   └─ preprocessing/calculate_normalization_stats.py → normalization_stats.json
   └─ preprocessing/prepare_dataset.py → temporal manifests (3-week windows), 80/10/10 random anchor split,
                                         balanced train set (24 pos : 72 neg)
   └─ models/build_weekly_graph.py     → 8-NN graph per split per week
   └─ models/train_earthswin.py        → EarthSwin training (BCE, AdamW, 15 epochs)
   └─ test.py / models/result_tester*.py → diagnostics, threshold sweeps
```

## 3. Architecture diagram vs code

| # | Diagram block | Status | Evidence |
|---|---|---|---|
| 1 | Multi-source data collection | **Partial (3 of ~9 sources)** | S1 VV, CHIRPS rain, SRTM slope. VH exists in `images/` but `create_patches.py:189` keeps bands (0,2,3). No S2, soil moisture, seismic, LULC, lithology, roads, hydrology |
| 2 | Preprocessing & harmonization | **Partial, with a critical bug** | Mosaic, weekly alignment, z-score, NaN masks exist. S1 no-data stored as `0`, not NaN (see C2). `merge.py:11` merges only weeks 1–5, 13, 14 but `merge.py:203` prints "ALL 14 WEEKS" |
| 3 | Multi-scale pyramid | ✅ Built | `create_patches.py`, `models/patch_pyramid.py` — centred 64/128/256 crops (~1.3/2.6/5 km) |
| 4 | Hybrid CNN + Swin (shared weights) | ⚠️ **Built but inert** | Frozen at random init (C1). Three *separate* Swin-T (`patch_pyramid.py`), not shared |
| 5 | Cross-modal attention fusion | ❌ **Mis-implemented** | `models/attention.py` attends CNN↔Swin (not SAR↔optical↔rain…) with a single token → degenerate (C5) |
| 6 | Dynamic graph construction | Partial | `models/graph.py`: 8-NN Euclidean + VV/rain/slope similarity, per week. No elevation/river/road/geology edges |
| 7 | GAT | Built, **bug** | `models/gat.py`: single-head, no self-loops / residual (C4) |
| 8 | Temporal transformer | Built (short) | `models/temporal.py`: fixed 3-week window, target = last input week (C7) |
| 8b | Physics-guided feature fusion | ❌ Missing | slope only as a raw channel |
| 9 | Adaptive memory module | ❌ Missing | — |
| 11 | Uncertainty-aware decoder | ❌ Missing | plain MLP + sigmoid (`earthswin_model.py`) |
| 12 | Risk prediction & early warning | ❌ Missing | no maps, levels or alerts |
| 13 | Explainable AI | ❌ Missing | — |
| 14 | Multi-objective loss | 1 of 4 terms | `train_earthswin.py:722` BCE only |
| 15 | Deployment pipeline | ❌ Missing | — |

Diagram housekeeping: two blocks are numbered **8**, there is no **10**, the warning level "MODERATE" is misspelt, block 1 says "InSAR" but only GRD backscatter is used, and block 4 says "Shared Weights" but the code builds three separate Swin encoders.

## 4. Critical issues (ranked)

### C1 — The CNN + Swin encoder never learns
- `models/swin.py:30` → `pretrained=False`; the Microsoft Swin submodule folder `earthswin/Swin-Transformer/` is **empty**, so the timm path is always used.
- `models/train_earthswin.py:81` → `FREEZE_PATCH_PYRAMID = True` (also true in commit `0e38bd9`, which produced the checkpoint).
- **Effect:** blocks 3–5 are a frozen *random* feature extractor. Only GAT + temporal + classifier train. Parameter count is dominated by 3 × Swin-T (~28 M each), consistent with the 397 MB checkpoint.
- **Fix:** unfreeze, share one encoder across scales/time, and initialise from pretrained EO weights or self-supervised pretraining (see `10_ADVANCED_ARCHITECTURES.md` A2/A3).

### C2 — Sentinel-1 no-data is `0`, so masks never fire
Measured on 111 sampled patches (`patch_256`, VV channel):

| Week | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VV == 0 (fraction) | .28 | .16 | **.76** | **1.00** | .04 | **.76** | .14 | .28 | .16 | .05 | .28 | **.59** | .01 | .14 |
| VV NaN (fraction) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- `create_missing_mask` (`create_pytorch_dataset.py:110`) tests `isfinite`, so all mask channels are zero.
- Zeros are treated as real backscatter (≈ +1σ after normalization) and contaminate `normalization_stats.json` (VV mean −4157 in dB×1000 units — far too bright for Himalayan terrain).
- Week 3 is the target week of a test event (anchor 2719) and is 76 % missing.
- 2016 coverage is sparse because Sentinel-1B only launched in April 2016.
- **Fix:** convert `VV == 0` (and VH) to NaN *before* masks and statistics; recompute stats on valid pixels of the training split only.

### C3 — Only 6 labels exist
- Source: GLC (`preprocessing/inspect_csv.py`, `tif_meta.py`) — media-derived points with a location-accuracy field (exact / 1 km / 5 km / …).
- 6 positive anchor-weeks out of 62,160 (4,440 anchors × 14 weeks); split 3 train / 1 val / 2 test (the two test events are adjacent anchors 2719/2720).
- Train "positives" = 3 events × 8 dihedral augmentations = 24, against 72 fixed negatives (sampled once from ~42.6k).
- **No architecture can learn landslide appearance from 3 events.** This is the root cause; see `04_DATA_SPECIFICATION.md`.

### C4 — GAT discards each node's own features
- `models/graph.py:48` sets the distance diagonal to ∞ → no self-edges.
- `models/gat.py:144–160` aggregates messages only from neighbours into the source node.
- After layer 1 a node's own patch features are gone; for a detector whose evidence is *at* the node, this is fatal.
- **Fix:** add self-loops and a residual connection: $h_i' = \mathrm{LN}(h_i + \mathrm{GAT}(h)_i)$.

### C5 — Cross-modal attention is mathematically a no-op
`models/attention.py:25–28` unsqueezes CNN and Swin vectors to sequence length 1. Softmax over one key is always 1, so output = $W_o W_v\,f_{\text{swin}} + f_{\text{cnn}}$ → a linear layer plus residual. It also fuses *encoders*, not *modalities* as the diagram intends.
**Fix:** attention over spatial tokens or across data modalities with masking (see `05_ARCHITECTURE_PMGST_NET.md` §Block 5).

### C6 — Graph geometry differs between train and test
Each split builds its graph only from its own randomly scattered anchors (`build_weekly_graph.py:21`). Measured mean 8-NN spacing: **train ≈ 7.1 km, val ≈ 19.3 km, test ≈ 18.9 km**. The GAT is evaluated on a graph it never saw.
**Fix:** spatial-block splits with one graph per region (labels masked), or slope-unit graphs (see `07_EVALUATION_PROTOCOL.md`).

### C7 — It is detection, not early warning
In every manifest `target_week == week_3` (last input). The model asks "is there a landslide in this week's image?" — that is (weak) detection. A warning needs `target > last input`.

### C8 — Compute is wasted
`compute_pyramid_features` (`train_earthswin.py:187`) re-encodes all 3,550 train anchors × 3 weeks × 3 scales at `BATCH_SIZE = 1` (line 72) for each of 33 groups, every epoch — through a *frozen* encoder whose outputs never change (except augmentation). Cache once.

### C9 — Evaluation cannot discriminate
- Val has 1 positive in 5,328; test has 2 in 5,352.
- The 96-row diagnostic (`balanced_diagnostic_results_96`) selects its threshold on the training rows.
- Random anchor split ignores spatial autocorrelation.

### C10 — Two of three channels carry almost no signal
- Slope is identical across weeks (static layer repeated).
- CHIRPS is 0.05° (~5.5 km); within a 64 px patch rainfall relative std ≈ 0.3 %.
- So the temporal signal is essentially VV alone, which is itself up to 100 % missing in some weeks.

### Minor
- 22 hard-coded `C:/Users/advit/...` paths; `config.py` is empty (0 bytes).
- README references `train.py` (doesn't exist).
- `torch.load` without `weights_only=True`; checkpoint saves optimizer state (397 MB).
- No torch seeding / determinism flags.
- Learnable temporal position embedding initialised with `torch.randn` (unit scale) — large relative to features.
- `timm` `swin_tiny_patch4_window7_224` resized to 64 px input gives a 2×2 final stage — very little context.

## 5. Results (reproduced from saved JSON/CSV)

| Split | Positives / total | AP | Threshold | Outcome |
|---|---|---|---|---|
| Train (balanced 96) | 24 / 96 | 1.00 | 0.99 (chosen on train) | memorised |
| Validation | 1 / 5,328 | 0.00024 (chance ≈ 0.00019) | 0.05 | 0 TP, 323 FP |
| Test | 2 / 5,352 | 0.0022 | 0.05 | 1 TP, 300 FP (precision 0.3 %) |

## 6. What is worth keeping

- The **multi-scale crop logic** and **dihedral augmentation** code (fix aspect handling when terrain channels are added).
- The **vectorised GAT softmax** (`gat.py`) — correct and efficient once self-loops are added.
- The **per-week graph builder** pattern (becomes the dynamic-graph block).
- The **component test scripts** under `models/test/`.
- The **14 weekly VV/VH stacks** — unlabeled data that is still useful for self-supervised pretraining.
- The team's familiarity with the end-to-end flow.

## 7. Verdict

About **a third of the diagram is coded, but nothing is learning a generalisable signal yet**. The binding constraint is **labels and input signal**, not architecture. Fixing C1–C4 is necessary but not sufficient; the project needs polygon-level landslide inventories and Sentinel-2 optical data (see `02_REQUIREMENTS.md`, `04_DATA_SPECIFICATION.md`).
