# 07 — Evaluation Protocol

> How we prove PMGST-Net works, without fooling ourselves. v0 reported AP 1.0 on training rows and near-chance on 1–2 held-out positives (audit C9); this protocol makes that impossible to repeat.

---

## 1. Principles
1. **No leakage:** no information from test areas (pixels, graph neighbours, memory items, normalisation stats, thresholds) touches training.
2. **Spatial autocorrelation is real:** neighbouring pixels/chips are not independent → split by **space**, buffer between folds, bootstrap by **block**.
3. **Thresholds and calibration are fitted on validation, never on test.**
4. **Every claim has a baseline and a confidence interval.**
5. **Report failures:** false positives and misses by category.

## 2. Splits
| Split | Construction | Answers |
|---|---|---|
| Spatial block CV (primary) | 10 km blocks, K = 5, stratified by landslide load, 2.56 km buffer (`03` §3.4) | In-region generalisation |
| Region hold-out | Train district A (e.g. Mandi) → test district B (e.g. Kullu), and swap | New valleys/lithology |
| Temporal hold-out | Train 2023 → test 2025 (if labels) | New season, new sensors (S1C) |
| External sanity | Landslide4Sense official test (S2+DEM model only) | Are we in a sane range vs literature? |

Within each training fold, hold out one block group as **validation** for early stopping, threshold selection and temperature scaling.

## 3. Pixel-level metrics
Computed over valid (non-ignore) pixels, aggregated over the whole test fold (not averaged per chip).

$$
P = \frac{TP}{TP+FP},\quad R = \frac{TP}{TP+FN},\quad F_1 = \frac{2PR}{P+R},\quad \mathrm{IoU} = \frac{TP}{TP+FP+FN}
$$

$$
\mathrm{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

- **PR-AUC (average precision)** — threshold-free, the primary model-selection metric (ROC-AUC is misleading at this imbalance).
- Report F1/IoU at the validation-selected threshold $\tau^\ast = \arg\max_\tau F_1^{\text{val}}(\tau)$.

## 4. Object-level metrics (what users care about)
1. Predicted objects = connected components of $\hat p \ge \tau^\ast$ after 3×3 opening, area ≥ 900 m².
2. Reference objects = inventory polygons ≥ 900 m².
3. Match if IoU ≥ 0.1 **or** predicted centroid inside reference polygon (landslides are small and boundaries uncertain).
4. Object precision = matched predictions / predictions; object recall = matched references / references; object F1.
5. Report by size class: < 0.01 km², 0.01–0.1 km², > 0.1 km².
6. **False alarms per 100 km²** of steep terrain (slope > 15°).

## 5. Calibration and uncertainty
- **Brier**, **ECE** (15 bins), reliability diagram, before and after temperature scaling (`06` §11).
- **Risk–coverage curve:** sort pixels by uncertainty, report error rate as the most uncertain are abstained; area under it (AURC).
- **Uncertainty usefulness:** AUROC of $u$ for predicting errors (misclassified vs correct).

## 6. Baselines (all on identical folds)

| ID | Baseline | Inputs | Why |
|---|---|---|---|
| B0 | Rule: ΔNDVI < −0.2 ∧ BSI_post > 0 ∧ slope > 15° | S2 + DEM | Simplest physical rule; is deep learning needed? |
| B1 | S1 log-ratio heatmap (Handwerger et al. 2022 style), thresholded on steep slopes | S1 + DEM | Cloud-proof classical method |
| B2 | Random forest / XGBoost on per-pixel features (bands, indices, LR, terrain) | all | Strong tabular baseline |
| B3 | U-Net (ResNet-34) on post-event S2 + DEM (Landslide4Sense style) | S2 + DEM | Standard DL baseline |
| B4 | Siamese U-Net on pre/post S2 + S1 + DEM | all | Change-aware DL baseline |
| B5 | U-TAE on S1 time series (+ S2) | all | Temporal SOTA-style baseline |
| B6 | Foundation model fine-tune (Prithvi-EO-2.0 or TerraMind) | S2 (+S1) | Pretrained-transformer baseline |
| **M** | **PMGST-Net-D** | all | Ours |

## 7. Ablations (each relative to full PMGST-Net-D)

| Ablation | What it tests |
|---|---|
| − Swin branch (CNN only) / − CNN branch (Swin only) | Block 4 hybrid value |
| Shared vs separate encoders across scales/time | Weight sharing |
| Concat instead of cross-modal attention | Block 5 |
| − S1 / − S2 / − DEM / − rainfall | Modality value |
| Pre/post only vs full S1 time series | Block 8 |
| Single scale vs 3-scale pyramid | Block 3 |
| − graph refinement; Euclidean-only edges vs full edge set | Blocks 6–7 |
| − physics fusion; − physics loss | Block 8b / 14.2 |
| − memory module | Block 9 |
| Sigmoid vs evidential vs MC-dropout vs ensemble | Block 11 |
| − contrastive loss; − EDL loss | Block 14 |
| Random init vs pretrained vs SSL-pretrained | Initialisation |
| Flips/rotations on vs off | Augmentation geometry |

Keep a block only if it improves PR-AUC/F1 by ≥ 1 point (CI not crossing 0) **or** improves calibration/robustness at equal accuracy.

## 8. Robustness tests
- **Cloud mode:** S2 removed entirely at test time (S1 + DEM only).
- **Partial cloud:** random 30–70 % S2 masking.
- **Sparse S1:** keep only 3 dates.
- **Unseen district / unseen year.**
- **Land-cover strata:** forest, shrub, cropland, bare, built-up, snow-adjacent.

## 9. Statistics
- **Block bootstrap:** resample test blocks with replacement (1,000×), recompute metrics → 95 % CI.
- **Paired comparison:** bootstrap the *difference* (model − baseline) on the same resamples; claim improvement only if the CI excludes 0.
- Report mean ± CI across the 5 folds, not the best fold.

## 10. Sanity checks (must pass before any result is reported)
- [ ] **Overfit one batch:** loss → ~0 on 8 chips (pipeline can learn).
- [ ] **Shuffled labels:** PR-AUC drops to positive-pixel prevalence (no leakage).
- [ ] **Blank input:** zeros in → constant out (no label leakage through masks/metadata).
- [ ] **Train/test geometry:** graph degree and edge-length distributions match across splits.
- [ ] **Memory leakage:** with memory on, test events are not in the bank.
- [ ] **Threshold source:** logged as "validation".

## 11. Error analysis (required)
- Confusion by WorldCover class, slope class, aspect, elevation band, distance to river/road, cloud fraction, S1 layover fraction.
- Gallery: 10 best TPs, 10 worst FPs, 10 worst FNs with S2 pre/post, S1 LR, Grad-CAM and modality weights.
- Top FP causes (expected: river bars, road construction, quarries, bare scree, snow edges).

## 12. Phase 2 (early warning) metrics
For forecast horizon $h$, per node-day:

| Metric | Formula |
|---|---|
| Probability of detection | $\mathrm{POD} = \frac{\text{hits}}{\text{hits}+\text{misses}}$ |
| False alarm ratio | $\mathrm{FAR} = \frac{\text{false alarms}}{\text{hits}+\text{false alarms}}$ |
| Critical success index | $\mathrm{CSI} = \frac{\text{hits}}{\text{hits}+\text{misses}+\text{false alarms}}$ |
| Brier skill score | $\mathrm{BSS} = 1 - \mathrm{BS}/\mathrm{BS}_{\text{climatology}}$ |
| Lead time | days between first ≥ HIGH alert and event |
| Alert burden | alerts per district per week |

Compare against: rainfall I–D threshold, LHASA v2 nowcast, XGBoost, at **matched alert rates**.

## 13. Reporting template

**Table A — Main results (spatial CV, mean [95 % CI])**

| Model | Pixel F1 | IoU | PR-AUC | Obj. F1 | ECE | FA/100 km² |
|---|---|---|---|---|---|---|
| B0 … B6 | | | | | | |
| PMGST-Net-D | | | | | | |

**Table B — Ablations**, **Table C — Robustness**, **Table D — Region/temporal hold-out**, **Figure set** — maps, reliability diagrams, risk–coverage, XAI gallery.
