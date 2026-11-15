# 10 — Advanced Architectures (beyond the diagram)

> Proposals to make PMGST-Net better than the original diagram. Each has: idea → why → sketch/maths → cost → priority → how to test.
> Priority: **P0** do for the major project · **P1** if time · **P2** Phase 2/3.
> The original diagram's blocks stay valid; most proposals are *how* to realise them well.

---

## Summary

| # | Proposal | Improves | Priority | Effort |
|---|---|---|---|---|
| A1 | Change-aware Siamese multi-modal U-Net with temporal attention (concrete PMGST-Net-D backbone) | Detection accuracy, fewer static false positives | **P0** | M |
| A2 | Foundation-model encoder (TerraMind / Prithvi-EO-2.0 / SSL4EO) with LoRA | Label efficiency, generalisation | **P0** | M |
| A3 | Self-supervised pretraining on our own HP time series | Uses unlabeled data (incl. v0 stacks) | P1 | M |
| A4 | Heterogeneous slope-unit graph (slope units, rivers, roads) | Spatial coherence, run-out, interpretability | P1 | H |
| A5 | Weak & semi-supervised learning (MIL on GLC points, pseudo-labels, noisy-label losses) | More labels | **P0** (pseudo-labels) / P1 | M |
| A6 | Active learning loop | Labelling efficiency | P1 → P2 | L |
| A7 | Differentiable physics layer (infinite slope + transient infiltration) | Early warning credibility | P2 | H |
| A8 | Discrete-time survival model for warning | Proper "when" modelling | P2 | M |
| A9 | Deep ensembles + temperature scaling + conformal alert thresholds | Trustworthy uncertainty | **P0** | L |
| A10 | Retrieval-augmented detection (FAISS memory of event signatures) | Rare events, explanations | P1 | M |
| A11 | Rainfall-forecast inputs (1–3-day lead) | Warning lead time | P2 | M |
| A12 | Agentic bulletin layer (tool-calling LLM + human approval) | Usability for authorities | P2 | M |
| A13 | Ground-sensor fusion (IIT Mandi-type sensors) | Site-level accuracy | P2 | M |
| A14 | Lightweight variant for low compute | Deployment | P1 | L |

---

## A1 — Change-aware Siamese multi-modal U-Net with temporal attention (P0)

**Why:** a landslide is a *change*. Single-date models confuse bare scree, river bars and construction with landslides. A Siamese design compares pre and post with shared weights.

**Sketch:**
```
S2_pre ─► E_s2 ─┐                         ┌─► [f_pre ‖ f_post ‖ |f_post − f_pre|] per stage
S2_post ─► E_s2 ─┘ (shared weights)       │
S1_t1..tT ─► E_s1 (shared over t) ─► U-TAE temporal attention ─► f_s1 per stage
DEM ─► E_dem ─────────────────────────────► f_dem per stage
                         ▼
          Block 5 masked modality attention (per stage)
                         ▼
       Block 3 multi-scale context (optional)  →  UPerNet decoder
                         ▼
          Block 6–7 graph refinement (optional) → Block 11 head (p, u)
```

**Change features per stage $k$:**

$$
f^{k}_{\text{chg}} = \mathrm{Conv}_{1\times1}\big([\,f^k_{\text{pre}}\ \Vert\ f^k_{\text{post}}\ \Vert\ |f^k_{\text{post}} - f^k_{\text{pre}}|\,]\big)
$$

**Test:** vs B3 (single-date U-Net) and B4 (Siamese U-Net) — expect fewer false positives on static bare surfaces.

## A2 — Foundation-model encoder with LoRA (P0)

**Why:** with a few thousand polygons, pretrained representations beat random init; v0's frozen *random* Swin is the opposite of this (audit C1).

**Options:**
- **TerraMind** (S1 + S2 + DEM natively) — best modality match.
- **Prithvi-EO-2.0** (HLS optical, multi-temporal) — evaluated on Landslide4Sense.
- **SSL4EO-S12** weights (ResNet/ViT; MoCo, DINO, MAE, data2vec) — lighter, has S1 weights.

**LoRA** on attention projections: $W' = W + \frac{\alpha}{r}BA$, $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times d}$, $r \in \{8, 16\}$; train LoRA + decoder; optionally unfreeze last 2 blocks.

**Test:** random init vs SSL4EO vs TerraMind/Prithvi, all with identical decoder; plus 10 %/25 %/50 % label-fraction curves (label efficiency).

## A3 — Self-supervised pretraining on our own data (P1)

**Why:** we have far more unlabeled imagery (all of HP, many dates; even v0's 14 weekly VV/VH stacks) than labels. Same playbook as an SSL-then-segment pipeline.

**Options:**
- **Multi-modal MAE:** mask 75 % of patches across S1/S2/DEM tokens; reconstruct each modality: $\mathcal{L}_{\text{MAE}} = \sum_m \frac{1}{|\Omega_m|}\sum_{p\in\Omega_m}\lVert \hat x^{(m)}_p - x^{(m)}_p\rVert_2^2$ over masked patches $\Omega_m$.
- **Seasonal contrast:** same location, different dates = positives (InfoNCE) → invariance to season/illumination, sensitivity to abrupt change.
- **Cross-modal contrast:** S1 and S2 embeddings of the same place/time pulled together.

**Test:** same as A2 label-fraction curves.

## A4 — Heterogeneous slope-unit graph (P1)

**Why:** chips are arbitrary squares; slope units follow ridges and channels, so nodes are geomorphologically meaningful, and run-out follows drainage.

**Graph:** node types {slope unit, river reach, road segment}; edge types {adjacent (SU–SU), drains-to (SU→river, river→river), crosses (road–SU), downslope (SU→SU)}.

**Relational attention (R-GAT / HGT style):** for edge type $r$,

$$
h_i' = \mathrm{LN}\Big(h_i + \sum_{r}\sum_{j\in\mathcal{N}_r(i)}\alpha^{r}_{ij}\,W^{r}_v h_j\Big),\qquad \alpha^{r}_{ij} = \mathrm{softmax}_{j}\big(\mathbf{a}_r^\top\mathrm{LeakyReLU}(W^r[h_i\Vert h_j])\big)
$$

Side benefits: road-segment nodes collect "risk to road" (exposure) for Phase 2; river nodes model debris flows.

**Test:** vs chip graph and no graph; object-level F1 and spatial coherence (fewer isolated speckle detections).

## A5 — Weak and semi-supervised learning (P0 pseudo-labels; P1 rest)

1. **Pseudo-labels with teacher–student (Mean Teacher / FixMatch-style):** teacher = EMA of student, $\theta_T \leftarrow \mu\theta_T + (1-\mu)\theta_S$; on unlabeled chips, keep pixels with $\hat p_T > \tau_{hi}$ (positive) or $< \tau_{lo}$ (negative) **and** physically plausible (slope ≥ $\beta_{\min}$) **and** low uncertainty; loss on strong augmentation of the student. Cap pseudo-positive share per batch.
2. **Multiple-instance learning on GLC points:** bag $B$ = chips within the point's location-accuracy radius; $\hat P(B) = 1-\prod_{c\in B}(1-\hat p_c)$ (noisy-OR), loss $\mathrm{BCE}(\hat P(B), 1)$.
3. **Noisy-label-robust losses** for self-digitised/automatic labels: generalised cross-entropy $\mathcal{L}_{\text{GCE}} = \frac{1-p_y^{q}}{q}$, $q\in(0,1]$.
4. **S1 timing labels** (Burrows-style step detection) give dates to undated polygons.

**Test:** gains on held-out blocks with vs without unlabeled data; audit 100 pseudo-labels manually.

## A6 — Active learning loop (P1 → P2)

Select chips to label next by **expected value**: $\text{score}(c) = \bar u(c)\cdot \text{area}(\hat p_c>0.5) + \lambda\,\text{diversity}(c)$ (core-set distance in embedding space). Label in batches of 50; retrain; track label-efficiency curve. Becomes the human-in-the-loop part of production (`11`).

## A7 — Differentiable physics layer for early warning (P2)

**Why:** rain does not cause failure directly — pore pressure does. Make that explicit and learnable.

**Model:**
1. Network $g_\psi$ predicts bounded soil parameters per slope unit from static features: $(c', \phi', z, K_z, D_1) = \text{bounds}\odot\sigma(g_\psi(\mathbf{s}_i))$.
2. Physics layer computes pressure head $\psi(Z,t)$ from observed rainfall via the Iverson transient solution and $FS_i(t)$ (`06` §3).
3. Hazard: $\hat p_i(t) = \sigma\big(\kappa\,(1 - FS_i(t)) + r_\omega(\mathbf{x}_{i,\le t})\big)$, where $r_\omega$ is a small learned residual (hybrid).
4. Loss: dated events (from Phase 1 + S1 timing); regulariser keeping parameters near lithology priors.

Inspired by Dahal & Lombardo's Newmark PINN, adapted to rainfall. Interpretable: maps of learned $c', \phi'$ can be sanity-checked against GSI geology.

## A8 — Discrete-time survival (hazard) model for warnings (P2)

Per node $i$ and day $t$: $h_i(t) = P(\text{event at } t \mid \text{none before}) = \sigma\big(f_\theta(\mathbf{x}_{i,\le t})\big)$.
Likelihood for event at $t_e$ (or censored at $t_c$):

$$
\log L_i = \sum_{t < t_e}\log\big(1-h_i(t)\big) + \log h_i(t_e)\quad\text{(event)},\qquad \sum_{t\le t_c}\log\big(1-h_i(t)\big)\quad\text{(censored)}
$$

Risk of ≥ 1 event in next $h$ days: $1-\prod_{k=1}^{h}(1-h_i(t+k))$. Handles "no event yet" correctly, unlike plain classification. Note: re-failure is possible; reset or use recurrent-event variants.

## A9 — Trustworthy uncertainty (P0)

- **Deep ensemble** of 5 PMGST-Net-D models (different seeds/folds) — usually the strongest practical uncertainty baseline.
- **Temperature scaling** per ensemble.
- **Conformal risk control for alert thresholds:** choose $\tau$ on calibration blocks so that expected false-negative rate ≤ $\alpha$: $\hat\tau = \sup\{\tau : \frac{n}{n+1}\widehat{\mathrm{FNR}}_n(\tau) + \frac{1}{n+1} \le \alpha\}$ (bounded loss, exchangeable blocks).
- Compare against evidential head (Block 11) — keep whichever is better calibrated.

## A10 — Retrieval-augmented detection (P1)

Realises Block 9 pragmatically: FAISS index of event-level embeddings with metadata (date, district, type, image chips). At inference: top-K analogues → (a) kNN prior ensembled with the network, (b) shown to analysts ("similar to these past events"). Strict leakage masking by event/block during training.

## A11 — Rainfall-forecast inputs (P2)

Add 1–3-day precipitation forecasts (e.g. IMD/NCMRWF or ECMWF products) as inputs to PMGST-Net-W, as LHASA and LHASKarnali do with forecast rainfall. Evaluate lead-time gain vs forecast error.

## A12 — Agentic bulletin layer (P2)

A tool-calling LLM agent that reads model outputs (hazard maps, uncertainty, top drivers from XAI), rainfall forecasts and GSI susceptibility, then drafts a district bulletin with evidence links. Architecture: planner → tools (map query, statistics, XAI summary, forecast fetch) → **policy/safety validation layer** (no claims beyond model evidence, mandatory uncertainty statement) → **human approval** → dissemination. Advisory only.

## A13 — Ground-sensor fusion (P2)

Where in-situ sensors exist (tilt, soil moisture, rain gauge), fuse late: $\text{logit}\,\hat p_{\text{site}} = \text{logit}\,\hat p_{\text{sat}} + w^\top \mathbf{s}_{\text{sensor}}$, or a Bayesian update of satellite prior with sensor likelihood. Natural collaboration with IIT Mandi.

## A14 — Lightweight variant (P1)

For limited GPUs or deployment: U-TAE-sized temporal encoder (~1 M params) + ResNet-18 S2 encoder + concat fusion + sigmoid head. Use as fallback and as a speed reference.

---

## Recommended stack for the major project

**PMGST-Net-D = A1 backbone + A2 (pretrained encoder) + Block 5 masked fusion + U-TAE temporal (Block 8) + physics features (Block 8b) + A9 uncertainty + A5 pseudo-labels**, with Blocks 3, 6–7 (A4 if time) and 9/A10 added only if ablations justify them.

**Phase 2:** A7 + A8 + A11 on top of Phase 1 detections. **Phase 3:** A6 + A12 + A13 (`11_EARLY_WARNING_AND_PRODUCTION_ROADMAP.md`).
