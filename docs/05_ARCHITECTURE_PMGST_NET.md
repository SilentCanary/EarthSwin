# 05 — PMGST-Net Architecture: Blocks, Logic and Maths

> **PMGST-Net**: Physics-guided Multi-scale Spatio-Temporal Graph Transformer Network for landslide detection and early warning.
> This document formalises every block of the team's architecture diagram (blocks 1–15), with tensor shapes, equations, design choices, the current code status, and which phase uses it.
> Physics derivations and remote-sensing maths live in `06_PHYSICS_AND_MATHEMATICS.md`; extra architectures beyond the diagram live in `10_ADVANCED_ARCHITECTURES.md`.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $m \in \mathcal{M}$ | modality: S1 (radar), S2 (optical), DEM (terrain), RAIN, SM (soil moisture), SEIS, STATIC (land cover, geology, roads, rivers) |
| $X^{(m)}_t \in \mathbb{R}^{C_m\times H\times W}$ | input of modality $m$ at time $t$ |
| $\mathcal{M}_p \subseteq \mathcal{M}$ | modalities *valid* at location $p$ (e.g. S2 missing under cloud) |
| $s \in \{1,2,3\}$ | pyramid scale (small, medium, large) |
| $d$ | feature width (e.g. 256) |
| $F \in \mathbb{R}^{d\times h\times w}$ | feature map; $F(p)\in\mathbb{R}^d$ at location $p$ |
| $i, j$ | graph nodes (chips or slope units); $\mathcal{N}(i)$ neighbours |
| $h_i \in \mathbb{R}^d$ | node embedding |
| $\sigma(\cdot)$, $\mathrm{LN}$, $\Vert$ | sigmoid, LayerNorm, concatenation |
| $\hat p(p)\in[0,1]$ | predicted landslide probability at pixel $p$; $y(p)\in\{0,1\}$ label |

## 1. Two operating modes

| | **PMGST-Net-D (Detection)** — Phase 1 | **PMGST-Net-W (Warning)** — Phase 2 |
|---|---|---|
| Question | Did a landslide occur here between $t_{pre}$ and $t_{post}$? | Will one occur in $(t, t+h]$? |
| Inputs | S2 pre/post, S1 time series, DEM (+ rainfall context) | Everything up to $t$ only (causal): rainfall, soil moisture, S1 trends, susceptibility, physics |
| Output | Pixel map $\hat p \in [0,1]^{H\times W}$ + uncertainty + polygons | Per-node hazard $P(\text{event}\in(t,t+h])$ + levels |
| Blocks emphasised | 1–8, 11–14 | 1, 2, 6–9, 8b (central), 11–15 |
| Label | Polygon inventory (pixel) | Dated events (node, time) — produced by PMGST-Net-D + inventories |

```
                ┌──────────── PMGST-Net-D (detection) ────────────┐
S2 pre/post ─┐  │ [2] preprocess → [3] pyramid → [4] CNN+Swin enc  │
S1 series  ──┼─►│ → [5] cross-modal fusion → [8] temporal attention │──► [11] decoder+uncertainty ─► [12] map/polygons
DEM/terrain ─┘  │ → [6,7] graph refinement → [8b] physics fusion    │         ▲            │
                │ → [9] memory (optional)                           │         │            ▼
                └───────────────────────────────────────────────────┘    [14] loss     [13] XAI
```

---

## Block 1 — Multi-source data collection

**Purpose:** gather complementary evidence: *what changed* (S1, S2), *where failure is physically plausible* (DEM, geology), *what triggered it* (rain, soil moisture, seismicity).

| Diagram item | Our concrete source (spec in `04`) | Mode |
|---|---|---|
| Sentinel-1 SAR (InSAR) | S1 GRD VV/VH amplitude time series (InSAR coherence/deformation later; NISAR L-band when available) | D, W |
| Sentinel-2 optical (10 m) | S2 L2A B2–B4, B8, B11, B12 | D |
| DEM & terrain (30 m) | Copernicus GLO-30 + derivatives | D, W |
| Rainfall (IMD / GPM) | GPM IMERG V07 (IMD gridded optional) | D (context), W |
| Soil moisture (SMAP / S1) | SMAP L4 | W |
| Seismic activity | USGS / NCS catalogues | W (later) |
| Static layers (LULC, geology, lithology, roads, hydrology) | WorldCover, GSI lithology, OSM, HydroRIVERS | D, W |

**Status:** v0 uses 3 sources (S1 VV, CHIRPS, slope). See audit C10.

## Block 2 — Preprocessing & harmonisation

Radiometric calibration → atmospheric correction (S2 L2A already) → speckle filtering (S1) → co-registration & resampling → cloud/shadow masking → normalisation → temporal alignment. Every operation and formula: `06_PHYSICS_AND_MATHEMATICS.md` §6–§9; pipeline: `04_DATA_SPECIFICATION.md` §5.

**Key rule:** missing data is *never* filled silently. Each modality carries a validity mask $V^{(m)}_t \in \{0,1\}^{H\times W}$; the model receives $(\tilde X^{(m)}_t, V^{(m)}_t)$ where $\tilde X$ is normalised with missing values set to 0.

**Status:** partial; S1 no-data bug (audit C2).

---

## Block 3 — Multi-scale pyramid patch generation

**Purpose:** landslides range from 10² m² rockfalls to km-long debris flows; run-out and drainage context lies outside a small chip.

For chip centre $c$ and scale $s$:

$$
P_s = \mathcal{R}_{256}\big(\mathrm{Crop}(X, c, 256\cdot 2^{s-1})\big), \qquad \mathrm{GSD}_s = 10\cdot 2^{s-1}\ \text{m}
$$

where $\mathcal{R}_{256}$ is area-average resampling to 256 × 256. Scales: 2.56 km @ 10 m, 5.12 km @ 20 m, 10.24 km @ 40 m. (The diagram's 64/128/256 px at 20 m in v0 are equivalent ideas at coarser resolution.)

**Scale alignment and fusion (pixel-level version).** Encode each scale with the shared encoder $E_\theta$ (Block 4) → $F_s$. Align coarser scales to the scale-1 footprint:

$$
\tilde F_s = \mathrm{Up}_{2^{s-1}}\big(\mathrm{CenterCrop}(F_s,\ 2^{-(s-1)})\big)
$$

Per-pixel scale attention:

$$
\beta_s(p) = \frac{\exp\big(w_\beta^\top \tilde F_s(p)\big)}{\sum_{s'}\exp\big(w_\beta^\top \tilde F_{s'}(p)\big)}, \qquad F(p) = \sum_s \beta_s(p)\,\tilde F_s(p)
$$

(v0's `ScaleFusion` is the global-vector special case.)

**Status:** ✅ crops built (v0); pixel-level fusion to implement.

---

## Block 4 — Hybrid CNN + Swin Transformer encoder (shared weights)

**Purpose:** CNNs capture local texture/edges (scarps, debris tone); Swin captures longer-range context (slope position, drainage) at near-linear cost.

### 4.1 CNN branch (local features)
Residual stages (ResNet-18/34 or ConvNeXt-T) producing features at strides 4, 8, 16, 32:

$$
F^{k}_{\text{cnn}} = \mathcal{C}_k\big(F^{k-1}_{\text{cnn}}\big),\qquad \mathcal{C}_k(x) = x + \mathrm{Conv}(\mathrm{ReLU}(\mathrm{BN}(\mathrm{Conv}(x))))
$$

### 4.2 Swin branch (global context)
- **Patch partition + linear embedding:** split into 4×4 patches, project to $C=96$: $z_0 = W_e\,\mathrm{vec}(x_{4\times4}) + b_e$.
- **Swin block pair** (window size $M$, use $M=8$ for 256-px chips so windows tile 64/32/16/8 grids exactly):

$$
\begin{aligned}
\hat z^{l} &= \text{W-MSA}\big(\mathrm{LN}(z^{l-1})\big) + z^{l-1}, &
z^{l} &= \mathrm{MLP}\big(\mathrm{LN}(\hat z^{l})\big) + \hat z^{l},\\
\hat z^{l+1} &= \text{SW-MSA}\big(\mathrm{LN}(z^{l})\big) + z^{l}, &
z^{l+1} &= \mathrm{MLP}\big(\mathrm{LN}(\hat z^{l+1})\big) + \hat z^{l+1}
\end{aligned}
$$

- **Window attention with relative position bias** $B\in\mathbb{R}^{M^2\times M^2}$ (indexed from a learnable table $\hat B\in\mathbb{R}^{(2M-1)\times(2M-1)}$):

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}} + B\Big)V
$$

- **Shifted windows:** SW-MSA cyclically shifts the map by $(\lfloor M/2\rfloor, \lfloor M/2\rfloor)$ so information crosses window borders.
- **Cost:** $\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C$ vs $\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC$ (linear in $hw$).
- **Patch merging:** concatenate 2×2 neighbours ($4C$) → linear → $2C$. Swin-T: depths [2, 2, 6, 2], heads [3, 6, 12, 24].

### 4.3 Fusion "Ⓒ" (concatenation) per stage

$$
F^{k} = \mathrm{Conv}_{1\times1}\big([\,F^{k}_{\text{cnn}} \,\Vert\, F^{k}_{\text{swin}}\,]\big) \in \mathbb{R}^{d_k\times \frac{H}{2^{k+1}}\times\frac{W}{2^{k+1}}}
$$

### 4.4 Weight sharing (what "Shared Weights" must mean)
The **same** encoder $E_\theta$ processes every scale $s$ and every time step $t$ (siamese). Modality-specific *stems* map $C_m$ channels to the common width; the S2 path uses the full hybrid encoder, S1 a lighter encoder (many dates), DEM a small CNN. Initialise from pretrained EO weights (Prithvi-EO-2.0 / SSL4EO-S12 / TerraMind) or self-supervised pretraining (`10_ADVANCED_ARCHITECTURES.md` A2–A3).

**Status:** ⚠️ v0 builds 3 *separate* Swin-T, frozen at random init (audit C1).

---

## Block 5 — Cross-modal attention fusion

**Purpose:** decide, *per location*, how much to trust each modality (e.g. S2 under cloud → weight 0; S1 in layover → low; DEM always available). The diagram's example weights (0.21, 0.18, 0.16, 0.15, 0.12, 0.10, 0.08) sum to 1.00 — exactly a per-location softmax.

### 5.1 Masked additive attention (recommended default)
Modality embeddings at location $p$: $e_m(p) = \phi_m(F^{(m)})(p) + \mu_m$ ($\mu_m$ learnable modality embedding).

$$
s_m(p) = v^\top \tanh\big(W e_m(p) + U\,\bar e(p) + b\big), \qquad \bar e(p) = \frac{1}{|\mathcal{M}_p|}\sum_{m\in\mathcal{M}_p} e_m(p)
$$

$$
\alpha_m(p) = \frac{\mathbb{1}[m\in\mathcal{M}_p]\,\exp(s_m(p))}{\sum_{m'}\mathbb{1}[m'\in\mathcal{M}_p]\,\exp(s_{m'}(p))}, \qquad
f(p) = \sum_{m}\alpha_m(p)\,e_m(p)
$$

"⊗" in the diagram = this weighting. DEM is always valid, so the denominator is never empty.

### 5.2 Token cross-attention (more expressive, at strides 16/32)
Stack tokens from all modalities in a local window; multi-head attention with a mask $\mathbf{M}$ ($-\infty$ on invalid modality tokens):

$$
\mathrm{head}_k = \mathrm{softmax}\!\Big(\frac{(QW^Q_k)(KW^K_k)^\top}{\sqrt{d_k}} + \mathbf{M}\Big)VW^V_k,\qquad
\mathrm{MHA} = [\mathrm{head}_1\Vert\cdots\Vert\mathrm{head}_H]\,W^O
$$

Typical: queries from S2 tokens, keys/values from S1 + DEM tokens; symmetric version alternates roles.

### 5.3 Training tricks
- **Modality dropout:** randomly remove S2 (p = 0.3) and S1 dates (p = 0.2) so the model learns to fall back.
- Optional entropy regulariser on $\alpha_m(p)$ to avoid collapse onto one modality.
- Log $\alpha_m$ maps — they double as explanations (Block 13).

**Status:** ❌ v0 attends CNN↔Swin with one token → softmax ≡ 1 (audit C5). Code for 5.1: `03_IMPLEMENTATION_PLAN.md` F4.

---

## Block 6 — Dynamic graph construction (spatial dependency)

**Purpose:** landslides cluster along valleys, rivers, roads and lithological belts; neighbouring slopes share triggers and run-out paths.

### 6.1 Nodes
- Detection: chips (2.56 km) or, better, **slope units** (terrain-derived polygons bounded by ridges and channels).
- Node features: $h^{(0)}_i = [\,\mathrm{meanpool}_{p\in i}F(p)\ \Vert\ \mathrm{maxpool}_{p\in i}F(p)\ \Vert\ \mathbf{p}_i\,]$ ($\mathbf{p}_i$ physics vector, Block 8b).

### 6.2 Edges and edge features (all diagram relations)

Candidate edges $\mathcal{E}$ = $k$-NN in space ($k=8$) ∪ hydrological edges (downslope/downstream neighbour) ∪ shared-road edges. For each edge, a feature vector $\mathbf{d}_{ij}(t)$:

| Relation (diagram) | Distance term |
|---|---|
| Euclidean distance | $d^{\text{geo}}_{ij} = \lVert c_i - c_j\rVert_2/\sigma_{\text{geo}}$ |
| Elevation difference | $d^{\text{elev}}_{ij} = \lvert z_i - z_j\rvert/\sigma_z$ |
| Slope similarity | $d^{\text{slope}}_{ij} = 1-\mathbf{n}_i^\top\mathbf{n}_j$, with surface normal $\mathbf{n} = (\sin\beta\sin a,\ \sin\beta\cos a,\ \cos\beta)$ |
| River connectivity | $d^{\text{river}}_{ij} = 0$ if connected along the flow network within $L$ km (same sub-catchment), else $\min(1, \ell^{\text{hydro}}_{ij}/L)$ |
| Road proximity | $d^{\text{road}}_{ij} = 0$ if both within 200 m of the same road segment, else $\lvert r_i-r_j\rvert/\sigma_r$ ($r$ = distance to road) |
| Geological similarity | $d^{\text{geol}}_{ij} = 1-\mathrm{Jaccard}(\text{lith}_i,\text{lith}_j)$ |
| Dynamic (time-varying) | $d^{\text{rain}}_{ij}(t) = \lvert R_i(t)-R_j(t)\rvert/\sigma_R$, $d^{\text{S1}}_{ij}(t)$ = change similarity |

**Learned edge weight** ("dynamic" = learned and time-dependent):

$$
w_{ij}(t) = \exp\Big(-\sum_k \lambda_k\, d^{k}_{ij}(t)\Big), \qquad \lambda_k = \mathrm{softplus}(\theta_k)\ \text{(learnable)}
$$

Self-loops $w_{ii}=1$ always. Optional directed **downslope edges** ($z_i > z_j$ and $j$ downstream of $i$) model run-out. Optional adaptive adjacency (Graph WaveNet style): $A^{\text{adp}} = \mathrm{softmax}(\mathrm{ReLU}(E_1E_2^\top))$.

**One graph per region** containing all nodes (test labels hidden) — avoids the train/test geometry shift in audit C6.

**Status:** partial (v0: Euclidean + VV/rain/slope only).

## Block 7 — Graph Attention Network (GAT)

Use **GATv2** (dynamic attention) with edge features and self-loops. For head $k$:

$$
e^{(k)}_{ij} = \mathbf{a}^{(k)\top}\,\mathrm{LeakyReLU}\big(W^{(k)}[h_i\Vert h_j] + W_e^{(k)}\mathbf{d}_{ij}\big)
$$

$$
\alpha^{(k)}_{ij} = \frac{\exp\big(e^{(k)}_{ij} + \log w_{ij}\big)}{\sum_{l\in\mathcal{N}(i)\cup\{i\}}\exp\big(e^{(k)}_{il} + \log w_{il}\big)}
$$

$$
h_i' = \mathrm{LN}\Big(h_i + \big\Vert_{k=1}^{K}\ \mathrm{ELU}\Big(\sum_{j\in\mathcal{N}(i)\cup\{i\}}\alpha^{(k)}_{ij}W^{(k)}_v h_j\Big)\Big)
$$

- 2–3 layers ($L$-hop receptive field), $K=4$ heads, dropout 0.1; residual + LN prevent over-smoothing and keep each node's own evidence (fixes audit C4).
- Original GAT scores $\mathbf{a}^\top[Wh_i\Vert Wh_j]$ before the nonlinearity ("static" attention); GATv2 applies the nonlinearity first, so rankings can depend on the query node.
- **Back to pixels (detection):** $F'(p) = F(p) + W_g\,h'_{\text{node}(p)}$ broadcast to the node's pixels.

**Status:** built (single head, no self-loops) — fix per `03` F3.

---

## Block 8 — Temporal transformer encoder (long-term spatio-temporal modelling)

**Purpose:** use *when* things changed: a landslide is an abrupt, persistent change; crops, snow and clouds are seasonal or transient.

Sequence per pixel group or node: $\{h^{t_1}, \dots, h^{t_T}\}$ with real acquisition dates (irregular).

**Continuous time encoding** (days since reference $\Delta t$):

$$
\mathrm{PE}(\Delta t)_{2k} = \sin\big(\Delta t/\tau^{2k/d}\big),\quad \mathrm{PE}(\Delta t)_{2k+1} = \cos\big(\Delta t/\tau^{2k/d}\big),\quad \tau = 1000
$$

plus learnable orbit/sensor embeddings. Prepend a [CLS] token: $Z^{(0)} = [\,z_{\text{cls}};\ h^{t_1}+\mathrm{PE}(\Delta t_1);\ \dots\,]$.

**Pre-norm transformer layer:**

$$
Z' = Z + \mathrm{MHA}\big(\mathrm{LN}(Z);\ \mathbf{M}_{\text{pad}} + \mathbf{M}_{\text{causal}}\big), \qquad Z'' = Z' + \mathrm{FFN}\big(\mathrm{LN}(Z')\big)
$$

- $\mathbf{M}_{\text{pad}}$: $-\infty$ for missing dates. $\mathbf{M}_{\text{causal}}$: only in warning mode (no peeking at the future).
- **Detection output:** $h^{\text{temp}} = Z''_{\text{cls}}$ and change token $\delta = \overline{h}^{\,\text{post}} - \overline{h}^{\,\text{pre}}$ (mean over dates in each window).
- **Pixel-dense variant (recommended for S1 series):** L-TAE / U-TAE style — attention computed at the lowest resolution gives masks $a_t(p)$ that are upsampled and applied at every scale: $F^k(p) = \sum_t a_t(p)\,F^k_t(p)$.
- Window length: v0 uses 3 weeks; target $T \le 16$ S1 dates (May–Oct 2023).

**Status:** built for $T=3$ with target = last input (audit C7).

---

## Block 8b — Physics-guided feature fusion

(Numbered "8" twice in the diagram; we call it 8b.)

**Physical knowledge inputs:** slope $\beta$, aspect $a$, curvature $\kappa$, lithology $\ell$, rainfall thresholds, hydrology index (TWI). Derived **physics vector** $\mathbf{p}$ (formulas in `06` §2–§5):

| Feature | Formula (summary) |
|---|---|
| Factor of safety, dry and wet | $FS(m) = \dfrac{c'}{\gamma z\sin\beta\cos\beta} + \Big(1 - m\dfrac{\gamma_w}{\gamma}\Big)\dfrac{\tan\phi'}{\tan\beta}$, with $(c',\phi',\gamma)$ from a lithology lookup |
| Wetness (steady state) | $m = \min\!\Big(1,\ \dfrac{q\,a}{T_s\,\sin\beta}\Big)$ |
| Topographic wetness index | $\mathrm{TWI} = \ln\big(a/\tan\beta\big)$ |
| Rainfall exceedance | $E = I_{\text{obs}}/I_{\text{thr}}(D)$ with $I_{\text{thr}} = \alpha D^{-\beta}$ |
| Curvature | profile/plan curvature (convergent hollows concentrate water) |

**Physics encoder:** $\mathbf{u} = \mathrm{MLP}_\psi(\mathbf{p})$ (or 1×1 convs on physics maps).

**Fusion "Ⓕ" — FiLM conditioning (default):**

$$
[\gamma_f,\ \beta_f] = W_f\,\mathbf{u}, \qquad f' = (1+\gamma_f)\odot f + \beta_f
$$

Alternative gated residual: $g = \sigma(W_g[f\Vert\mathbf{u}])$, $f' = f + g\odot W_u\mathbf{u}$.

Role by mode: **D** — plausibility prior (landslides need relief; suppress detections on flat river terraces); **W** — central driver (rain × wetness × FS).

**Status:** ❌ missing.

## Block 9 — Adaptive memory module (retrieval of historic events)

Memory bank of historical landslide (and hard-negative) signatures:
$\mathcal{B} = \{(k_n, v_n, y_n, \text{meta}_n)\}_{n=1}^{N}$, keys $\lVert k_n\rVert = 1$.

**Query and similarity search:** $q = \mathrm{normalize}(W_q f')$, $s_n = q^\top k_n$, $\mathcal{T} = \mathrm{TopK}_n(s_n)$.

**Read:**

$$
r = \sum_{n\in\mathcal{T}} \frac{\exp(s_n/\tau)}{\sum_{n'\in\mathcal{T}}\exp(s_{n'}/\tau)}\,v_n, \qquad
f'' = f' + \sigma\big(W_g[f'\Vert r]\big)\odot W_r\,r
$$

Retrieval prior (kNN classifier) for ensembling: $\hat y_{\text{mem}} = \sum_{n\in\mathcal{T}}\mathrm{softmax}(s_n/\tau)\,y_n$.

**Write/update:** after each epoch insert event-level embeddings of training positives (one per event, not augmented copies) up to capacity $N_{\max}$ (e.g. 4,096; FIFO or reservoir). Matched slots: $k_n \leftarrow \mathrm{normalize}(\mu k_n + (1-\mu)q)$, $\mu = 0.9$.

**Leakage control (critical):** during training, mask memory items from the same event or spatial block as the query. FAISS `IndexFlatIP` for search.

**Value:** few-shot recall of rare event types; "this looks like event X" explanations. **Priority P2** — only if ablation shows a gain.

**Status:** ❌ missing.

---

## Block 11 — Uncertainty-aware decoder

(The diagram has no block 10.)

**Decoder:** UPerNet/FPN over encoder stages → full-resolution features → head.

### 11.1 Evidential (Beta) head — matches the diagram's "Variance / Evidential"
Two non-negative evidences: $e^+ = \mathrm{softplus}(z^+)$, $e^- = \mathrm{softplus}(z^-)$; $\alpha = e^+ + 1$, $\beta = e^- + 1$, $S = \alpha+\beta$.

$$
\hat p = \frac{\alpha}{S}, \qquad u_{\text{vacuity}} = \frac{2}{S}, \qquad \mathrm{Var}[p] = \frac{\alpha\beta}{S^2(S+1)}
$$

### 11.2 Sampling alternatives (often better calibrated)
MC dropout or a deep ensemble of $T$ members:

$$
\bar p = \frac{1}{T}\sum_t \sigma(z_t),\quad \mathrm{Var} = \frac1T\sum_t(\sigma(z_t)-\bar p)^2,\quad
\mathrm{MI} = H(\bar p) - \frac1T\sum_t H(\sigma(z_t))
$$

$H(p) = -p\log p - (1-p)\log(1-p)$; MI = epistemic part.

### 11.3 Calibration
Temperature scaling on validation: $\hat p = \sigma(z/T^\ast)$, $T^\ast = \arg\min_T \mathrm{NLL}_{\text{val}}$. Report ECE before/after (`07`).

**Status:** ❌ plain sigmoid head in v0.

## Block 12 — Risk prediction & early warning

**Detection outputs (Phase 1):**
1. Probability COG $\hat p$ (diagram colour bar 0–1) and uncertainty COG.
2. Polygons: $\hat p \ge \tau_{\det}$ → morphological opening (3×3) → connected components → drop < 9 px (900 m²) → vectorise; attributes: area, mean/max $\hat p$, mean $u$, date range (from S1 timing).

**Warning levels (Phase 2):**

$$
\text{Level}(\hat p, u) = \begin{cases}
\text{VERY HIGH} & \hat p \ge \tau_3\\
\text{HIGH} & \tau_2 \le \hat p < \tau_3\\
\text{MODERATE} & \tau_1 \le \hat p < \tau_2\\
\text{LOW} & \hat p < \tau_1
\end{cases}
\quad + \text{ flag "REVIEW" if } u > u_{\max}
$$

Thresholds set on a validation season: $\tau_3$ for precision ≥ 0.8, $\tau_2$ ≥ 0.6, $\tau_1$ for recall ≥ 0.9. High uncertainty never silently downgrades a level.

**Alert channels** (SMS/e-mail, dashboard, sirens/IoT, web/app) are Phase 3 and go through human review (`11`).

**Status:** ❌ missing.

## Block 13 — Explainable AI (XAI)

| Method | Formula | Use |
|---|---|---|
| Grad-CAM (spatial) | $\alpha^c_k = \frac1Z\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}$, $\ L^c = \mathrm{ReLU}\big(\sum_k\alpha^c_kA^k\big)$; for segmentation $y^c = \sum_{p\in\text{region}} z(p)$ | Which pixels drove a detection |
| Built-in attention | $\alpha_m(p)$ (Block 5), $\beta_s(p)$ (Block 3), $\alpha_{ij}$ (Block 7), temporal $a_t(p)$ (Block 8) | Which modality / scale / neighbour / date |
| SHAP (feature importance) | $\phi_i = \sum_{S\subseteq F\setminus\{i\}}\frac{\lvert S\rvert!(\lvert F\rvert-\lvert S\rvert-1)!}{\lvert F\rvert!}\big[f(S\cup\{i\}) - f(S)\big]$ over channel groups (rainfall, slope, S1/InSAR, soil moisture, lithology, S2) | Global/regional driver ranking (diagram bar chart) |
| Integrated Gradients | $\mathrm{IG}_i(x) = (x_i-x'_i)\int_0^1\frac{\partial f(x'+a(x-x'))}{\partial x_i}\,da$ | Pixel attributions with a baseline |

Faithfulness check: deletion/insertion curves (remove top-attributed pixels → score should drop fast).

**Status:** ❌ missing.

## Block 14 — Multi-objective loss

$$
\mathcal{L}_{\text{total}} = \lambda_1\mathcal{L}_{\text{cls}} + \lambda_2\mathcal{L}_{\text{physics}} + \lambda_3\mathcal{L}_{\text{contrastive}} + \lambda_4\mathcal{L}_{\text{uncertainty}}
$$

### 14.1 Classification (BCE / focal + Dice) — mandatory
Over valid (non-ignore) pixels $\mathcal{P}$, $p_t = \hat p$ if $y=1$ else $1-\hat p$:

$$
\mathcal{L}_{\text{focal}} = -\frac{1}{|\mathcal{P}|}\sum_{p}\alpha_t(1-p_t)^{\gamma}\log p_t,\qquad
\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_p\hat p\,y + \epsilon}{\sum_p\hat p + \sum_p y + \epsilon}
$$

$\mathcal{L}_{\text{cls}} = \mathcal{L}_{\text{focal}} + \mathcal{L}_{\text{Dice}}$, $\gamma = 2$, $\alpha_t$ from class frequency.

### 14.2 Physics (constraint-based)
- **Implausibility:** $\mathcal{L}_{\text{imp}} = \frac1{|\mathcal{P}|}\sum_p \hat p(p)\,\mathbb{1}[\beta(p) < \beta_{\min}\ \lor\ p\in\text{water}]$, $\beta_{\min}$ = 1st percentile of slope inside training polygons.
- **Monotonicity (W mode):** $\mathcal{L}_{\text{mono}} = \mathbb{E}\big[\mathrm{ReLU}\big(\hat p(x) - \hat p(x\oplus\Delta R)\big)\big]$ — more antecedent rain must not lower hazard, all else equal.
- **Factor-of-safety consistency (PINN-style):** a head predicts bounded $(c', \phi')$ per node; compute $FS$ (Block 8b), $\hat p_{\text{phys}} = \sigma(\kappa(1-FS))$; $\mathcal{L}_{FS} = \mathrm{BCE}(\hat p_{\text{phys}}, y_{\text{node}})$, small weight.

$\mathcal{L}_{\text{physics}} = \mathcal{L}_{\text{imp}} + \mathcal{L}_{\text{mono}} + \mathcal{L}_{FS}$ (use the terms valid for the mode).

### 14.3 Contrastive (representation learning)
Supervised contrastive on node/region embeddings $z$ (unit norm), temperature $\tau=0.1$:

$$
\mathcal{L}_{\text{SupCon}} = \sum_i\frac{-1}{|P(i)|}\sum_{p\in P(i)}\log\frac{\exp(z_i\cdot z_p/\tau)}{\sum_{a\neq i}\exp(z_i\cdot z_a/\tau)}
$$

Positives $P(i)$: same class (landslide vs hard negative) from *different* events/blocks; or unchanged-area pre/post pairs (teaches invariance to season/illumination).

### 14.4 Uncertainty (evidential NLL / Bayes risk + KL)

$$
\mathcal{L}_{\text{EDL}} = y\big[\psi(S)-\psi(\alpha)\big] + (1-y)\big[\psi(S)-\psi(\beta)\big] + \lambda_{KL}(e)\ \mathrm{KL}\big[\mathrm{Beta}(\tilde\alpha,\tilde\beta)\,\Vert\,\mathrm{Beta}(1,1)\big]
$$

with $\tilde\alpha = y + (1-y)\alpha$, $\tilde\beta = (1-y) + y\beta$ (remove evidence for the correct class), $\lambda_{KL}(e) = \min(1, e/10)$ annealed by epoch $e$, and

$$
\mathrm{KL}\big[\mathrm{Beta}(a,b)\Vert\mathrm{Beta}(1,1)\big] = \ln\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} + (a-1)\big[\psi(a)-\psi(a+b)\big] + (b-1)\big[\psi(b)-\psi(a+b)\big]
$$

### 14.5 Weighting
Start $\lambda = (1, 0.1, 0.1, 0.5)$; or learn them with homoscedastic-uncertainty weighting $\mathcal{L} = \sum_k \frac{1}{2\sigma_k^2}\mathcal{L}_k + \log\sigma_k$. Keep a term only if its ablation helps.

**Status:** BCE only in v0.

## Block 15 — Deployment pipeline

Real-time ingestion → model inference → decision engine & alerting → end users (authorities & public). Detailed design, self-training loop and safeguards: `11_EARLY_WARNING_AND_PRODUCTION_ROADMAP.md`.

---

## 2. Tensor shapes (PMGST-Net-D, default config)

| Stage | Shape (B = batch) |
|---|---|
| S2 input | [B, 2, 6, 256, 256] + mask [B, 2, 1, 256, 256] |
| S1 input | [B, T≤16, 2, 256, 256] + mask + Δt [B, T] |
| DEM input | [B, 10, 256, 256] |
| Encoder stages (Swin-T, C=96) | [B,96,64,64] → [B,192,32,32] → [B,384,16,16] → [B,768,8,8] |
| Fused features per stage | same spatial sizes, width d_k |
| Temporal (U-TAE style on S1) | attention masks [B, heads, T, 16, 16] → applied at all stages |
| Graph nodes | [N_nodes, d] per region batch |
| Decoder output | logits/evidence [B, 2, 256, 256] |
| Pixel outputs | $\hat p$, $u$: [B, 256, 256] |

Memory: Swin-T (~28 M params) for S2 + light S1 encoder + small DEM CNN; with AMP and gradient checkpointing, batch 8 is expected to fit in ~10–12 GB — **measure on your GPU** and scale batch/backbone accordingly.

## 3. Diagram corrections to carry into the report
1. Renumber: 8 (temporal) / 8b (physics) or 8→9 and shift; there is no block 10.
2. "Shared Weights" must be true in code (one encoder for all scales/time).
3. Block 1 "InSAR": we use S1 amplitude; InSAR coherence/deformation is future work (NISAR).
4. Fix spelling "MODERATE" in block 12.
5. Show the two modes (D/W) explicitly, since early warning needs a forecast horizon.

## 4. Code map (target)

| Block | Module |
|---|---|
| 2 | `pmgst/data/transforms.py`, `pmgst/data/datasets.py` |
| 3 | `pmgst/models/pyramid.py` |
| 4 | `pmgst/models/encoders.py` |
| 5 | `pmgst/models/fusion.py` |
| 6–7 | `pmgst/models/graph.py`, `pmgst/models/gat.py` |
| 8 | `pmgst/models/temporal.py` |
| 8b | `pmgst/models/physics.py` |
| 9 | `pmgst/models/memory.py` |
| 11–12 | `pmgst/models/heads.py`, `pmgst/infer.py` |
| 13 | `pmgst/xai.py` |
| 14 | `pmgst/losses.py` |
| all | `pmgst/models/pmgst_net.py` (assembly, config-driven on/off per block for ablations) |
