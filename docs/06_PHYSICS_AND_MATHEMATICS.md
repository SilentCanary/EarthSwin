# 06 — Physics and Mathematics Reference

> The science behind PMGST-Net: slope-stability physics, hydrology, rainfall thresholds, seismic triggering, radar and optical remote sensing, terrain analysis, and the statistics we use for imbalance and calibration.
> Numbers marked **(verify)** are from the literature as we recall it — check the original paper before using them numerically in the report.

---

## 1. Why slopes fail (one paragraph)

A slope fails when the **shear stress** driving material downslope exceeds the **shear strength** resisting it. Rainfall rarely "pushes" a slope; it raises **pore-water pressure** $u$, which reduces the effective normal stress and therefore friction. Earthquakes add transient inertial forces. Vegetation adds root cohesion; its removal (fire, deforestation, road cuts) weakens slopes. Everything in Block 8b is a quantified version of this sentence.

**Mohr–Coulomb strength (effective stress):**

$$
\tau_f = c' + (\sigma_n - u)\tan\phi'
$$

$c'$ effective cohesion (kPa), $\phi'$ effective friction angle, $\sigma_n$ normal stress, $u$ pore pressure.

## 2. Infinite-slope model (shallow landslides)

For a soil layer of vertical thickness $z$ on a planar slope of angle $\beta$, with a water table at height $m z$ above the failure plane ($0 \le m \le 1$), unit weights $\gamma$ (soil) and $\gamma_w$ (water, 9.81 kN/m³):

- Driving stress: $\tau = \gamma z \sin\beta\cos\beta$
- Normal stress: $\sigma_n = \gamma z\cos^2\beta$
- Pore pressure (slope-parallel seepage): $u = \gamma_w m z\cos^2\beta$

**Factor of safety:**

$$
FS = \frac{c' + (\gamma - m\gamma_w)\,z\cos^2\beta\,\tan\phi'}{\gamma z\sin\beta\cos\beta}
= \frac{c'}{\gamma z\sin\beta\cos\beta} + \Big(1 - m\frac{\gamma_w}{\gamma}\Big)\frac{\tan\phi'}{\tan\beta}
$$

$FS < 1$ ⇒ failure. Root cohesion $c_r$ adds to $c'$.

**Illustrative parameter ranges** (calibrate per lithology; use as bounded priors, not truth):

| Material | $c'$ (kPa) | $\phi'$ (°) | $\gamma$ (kN/m³) | $z$ (m) |
|---|---|---|---|---|
| Colluvium / residual soil | 0–10 | 28–38 | 17–20 | 0.5–3 |
| Weathered schist/phyllite | 5–20 | 25–35 | 19–22 | 1–5 |
| Sandstone/shale debris | 0–15 | 30–40 | 18–21 | 1–3 |

Use in PMGST-Net: (a) physics features $FS_{\text{dry}} = FS(m{=}0)$ and $FS_{\text{wet}} = FS(m{=}1)$ (Block 8b); (b) PINN-style loss where the network predicts $(c', \phi')$ (Block 14.2).

## 3. Hydrology: wetness, TWI, critical rainfall

**Specific catchment area** $a$ = upslope contributing area per unit contour width (m²/m), from D8/D∞ flow accumulation × cell size.

**Topographic Wetness Index** (Beven & Kirkby 1979):

$$
\mathrm{TWI} = \ln\!\Big(\frac{a}{\tan\beta}\Big)
$$

**Steady-state wetness** (TOPOG / SHALSTAB, Montgomery & Dietrich 1994), with steady recharge $q$ (m/day) and soil transmissivity $T_s$ (m²/day):

$$
m = \min\!\Big(1,\ \frac{q\,a}{T_s\sin\beta}\Big)
$$

Substituting into $FS = 1$ with $c'=0$ gives the **critical steady-state rainfall** for failure (cohesionless case):

$$
q_c = \frac{T_s\sin\beta}{a}\cdot\frac{\gamma}{\gamma_w}\Big(1 - \frac{\tan\beta}{\tan\phi'}\Big)
$$

Unconditionally stable if $\tan\beta < (1-\gamma_w/\gamma)\tan\phi'$; unconditionally unstable if $\tan\beta \ge \tan\phi'$ (dry). Low $q_c$ = high susceptibility.

**Transient infiltration** (Iverson 2000; basis of TRIGRS) — pressure head at depth $Z$ during rainfall of intensity $I_Z$ lasting $T$, with diffusivity $D_1$ and hydraulic conductivity $K_z$:

$$
\psi(Z,t) = (Z-d)\,\beta_I + 2\frac{I_Z}{K_z}\sqrt{D_1 t}\ \mathrm{ierfc}\!\Big(\frac{Z}{2\sqrt{D_1 t}}\Big),\quad t\le T
$$

$$
FS(Z,t) = \frac{\tan\phi'}{\tan\alpha} + \frac{c' - \psi(Z,t)\,\gamma_w\tan\phi'}{\gamma Z\sin\alpha\cos\alpha}
$$

($d$ = initial water-table depth, $\alpha$ = slope angle, $\beta_I$ a steady-state term; **verify** constants in Iverson 2000 before implementing.) This is the natural "physics layer" for the warning model (`10_ADVANCED_ARCHITECTURES.md` A7).

## 4. Rainfall thresholds (empirical triggering)

**Intensity–duration (I–D) power law:**

$$
I = \alpha\,D^{-\beta}
$$

$I$ mean rainfall intensity (mm/h), $D$ duration (h). Examples **(verify)**:

| Threshold | $\alpha$ | $\beta$ | Scope |
|---|---|---|---|
| Caine (1980) | 14.82 | 0.39 | global, shallow landslides/debris flows |
| Guzzetti et al. (2008) | 2.20 | 0.44 | global minimum |
| Dahal & Hasegawa (2008) | 73.90 | 0.79 | Nepal Himalaya |

**Event rainfall–duration (E–D):** $E = \alpha D^{\beta}$ (cumulative mm).

**Antecedent rainfall index** with decay $k \in (0,1)$ (typically 0.8–0.95 per day):

$$
\mathrm{ARI}_t = \sum_{i=1}^{n} k^{\,i}\,R_{t-i}
$$

**Exceedance ratio** used as a physics feature: $E_t = I_{\text{obs}} / (\alpha D^{-\beta})$; $E_t > 1$ means above threshold.

Satellite rainfall caveats: IMERG (0.1°) and CHIRPS (0.05°, daily values downscaled from pentads) smooth convective cloudbursts; thresholds derived from gauges are not directly transferable — refit $\alpha,\beta$ on satellite rainfall with our dated inventory.

## 5. Seismic triggering (documented, out of Phase 1 scope)

**Newmark sliding block:** critical (yield) acceleration

$$
a_c = (FS - 1)\,g\sin\beta
$$

Permanent displacement $D_n$ accumulates while ground acceleration exceeds $a_c$. Regression form (Jibson 2007, **verify**):

$$
\log D_n = 0.215 + \log\Big[\Big(1-\frac{a_c}{a_{\max}}\Big)^{2.341}\Big(\frac{a_c}{a_{\max}}\Big)^{-1.438}\Big]
$$

Dahal & Lombardo's physics-informed network learns geotechnical parameters under this Newmark constraint from coseismic inventories — a template for our rainfall version.

## 6. Radar (Sentinel-1) physics and maths

**Backscatter:** $\sigma^0$ (linear, m²/m²); in decibels $\sigma^0_{\text{dB}} = 10\log_{10}\sigma^0$. Terrain-flattened $\gamma^0 \approx \sigma^0/\cos\theta_i$ (with area normalisation in RTC). **Always average in linear units.**

**Why landslides change backscatter:**
- Vegetation removal → less volume scattering → **VH drops** (cross-pol sensitive to vegetation).
- New rough debris → surface scattering changes → **VV can rise or fall**.
- Changed geometry (new scarp) → local incidence angle changes.

**Geometric distortions** (steep Himalayan terrain) with incidence $\theta$, slope $\beta$ facing the sensor or away:
- **Layover** if slope facing sensor and $\beta > \theta$.
- **Shadow** if slope facing away and $\beta > 90^\circ - \theta$.
- **Foreshortening** for slopes facing the sensor with $\beta < \theta$.
Mask layover/shadow; combine ascending + descending orbits to see both sides of valleys.

**Speckle** — multiplicative model $y = x\cdot n$, $\mathbb{E}[n]=1$, $\mathrm{Var}[n]=1/L$ ($L$ looks).
**Lee filter** in a local window:

$$
\hat x = \bar y + k\,(y - \bar y),\qquad k = \max\!\Big(0,\ 1 - \frac{C_u^2}{C_y^2}\Big),\quad C_u = \frac{1}{\sqrt L},\ C_y = \frac{\sigma_y}{\bar y}
$$

Temporal multi-looking (median of ≥ 3 dates) is simpler and preserves edges.

**Change detection (Handwerger et al. 2022 style):** per orbit,

$$
\mathrm{LR} = 10\log_{10}\frac{\mathrm{median}_{\text{post}}(\sigma^0)}{\mathrm{median}_{\text{pre}}(\sigma^0)}
$$

Large $|\mathrm{LR}|$ on steep slopes → candidate landslide; aggregate to density heatmaps.

**Interferometric coherence** (future, needs SLC):

$$
\gamma_{12} = \frac{\big|\mathbb{E}[s_1 s_2^*]\big|}{\sqrt{\mathbb{E}[|s_1|^2]\,\mathbb{E}[|s_2|^2]}}
$$

A landslide destroys coherence between pre- and co-event pairs ($\Delta\gamma$ drop). NISAR's L-band keeps coherence better under vegetation than C-band.

**Timing:** amplitude time series inside a mapped polygon shows a step change at failure; fitting a step function (Burrows et al. 2022) dates landslides within days–weeks → dated labels for Block 8 and Phase 2.

## 7. Optical (Sentinel-2) indices

| Index | Formula (S2 bands) | Landslide signal |
|---|---|---|
| NDVI | $(B8-B4)/(B8+B4)$ | Drops sharply on fresh scars |
| ΔNDVI | $\mathrm{NDVI}_{\text{post}}-\mathrm{NDVI}_{\text{pre}}$ | Strongly negative = vegetation loss |
| BSI | $\frac{(B11+B4)-(B8+B2)}{(B11+B4)+(B8+B2)}$ | Rises on bare soil/rock |
| NDWI | $(B3-B8)/(B3+B8)$ | Separates water/rivers (false positives) |
| Brightness | mean of B2–B4 | Fresh debris is bright |

**Cloud masking:** drop SCL classes 3 (cloud shadow), 8 (cloud medium probability), 9 (cloud high probability), 10 (thin cirrus); additionally s2cloudless probability > 40 %. Snow (SCL 11) handled as its own stratum.

Confusers: bare river bars, construction, quarries, harvested fields, fresh snow, burnt areas — this is why change + terrain + physics + hard-negative mining matter.

## 8. Terrain derivatives

With elevation grid $z$ and cell size $\Delta$, Horn (1981) gradients from a 3×3 window $\begin{smallmatrix}a&b&c\\d&e&f\\g&h&i\end{smallmatrix}$:

$$
\frac{\partial z}{\partial x} = \frac{(c+2f+i)-(a+2d+g)}{8\Delta},\qquad
\frac{\partial z}{\partial y} = \frac{(g+2h+i)-(a+2b+c)}{8\Delta}
$$

$$
\beta = \arctan\sqrt{\Big(\frac{\partial z}{\partial x}\Big)^2+\Big(\frac{\partial z}{\partial y}\Big)^2},\qquad
\text{aspect} = \operatorname{atan2}\Big(\frac{\partial z}{\partial y}, -\frac{\partial z}{\partial x}\Big)\ \text{(convention-dependent; verify against GIS)}
$$

Encode aspect as $(\sin a, \cos a)$ (no 0°/360° discontinuity).

- **Profile curvature** (along steepest descent): controls flow acceleration; **plan curvature** (across slope): convergent hollows concentrate water → shallow failures.
- **TPI** $= z - \overline{z}_{\text{window}}$ (ridge > 0, valley < 0). **Roughness** = std of $z$ in window.

## 9. Normalisation and missing data

- Robust z-score: $\tilde x = (x - \mathrm{median})/\mathrm{IQR}$, statistics from training folds' **valid** pixels only.
- Validity mask $V\in\{0,1\}$ per modality; after normalising, set $\tilde x = 0$ where $V=0$ and feed $V$ as a channel.
- Never compute statistics including no-data sentinel values (v0 bug: S1 zeros, audit C2).

## 10. Class imbalance

Landslide pixels are typically a few percent of a chip, and far less of a region.

- **Focal loss** (Lin et al. 2017): $-\alpha_t(1-p_t)^\gamma\log p_t$ down-weights easy negatives.
- **Dice / Tversky:** $\mathrm{TI} = \frac{TP}{TP+\alpha_{T}FN+\beta_{T}FP}$, loss $1-\mathrm{TI}$; $\alpha_T>\beta_T$ favours recall.
- **Class-balanced weights** (Cui et al. 2019): $w_c = \frac{1-\beta_{cb}}{1-\beta_{cb}^{n_c}}$, $\beta_{cb}\in[0.99, 0.9999]$.
- **Sampling:** positive-aware chip sampler + hard-negative mining (`04` §9).

## 11. Calibration and conformal prediction

- **Temperature scaling:** $\hat p = \sigma(z/T)$, $T$ fitted by minimising validation NLL.
- **Expected Calibration Error:** $\mathrm{ECE} = \sum_{b=1}^{B}\frac{|S_b|}{n}\big|\mathrm{acc}(S_b) - \mathrm{conf}(S_b)\big|$ (binary: fraction positive vs mean $\hat p$ in bin $b$).
- **Brier score:** $\frac1n\sum(\hat p_i - y_i)^2$.
- **Split conformal (for alert thresholds):** non-conformity $s_i = 1-\hat p_{y_i}$ on a calibration set of size $n$; $\hat q = $ the $\lceil (n+1)(1-\alpha)\rceil/n$ quantile; prediction set $\{y: 1-\hat p_y \le \hat q\}$ has coverage ≥ $1-\alpha$ under exchangeability. Spatial autocorrelation weakens exchangeability → calibrate by spatial block.

## 12. Graph maths used in Blocks 6–7

- Gaussian-kernel edge weight $w_{ij} = \exp(-d_{ij}^2/2\sigma^2)$ or the learned form $\exp(-\sum_k\lambda_k d^k_{ij})$.
- Symmetric normalised adjacency (if using GCN layers): $\hat A = \tilde D^{-1/2}(A+I)\tilde D^{-1/2}$.
- Hydrological connectivity from D8 flow directions: build a directed tree; $i\to j$ if $j$ is downstream of $i$ within $L$ km of channel distance.
- Slope units: partition terrain along ridges and channels (e.g. `r.slopeunits`), giving geomorphologically meaningful nodes.

## 13. Constants and units

| Symbol | Value / unit |
|---|---|
| $g$ | 9.81 m/s² |
| $\gamma_w$ | 9.81 kN/m³ |
| S2 reflectance scale | DN / 10,000 |
| S1 GEE GRD | $\sigma^0$ in dB |
| v0 2016 export | int16, VV apparently dB × 1000 (verify) |
| IMERG | mm/h (half-hourly) → aggregate to mm/day |
| Slope | degrees in features; radians in formulas |
