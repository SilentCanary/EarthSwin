# 11 — Early Warning (Phase 2) and Production / Self-Training (Phase 3)

> Starts **only after** Phase 1 detection passes its exit criteria (`02_REQUIREMENTS.md` §7). Detection is the foundation: its dated detections are the labels for warning, and its confirmed outputs feed self-training.

---

## Part A — Phase 2: Early warning ("be wary of this place")

### A.1 Problem formulation
For each **slope unit** (or 1 km cell) $i$ and issue day $t$, estimate

$$
P_i(t, h) = P\big(\text{landslide in } (t, t+h] \mid \mathbf{x}_{i,\le t}\big),\qquad h \in \{1, 3, 7\}\ \text{days}
$$

using only information available at $t$ (causal). Model family: PMGST-Net-W (`05` §1) with survival formulation (`10` A8) and physics layer (`10` A7).

### A.2 Labels
- Dated events from Phase 1 detections + S1 timing (`06` §6) + inventories with dates (GSI, NRSC, IIT Mandi), deduplicated per slope unit.
- Negatives: slope-unit-days without events (censoring handled by the survival likelihood).
- Expect noisy dates (± 6–12 days with S1-only timing): use interval-censored likelihood or label smoothing over the date window.

### A.3 Inputs

| Group | Variables | Source |
|---|---|---|
| Static predisposition | slope, curvature, TWI, lithology, land cover, distance to rivers/roads, GSI susceptibility, Phase 1 learned susceptibility embedding | DEM, GSI, WorldCover, OSM |
| Antecedent state | ARI (decay 0.8–0.95), 3/7/14/30-day rain, SMAP root-zone soil moisture, snowmelt (high elevations) | IMERG, SMAP |
| Trigger | daily/3-hour rain intensity, I–D exceedance $E$ | IMERG |
| Forecast (optional) | 1–3-day forecast rainfall | IMD/NCMRWF or ECMWF products |
| Precursors (optional) | S1 backscatter/coherence trends, InSAR velocity (NISAR later) | S1, NISAR |
| Physics | $FS(t)$ from transient infiltration | `06` §3 |

### A.4 Baselines
1. Rainfall I–D threshold fitted to our data (`06` §4).
2. LHASA v2 daily nowcast (download for HP).
3. XGBoost on the same features (LHASA-style, regional).
4. GSI susceptibility × rainfall exceedance (simple product).

### A.5 Evaluation
POD, FAR, CSI, Brier skill score, lead time, alert burden at **matched alert rates**; train on one monsoon, test on another (`07` §12).

### A.6 Warning levels and communication
- Levels LOW / MODERATE / HIGH / VERY HIGH from calibrated thresholds (`05` Block 12), with a REVIEW flag for high uncertainty.
- Each bulletin states: model version, data dates, probability, uncertainty, top drivers (XAI), and "research product — follow official advisories (GSI NLFC / state authorities)".
- Watch **alert fatigue**: target a maximum alerts-per-district-per-week budget and report it.

---

## Part B — Phase 3: Production system that trains itself (with humans in the loop)

### B.1 Architecture

```
            ┌───────────────────────── scheduled (Prefect/Airflow) ─────────────────────────┐
 Ingestion  │ S1 (Copernicus Data Space / ASF), S2, IMERG, SMAP, forecasts ─► STAC catalog  │
            │                    ▼                                                          │
 Features   │ preprocessing workers (same code as training) ─► Zarr/COG feature store       │
            │                    ▼                                                          │
 Inference  │ PMGST-Net-D (new imagery) ─► detections + uncertainty                         │
            │ PMGST-Net-W (daily)       ─► hazard levels per slope unit                     │
            │                    ▼                                                          │
 Decision   │ rules + thresholds + uncertainty gates ─► review queue (analyst)              │
            │                    ▼                                                          │
 Publish    │ dashboard (map tiles via TiTiler, PostGIS), e-mail/SMS/webhook to partners    │
            │                    ▼                                                          │
 Feedback   │ analyst confirm/reject, field reports, news/GLC matches, sensor data          │
            │                    ▼                                                          │
 Learning   │ label store ─► pseudo-label gate ─► retrain ─► evaluation gate ─► registry    │
            └───────────────────────────────────────────────────────────────────────────────┘
```

### B.2 Self-training loop ("trains itself", safely)
1. **Collect:** new detections with $\hat p$, $u$, physics plausibility, and any human/field feedback.
2. **Gate pseudo-labels:** accept automatically only if $\hat p > \tau_{hi}$ **and** $u < u_{\max}$ **and** slope ≥ $\beta_{\min}$ **and** persistent across ≥ 2 S1 dates; reject automatically if $\hat p < \tau_{lo}$; everything else → **human review queue** (active learning order from `10` A6).
3. **Mix:** retraining batches = verified labels + gated pseudo-labels (capped, e.g. ≤ 30 % of positives) + **replay** of the original curated set (prevents drift/forgetting).
4. **Train challenger** with the same config + new data version (DVC).
5. **Evaluation gate:** challenger must match or beat the champion on a **frozen benchmark** (never used for training) on PR-AUC, ECE and false-alarm rate, with bootstrap CI; plus no regression on any district by > 3 points.
6. **Promote** via model registry (MLflow) with shadow mode for one cycle; keep one-click **rollback**.
7. **Audit:** monthly manual audit of 100 random pseudo-labels; if precision < 0.9, tighten gates.

**Why gates matter:** naive self-training amplifies its own mistakes (confirmation bias). Physics plausibility + uncertainty + humans + a frozen benchmark break that loop.

### B.3 Monitoring and drift

| Signal | Metric | Alert when |
|---|---|---|
| Input drift | Population Stability Index per channel vs training | PSI > 0.2 |
| Data health | missing-S1/S2 fraction, cloud fraction, ingestion lag | above season norm |
| Output drift | detections/km²/week, level distribution | > 3σ from seasonal baseline without matching rainfall |
| Calibration | ECE on newly verified labels (rolling) | ECE > 0.08 |
| Sensor changes | new satellites/processing baselines (e.g. S1C, S1D) | any change → re-validate |

### B.4 Infrastructure (lean, student-feasible)
- **Compute:** GEE for heavy preprocessing where possible; one GPU box or cloud spot instances for inference/retraining.
- **Services:** FastAPI (model API), TiTiler (COG map tiles), PostGIS (polygons, slope units, alerts), Prefect (orchestration), MLflow (registry), DVC (data), Docker Compose.
- **Front end:** web map (MapLibre/Leaflet) with layers: probability, uncertainty, polygons, hazard level, XAI panel.
- **Notifications:** e-mail/webhook first; SMS only through an official partner.

### B.5 Governance and ethics
- Not an official warning service; coordinate with HPSDMA and GSI NLFC before any public-facing alert.
- Publish performance (POD/FAR) each season; document known failure modes (clouds, layover, snow).
- Data licences respected; inventories from partners not redistributed without consent.
- Keep a human decision-maker in the loop for every alert above MODERATE.

### B.6 Timeline (indicative)
| Period | Work |
|---|---|
| Apr–May 2027 | Phase 2 prototype on 2023 → test 2025 |
| Jun–Jul 2027 | Pipeline automation, dashboard, shadow mode during 2027 monsoon |
| Aug–Oct 2027 | Self-training cycles with review; season evaluation |
| After | Partner discussions (HPSDMA / IIT Mandi / GSI) |
