# 09 — What NASA, ISRO, GSI and Others Have Built

> Operational systems and programmes relevant to landslide detection and early warning, how they work, and how PMGST-Net relates to each. Citations: `12_REFERENCES.md`.

---

## 1. Summary table

| System / programme | Organisation | Type | Data | Method | Scale | Status |
|---|---|---|---|---|---|---|
| Global Landslide Catalog / COOLR | NASA GSFC | Inventory | News media, reports, citizen science | Manual compilation since 2007 | Global points | Active archive |
| LHASA 2.0 | NASA GSFC | Hazard nowcast (+ forecasts evaluated) | GPM rainfall, SMAP soil moisture, static factors | XGBoost, probabilistic | 1 km, daily (≈30-min capable) | Operational |
| S1 backscatter heatmaps (GEE) | NASA JPL / UCLA et al. | Rapid detection | Sentinel-1 | Backscatter change | Regional | Open tools |
| OPERA DIST-S1 | NASA JPL | Surface disturbance alerts | Sentinel-1 | Transformer-based anomaly vs historical baseline | 30 m | Production collection released Oct 2026 |
| Prithvi-EO-2.0 | IBM + NASA | Foundation model | HLS (Landsat + S2) | ViT MAE | Global | Open weights |
| NISAR | NASA + ISRO | SAR mission | L-band (NASA) + S-band (ISRO) | InSAR / backscatter | Global, 12-day | Launched 30 Jul 2025 |
| Landslide Atlas of India | ISRO NRSC | Inventory + exposure | Indian & international satellites | Visual + validated mapping | ~80k landslides, 1998–2022 | Published 2023 |
| DL landslide mapping | ISRO NRSC | Detection | Optical + terrain | DeepLabV3+ | Himalaya, Western Ghats | Research/operational support |
| National Landslide Forecasting Centre | GSI | Regional early warning | Rainfall, susceptibility, inventory | Thresholds / expert system (AI under research) | District | Operational since Jul 2024 |
| NLSM | GSI | Susceptibility | Field + thematic layers | Mapping | 1:50k, 4.3 lakh km² | Published (Bhukosh) |
| Slope-scale sensor EWS | IIT Mandi | Site early warning | In-situ sensors | ML on sensor streams | Sites in HP | Deployed |
| SERVIR-HKH tools | ICIMOD + NASA (+ USAID) | Regional hazard tools | LHASA-based, SAR | Adapted LHASA, slope-motion monitoring | Nepal/HKH | Programme funding ended Jan 2025; tools at ICIMOD |

---

## 2. NASA

### 2.1 Global Landslide Catalog (GLC) and COOLR
- GLC has been compiled at NASA Goddard since 2007; its main information sources are online news found via Google Alerts and search engines, plus disaster databases and reports.
- COOLR (Cooperative Open Online Landslide Repository) merges GLC with citizen-science reports from Landslide Reporter.
- Each record has a **location accuracy** field (exact, within 1 km, within 5 km, …), date, type, trigger and fatalities.
- **Relevance:** this is where EarthSwin v0's 6 labels came from. Good for *when/where roughly*; not pixel truth. Use as weak labels only.

### 2.2 LHASA (Landslide Hazard Assessment for Situational Awareness)
- **v1:** decision tree combining a static susceptibility map (slope, rock type, forest loss, distance to faults, …) with satellite rainfall exceedance.
- **v2:** replaces the decision tree with **XGBoost**, feeding each dynamic and static variable separately instead of merging non-rain factors into one susceptibility map; combines **GPM** rainfall with **SMAP** soil moisture; outputs a **probabilistic** nowcast so users can choose thresholds; daily 1 km archive; 1–3-day forecasts with global rainfall forecasts were being evaluated; exposure (population, roads) added.
- **Relevance:** the natural **Phase 2 baseline** and a feature source (download daily nowcast for HP). Its design lesson — keep variables separate and let the model learn interactions — mirrors our Block 5/8b.

### 2.3 JPL: SAR detection, ARIA and OPERA
- **Handwerger et al. 2022:** Sentinel-1 backscatter-change approach in Google Earth Engine that turns multi-temporal stacks into landslide-density heatmaps for rapid response, with topographic masks for flat areas → our **B1 baseline**.
- **OPERA DIST-S1:** maps surface disturbance outside a historical norm from Sentinel-1; statistics estimated with a deep model using spatiotemporal (vision-transformer) encoding; DIST-ALERT-S1 production collection announced Oct 2026 at ASF DAAC. → candidate **weak label / feature**; check India coverage.

### 2.4 Foundation models
- **Prithvi-EO-2.0** (with IBM): ViT pretrained by masked autoencoding on 4.2 M multi-temporal HLS sequences; landslide fine-tuning on Landslide4Sense is part of its evaluation suite → **B6 baseline**.
- **TerraMind** (IBM + ESA Φ-lab + Jülich + NASA IMPACT and others): multimodal any-to-any model covering S1, S2, DEM and land cover → strong encoder candidate.

### 2.5 NISAR (with ISRO)
- Launched **30 Jul 2025** on GSLV-F16 from Sriharikota; scans nearly all land and ice twice every 12 days; designed to help study earthquakes, volcanoes and **landslides**.
- **L-band (NASA JPL) + S-band (ISRO)** — first radar imaging satellite with two frequencies; 240 km swath.
- **Relevance:** L-band penetrates vegetation better than Sentinel-1's C-band → better coherence on forested Himalayan slopes. Plan a future S1 → NISAR transfer experiment. ISRO distributes NISAR data via Bhoonidhi.

### 2.6 SERVIR-HKH (with ICIMOD)
- **LHASKarnali** (Nepal): LHASA v2 adapted with HIWAT forecasts, giving hazard estimates up to one day ahead.
- Slow-moving landslide (unstable slope motion) monitoring tools built by ICIMOD, Univ. of Alaska Fairbanks and NASA Marshall/Goddard.
- US funding ended in January 2025; tools remain with ICIMOD.
- **Relevance:** the closest regional precedent to "LHASA for a Himalayan basin" → template for our Phase 2.

## 3. ISRO / NRSC

### 3.1 Landslide Atlas of India (2023)
- ~80,000 landslides mapped by NRSC under its Disaster Management Support programme, 1998–2022, covering 17 states and 2 UTs in the Himalaya and Western Ghats.
- Three inventory types: **seasonal** (pan-India 2014 and 2017 rainy seasons), **event-based** (e.g. Kedarnath, Kerala, Sikkim earthquake), **route-wise** (tourist/pilgrimage routes).
- Built from IRS-1D, Resourcesat-1/2/2A LISS-IV, Cartosat-1/2S, plus Sentinel-1/2, Pléiades, WorldView and aerial images.
- Ranks 147 districts by landslide exposure.
- **Relevance:** label source L4 (2017 seasonal inventory could give us a second year); exposure ranking for Phase 2 prioritisation.

### 3.2 Deep-learning detection
- NRSC developed a **DeepLabV3+** model for mapping landslides in the Himalaya and Western Ghats, with inputs such as NDVI and terrain. → Indian precedent; compare our numbers if they publish metrics.

### 3.3 Rapid mapping (Wayanad, 30 Jul 2024)
- NRSC mapped the Chooralmala debris flow using **Cartosat-3** (pre-event, May 2023) and very-high-resolution **RISAT** SAR (31 Jul 2024); the flow was ~8 km from crown to run-out.
- **Relevance:** shows SAR + optical pairing for rapid response — the same logic as our S1 + S2 fusion, at much higher resolution.

### 3.4 Portals
- **Bhuvan / NDEM** (maps, disaster products), **Bhoonidhi** (satellite data incl. NISAR).

## 4. Geological Survey of India (GSI)

- **National Landslide Susceptibility Mapping (NLSM):** 1:50k susceptibility for ~4.3 lakh km², including ~42,093 km² in Himachal Pradesh; inventory and maps on **Bhukosh** and shared with NDMA and HPSDMA.
- **National Landslide Forecasting Centre (NLFC):** inaugurated **19 Jul 2024** in Kolkata with the **Bhusanket** portal and **Bhooskhalan** app; live forecasts for Kalimpong, Darjeeling and Nilgiris from 20 Jul 2024.
- Expanded from 16 districts in 6 states to **21 districts in 8 states** in 2025; research underway on an **AI-based** forecasting model and expert system; goal of a nationwide regional LEWS by **2030**.
- Himachal districts (Shimla, Kinnaur) listed as experimental in GSI's forecasting coverage.
- The LEWS prototype comes from **LANDSLIP** (2016–2022, India–UK–Italy; Darjeeling and Nilgiris).
- **Relevance:** the official channel for warnings in India. PMGST-Net should align its warning levels and verification with NLFC practice and position itself as research input (detected-event maps for verification, an AI model candidate), not a competing alert service.

## 5. Himachal Pradesh state bodies
- **HPSDMA** (state disaster management authority) hosts GSI site studies and hazard reports.
- **HPSCCC / HP Council for Science, Technology & Environment** produced the **2023 post-monsoon landslide inventory** (Feb 2025) using NRSC's mapping tool plus field mapping → label source L2.

## 6. Academic groups in India

### IIT Mandi (in our study area)
- Low-cost sensor-based landslide monitoring by Varun Dutt and K. V. Uday, started after the Aug 2017 Kotropi landslide; sensors measure ground movement, rainfall, soil moisture; SMS alerts.
- Reported deployment at 60+ sites in HP with up to **3-hour** advance prediction and >90 % accuracy (NITI Frontier Tech); a 2026 report describes sub-millimetre movement detection deployed in Mandi district with the National Mission on Himalayan Studies.
- Field reality: of 60 sensors installed in Mandi, Kinnaur and Kangra, only 45 were working after malfunctions (Down To Earth).
- With Durham University: the **2023 Mandi/Kullu inventory** (~6,500 landslides).
- **Relevance:** best potential collaborator — their inventory (labels) and sensor data (ground truth for timing/warning) complement our satellite model.

### Amrita Vishwa Vidyapeetham
- Wireless sensor network in Munnar (Kerala); warned of a possible landslide during July 2009 rains.

### IIT Roorkee, IIT Indore
- Also experimenting with or deploying landslide early warning systems.

## 7. MIT and Harvard

### MIT
- **Draper Laboratory + MIT**, under NASA contract: a statistical model of where landslides are most likely, finding that heavy rain matters most when cumulative precipitation is high and soil is already saturated (lead: Natasha Markuzon).
- **MIT EAPS + Draper** (Bellugi, Perron, Schlosser et al.): physically based shallow-landslide prediction from repeat-mapped inventories, with climate-change rainfall projections.
- **Theses:** Gupta (2022) — interpretable ML for landslide detection and susceptibility (Mocoa, Colombia); CNN best with 87.3 % detection accuracy. Adebi (2023) — susceptibility adaptive to triggering events, adapted for MIT Environmental Solutions Initiative's DECCA drone project.
- **Assessment:** no operational MIT landslide system; the useful idea for us is **antecedent saturation × storm rainfall interaction** (Block 8b / Phase 2).

### Harvard
- We found **no substantive Harvard landslide programme**. The only link is the 2026 Clay-CNN hybrid paper on Landslide4Sense, one of whose authors was formerly at Harvard. Don't cite Harvard as a source of landslide systems.

## 8. Others worth knowing
- **ESA / Copernicus:** Sentinel-1 and -2; Sentinel-1B ended Aug 2022; Sentinel-1C launched 5 Dec 2024 restoring two-satellite revisit; TerraMind with IBM.
- **IARAI** (Landslide4Sense), **TUM / DLR** (Sen12Landslides), **CAS IMHE** (CAS dataset), **University of Twente ITC** (Dahal & Lombardo), **Université Toulouse** (Burrows — S1 timing).

## 9. Where PMGST-Net fits

| Need | Existing | Gap we fill |
|---|---|---|
| Where did landslides happen (HP, monsoon)? | Manual inventories (NRSC, HPSCCC, IIT Mandi) — months later | **Automated, uncertainty-aware detection maps within days** of cloud-free/S1 data |
| When did they happen? | Mostly unknown at season scale | **S1-based dating** → dated events |
| Where next (days)? | LHASA (1 km global), GSI NLFC (district, thresholds), IIT Mandi (site sensors) | **Slope-unit hazard for HP** trained on our dated detections, physics-guided, calibrated |
| Trust | Rarely quantified | **Calibrated probabilities + uncertainty + XAI** |
