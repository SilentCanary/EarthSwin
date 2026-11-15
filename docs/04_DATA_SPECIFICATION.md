# 04 — Data Specification

> Everything the model eats: sources, labels, preprocessing, tensors, splits, QA.
> The audit (`01_CURRENT_STATE_AUDIT.md`) showed the 2016 GLC-based dataset cannot train a detector (6 labels). This spec replaces it.

---

## 1. Area, grid and time

| Item | Choice | Why |
|---|---|---|
| AOI | Himachal Pradesh boundary (`preprocessing/Himachal_GeoJSON.geojson`), bbox ≈ 75.58–79.00 °E, 30.38–33.17 °N | Same state as v0 |
| Primary event season | **2023 monsoon (Jul–Sep)** | Exceptional, well-documented event; polygon inventories exist (§3) |
| Secondary / test season | 2025 monsoon (if labels obtainable) or 2017 (NRSC seasonal inventory) | Temporal generalisation test |
| Working CRS | **EPSG:32643 (UTM 43N)**, 10 m pixels | Metric pixels for slope/area; HP's eastern edge spills into zone 44 — acceptable distortion, document it |
| Pre-event window | 1 May – 30 Jun 2023 | Before monsoon failures |
| Post-event window | 15 Sep – 31 Oct 2023 | Clearer skies for Sentinel-2 |
| S1 time series | Every acquisition 1 May – 31 Oct 2023 | For timing and cloud-robust detection |

Sentinel-1 constellation context: Sentinel-1B stopped operating in **August 2022**, so 2023 has Sentinel-1A only (12-day repeat per orbit). Sentinel-1C launched on **5 Dec 2024**, restoring two-satellite coverage for 2025 onward.

## 2. Input sources

| # | Layer | Source / GEE ID (check catalog) | Native res. | Revisit | Bands / variables | Use |
|---|---|---|---|---|---|---|
| S2 | Sentinel-2 L2A | `COPERNICUS/S2_SR_HARMONIZED` + `COPERNICUS/S2_CLOUD_PROBABILITY` | 10/20 m | 5 d | B2, B3, B4, B8, B11, B12, SCL | Primary detection signal (vegetation loss, fresh soil) |
| S1 | Sentinel-1 GRD IW | `COPERNICUS/S1_GRD` | 10 m | 12 d/orbit (2023) | VV, VH (dB), angle | Cloud-proof change; timing |
| S1-RTC | Radiometric terrain-corrected γ⁰ (optional) | ASF HyP3 RTC on demand | 10–30 m | per acquisition | VV, VH γ⁰ | Better in steep terrain |
| DEM | Copernicus GLO-30 | `COPERNICUS/DEM/GLO30` | 30 m | static | elevation → derivatives (§5.3) | Terrain, physics |
| RAIN | GPM IMERG V07 | `NASA/GPM_L3/IMERG_V07` | 0.1° | 30 min | precipitation | Antecedent rainfall, physics |
| RAIN-alt | CHIRPS daily | `UCSB-CHG/CHIRPS/DAILY` | 0.05° | daily (downscaled from pentads) | precipitation | Fallback; long climatology |
| SM | SMAP L4 | `NASA/SMAP/SPL4SMGP/007` | 9 km | 3 h | surface/root-zone soil moisture | Phase 2 (warning) |
| LC | ESA WorldCover 2021 | `ESA/WorldCover/v200` | 10 m | static | land-cover class | Hard-negative strata, error analysis |
| HYD | HydroRIVERS / derived drainage | HydroSHEDS; or flow accumulation from DEM | 30 m+ | static | river network | Graph edges, false-positive control |
| ROAD | OpenStreetMap roads | OSM extract | vector | static | road segments | Graph edges, exposure |
| GEO | Lithology | GSI Bhukosh geological maps | 1:50k–1:2M | static | rock type | Physics, graph edges |
| SUS | GSI national landslide susceptibility (NLSM) | GSI Bhukosh | 1:50k | static | susceptibility zone | Baseline / prior |
| DIST | OPERA DIST-ALERT-S1 (check India coverage and dates) | NASA ASF DAAC | 30 m | per acquisition | disturbance status/confidence | Weak labels / feature |

## 3. Labels

### 3.1 Sources, in priority order

| # | Inventory | Content | Access |
|---|---|---|---|
| L1 | **IIT Mandi–Durham 2023 inventory** (Sana et al., *Landslides* 2025) | ~6,500 landslides mapped across ~3,100 km² of Mandi and Kullu districts, 2023 monsoon | Request from authors (template: `03_IMPLEMENTATION_PLAN.md` Appendix A) |
| L2 | **HPSCCC 2023 post-monsoon inventory** (HP Council for Science, Technology & Environment, Feb 2025) | Statewide 2023 inventory made with NRSC's mapping tool + field mapping | Request via HPSCCC / HPSDMA |
| L3 | **GSI national inventory** | Field-studied landslides, susceptibility maps; HP included | Download from Bhukosh |
| L4 | **NRSC/ISRO Landslide Atlas data** | ~80k landslides 1998–2022; seasonal inventories for 2014 and 2017 | Request via NRSC / Bhuvan |
| L5 | **Self-digitised polygons** | ≥ 300–500 polygons (protocol §6) | Team |
| L6 | NASA GLC / COOLR | Media-derived points with location accuracy (exact / 1 km / 5 km / …) | Weak labels only (MIL, §3.3) |

### 3.2 External datasets (pretraining / benchmarking)

| Dataset | What it is | Use |
|---|---|---|
| Landslide4Sense (2022) | 3,799 patches, 128×128, 14 layers: Sentinel-2 bands + ALOS PALSAR DEM and slope; 4 regions incl. Kodagu (India) | Pretrain S2+DEM branch; benchmark sanity check |
| Sen12Landslides (2025) | ~75k annotations, 15 regions, >12k patches of S1 + S2 + Copernicus DEM time series with event dates | Pretrain multi-modal temporal model — closest match to our design |
| CAS Landslide (2024) | 20,865 RGB images from satellite + UAV, 9 regions | Optional RGB pretraining |
| GVLM (2023) | 17 bitemporal VHR (0.59 m) pairs | Change-detection ideas; not our resolution |

### 3.3 Label definitions

- **Positive pixel:** inside a mapped polygon of a *new* failure in the event window (source scar + transport + deposit, i.e. the whole affected area, as in Landslide4Sense).
- **Negative pixel:** outside all polygons and outside the ignore band.
- **Ignore (255):** 1-pixel band around polygon boundaries; S2 cloud/shadow in the post composite; S1 layover/shadow (for S1-only experiments); no-data.
- **Event date:** exact date if known, else `[date_min, date_max]`; refine with S1 amplitude time series (Burrows et al. 2022 approach) — target ≤ 24-day ranges.
- **Weak point labels (L6):** a point with location accuracy *r* defines a bag of chips within radius *r*; multiple-instance learning assumes ≥ 1 positive in the bag. Never rasterised as pixel truth.

## 4. What changes from v0

| v0 (2016) | v1 (this spec) |
|---|---|
| 6 GLC points mapped to 5 km patches | Thousands of polygons, pixel labels at 10 m |
| VV only (VH dropped), no-data = 0 | VV + VH, no-data = NaN + masks |
| No optical data | Sentinel-2 pre/post |
| Slope only | Full terrain derivatives |
| CHIRPS weekly, 5.5 km | IMERG daily + antecedent indices (physics branch) |
| Random anchor split | Spatial block CV + buffers |

## 5. Preprocessing chain

Maths for each step: `06_PHYSICS_AND_MATHEMATICS.md`.

### 5.1 Sentinel-2
1. Filter by AOI, date, `CLOUDY_PIXEL_PERCENTAGE < 60`.
2. Mask clouds: SCL ∈ {3 cloud shadow, 8 cloud medium, 9 cloud high, 10 thin cirrus} **and** s2cloudless probability > 40 %.
3. Scale to reflectance (÷ 10,000), median composite per window.
4. Indices: NDVI, BSI, NDWI; differences ΔNDVI = NDVI_post − NDVI_pre.
5. Resample 20 m bands (B11, B12) to 10 m (bilinear).

### 5.2 Sentinel-1
1. Filter IW mode, VV+VH, by orbit direction; keep `relativeOrbitNumber_start`.
2. GEE GRD values are dB (thermal noise removed, calibrated, ortho-rectified). **Average in linear power, never in dB**; medians are safe in either.
3. Speckle: temporal multi-looking (median of ≥ 3 acquisitions) or Refined Lee 7×7 for single dates.
4. Optional radiometric terrain flattening (γ⁰) via ASF HyP3 RTC.
5. Log-ratio change per orbit: LR = 10·log₁₀(median_post / median_pre) in linear units.
6. Layover/shadow mask from local incidence angle (§5.3 of `06`).
7. No-data → NaN + validity mask (fixes audit C2).

### 5.3 Terrain (from GLO-30)
Elevation, slope (Horn), aspect as (sin, cos), profile and plan curvature, TPI (300 m window), terrain roughness, D8/D∞ flow accumulation, specific catchment area, TWI. Compute at 30 m, then bilinear to 10 m (nearest for categorical).

### 5.4 Rainfall
Daily IMERG totals per 0.1° cell; per chip/slope unit: event total, max 1-day, max 3-day, antecedent indices (1, 3, 7, 14, 30 d), rainfall-threshold exceedance ratio (§4 of `06`). Treat as coarse context — never as pixel texture.

### 5.5 Normalisation
- Per channel: robust z-score using median and IQR (or mean/std after clipping to 1st–99th percentile), **computed on training folds only**, valid pixels only.
- Missing values → 0 after normalisation, with a validity mask channel per modality.

## 6. Digitisation protocol (self-made labels)

1. Tools: QGIS + S2 pre/post true/false-colour composites (B12-B8-B4), Google Earth historical imagery for confirmation.
2. Map **new** failures only (bare scar absent in pre-image, present in post-image), including run-out.
3. Minimum mapping unit: **900 m²** (≈ 9 pixels at 10 m).
4. Attributes per polygon: `id, date_min, date_max, type {slide, debris_flow, rockfall, unknown}, source_image_ids, confidence {1 low, 2 medium, 3 high}, digitiser, notes`.
5. Exclude: river-bank erosion without slope failure, road cuts/construction, quarries, seasonal snow/ice changes.
6. Quality: 10 % of chips double-labelled by both team members; inter-annotator IoU ≥ 0.6, disagreements discussed and logged.

## 7. Chips and tensors

| Item | Value |
|---|---|
| Chip | 256 × 256 px at 10 m (2.56 km), stride 192 px (64 px overlap) |
| Pyramid context | 512 px (5.12 km) and 1024 px (10.24 km) crops centred on the chip, resampled to 256 × 256 |
| S2 tensor | `[2 (pre/post), 6 bands, 256, 256]` + validity mask `[2, 1, 256, 256]` |
| S1 tensor | `[T ≤ 16 dates, 2 (VV, VH), 256, 256]` + mask + day offsets `[T]` + orbit id `[T]` |
| Terrain tensor | `[10, 256, 256]`: elev, slope, sin asp, cos asp, prof curv, plan curv, TPI, roughness, log flowacc, TWI |
| Rain vector | `[R]` antecedent indices per chip (plus per-slope-unit for graph) |
| Label | `[256, 256]` uint8 {0, 1, 255} |
| Storage | Zarr (chunk = 1 chip) or Cloud-Optimised GeoTIFF; index `chips.parquet` |

`chips.parquet` columns: `chip_id, geometry, district, block, fold, n_pos_px, n_ignore_px, s2_valid_frac, s1_n_dates, landcover_majority, mean_slope, label_source`.

## 8. Splits
- **Spatial block K-fold:** 10 km × 10 km blocks, K = 5, stratified by landslide load, 2.56 km buffer between training chips and test blocks (code: `03_IMPLEMENTATION_PLAN.md` §3.4).
- **Region hold-out:** e.g. train on Mandi, test on Kullu (and vice versa).
- **Temporal hold-out:** train 2023 → test 2025 (if labels exist).
- One graph per region over **all** nodes (labels of test nodes hidden), so train and test graphs share geometry (fixes audit C6).

## 9. Sampling and imbalance
- Chip sampler per epoch: 50 % chips with ≥ 1 landslide pixel, 30 % random, 20 % hard negatives (WorldCover bare/sparse, river beds within 100 m of drainage, road buffers, high-slope scree).
- Hard-negative mining refresh every 5 epochs from current false positives on training folds.
- Pixel loss weighting via focal + Dice (see `05`, Block 14).

## 10. Augmentation
- **Geometric:** D4 group (flips, 90° rotations) on all rasters *consistently*. Transform aspect channels: horizontal flip (sin a → −sin a), vertical flip (cos a → −cos a), 90° counter-clockwise rotation (a → a − 90°: sin → −cos, cos → sin). Slope, curvature, TPI, roughness, TWI are rotation-invariant.
  - Caveat: flips/rotations also change the apparent SAR look direction relative to terrain. Run one ablation with flips disabled for S1-dependent models.
- **Radiometric:** per-band gain/offset jitter (±5 %), S1 speckle-like multiplicative noise.
- **Modality dropout:** drop S2 (p = 0.3) to simulate clouds; drop S1 dates (p = 0.2).
- **Copy-paste:** paste landslide polygons onto plausible slopes (slope > 15°) — optional, check realism.

## 11. GEE export sketch (Python API)

```python
import ee
ee.Initialize(project="<your-gee-project>")
aoi = ee.FeatureCollection("projects/<your-gee-project>/assets/hp_boundary").geometry()

def mask_s2(img):
    scl = img.select("SCL")
    clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(clear).select(["B2", "B3", "B4", "B8", "B11", "B12"]).divide(10000)

def s2_composite(start, end):
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
            .map(mask_s2).median().clip(aoi))

s2_pre, s2_post = s2_composite("2023-05-01", "2023-06-30"), s2_composite("2023-09-15", "2023-10-31")

s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
      .filterBounds(aoi).filterDate("2023-05-01", "2023-10-31")
      .filter(ee.Filter.eq("instrumentMode", "IW"))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
      .select(["VV", "VH", "angle"]))

task = ee.batch.Export.image.toDrive(
    image=s2_post.toFloat(), description="s2_post_2023", region=aoi,
    scale=10, crs="EPSG:32643", maxPixels=1e13, fileFormat="GeoTIFF")
task.start()
```

Volume check: HP ≈ 55,700 km² → ~557 M pixels per band at 10 m; one 10-band float16 statewide layer ≈ 11 GB. For training, export only labelled regions + 5 km buffers; export statewide only for final inference.

## 12. QA checklist (run after every data version)
- [ ] No S1 zeros survive; validity masks non-zero where expected.
- [ ] S2 post composite cloud-free fraction ≥ 95 % in labelled regions.
- [ ] Layers co-registered within 1 px (5 tie points).
- [ ] Label polygons overlay correctly on post-event S2 in 20 random chips.
- [ ] Positive-pixel fraction per fold within ±30 % of mean.
- [ ] Normalisation stats computed on training folds only.
- [ ] No test-block chip within buffer of a training chip.
- [ ] `chips.parquet` row count = number of stored chips.
