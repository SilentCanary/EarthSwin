# 08 — Research Landscape: What the Papers Use

> What published work uses (data, models, metrics) and what we take from each. Full citations and links: `12_REFERENCES.md` (numbers in [brackets]).
> Agency/institution programmes (NASA, ISRO, GSI, IIT Mandi, MIT, …) are in `09_NASA_ISRO_AND_INSTITUTIONS.md`.

---

## 1. Benchmarks and datasets

| Dataset | Data | Size / scope | Task | What we take |
|---|---|---|---|---|
| **Landslide4Sense** [1] (Ghorbanzadeh et al., IEEE TGRS 2022) | Sentinel-2 (12 bands) + ALOS PALSAR DEM and slope, 128×128 patches, 14 layers | 3,799 patches; Iburi 2018, **Kodagu (India) 2018**, Gorkha 2015, Taiwan 2009 | Pixel segmentation | S2+DEM pretraining; external sanity benchmark; channel schema |
| **L4S competition outcome** [2] (Ghorbanzadeh et al., IEEE JSTARS 2022) | Same | 85 teams, 7,775 submissions; winners used Swin Transformer, SegFormer, U-Net; 1st place F1 80.41 % (val), 74.54 % (test) [3] | Segmentation | Realistic F1 ceiling for S2-based detection; tricks: hard-example mining, self-training, mix-up |
| **Sen12Landslides** [4] (Höhn, Heidler, Behling, Zhu, Sci. Data 2025) | **S1 + S2 time series + Copernicus DEM**, pre/post dates | ~75,000 annotations, 15 regions, >12,000 patches; benchmarks U-ConvLSTM, 3D-UNet, U-TAE; best F1 > 83 % on S2 | Spatio-temporal segmentation | **Closest match to PMGST-Net-D** — pretrain multi-modal temporal model; reuse their temporal baselines |
| **CAS Landslide** [5] (Xu et al., Sci. Data 2024) | RGB satellite + UAV | 20,865 images, 9 regions | Segmentation | Optional RGB pretraining |
| **GVLM** [6] (Zhang et al., ISPRS 2023) | Bitemporal very-high-res (0.59 m) Google Earth | 17 image pairs | Change detection | Change-detection architectures (Siamese, difference enhancement) |
| HR-GLDD [7] (Meena et al., 2023) | PlanetScope (4 bands) | 13 events, 1,756 tiles, 7,193 landslides | Segmentation | Higher-res generalisation reference |
| Bijie [8] (Ji et al., 2020) | 3-band optical | 1 event, 2,773 tiles, 770 landslides | Classification/segmentation | Historical reference only |
| GDCLD [9] (Fang et al., ESSD 2024) | Multi-source high-res, coseismic | Globally distributed | Segmentation | Earthquake-trigger generalisation (out of scope) |

**Takeaway:** public datasets mix sensors and triggers; none covers Himachal monsoon landslides. We pretrain on them and fine-tune/evaluate on HP inventories (`04` §3).

## 2. Detection models (optical / SAR)

| Paper | Data | Model | Key result / idea | Take |
|---|---|---|---|---|
| Ghorbanzadeh et al. 2019 [10] | Optical (Nepal) | CNNs vs SVM/RF/ANN | Early evidence CNNs compete with ML for landslide detection | Baseline lineage |
| Prakash et al. 2020 [11] | Optical + DEM | U-Net vs RF/LR | U-Net outperforms pixel classifiers | B3 baseline design |
| Nava et al. 2022 [12] | **Sentinel-1 SAR** | Attention U-Net | Rapid mapping from SAR when optical is cloud-covered | S1 branch + attention gates |
| Le et al. 2023 [13] | L4S | U-Net + augmentation/loss/ensemble improvements | Systematic improvements over CE-trained U-Net | Training-recipe checklist |
| NRSC/ISRO [14] | Indian Himalaya & Western Ghats | DeepLabV3+ with NDVI and terrain inputs | Operational-style DL mapping in India | Indian precedent; DeepLabV3+ baseline option |
| MS-Former [15] | GVLM (bitemporal VHR) | Memory-supported transformer, weak patch-level labels | Memory bank helps weakly supervised change detection | Supports Block 9 (memory) idea |
| Sharma, Tiwari & James (EGU 2026) [16] | S2 NDVI change + SRTM slope/roughness in GEE, HP 2022 | Rule-based change detection | NDVI change rapidly highlights probable failures in HP | Our B0 baseline is the same idea |

## 3. SAR-specific detection and timing

| Paper | Contribution | Take |
|---|---|---|
| **Handwerger et al. 2022** [17] (NHESS 22:753) | Sentinel-1 backscatter-change heatmaps in Google Earth Engine; tested on rainfall- and earthquake-triggered events; more images and both orbit geometries improve detection | B1 baseline; S1 preprocessing choices |
| **Burrows et al. 2022** [18] (NHESS 22:2637) | Four techniques using S1 amplitude time series to constrain landslide timing to within a few days | Dating our polygons (FR-LBL-03); Block 8 design |
| Burrows et al. 2023 [19] (GRL) | Timings for 579 Nepal landslides (2015, 2017–2019); clusters linked to intense rainfall | Rain–timing analysis template for Phase 2 |
| Mondini et al. 2021 [20] (review) | SAR for landslide detection: past, present, future | Background for report |

## 4. Foundation models and self-supervision

| Model | What it is | Landslide evidence | Take |
|---|---|---|---|
| **Prithvi-EO-2.0** [21] (IBM–NASA, 2024) | ViT-MAE pretrained on 4.2 M multi-temporal Harmonized Landsat-Sentinel-2 sequences; 300 M / 600 M versions | Fine-tuned on L4S: F1 60.7 % vs U-Net 59.7 % (as cited in [23]); robust under label scarcity and band changes [22] | B6 baseline; LoRA fine-tune |
| **TerraMind** [24] (IBM, ESA Φ-lab, Jülich, NASA IMPACT; ICCV 2025) | Any-to-any generative multimodal model: S1 GRD/RTC, S2 L1C/L2A, DEM, LULC, NDVI, captions, coordinates; trained on TerraMesh (9 M samples) | General EO benchmarks; natively multi-modal (S1 + S2 + DEM like us) | Strong candidate encoder for Blocks 4–5 |
| Clay-CNN hybrids [23] (2026) | Clay v1.5 GFM features as auxiliary context for a CNN on L4S | GFM context + CNN spatial precision | Hybrid design for Block 4 |
| **SSL4EO-S12** [25] (Wang et al., 2023) | Unlabeled S1/S2 corpus (251k+ locations) with MoCo-v2, DINO, MAE, data2vec weights | General EO | S1 encoder init; SSL recipe for our own data |

Honest read: on L4S, foundation models give **small** gains over a well-trained U-Net; their value is larger when labels are scarce or sensors differ — which is our situation.

## 5. Spatio-temporal and graph models

| Paper | Task | Model | Take |
|---|---|---|---|
| **U-TAE** [26] (Garnot & Landrieu, ICCV 2021) | Satellite image time-series segmentation (PASTIS) | Shared conv encoder + lightweight temporal attention at lowest resolution, masks upsampled to all scales; ~1.1 M params, 63.1 mIoU | Default pixel-dense Block 8 for S1 series |
| KNN-GCN (Xia et al., EGU 2023) [27] | Susceptibility on slope units | GCN on KNN graph of slope units | Slope units as nodes (Block 6) |
| GNN on slope units, Taiwan [28] | Susceptibility | GNN with conditioning factors as node features | More spatially coherent maps than independent units |
| LandGNN [29] (Remote Sensing 2022) | InSAR displacement prediction | Attentive GNN + "locally historical" transformer | GNN + temporal transformer pattern ≈ Blocks 6–8 |
| Lan et al. [30] | Spatial prediction with deformation | GNN guided by "Third Law of Geography" + temporal transformer | Separating static predisposition vs dynamic triggers |

**Gap / opportunity:** GNNs are used for susceptibility and displacement *forecasting*, rarely for *detection*. Using graph refinement on top of a pixel detector is a defensible novelty — but must earn its place in ablations.

## 6. Physics-informed and space–time hazard

| Paper | Contribution | Take |
|---|---|---|
| **Dahal et al. 2024** [31] (NHESS 24:823) | Space–time hazard via Ensemble Neural Networks; susceptibility AUC 0.93, density Pearson r 0.93; code public | Joint "where + how much" formulation for Phase 2 |
| **Dahal & Lombardo 2024** [32] (Eng. Geol.) | Physics-informed NN: network retrieves geotechnical parameters, Newmark constraint in the loss, trained on coseismic inventories | Template for our FS-consistency loss (Block 14.2) with rainfall physics |
| Montgomery & Dietrich 1994; Iverson 2000 [33, 34] | SHALSTAB steady-state and transient-infiltration slope stability | Physics features and differentiable physics layer (`06` §3) |

## 7. Early warning and thresholds

| Work | Contribution | Take |
|---|---|---|
| Caine 1980; Guzzetti et al. 2008 [35, 36] | Rainfall I–D thresholds (global) | Phase 2 baseline |
| Guzzetti et al. 2020 [37] (review) | Geographical (regional) landslide early warning systems | Design principles for levels and verification |
| **LHASA 2.0** [38] (Stanley et al. 2021) | Global ML (XGBoost) nowcast with separate dynamic/static inputs; GPM + SMAP | Phase 2 baseline; feature ideas |

## 8. India / Himachal-specific

| Work | Finding | Take |
|---|---|---|
| Sana et al. 2024 [39] (*Landslides* 21(4)) | July 2023 HP: Shimla, Kullu, Mandi hardest hit; 223 fatalities, 1,300 road closures | Context, event dates |
| **Sana et al. 2025** [40] (*Landslides* 22(2)) | ~6,500 landslides mapped over ~3,100 km² of Mandi & Kullu (2023 monsoon); sandstone lithology the most critical combination | **Primary label source** (L1); lithology edge feature |
| 2023 HP geospatial assessment [41] | Jul–Sep 2023: 428 fatalities, > 40 % from slope failures | Motivation statistics |
| HPSCCC 2023 inventory [42] | Statewide post-monsoon inventory with NRSC tool + field mapping | Label source L2 |
| GSI NLSM / national inventory [43] | 1:50k susceptibility, incl. ~42,093 km² of HP | Prior / baseline |

## 9. Reviews to cite
- Jiang, Ma & Mei 2026 [44] (NHESS 26:487): deep learning for potential landslide identification; open problems are **data imbalance, generalisation and mechanism complexity**; recommends combining knowledge-driven and data-driven methods.
- Guzzetti et al. 2020 [37] (early warning); Mondini et al. 2021 [20] (SAR).

---

## 10. What the literature implies for PMGST-Net

| Our design choice | Literature support | Risk | Verdict |
|---|---|---|---|
| S1 + S2 + DEM fusion | Strong (L4S, Sen12Landslides, Handwerger) | Low | **Keep — core** |
| Swin/transformer encoder | Competition winners used Swin/SegFormer | Overfitting with few labels | Keep, pretrained only |
| Temporal modelling | Sen12Landslides, U-TAE, Burrows timing | Irregular S1 dates | Keep (U-TAE style) |
| Multi-scale pyramid | Common in segmentation | Memory | Keep if ablation helps |
| Graph over slope units | Susceptibility literature | Novel for detection; may add little | Test; P1 |
| Physics guidance | PINN for seismic (Dahal & Lombardo); SHALSTAB/TRIGRS classics | Parameter uncertainty | Features now; loss later |
| Memory module | MS-Former (change detection) | Leakage | P2 |
| Evidential uncertainty | Rare in landslide papers | Calibration can be poor | Compare with ensembles |
| Foundation-model init | Prithvi/TerraMind/SSL4EO | Small gains on L4S | Use for label scarcity |

**Defensible novelty for a paper:** a physics-guided, uncertainty-aware, multi-modal spatio-temporal detector, evaluated with spatial cross-validation on the 2023 Himachal monsoon — with honest ablations showing which blocks matter.
