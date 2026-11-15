# 13 — Build & Run (PMGST-Net-D, the code)

> What is implemented, how to run it on the RTX 3060, and exactly what is **not** proven yet.
> Architecture and maths: `05`. Data spec: `04`. Plan: `03`. This file is the "how do I run it" page.

---

## 1. What exists (all 15 blocks of the diagram)

| Block | File | Status |
|---|---|---|
| 1 Multi-source data | `pmgst/data/gee_export.py` | Written, **not run against live Earth Engine** (needs your account) |
| 2 Preprocessing | `pmgst/data/chips.py`, `terrain.py`, `formats.py` | Tested on fabricated GEE-format tiles |
| 3 Multi-scale pyramid | `pmgst/models/pyramid.py` | Tested (3 scales, per-pixel scale attention) |
| 4 Hybrid CNN + Swin, shared weights | `pmgst/models/encoders.py` | Tested; one S2 encoder serves every scale and both dates |
| 5 Cross-modal attention fusion | `pmgst/models/fusion.py` | Tested: clouded modality gets weight 0 |
| 6 Dynamic graph | `pmgst/models/graph.py` | Tested; 7 edge relations, learned weights λ |
| 7 GATv2 | `pmgst/models/gat.py` | Tested; own implementation, **no torch_geometric needed** |
| 8 Temporal transformer | `pmgst/models/temporal.py` | Tested; irregular dates, missing-date masks, pixel-dense |
| 8b Physics fusion | `pmgst/models/physics.py` | Tested; differentiable factor of safety, FiLM |
| 9 Memory | `pmgst/models/memory.py` | Tested incl. leakage control; **off by default** (P2) |
| 11 Evidential decoder | `pmgst/models/heads.py` | Tested; Beta evidence → p, vacuity u |
| 12 Risk / warning | `pmgst/infer.py` | Tested; calibrated maps, objects, levels, alerts |
| 13 XAI | `pmgst/xai.py` | Written; attention maps, Grad-CAM, IG, Shapley groups |
| 14 Multi-objective loss | `pmgst/losses.py` | Tested; focal+Dice, physics, SupCon, EDL |
| 15 Deployment | `pmgst/serve.py` | Minimal API; full Phase 3 loop is `11_…` |
| Baselines B0-B2 / B3-B5 | `pmgst/baselines.py` / `pmgst/models/baselines_dl.py` | Tested; B6 foundation model not built |

Deliberate departures from the diagram: the graph runs over **chips in a spatial cluster** (not 5 km weekly nodes); detection is **pixel-level at 10 m**; there is no block 10 (as in your diagram). Everything else follows `05`.

## 2. Setup on your PC (Windows, GPU)

```bat
conda activate landslide
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```
Must print `True`. If it prints `False`, install the CUDA build from pytorch.org (pick your CUDA version) into that env.

```bat
cd E:\major\EarthSwin
pip install pyyaml scipy scikit-learn pytest tqdm
:: only for the real-data pipeline:
pip install rasterio geopandas shapely pyproj earthengine-api
```

## 3. Verify everything works (10 minutes, no real data)

```bat
python -m pytest tests -q                       :: 44 tests, CPU, ~1 min
python -m pmgst.profile_gpu --config configs/rtx3060.yaml
```
`profile_gpu` prints the peak memory per batch size on **your** card. Use the largest batch that fits for `train.batch`
and raise `train.accum` so that `batch x accum` stays about 16. If even batch 4 fails: `--set model.ckpt=true`, then lower `data.chip` to 96 or `data.t_max` to 6.

Then train the full-size model on synthetic data to see real speed and memory:
```bat
python -m pmgst.data.synth --out data/synth128 --chips 16 --chip 128 --ctx 64
python -m pmgst.train --config configs/rtx3060_synth.yaml
```

## 4. Real data (the part that decides everything)

1. **Labels** (see `03` Appendix A): get a polygon inventory of 2023 landslides + the footprint of the area it covers.
2. **Export** from Earth Engine (needs a Google account with Earth Engine and a Cloud project registered for it). Nothing is downloaded from a website:
   Earth Engine writes GeoTIFFs to your Google Drive folder, and you download that folder.
   ```bat
   pip install earthengine-api pyproj rasterio geopandas shapely
   earthengine authenticate
   :: pilot: one central tile (~0.2 GB), checks dataset IDs and bands before the big job
   python -m pmgst.data.gee_export --project YOUR_GEE_PROJECT --bbox 76.5 31.4 77.9 32.4 --year 2023 --folder pmgst_2023 --max-tiles 1
   :: full job: ONLY tiles that touch the labelled area (about 35 tiles = 7-8 GB for ~3,100 km2)
   python -m pmgst.data.gee_export --project YOUR_GEE_PROJECT --extent labels/extent.gpkg --year 2023 --folder pmgst_2023
   ```
   Do **not** export the whole Mandi+Kullu bounding box: it is 168 tiles (1,008 tasks, ~37 GB) and most of it has no labels.
   Watch progress at https://code.earthengine.google.com/tasks, then download the Drive folder into `data\raw\`.
3. **Chips:**
   ```bat
   python -m pmgst.data.chips --raw data/raw --out data/chips --labels labels/inventory.gpkg --label-extent labels/extent.gpkg --chip 128 --ctx 64
   ```
   Prints chips, chips with landslides, and positive-pixel %. Open 20 chips and overlay labels on post-event S2 (QA checklist in `04` §12).
4. **Baselines first** (so you know what "good" is): `python -m pmgst.baselines --config configs/rtx3060.yaml`
5. **Train:** `python -m pmgst.train --config configs/rtx3060.yaml`
6. **Evaluate (with CIs):** `python -m pmgst.evaluate --ckpt runs/pmgst_d_v1/best.pt --split test`
7. **Maps + warnings:** `python -m pmgst.infer --ckpt runs/pmgst_d_v1/best.pt --split test --out outputs/v1`
   → `probability.tif`, `uncertainty.tif`, `landslides.geojson`, `warnings.csv`, `alerts.json`.

## 5. Ablations (each block must earn its place)

Same split, same seed; change one switch:
```bat
python -m pmgst.train --config configs/rtx3060.yaml --set train.out_dir=runs/abl_nograph  model.use_graph=false
python -m pmgst.train --config configs/rtx3060.yaml --set train.out_dir=runs/abl_nophys   model.use_physics=false
python -m pmgst.train --config configs/rtx3060.yaml --set train.out_dir=runs/abl_nopyr    model.use_pyramid=false
python -m pmgst.train --config configs/rtx3060.yaml --set train.out_dir=runs/abl_s2only   model.use_s1=false
python -m pmgst.train --config configs/rtx3060.yaml --set train.out_dir=runs/abl_s1only   model.use_s2=false
python -m pmgst.train --config configs/rtx3060.yaml --set train.out_dir=runs/abl_mem      model.use_memory=true
python -m pmgst.train --config configs/rtx3060.yaml --set train.out_dir=runs/abl_logit    model.head=logit loss.lam_unc=0
```
Keep a block only if val AP / F1 improves by about +1 point or robustness improves (docs `03` §3.3, `07`).
Loss terms: `--set loss.lam_phys=0` / `loss.lam_con=0` / `loss.lam_unc=0`.

## 5b. Deep baselines B3-B5 (docs/07) and evaluation modes

Same data, same recipe, same evaluation. Only the architecture changes (`models/baselines_dl.py`):
```bat
python -m pmgst.train --config configs/rtx3060.yaml --set model.arch=unet  train.out_dir=runs/b3_unet   :: post S2 + DEM
python -m pmgst.train --config configs/rtx3060.yaml --set model.arch=siam  train.out_dir=runs/b4_siam   :: pre/post S2 + S1 stats + DEM
python -m pmgst.train --config configs/rtx3060.yaml --set model.arch=utae  train.out_dir=runs/b5_utae   :: + S1 time series, temporal attention
```
`evaluate` works on all of them (`python -m pmgst.evaluate --ckpt runs/b3_unet/best.pt`). Phase 1 passes only if PMGST-Net-D beats the best of these
by the margin in `02` section 7 (+3 F1, or equal F1 with better calibration and cloud robustness). B6 (foundation model) is not built.

Training now logs `grad_norm_med` / `grad_norm_max` per epoch in `metrics.jsonl` and prints `[warn] gradient norm ...` if it exceeds 1e4.
That warning means a layer is exploding and the global clip is freezing every other weight (a flat loss and a constant val AP are the symptoms).
It happened once: the siamese baseline's optical encoder on cloudy/dropped chips (fixed by masking invalid S2 features; regression test in `tests/test_baselines.py`).

Cloud-robust metric (no optical at all): `python -m pmgst.evaluate --ckpt runs/pmgst_d_v1/best.pt --drop s2`. `--drop s1` removes radar.

Unseen-district test (docs/07, "unseen-district F1 drop"): build chips per district with `--region mandi` / `--region kullu`, put both in one index, then
`--set data.holdout_region=kullu` at train time. All Kullu chips become the test set, train/val come from Mandi only (buffered).

Hard negatives: the sampler seeds 50% of clusters on landslide chips, 20% on steep landslide-free chips (slope > 15 deg on at least 20% of the chip), the rest random.
`train.hard_frac=0` turns it off. The steepness table is cached as `hard_neg_15_20.json` in the chip folder.

## 6. Data format (what the model eats)

One `.npz` per chip, documented at the top of `pmgst/data/formats.py`. Physical units on disk; the model normalises.
`index.csv` lists chips with UTM centre, spatial block and positive-pixel count. Splits are spatial blocks (10 km) with a buffer; `data.test_fold` / `data.val_fold` pick the folds.

## 7. How training works (so you can debug it)

* One step = one **spatial cluster** of `train.batch` chips = one graph. Neighbouring chips are always in the same batch, so graph attention has real neighbours. Half the clusters are seeded on landslide chips.
* Validation sweeps clusters that cover every chip once. Final inference (`infer`) uses **one graph over the whole region** (two passes: embed all chips → GAT once → decode).
* Early stopping and the best checkpoint use val **AP (PR-AUC)**, not accuracy.
* After training, a slope+bias calibration (Platt) is fitted on val and stored in the checkpoint; the F1-optimal threshold is stored too. `infer` applies both.
* bf16 autocast on RTX 30-series (no loss scaling needed). GAT and losses run in fp32.

## 8. Honest status — read this before trusting any number

* **Everything above is verified on synthetic data only** (44 tests, CPU). On real Himachal data nothing has been trained yet, so there is no accuracy claim.
* The synthetic benchmark proves the code learns and that blocks respond sensibly (see §9), not that they help on real terrain.
* **Not yet measured:** GPU memory/speed on your RTX 3060 (run `profile_gpu`); `gee_export.py` against the live Earth Engine API (dataset IDs may need a tweak); `xai.py` on a trained real model.
* Known simplifications: S1 layover/shadow not masked (needs local incidence angle); no S1 terrain flattening; lithology = soil-texture class until you get GSI geology; roads optional; normalisation constants are fixed (`models/pmgst_net.py`) rather than fitted on training folds — replace once real data exists.
* Phase 2 (early warning, causal, forecast horizon) and Phase 3 (self-training) are **not** implemented; they need dated events and a trained detector first (`11_…`).

## 9. Synthetic sanity benchmark (reference, CPU, 144 chips, 22 epochs)

Tiny model (1.3 M params, `configs/smoke.yaml`), one seed, scored on the **val** split (checkpoint picked on the same split, so slightly optimistic).
Baselines are fitted on train and scored on the same val split.

| Run | val AP | object F1 |
|---|---|---|
| B0 slope threshold | 0.022 | |
| B1 dNDVI threshold | 0.239 | |
| B1 S1 backscatter change | 0.090 | |
| B2 gradient boosting (all features) | 0.414 | |
| B3 U-Net (post S2 + DEM) | 0.687 | 0.592 |
| B4 siamese (pre/post S2 + S1 stats + DEM) | 0.581 | 0.667 |
| B5 U-TAE-lite (+ S1 time series) | 0.618 | 0.529 |
| PMGST full | 0.594 | 0.596 |
| no physics | 0.580 | 0.588 |
| no graph | 0.586 | 0.576 |
| no pyramid | 0.632 | 0.603 |
| S2 only | 0.640 | 0.655 |
| S1 only | 0.097 | 0.186 |

How to read it: PMGST beats the cheap baselines B0-B2 by a wide margin, but **it does not beat the plain U-Net (B3) here**.
All the deep models sit between 0.58 and 0.69 AP, and the gaps among them (about 0.05-0.10) are within what one seed on a 30-chip val split can move, so this is not a ranking.
What it does say: the extra blocks have not earned their place on synthetic data, and a bigger model alone is not an argument.
The real test is section 5 and 5b on real Himachal chips: 3 seeds each, block-bootstrap CIs (`pmgst.evaluate`), same split.
On the GPU (full-size model, synth128) run B3 next to PMGST before reading anything into either number.
