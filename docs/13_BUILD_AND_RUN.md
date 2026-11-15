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
python -m pytest tests -q                       :: 32 tests, CPU, ~1 min
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
2. **Export** from Earth Engine (pilot = Mandi + Kullu first, where the best inventory is):
   ```bat
   python -m pmgst.data.gee_export --project YOUR_GEE_PROJECT --bbox 76.5 31.4 77.9 32.4 --year 2023 --folder pmgst_2023 --dry-run
   python -m pmgst.data.gee_export --project YOUR_GEE_PROJECT --bbox 76.5 31.4 77.9 32.4 --year 2023 --folder pmgst_2023
   ```
   Download the Drive folder into `data\raw\`. The bbox is approximate: adjust it. Size is roughly 10–12 GB for both districts.
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

* **Everything above is verified on synthetic data only** (32 tests, CPU). On real Himachal data nothing has been trained yet, so there is no accuracy claim.
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
| PMGST full | 0.594 | 0.596 |
| no physics | 0.580 | 0.588 |
| no graph | 0.586 | 0.576 |
| no pyramid | 0.632 | 0.603 |
| S2 only | 0.640 | 0.655 |
| S1 only | 0.097 | 0.186 |

How to read it: the model clearly beats every cheap baseline, and S1 alone is weak, which is what the synthetic world was built to do.
The differences between full / no-physics / no-graph / no-pyramid / S2-only are about 0.05 AP, which is inside what one seed on a 30-chip val split can move.
**So this table does not show that pyramid, graph or physics help.** On synthetic data it only shows the switches work and nothing is broken.
The real ablation is section 5 on real Himachal chips, with 3 seeds and block-bootstrap CIs (`pmgst.evaluate`).

