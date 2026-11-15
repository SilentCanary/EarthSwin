# 14 — Real-data runbook (everything is coded; this is the order to run it)

All commands run from `E:\major\EarthSwin`. Nothing below needs new code. Install once:
`pip install rasterio geopandas shapely pyproj earthengine-api h5py matplotlib`

Disk is the constraint, so the pipeline is built to work in batches: export few tiles → download → chip → delete raw → repeat.
A chip is ~1-2 MB compressed; the 2023 Mandi+Kullu area (~35 tiles x 64 chips) is ~3-4 GB of chips, while the raw GeoTIFFs are ~7.6 GB.

## 0. Wait-for-labels path (no download needed): Landslide4Sense  (optional, 2.5 GB)
```
curl.exe -L -o data\l4s\TrainData.zip "https://zenodo.org/records/10463239/files/TrainData.zip?download=1"
tar -xf data\l4s\TrainData.zip -C data\l4s
python -m pmgst.data.l4s --src data\l4s\TrainData --inspect
python -m pmgst.data.l4s --src data\l4s\TrainData --out data\l4s_chips --compress
python -m pmgst.data.qa --root data\l4s_chips
python -m pmgst.train --config configs\l4s.yaml
python -m pmgst.evaluate --ckpt runs\l4s_pmgst\best.pt
```
Single post-event optical + DEM only, no Himachal, proxy spatial split: validates the training code on real scars, nothing more.
If `--inspect` shows the bands are not as assumed (docstring of `pmgst/data/l4s.py`), say so before trusting the numbers.

## 1. Labels arrive (Mandi/Kullu 2023 polygons)
Put the file in `labels\`. If you were not given the mapped-area boundary, make a fallback (a convex hull over-covers unmapped gaps, so prefer the official one):
```
python -c "import geopandas as g; d=g.read_file('labels/inventory.gpkg').to_crs(32643); g.GeoSeries([d.unary_union.convex_hull.buffer(500)],crs=32643).to_file('labels/extent.gpkg',driver='GPKG')"
```

## 2. Pilot: ONE tile end to end (do this before anything big)
```
earthengine authenticate
python -m pmgst.data.gee_export --project YOUR_GEE_PROJECT --extent labels\extent.gpkg --year 2023 --folder pmgst_2023 --max-tiles 1
```
Wait for the 6 Drive tasks (code.earthengine.google.com/tasks), download them into `data\raw\`, then:
```
python -m pmgst.data.chips --raw data\raw --out data\chips --labels labels\inventory.gpkg --label-extent labels\extent.gpkg --region mandi --compress --delete-raw --append
python -m pmgst.data.qa --root data\chips --png qa.png
```
Open `qa.png`: red outlines must sit on bare scars, cyan on steep slopes. `qa` prints an **alignment** line: landslide pixels must be steeper
than background (+) and show an NDVI drop (−). If either sign is wrong, stop: labels are shifted, in another CRS, or from another season.
Exit code 1 = a hard error (NaN, bad label values, S1 empty). Warnings about folds with < 5 landslide chips are expected with one tile.

Optional with geology polygons: `--lith-vector geology.gpkg --lith-col LITHO` on every chips run (the class table is kept in `lith_classes.json`).
Orbit: `--orbit` must equal the export's (`gee_export --orbit`, default ASCENDING). Layover/shadow is masked from the DEM; `--no-sar-mask` disables it.

## 3. Everything else: batches of tiles
```
python -m pmgst.data.gee_export --project YOUR_GEE_PROJECT --extent labels\extent.gpkg --year 2023 --folder pmgst_2023
```
Download ~5-8 tiles at a time (names `*_x<X0>_y<Y1>.tif`), run the `chips ... --append --delete-raw` line, repeat. `--append` merges `index.csv`.
When all tiles are in: `python -m pmgst.data.qa --root data\chips --png qa.png` again (folds need >= 5 landslide chips each).

## 4. Train, evaluate, compare
```
python -m pmgst.profile_gpu --config configs\rtx3060.yaml
python -m pmgst.train --config configs\rtx3060.yaml
python -m pmgst.evaluate --ckpt runs\pmgst_d_v1\best.pt --split test
python -m pmgst.evaluate --ckpt runs\pmgst_d_v1\best.pt --split test --drop s2          # cloud-robust: no optical at all
python -m pmgst.ablate --config configs\rtx3060.yaml --out runs\ablate --seeds 3        # core variants x 3 seeds, mean ± sd -> runs\ablate\ablation.md
python -m pmgst.evaluate --ckpt runs\ablate\full_s0\best.pt,runs\ablate\full_s1\best.pt,runs\ablate\full_s2\best.pt   # 3-seed ensemble
python -m pmgst.infer --ckpt runs\pmgst_d_v1\best.pt --split all --out outputs\mandi
```
`configs\rtx3060.yaml` has `data.fit_norm: true` (DEM normalisation fitted on the training chips, stored in the checkpoint).
`ablate` is resumable (a run with an `eval_test.json` is skipped). Time: PMGST ~13 s/epoch on 82 synthetic chips; real data will be ~20-50x more chips, plan hours, not minutes.

Unseen-district test (docs/07): chip Mandi and Kullu with different `--region`, then `--set data.holdout_region=kullu` at train time.

## 5. Reading the result honestly
* Phase-1 passes only if PMGST beats the best of B3/B4/B5 by the docs/07 margin (+3 F1, or equal F1 with better ECE and cloud robustness) and the `ablation.md` mean ± sd gaps exceed the seed spread.
* Report block-bootstrap CIs (`evaluate` prints them). One test fold with few landslide blocks gives wide intervals; say so.
* Everything before this step was verified on synthetic data. A drop from synthetic to real numbers is expected and is the real result.
