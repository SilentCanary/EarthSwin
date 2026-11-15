"""QA tool, geology rasterisation, SAR geometry mask, disk-saving options, fitted normalisation, ablation runner,
checkpoint ensembles and the Landslide4Sense adapter."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pmgst.config import load_config
from pmgst.data.formats import load_chip
from pmgst.data.terrain import sar_geometry_mask

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ SAR geometry
def test_sar_mask_layover_shadow_flat():
    # radar looks east (az 90): the sensor is to the WEST, so slopes facing west (downslope = west) face the sensor
    one = np.ones((1, 1))
    f = lambda slope, ax, ay: bool(sar_geometry_mask(one * slope, one * ax, one * ay, 90.0, inc_deg=39.0)[0, 0])
    assert f(60, -1, 0) is True            # steep slope facing the sensor -> layover
    assert f(60, +1, 0) is True            # steep slope facing away (60 > 90-39) -> shadow
    assert f(10, -1, 0) is False and f(10, +1, 0) is False and f(0, 0, 0) is False
    assert f(60, 0, 1) is False            # slope facing north: side-on to the look direction, neither


# ------------------------------------------------------------------ chips: geology, compression, CLI
from test_real_pipeline import tile, X0, Y1  # noqa: E402,F401  (fixture: a fabricated GEE tile + polygons)


def test_geology_and_compress(tile, tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box
    from pmgst.data import chips as C
    raw, gdf, R = tile
    geo = gpd.GeoDataFrame({"rock": ["granite", "slate", "granite"]},
                           geometry=[box(X0, Y1 - 5120, X0 + 2560, Y1), box(X0 + 2560, Y1 - 5120, X0 + 4000, Y1), box(X0 + 4000, Y1 - 5120, X0 + 5120, Y1)],
                           crs="EPSG:32643")
    table = C.lith_table(geo, "rock", 13)
    assert table == {"granite": 1, "slate": 2}                                  # most extensive class first
    out = tmp_path / "c"; out.mkdir()
    args = (raw, X0 + 1280, Y1 - 1280, gdf, None, None, 64, 32, 3, 1280.0, 1.0, np.random.default_rng(0))
    rows = C.build_tile(*args, out, "t", geo, "rock", table, "ASCENDING", 39.0, True)
    ids = set(np.unique(np.concatenate([load_chip(out / f"{r['chip_id']}.npz")["lith"].ravel() for r in rows])).tolist())
    assert ids == {1, 2}
    out2 = tmp_path / "plain"; out2.mkdir()
    rows2 = C.build_tile(*args, out2, "t", None, None, None, None, 39.0, False)  # no SAR mask, no geology
    s_masked = load_chip(out / f"{rows[0]['chip_id']}.npz"); s_plain = load_chip(out2 / f"{rows2[0]['chip_id']}.npz")
    assert s_masked["s1_valid"].sum() <= s_plain["s1_valid"].sum()              # layover/shadow only ever removes pixels
    assert (out / f"{rows[0]['chip_id']}.npz").stat().st_size < (out2 / f"{rows2[0]['chip_id']}.npz").stat().st_size


def test_chips_cli_delete_raw(tile, tmp_path):
    gpd = pytest.importorskip("geopandas")
    raw, gdf, R = tile
    import shutil
    raw2 = tmp_path / "raw"; shutil.copytree(raw, raw2)
    gdf.to_file(tmp_path / "labels.geojson", driver="GeoJSON")
    out = tmp_path / "chips"
    r = subprocess.run([sys.executable, "-m", "pmgst.data.chips", "--raw", str(raw2), "--out", str(out), "--labels", str(tmp_path / "labels.geojson"),
                        "--chip", "64", "--ctx", "32", "--buffer", "1280", "--compress", "--delete-raw"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-800:]
    assert (out / "index.csv").exists() and not list(raw2.glob("*.tif"))        # raw tiles removed once chipped


def test_chips_cli_append(tile, tmp_path):
    pytest.importorskip("geopandas")
    import shutil
    from pmgst.data.formats import read_index
    raw, gdf, R = tile
    gdf.to_file(tmp_path / "labels.geojson", driver="GeoJSON")
    out = tmp_path / "chips"
    cmd = lambda raw_dir, *extra: [sys.executable, "-m", "pmgst.data.chips", "--raw", str(raw_dir), "--out", str(out), "--labels", str(tmp_path / "labels.geojson"),
                                   "--chip", "64", "--ctx", "32", "--buffer", "1280", "--region", "r", *extra]
    for k in range(2):                                                          # same tile twice: second run must not duplicate rows
        raw_k = tmp_path / f"raw{k}"; shutil.copytree(raw, raw_k)
        r = subprocess.run(cmd(raw_k, "--append"), capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-600:]
    rows = read_index(out / "index.csv")
    assert len(rows) == len({r["chip_id"] for r in rows}) == 16


# ------------------------------------------------------------------ QA
def test_qa_clean_and_corrupt(synth_dir, tmp_path):
    import shutil
    from pmgst.data.qa import run_qa
    d = tmp_path / "s"; shutil.copytree(synth_dir, d)
    rep = run_qa(d, png=str(tmp_path / "qa.png"))
    assert rep["errors"] == [] and rep["chips_with_landslides"] > 0
    assert rep["alignment"]["slope_pos_minus_bg_deg"] > 0 and rep["alignment"]["dndvi_pos_minus_bg"] < 0   # synthetic scars are steep + bare
    assert (tmp_path / "qa.png").exists() and (d / "qa_report.json").exists()
    f = sorted((d / "chips").glob("*.npz"))[0]
    z = dict(np.load(f)); z["dem"] = z["dem"].astype(np.float32); z["dem"][0, 0, 0] = np.nan; z["label"][0, 0] = 7
    np.savez(f, **z)
    rep = run_qa(d)
    assert any("NaN" in e for e in rep["errors"]) and any("label values" in e for e in rep["errors"])


# ------------------------------------------------------------------ fitted normalisation
def test_fit_norm_lands_in_checkpoint(cfg):
    import torch
    from pmgst.train import train
    cfg.train.epochs = 1; cfg.train.steps_per_epoch = 3; cfg.data.fit_norm = True
    train(cfg)
    ck = torch.load(Path(cfg.train.out_dir) / "best.pt", map_location="cpu", weights_only=False)
    mu = ck["model"]["dem_mu"].flatten().numpy()
    assert (Path(cfg.train.out_dir) / "norm_stats.json").exists()
    assert abs(mu[0] - 2500.0) > 1.0 and mu[2] == 0.0 and mu[3] == 0.0           # elevation refitted; aspect means stay 0


# ------------------------------------------------------------------ ablation runner + ensemble
def test_ablate_runner_and_ensemble(cfg, tmp_path):
    from pmgst import ablate
    from pmgst.evaluate import evaluate_ckpt
    assert ablate.plan(["full", "b3_unet"], 2) == [("full", 0), ("full", 1), ("b3_unet", 0), ("b3_unet", 1)]
    base = [f"data.root={cfg.data.root}", "train.steps_per_epoch=3", "train.epochs=1"]
    res = {}
    for name in ("full", "b3_unet"):
        res[name] = [ablate.run_one(ROOT / "configs" / "smoke.yaml", base, name, s, tmp_path / "abl") for s in (0, 1)]
    md = ablate.summarise(res)
    assert "| full | 2 |" in md and "| b3_unet | 2 |" in md and "±" in md
    again = ablate.run_one(ROOT / "configs" / "smoke.yaml", base, "full", 0, tmp_path / "abl")   # resumable: reads the stored result
    assert again["calibrated"]["ap"] == res["full"][0]["calibrated"]["ap"]
    ens = evaluate_ckpt(f"{tmp_path / 'abl' / 'full_s0' / 'best.pt'},{tmp_path / 'abl' / 'full_s1' / 'best.pt'}", "test")
    assert ens["n_models"] == 2 and 0.0 <= ens["calibrated"]["ap"] <= 1.0
    assert ens["n_models"] == 2 and 0.0 <= ens["calibrated"]["ap"] <= 1.0


# ------------------------------------------------------------------ Landslide4Sense adapter
def test_l4s_adapter_and_training(tmp_path):
    h5py = pytest.importorskip("h5py")
    from pmgst.data import l4s
    src = tmp_path / "TrainData"; (src / "img").mkdir(parents=True); (src / "mask").mkdir()
    rng = np.random.default_rng(0)
    for i in range(1, 81):
        img = rng.uniform(300, 2500, (64, 64, 14)).astype(np.float32)           # raw DN-like reflectance*10000
        yy, xx = np.mgrid[:64, :64]
        m = ((yy - 32) ** 2 + (xx - 32) ** 2 < 100).astype(np.uint8) if i % 3 else np.zeros((64, 64), np.uint8)
        img[..., 12] = 20 + 10 * m + rng.normal(0, 2, (64, 64)); img[..., 13] = 2000 + 3 * xx   # slope, DEM
        img[..., 3][m > 0] += 1500                                              # something to learn
        with h5py.File(src / "img" / f"image_{i}.h5", "w") as f:
            f.create_dataset("img", data=img)
        with h5py.File(src / "mask" / f"mask_{i}.h5", "w") as f:
            f.create_dataset("mask", data=m)
    out = tmp_path / "chips"
    l4s.convert(src, out, compress=True)
    c = load_chip(out / "chips" / "l4s_00001.npz")
    assert c["s2"].shape == (2, 6, 64, 64) and c["s2_valid"][0].sum() == 0 and c["s2_valid"][1].all()   # pre = missing
    assert c["s2"].max() <= 1.0 and c["s2_valid"].dtype == np.uint8 and c["dem"].shape == (8, 64, 64)
    assert abs(float(c["dem"][0].mean()) - 2000) < 200 and c["dem"][1].mean() < 60      # DEM/slope not swapped
    cfg = load_config(ROOT / "configs" / "smoke.yaml", [f"data.root={out}", "model.use_s1=false", "model.use_pyramid=false",
                                                       "model.use_graph=false", "data.fit_norm=true", "train.epochs=3",
                                                       "train.steps_per_epoch=6", "data.buffer_m=1280", f"train.out_dir={tmp_path / 'run'}"])
    from pmgst.train import train
    r = train(cfg)
    assert np.isfinite(r["best_ap"])
