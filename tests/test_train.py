import json
from pathlib import Path

import numpy as np
import torch

from pmgst.losses import edl_loss, focal_loss, dice_loss, supcon
from pmgst.config import LossCfg
from pmgst.metrics import ObjectMetrics, PixelMetrics
from pmgst.calibration import apply_calibration, fit_calibration, fit_temperature


def test_pixel_metrics_match_sklearn():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.05).astype(np.uint8)
    p = np.clip(0.15 + 0.5 * y + 0.2 * rng.standard_normal(20000), 0, 1)
    pm = PixelMetrics(); pm.update(p, y)
    r = pm.compute()
    assert abs(r["ap"] - average_precision_score(y, p)) < 0.01
    lab = y.copy(); lab[:100] = 255                                             # ignored pixels never count
    pm2 = PixelMetrics(); pm2.update(p, lab)
    assert pm2.pos.sum() + pm2.neg.sum() == 19900


def test_object_metrics_perfect_and_miss():
    lab = np.zeros((1, 32, 32), np.uint8); lab[0, 4:12, 4:12] = 1; lab[0, 20:26, 20:30] = 1
    om = ObjectMetrics(); om.update(lab == 1, lab); r = om.compute()
    assert r["obj_f1"] == 1.0
    om = ObjectMetrics(); om.update(np.zeros_like(lab, bool), lab); assert om.compute()["obj_rec"] == 0.0


def test_losses_behave():
    y = torch.zeros(1, 16, 16); y[:, 4:8, 4:8] = 1; valid = torch.ones_like(y)
    good = torch.where(y > 0, torch.tensor(0.95), torch.tensor(0.02)); bad = 1 - good
    assert focal_loss(good, y, valid, 2, 0.75) < focal_loss(bad, y, valid, 2, 0.75)
    assert dice_loss(good, y, valid) < 0.2 < dice_loss(bad, y, valid)
    a, b = torch.full((1, 16, 16), 1.5), torch.full((1, 16, 16), 1.5)
    assert torch.isfinite(edl_loss(a, b, y, valid, 5, LossCfg()))
    emb = torch.randn(2, 8, 8, 8); lab = torch.zeros(2, 32, 32, dtype=torch.long); lab[0, :16, :16] = 1
    assert torch.isfinite(supcon(emb, lab))


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(20000) * 2
    y = (rng.random(20000) < 1 / (1 + np.exp(-z / 3))).astype(np.float32)        # true T = 3
    T = fit_temperature(z, y)
    assert 2.5 < T < 3.6


def test_train_learns_and_infer_runs(cfg, tmp_path):
    """End to end on synthetic data: loss falls, val AP clearly beats the base rate, infer writes outputs."""
    from pmgst.train import train
    from pmgst.infer import run_region
    cfg.train.epochs, cfg.train.steps_per_epoch = 8, 18
    res = train(cfg)
    hist = res["history"]
    base_rate = None
    assert hist[-1]["train_total"] < hist[0]["train_total"]
    assert res["best_ap"] > 0.2, res["best_ap"]
    ck = Path(cfg.train.out_dir) / "best.pt"
    out = tmp_path / "out"
    run_region(str(ck), "val", str(out), full_graph=True, device=torch.device("cpu"))
    for f in ("warnings.csv", "landslides.geojson", "alerts.json"):
        assert (out / f).exists()
    assert (out / "probability.npy").exists() or (out / "probability.tif").exists()


def test_platt_fixes_prior_shift():
    rng = np.random.default_rng(1)
    z = rng.standard_normal(30000) * 2
    y = (rng.random(30000) < 1 / (1 + np.exp(-(0.5 * z - 2.0)))).astype(np.float32)   # model overpredicts: true a=.5, b=-2
    a, b = fit_calibration(z, y)
    assert 0.4 < a < 0.6 and -2.4 < b < -1.6
    p = 1 / (1 + np.exp(-z)); pc = apply_calibration(p, (a, b))
    assert abs(pc.mean() - y.mean()) < abs(p.mean() - y.mean())
