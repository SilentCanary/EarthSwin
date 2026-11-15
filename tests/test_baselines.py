import shutil

import numpy as np
import pytest
import torch

from pmgst.data.dataset import ChipDataset, ClusterBatchSampler, collate, hard_negatives, to_device
from pmgst.data.formats import read_index, write_index
from pmgst.losses import PMGSTLoss
from pmgst.models.build import build_model
from pmgst.infer import predict


@pytest.mark.parametrize("arch", ["unet", "siam", "utae"])
def test_baseline_forward_backward_and_region_inference(cfg, arch):
    """Baselines share the batch/output contract: loss + backward work, two-pass region inference works."""
    cfg.model.arch = arch
    cfg.data.test_fold, cfg.data.val_fold = 0, 1
    ds = ChipDataset(cfg, "all", augment=False)
    b = to_device(collate([ds[i] for i in range(4)]), torch.device("cpu"))
    model = build_model(cfg); model.train()
    out = model(b)
    assert out["p"].shape == (4, 64, 64) and torch.isfinite(out["p"]).all()
    loss, parts = PMGSTLoss(cfg.loss)(out, b, epoch=20)
    assert torch.isfinite(loss) and "fs" not in parts          # no physics term for baselines
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)      # nothing dead
    assert model.n_params() < build_model(_pmgst(cfg)).n_params()
    model.eval()
    p, u = predict(model, ds, cfg, torch.device("cpu"), None, full_graph=True)
    assert p.shape == (len(ds), 64, 64) and np.isfinite(p).all()


def _pmgst(cfg):
    import copy
    c = copy.deepcopy(cfg); c.model.arch = "pmgst"
    return c


def test_baseline_train_one_epoch(cfg):
    from pmgst.train import train
    cfg.model.arch = "siam"; cfg.train.epochs, cfg.train.steps_per_epoch = 1, 4
    res = train(cfg)
    assert np.isfinite(res["history"][-1]["train_total"])


def test_hard_negative_sampler(cfg):
    cfg.data.test_fold, cfg.data.val_fold = 0, 1
    ds = ChipDataset(cfg, "all", augment=False)
    hard = hard_negatives(ds, slope_deg=0.0, min_frac=0.0)       # every landslide-free chip
    assert set(hard.tolist()) == set(np.where(ds.n_pos == 0)[0].tolist())
    if len(hard) == 0:
        pytest.skip("synthetic region has no landslide-free chip")
    s = ClusterBatchSampler(ds, 4, 12, pos_frac=0.0, seed=1, hard_frac=1.0)
    s.hard_idx = hard
    assert all(idx[0] in set(hard.tolist()) for idx in s)
    s2 = ClusterBatchSampler(ds, 4, 12, pos_frac=1.0, seed=1, hard_frac=0.0)
    assert all(ds.n_pos[idx[0]] > 0 for idx in s2)


@pytest.mark.parametrize("drop", ["s2", "s1"])
def test_drop_modality_at_eval(cfg, drop):
    cfg.data.test_fold, cfg.data.val_fold = 0, 1
    ds = ChipDataset(cfg, "all", augment=False, drop=drop)
    it = ds[0]
    if drop == "s2":
        assert it["s2_valid"].sum() == 0 and it["ctx_valid"].sum() == 0 and not bool(it["s1_pad"].all())
    else:
        assert bool(it["s1_pad"].all()) and it["s2_valid"].sum() > 0


def test_region_holdout_split(cfg, tmp_path):
    root = tmp_path / "two_regions"
    shutil.copytree(cfg.data.root, root)
    rows = read_index(root / "index.csv")
    xs = np.array([r["x"] for r in rows]); cut = np.median(xs)
    for r in rows:
        r["region"] = "east" if r["x"] > cut else "west"
        r["block"] = int(r["x"] // 1280) * 100 + int(r["y"] // 1280)
    write_index(root / "index.csv", rows)
    cfg.data.root = str(root); cfg.data.holdout_region = "east"; cfg.data.buffer_m = 640.0
    tr = ChipDataset(cfg, "train", augment=False); va = ChipDataset(cfg, "val", augment=False)
    te = ChipDataset(cfg, "test", augment=False)
    assert len(tr) and len(va) and len(te)
    assert {r["region"] for r in te.rows} == {"east"}
    assert all(r["region"] == "west" for r in list(tr.rows) + list(va.rows))
    cfg.data.holdout_region = "nowhere"
    with pytest.raises(ValueError):
        ChipDataset(cfg, "test")
