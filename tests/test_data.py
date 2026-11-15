import numpy as np

from pmgst.data.dataset import ChipDataset, ClusterBatchSampler, collate, d4_apply, sweep_clusters
from pmgst.data.formats import IDX_ASPX, IDX_ASPY, IDX_SLOPE, load_chip
from pmgst.data.splits import assign_folds, block_ids, make_split
from pmgst.data.terrain import terrain_channels


def test_chip_format_and_labels(synth_dir):
    ds = ChipDataset.__new__(ChipDataset)
    from pmgst.data.formats import read_index
    rows = read_index(synth_dir / "index.csv")
    assert len(rows) == 36
    c = load_chip(synth_dir / rows[0]["path"])
    assert c["s2"].shape == (2, 6, 64, 64) and c["dem"].shape == (8, 64, 64) and c["ctx_s2"].shape == (2, 2, 6, 32, 32)
    assert set(np.unique(c["label"])) <= {0, 1, 255}
    assert sum(r["n_pos_px"] for r in rows) > 100             # synthetic region has landslides


def test_d4_aspect_equivariance():
    """aspect channels after D4 must equal the aspect recomputed from the transformed DEM."""
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    z = gaussian_filter(rng.standard_normal((80, 80)), 6) * 400
    dem = terrain_channels(z)
    H = 80
    base = dict(label=np.zeros((H, H), np.uint8), lith=np.zeros((H, H), np.uint8), water=np.zeros((H, H), np.uint8),
                s2=np.zeros((2, 6, H, H)), s2_valid=np.zeros((2, 1, H, H)), s1=np.zeros((3, 2, H, H)),
                s1_valid=np.zeros((3, 1, H, H)), dem=dem, ctx_s2=np.zeros((1, 2, 6, 8, 8)), ctx_valid=np.zeros((1, 2, 1, 8, 8)),
                ctx_dem=np.zeros((1, 8, 8, 8)))
    for k in range(4):
        for flip in (False, True):
            t = d4_apply(base, k, flip)
            zz = z[:, ::-1] if flip else z
            zz = np.rot90(zz, k)
            ref = terrain_channels(np.ascontiguousarray(zz))
            sl = (slice(None), slice(3, -3), slice(3, -3))      # ignore borders (finite-difference edge effects)
            m = ref[IDX_SLOPE][3:-3, 3:-3] > 8
            for ch in (IDX_ASPX, IDX_ASPY):
                assert np.allclose(t["dem"][ch][3:-3, 3:-3][m], ref[ch][3:-3, 3:-3][m], atol=2e-2), (k, flip, ch)


def test_splits_no_leakage(synth_dir):
    from pmgst.data.formats import read_index
    rows = read_index(synth_dir / "index.csv")
    x = np.array([r["x"] for r in rows]); y = np.array([r["y"] for r in rows])
    blk = block_ids(x, y, 1280.0); npos = np.array([r["n_pos_px"] for r in rows])
    fold = assign_folds(blk, npos, 5, 0)
    assert set(np.unique(fold)) <= set(range(5))
    for b in np.unique(blk):                                   # a block never spans two folds
        assert len(set(fold[blk == b])) == 1
    tr, va, te = make_split(x, y, fold, 0, 1, buffer_m=1280.0, chip_m=640.0)
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    held = np.where(va | te)[0]
    for i in np.where(tr)[0]:
        d = np.max(np.abs(np.stack([x[held] - x[i], y[held] - y[i]], 1)), axis=1).min()
        assert d >= 640.0 + 1280.0 - 1e-6


def test_dataset_batch_and_sampler(cfg):
    cfg.data.test_fold, cfg.data.val_fold = 0, 1
    ds = ChipDataset(cfg, "all")
    s = ClusterBatchSampler(ds, 4, 5, 0.5, 0)
    idx = next(iter(s))
    assert len(idx) == 4 and len(set(idx)) == 4
    d = np.hypot(*(ds.xy[idx] - ds.xy[idx[0]]).T)
    assert d.max() <= 3 * 640.0 * 1.5                          # a spatial neighbourhood, not random chips
    b = collate([ds[i] for i in idx])
    assert b["s2"].shape == (4, 2, 6, 64, 64) and b["s1"].shape[1] == cfg.data.t_max and b["label"].dtype.is_floating_point is False
    cover = sum(sweep_clusters(ds, 4), [])
    assert sorted(cover) == list(range(len(ds)))               # val/test sweep visits every chip exactly once
