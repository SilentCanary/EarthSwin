import itertools

import numpy as np
import pytest
import torch

from pmgst.config import load_config
from pmgst.data.dataset import ChipDataset, collate
from pmgst.losses import PMGSTLoss
from pmgst.models.encoders import HybridEncoder
from pmgst.models.fusion import ModalityFusion
from pmgst.models.gat import GATStack, scatter_softmax
from pmgst.models.graph import DynamicGraph, knn_edges
from pmgst.models.memory import MemoryBank
from pmgst.models.physics import PhysicsModule
from pmgst.models.pmgst_net import PMGSTNet
from pmgst.models.temporal import TemporalEncoder


def _batch(cfg, n=4, split="all"):
    ds = ChipDataset(cfg, split, augment=False)
    return collate([ds[i] for i in range(n)])


@pytest.mark.parametrize("size", [64, 32, 24 * 4])
def test_hybrid_encoder_shapes_any_scale(size):
    enc = HybridEncoder(7, [16, 32, 64, 128], [1, 1, 1, 1], [2, 2, 4, 4], 8)
    f = enc(torch.randn(2, 7, size, size))
    assert [x.shape[-1] for x in f] == [size // 4, size // 8, size // 16, size // 32]
    assert [x.shape[1] for x in f] == [16, 32, 64, 128]


def test_gat_keeps_node_features_without_edges():
    """audit C4: a node with no neighbours must keep its own evidence (self loop + residual)."""
    g = GATStack(16, layers=2, heads=2, edge_dim=7, drop=0.0)
    x = torch.randn(5, 16)
    loops = torch.arange(5)
    out, _ = g(x, torch.stack([loops, loops]), torch.zeros(5, 7), torch.zeros(5))
    assert torch.isfinite(out).all() and out.abs().sum() > 0
    assert torch.corrcoef(torch.stack([x.flatten(), out.detach().flatten()]))[0, 1] > 0.3


def test_scatter_softmax_sums_to_one():
    logit = torch.randn(10, 3); idx = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
    a = scatter_softmax(logit, idx, 4)
    s = torch.zeros(4, 3).index_add_(0, idx, a)
    assert torch.allclose(s, torch.ones(4, 3), atol=1e-5)


def test_graph_edges_use_centred_coordinates():
    xy = torch.tensor([[7e5, 3.5e6], [7e5 + 1000, 3.5e6], [7e5 + 2000, 3.5e6], [7e5 + 50000, 3.5e6]], dtype=torch.float32)
    src, dst = knn_edges(xy, 1)
    nn_of = {int(d): int(s) for s, d in zip(src, dst)}
    assert nn_of[0] == 1 and nn_of[2] == 1 and nn_of[3] == 2


def test_modality_fusion_masks_invalid():
    f = ModalityFusion(8, 3)
    feats = torch.randn(2, 3, 8, 4, 4)
    present = torch.ones(2, 3, 1, 4, 4); present[:, 0] = 0                     # modality 0 (e.g. clouded S2) invalid
    out, alpha = f(feats, present)
    assert alpha[:, 0].abs().max() == 0 and torch.allclose(alpha.sum(1), torch.ones(2, 1, 4, 4), atol=1e-5)
    feats2 = feats.clone(); feats2[:, 0] = 100.0                                # changing an invalid modality changes nothing
    assert torch.allclose(out, f(feats2, present)[0], atol=1e-5)
    present[:, :, :, 0, 0] = 0                                                  # a pixel with no valid modality: no NaN
    assert torch.isfinite(f(feats, present)[0]).all()


def test_temporal_masks_handle_missing():
    te = TemporalEncoder(32, heads=4, layers=1)
    low = torch.randn(2, 5, 32, 3, 3); dt = torch.arange(5).float().mul(12).expand(2, 5)
    pad = torch.zeros(2, 5, 3, 3, dtype=torch.bool); pad[0, 2:] = True; pad[1] = True   # sample 1 has no data at all
    a = te.masks(low, dt, pad)
    assert torch.isfinite(a).all()
    assert a[0, :, 2:].abs().max() == 0 and torch.allclose(a[0].sum(1), torch.ones(4, 3, 3), atol=1e-5)
    assert a[1].abs().max() == 0
    out = TemporalEncoder.apply(a, torch.randn(2, 5, 32, 12, 12))
    assert out.shape == (2, 32, 12, 12)


def test_physics_monotonic():
    ph = PhysicsModule(13, 4)
    dem = torch.zeros(1, 8, 4, 4); dem[:, 6] = 8.0
    lith = torch.full((1, 4, 4), 5, dtype=torch.long)
    outs = []
    for s in (15.0, 30.0, 45.0):
        d = dem.clone(); d[:, 1] = s
        outs.append(ph(d, lith, torch.zeros(1, 4))[1].mean())
    assert outs[0] > outs[1] > outs[2]                                          # steeper => less stable
    d = dem.clone(); d[:, 1] = 30.0
    wet = ph(d, lith, torch.full((1, 4), 300.0))[1].mean()
    dry = ph(d, lith, torch.zeros(1, 4))[1].mean()
    assert wet <= dry + 1e-6                                                    # more rain never raises the factor of safety


def test_memory_excludes_same_block():
    mem = MemoryBank(16, 16, size=8, topk=3)
    h = torch.randn(3, 16)
    mem.write(h, torch.ones(3), torch.tensor([7, 7, 8]))
    out, ymem, q = mem(h[:1], torch.tensor([7]))                                # entries of block 7 must be hidden
    assert torch.isfinite(out).all()
    sims_visible = (q @ mem.keys[:3].T)[0]
    out2, y2, _ = mem(h[:1], torch.tensor([99]))                                # no exclusion -> uses all three
    assert not torch.allclose(out, out2)


ABL = ["use_s2", "use_s1", "use_pyramid", "use_graph", "use_physics", "use_memory"]


@pytest.mark.parametrize("off", [None] + ABL)
def test_full_forward_backward_each_ablation(cfg, off):
    if off:
        setattr(cfg.model, off, off == "use_memory")
    else:
        cfg.model.use_memory = True
    model = PMGSTNet(cfg); model.train()
    b = _batch(cfg, 4)
    out = model(b)
    assert out["p"].shape == (4, 64, 64) and torch.isfinite(out["p"]).all() and (out["p"] >= 0).all() and (out["p"] <= 1).all()
    loss, parts = PMGSTLoss(cfg.loss)(out, b, epoch=3)
    assert torch.isfinite(loss)
    loss.backward()
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, bad[:3]


def test_logit_head_and_mc_dropout(cfg):
    cfg.model.head = "logit"; cfg.model.mc_dropout = 0.2
    model = PMGSTNet(cfg); out = model(_batch(cfg, 3))
    loss, parts = PMGSTLoss(cfg.loss)(out, _batch(cfg, 3), 0)
    assert torch.isfinite(loss) and "edl" not in parts


def test_s2_clouds_do_not_change_s1_only_prediction(cfg):
    """With S2 fully invalid, S2 pixel values must not matter (modality + skip masking)."""
    cfg.model.use_graph = False
    model = PMGSTNet(cfg).eval()
    b = _batch(cfg, 3)
    b["s2_valid"] = torch.zeros_like(b["s2_valid"]); b["ctx_valid"] = torch.zeros_like(b["ctx_valid"])
    with torch.no_grad():
        p1 = model(b)["p"]
        b2 = dict(b); b2["s2"] = torch.rand_like(b["s2"])
        p2 = model(b2)["p"]
    assert torch.allclose(p1, p2, atol=1e-4)


def test_region_two_pass_equals_cluster_pass_for_same_graph(cfg):
    """Two-pass inference with node_override must reproduce the single-pass output when the graph is the same."""
    model = PMGSTNet(cfg).eval()
    b = _batch(cfg, 5)
    with torch.no_grad():
        full = model(b)["p"]
        h0 = model.embed_only(b)
        ref, _ = model.refine_nodes(h0, b["meta"], model.lith_hist(b["lith"]), b["meta"][:, 8].long())
        two = model(b, node_override=ref)["p"]
    assert torch.allclose(full, two, atol=1e-5)
