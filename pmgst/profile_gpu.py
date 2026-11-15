"""Find the largest batch that fits your GPU (run this FIRST on the RTX 3060).

  python -m pmgst.profile_gpu --config configs/rtx3060.yaml
  python -m pmgst.profile_gpu --config configs/rtx3060.yaml --set model.ckpt=true     # gradient checkpointing
Uses random tensors in the real shapes; no data needed.
"""
from __future__ import annotations

import argparse
import time

import torch

from .config import load_config
from .losses import PMGSTLoss
from .models.build import build_model
from .train import amp_setup, pick_device


def fake_batch(cfg, B, device):
    d = cfg.data; H = d.chip; h = d.ctx; S1 = d.n_scales - 1; T = d.t_max
    g = lambda *s: torch.randn(*s, device=device)
    b = dict(
        s2=torch.rand(B, 2, d.s2_bands, H, H, device=device) * 0.3, s2_valid=torch.ones(B, 2, 1, H, H, device=device),
        s1=g(B, T, 2, H, H) * 3 - 12, s1_valid=torch.ones(B, T, 1, H, H, device=device),
        s1_dt=torch.arange(T, device=device).float().mul(12).expand(B, T).contiguous(), s1_pad=torch.zeros(B, T, dtype=torch.bool, device=device),
        dem=torch.cat([g(B, 1, H, H) * 800 + 2500, g(B, 1, H, H).abs() * 20 + 5, g(B, 2, H, H) * 0.7, g(B, 2, H, H) * 0.05,
                       g(B, 1, H, H) * 3 + 8, g(B, 1, H, H).abs() * 100], 1),
        lith=torch.randint(0, d.n_lith, (B, H, H), device=device), water=torch.zeros(B, 1, H, H, device=device),
        rain=torch.rand(B, d.n_rain, device=device) * 100,
        ctx_s2=torch.rand(B, S1, 2, d.s2_bands, h, h, device=device) * 0.3, ctx_valid=torch.ones(B, S1, 2, 1, h, h, device=device),
        ctx_dem=torch.cat([g(B, S1, 1, h, h) * 800 + 2500, g(B, S1, 1, h, h).abs() * 20 + 5, g(B, S1, 2, h, h) * 0.7,
                           g(B, S1, 2, h, h) * 0.05, g(B, S1, 1, h, h) * 3 + 8, g(B, S1, 1, h, h).abs() * 100], 2),
        label=(torch.rand(B, H, H, device=device) > 0.98).long(), meta=torch.zeros(B, 10, device=device))
    b["meta"][:, 0] = 7e5 + torch.arange(B, device=device) * H * 10.0
    b["meta"][:, 1] = 3.5e6; b["meta"][:, 2] = 2500; b["meta"][:, 5] = 1
    return b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=None); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--max", type=int, default=32)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set); device = pick_device(cfg); _, dtype, _ = amp_setup(cfg, device)
    model = build_model(cfg).to(device); model.train(); crit = PMGSTLoss(cfg.loss)
    print(f"device {device} | params {model.n_params() / 1e6:.1f}M | chip {cfg.data.chip} ctx {cfg.data.ctx} T {cfg.data.t_max} | ckpt {cfg.model.ckpt}")
    B, ok = 1, 0
    while B <= a.max:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            b = fake_batch(cfg, B, device); t = time.time()
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                out = model(b)
            loss, _ = crit(out, b, 5); loss.backward(); model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
                print(f"batch {B:3d}: OK  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB  {time.time() - t:.2f}s/step")
            else:
                print(f"batch {B:3d}: OK  {time.time() - t:.2f}s/step (cpu)")
            ok = B; B *= 2
        except torch.cuda.OutOfMemoryError:
            print(f"batch {B:3d}: OUT OF MEMORY"); break
    print(f"=> largest working batch: {ok}. Use train.batch <= {ok} (and raise train.accum to keep ~16 chips per update). "
          f"If {ok} < 4 set model.ckpt=true or lower data.chip / data.t_max.")


if __name__ == "__main__":
    main()
