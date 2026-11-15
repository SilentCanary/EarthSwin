"""Training loop. One step = one spatial cluster of chips (= one graph).

  python -m pmgst.train --config configs/rtx3060.yaml
  python -m pmgst.train --config configs/smoke.yaml --set train.epochs=3
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .calibration import fit_calibration
from .config import Config, load_config, to_dict
from .data.dataset import ChipDataset, ClusterBatchSampler, collate, sweep_clusters, to_device
from .losses import PMGSTLoss
from .metrics import ObjectMetrics, PixelMetrics
from .models.build import build_model


def _worker_init(wid: int):
    ds = torch.utils.data.get_worker_info().dataset
    ds.rng = np.random.default_rng(torch.initial_seed() % (2 ** 32) + wid)


def pick_device(cfg: Config):
    if cfg.train.device != "auto":
        return torch.device(cfg.train.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_setup(cfg: Config, device):
    mode = cfg.train.amp
    if mode == "auto":
        mode = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "none"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mode)
    scaler = torch.amp.GradScaler("cuda") if (mode == "fp16" and device.type == "cuda") else None
    return mode, dtype, scaler


def make_loader(ds, batch_sampler, cfg: Config, device):
    nw = cfg.data.num_workers
    return DataLoader(ds, batch_sampler=batch_sampler, collate_fn=collate, num_workers=nw,
                      pin_memory=device.type == "cuda", persistent_workers=nw > 0,
                      worker_init_fn=_worker_init if nw > 0 else None)


@torch.no_grad()
def eval_loader(ds: ChipDataset, cfg: Config, device):
    """Deterministic sweep of spatial clusters covering every chip once (same graph structure as training)."""
    return make_loader(ds, sweep_clusters(ds, cfg.train.batch), cfg, device)


@torch.no_grad()
def evaluate(model, loader, cfg: Config, device, dtype, collect_calib=False, thr=None):
    model.eval()
    pm = PixelMetrics()
    crit = PMGSTLoss(cfg.loss); loss_sum, n = 0.0, 0
    cal_z, cal_y, cal_w = [], [], []
    rng = np.random.default_rng(0)
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            out = model(batch)
        loss, _ = crit(out, batch, epoch=10 ** 6)
        loss_sum += float(loss); n += 1
        p = out["p"].float().cpu().numpy(); lab = batch["label"].cpu().numpy()
        pm.update(p, lab)
        if collect_calib:
            v = lab != 255
            z = out["logit"].float().cpu().numpy()[v]; yy = (lab[v] == 1)
            keep = yy | (rng.random(len(yy)) < 0.02)
            cal_z.append(z[keep]); cal_y.append(yy[keep]); cal_w.append(np.where(yy[keep], 1.0, 50.0))
    res = pm.compute(thr)
    res["val_loss"] = loss_sum / max(n, 1)
    if collect_calib and cal_z:
        res["_calib"] = (np.concatenate(cal_z), np.concatenate(cal_y), np.concatenate(cal_w))
    return res


@torch.no_grad()
def evaluate_objects(model, loader, cfg, device, dtype, thr: float):
    model.eval(); om = ObjectMetrics()
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            out = model(batch)
        om.update((out["p"].float() >= thr).cpu().numpy(), batch["label"].cpu().numpy())
    return om.compute()


def train(cfg: Config) -> dict:
    torch.manual_seed(cfg.train.seed); np.random.seed(cfg.train.seed)
    device = pick_device(cfg)
    mode, dtype, scaler = amp_setup(cfg, device)
    out_dir = Path(cfg.train.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=1))
    tr, va = ChipDataset(cfg, "train"), ChipDataset(cfg, "val")
    print(f"[data] train {len(tr)} chips ({(tr.n_pos > 0).sum()} with landslides) | val {len(va)} ({(va.n_pos > 0).sum()}) | device {device} amp {mode}")
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("empty split - check data.root, folds and buffer_m")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    model = build_model(cfg).to(device)
    if cfg.data.fit_norm:
        from .data.stats import fit_dem_stats
        mu, sd = fit_dem_stats(tr)
        model.dem_mu.copy_(torch.from_numpy(mu).view(1, -1, 1, 1)); model.dem_sd.copy_(torch.from_numpy(sd).view(1, -1, 1, 1))
        (out_dir / "norm_stats.json").write_text(json.dumps(dict(dem_mu=mu.tolist(), dem_sd=sd.tolist())))
        print("[norm] DEM mean/std fitted on training chips")
    print(f"[model] {model.n_params() / 1e6:.2f} M params")
    crit = PMGSTLoss(cfg.loss)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
    steps_ep = cfg.train.steps_per_epoch or max(1, len(tr) // cfg.train.batch)
    total_opt = max(1, cfg.train.epochs * steps_ep // cfg.train.accum)
    warm = cfg.train.warmup_epochs * steps_ep // cfg.train.accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total_opt - warm, 1))) * 0.99 + 0.01)
    log = open(out_dir / "metrics.jsonl", "a")
    best, bad, history = -1.0, 0, []
    sampler = ClusterBatchSampler(tr, cfg.train.batch, steps_ep, cfg.train.pos_frac, cfg.train.seed, cfg.train.hard_frac)
    train_loader, val_loader = make_loader(tr, sampler, cfg, device), eval_loader(va, cfg, device)
    for epoch in range(cfg.train.epochs):
        model.train(); t0 = time.time()
        agg, k, gnorms, warned = {}, 0, [], False
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                out = model(batch)
            loss, parts = crit(out, batch, epoch)
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at epoch {epoch} step {step}, skipping"); opt.zero_grad(set_to_none=True); continue
            (scaler.scale(loss / cfg.train.accum) if scaler else loss / cfg.train.accum).backward()
            if (step + 1) % cfg.train.accum == 0:
                if scaler:
                    scaler.unscale_(opt)
                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad))
                if math.isfinite(gnorm):
                    gnorms.append(gnorm)
                    if gnorm > 1e4 and not warned:
                        print(f"[warn] gradient norm {gnorm:.2g}: global clipping will freeze every other weight "
                              f"(some layer is exploding, e.g. GroupNorm on constant input)"); warned = True
                    (scaler.step(opt), scaler.update()) if scaler else opt.step()
                else:
                    print(f"[warn] non-finite gradient at epoch {epoch} step {step}, update skipped")
                    if scaler:
                        scaler.update()
                opt.zero_grad(set_to_none=True); sched.step()
            if cfg.model.use_memory:
                pos = (batch["label"] == 1).flatten(1).any(1)
                with torch.no_grad():
                    model.memory.write(out["node_ref"][pos], torch.ones(int(pos.sum()), device=device),
                                       batch["meta"][pos][:, 8].long())
            for kk, v in parts.items():
                agg[kk] = agg.get(kk, 0.0) + float(v.detach())
            k += 1
            if (step + 1) % cfg.train.log_every == 0:
                print(f"  ep{epoch} step {step + 1}/{steps_ep} loss {agg['total'] / k:.4f}")
        tr_stats = {kk: v / max(k, 1) for kk, v in agg.items()}
        val = evaluate(model, val_loader, cfg, device, dtype)
        rec = dict(epoch=epoch, time=round(time.time() - t0, 1), lr=sched.get_last_lr()[0],
                   grad_norm_med=round(float(np.median(gnorms)), 3) if gnorms else None, grad_norm_max=round(max(gnorms), 3) if gnorms else None,
                   **{f"train_{a}": round(b, 4) for a, b in tr_stats.items()},
                   **{f"val_{a}": (round(b, 4) if isinstance(b, float) else b) for a, b in val.items() if not a.startswith("_")})
        history.append(rec); log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[ep {epoch}] train {tr_stats.get('total', float('nan')):.4f} | val AP {val['ap']:.4f} F1 {val['f1']:.4f} "
              f"IoU {val['iou']:.4f} ECE {val['ece']:.4f} | {rec['time']}s")
        ck = dict(model=model.state_dict(), cfg=to_dict(cfg), epoch=epoch, val={a: b for a, b in val.items() if not a.startswith('_')})
        torch.save(ck, out_dir / "last.pt")
        score = val["ap"] if not math.isnan(val["ap"]) else -1
        if score > best:
            best, bad = score, 0
            torch.save(ck, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= cfg.train.patience:
                print(f"[early stop] no val AP gain for {bad} epochs"); break
    # final: temperature scaling + object metrics on val with the best checkpoint
    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    val = evaluate(model, val_loader, cfg, device, dtype, collect_calib=True)
    calib = fit_calibration(*val["_calib"]) if "_calib" in val else (1.0, 0.0)
    obj = evaluate_objects(model, val_loader, cfg, device, dtype, val["thr"])
    ck["calib"], ck["thr"], ck["val"] = list(calib), val["thr"], {**{a: b for a, b in val.items() if not a.startswith("_")}, **obj}
    torch.save(ck, out_dir / "best.pt")
    print(f"[done] best val AP {best:.4f} | thr {val['thr']:.3f} | calib a={calib[0]:.3f} b={calib[1]:.3f} | object F1 {obj['obj_f1']:.3f}")
    return dict(best_ap=best, history=history, calib=calib, thr=val["thr"], **obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. model.use_graph=false train.epochs=5")
    a = ap.parse_args()
    train(load_config(a.config, a.set))


if __name__ == "__main__":
    main()
