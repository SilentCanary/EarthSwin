"""Block 15 (deployment, minimal): inference API + decision engine.

  pip install fastapi uvicorn
  set PMGST_CKPT=runs/exp/best.pt && uvicorn pmgst.serve:app --port 8000

POST /predict   upload a chip .npz (see data/formats.py) -> probability stats, polygons-as-points, warning level
GET  /health
Phase 3 (scheduler -> ingestion -> inference -> alert dispatch -> retraining) builds on run_region(); see docs/11.
"""
from __future__ import annotations

import io
import os

import numpy as np
import torch

from .calibration import apply_calibration
from .data.dataset import ChipDataset, collate
from .infer import extract_objects, load_model, warning_levels
from .train import amp_setup

try:
    from fastapi import FastAPI, File, UploadFile
except ImportError:                                            # keep the module importable without fastapi
    FastAPI = None

_state = {}


def _load():
    if "model" not in _state:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg, calib, thr = load_model(os.environ.get("PMGST_CKPT", "runs/exp/best.pt"), device)
        _state.update(model=model, cfg=cfg, calib=calib, thr=thr, device=device, dtype=amp_setup(cfg, device)[1])
    return _state


def predict_npz(raw: bytes) -> dict:
    st = _load()
    z = np.load(io.BytesIO(raw))
    chip = {k: z[k] for k in z.files}
    ds = ChipDataset.__new__(ChipDataset)                      # reuse the tensor conversion without an index on disk
    ds.cfg, ds.rng, ds.augment = st["cfg"], np.random.default_rng(0), False
    item = ds._to_tensors(ds._pick_dates(chip, train=False), 0)
    batch = {k: v.to(st["device"]) for k, v in collate([item]).items()}
    with torch.no_grad(), torch.autocast(device_type=st["device"].type, dtype=st["dtype"], enabled=st["dtype"] is not None):
        out = st["model"](batch)
    p = apply_calibration(out["p"][0].float().cpu().numpy(), st["calib"]); u = out["u"][0].float().cpu().numpy()
    thr = float(apply_calibration(np.array([st["thr"]]), st["calib"])[0])
    _, objs = extract_objects(p, u, thr)
    level, review = warning_levels(float(p.max()), float(u.mean()))
    return dict(level=level, review=review, p_max=float(p.max()), p_mean=float(p.mean()), u_mean=float(u.mean()),
                threshold=thr, n_objects=len(objs), objects=objs)


if FastAPI is not None:
    app = FastAPI(title="PMGST-Net-D")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": "model" in _state}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        return predict_npz(await file.read())
