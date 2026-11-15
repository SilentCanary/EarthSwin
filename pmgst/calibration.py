"""Temperature scaling on the validation set (block 11.3): p_cal = sigmoid(logit(p) / T), T = argmin NLL."""
from __future__ import annotations

import numpy as np
import torch


def fit_temperature(logit: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """logit, y (0/1), optional importance weights w (negatives were subsampled)."""
    z = torch.from_numpy(logit.astype(np.float32)); t = torch.from_numpy(y.astype(np.float32))
    wt = torch.from_numpy((w if w is not None else np.ones_like(logit)).astype(np.float32))
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(z / log_T.exp(), t, reduction="none") * wt).sum() / wt.sum()
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.detach().exp().clamp(0.05, 20.0))


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.log(p / (1 - p)) / T))


def fit_calibration(logit: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    """Platt scaling p_cal = sigmoid(a*logit + b). Needed because focal/Dice/pos-weighted training shifts the prior,
    which a pure temperature cannot undo. Weights w compensate for subsampled negatives."""
    z = torch.from_numpy(logit.astype(np.float32)); t = torch.from_numpy(y.astype(np.float32))
    wt = torch.from_numpy((w if w is not None else np.ones_like(logit)).astype(np.float32))
    ab = torch.tensor([1.0, 0.0], requires_grad=True)
    opt = torch.optim.LBFGS([ab], lr=0.5, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(ab[0] * z + ab[1], t, reduction="none") * wt).sum() / wt.sum()
        loss.backward()
        return loss

    opt.step(closure)
    a, b = ab.detach().tolist()
    return float(np.clip(a, 0.05, 20.0)), float(np.clip(b, -20.0, 20.0))


def apply_calibration(p: np.ndarray, ab) -> np.ndarray:
    a, b = ab
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(a * np.log(p / (1 - p)) + b)))
