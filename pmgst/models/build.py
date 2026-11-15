"""One entry point for every architecture (PMGST-Net-D and the deep baselines)."""
from __future__ import annotations

from ..config import Config


def build_model(cfg: Config):
    if cfg.model.arch == "pmgst":
        from .pmgst_net import PMGSTNet
        return PMGSTNet(cfg)
    from .baselines_dl import DLBaseline
    return DLBaseline(cfg)
