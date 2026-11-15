"""Config: plain dataclasses + YAML. Override from CLI with  --set model.use_graph=false  etc."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataCfg:
    root: str = "data/chips"          # folder with index.csv + chips/*.npz
    chip: int = 128                   # chip side in px (10 m) -> 1.28 km. 256 is the full-size option.
    ctx: int = 64                     # context crops (scales 2,3) are resampled to ctx x ctx
    n_scales: int = 3                 # block 3: scale s crops chip*2^(s-1) px, resampled to ctx
    s2_bands: int = 6                 # B2 B3 B4 B8 B11 B12
    t_max: int = 8                    # S1 dates per sample (random subset of what exists, max stored 16)
    dem_ch: int = 8
    n_lith: int = 13                  # classes of the static "ground" map (lithology or soil texture)
    n_rain: int = 4                   # max1d, max3d, sum7d, sum30d
    folds: int = 5
    test_fold: int = 0
    val_fold: int = 1
    buffer_m: float = 1280.0          # no train chip within this distance of a val/test chip
    aug: bool = True
    p_drop_s2: float = 0.3            # modality dropout (block 5 training trick)
    p_drop_s1: float = 0.2            # per S1 date
    num_workers: int = 2
    holdout_region: str = ""          # "" = spatial-block folds; else all chips of this region (district) are the test set


@dataclass
class ModelCfg:
    arch: str = "pmgst"               # "pmgst" | "unet" (B3) | "siam" (B4) | "utae" (B5): same I/O, see models/baselines_dl.py
    dims: list = field(default_factory=lambda: [48, 96, 192, 384])
    depths: list = field(default_factory=lambda: [2, 2, 2, 2])
    heads: list = field(default_factory=lambda: [3, 6, 12, 24])
    window: int = 8
    drop_path: float = 0.05
    d_dec: int = 128
    d_graph: int = 128
    gat_layers: int = 2
    gat_heads: int = 4
    knn: int = 8
    t_layers: int = 1
    t_heads: int = 4
    # block switches (for ablations)
    use_s2: bool = True
    use_s1: bool = True
    use_pyramid: bool = True          # block 3
    use_graph: bool = True            # blocks 6-7
    use_physics: bool = True          # block 8b
    use_memory: bool = False          # block 9 (P2: only if ablation shows gain)
    head: str = "evidential"          # block 11: "evidential" | "logit"
    mc_dropout: float = 0.0
    ckpt: bool = False                # gradient checkpointing of encoder stages (saves VRAM)
    mem_size: int = 2048
    mem_dim: int = 128
    mem_topk: int = 8


@dataclass
class LossCfg:
    lam_cls: float = 1.0
    lam_phys: float = 0.1
    lam_con: float = 0.1
    lam_unc: float = 0.2
    lam_ent: float = 0.0              # entropy regulariser on modality weights (block 5.3), 0 = off
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    beta_min_deg: float = 5.0         # implausibility: slopes flatter than this
    edl_pos_w: float = 10.0
    edl_anneal_epochs: int = 10
    fs_kappa: float = 4.0


@dataclass
class TrainCfg:
    out_dir: str = "runs/exp"
    epochs: int = 60
    batch: int = 8                    # chips per step = one spatial cluster = one graph
    accum: int = 2
    lr: float = 3e-4
    wd: float = 0.05
    warmup_epochs: int = 3
    steps_per_epoch: int = 0          # 0 = n_train_chips // batch
    amp: str = "auto"                 # auto | bf16 | fp16 | none
    patience: int = 12
    pos_frac: float = 0.5             # share of training clusters seeded on a landslide chip
    hard_frac: float = 0.2            # share seeded on hard negatives (steep, no landslide); the rest random
    seed: int = 42
    device: str = "auto"
    clip_grad: float = 1.0
    log_every: int = 20


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


def _fill(dc: Any, d: dict) -> Any:
    for f in fields(dc):
        if f.name in d:
            cur = getattr(dc, f.name)
            if is_dataclass(cur):
                _fill(cur, d[f.name] or {})
            else:
                setattr(dc, f.name, d[f.name])
    unknown = set(d) - {f.name for f in fields(dc)}
    if unknown:
        raise KeyError(f"unknown config keys for {type(dc).__name__}: {sorted(unknown)}")
    return dc


def to_dict(cfg: Config) -> dict:
    from dataclasses import asdict
    return asdict(cfg)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            _fill(cfg, yaml.safe_load(fh) or {})
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        parts = key.strip().split(".")
        node = cfg
        for p in parts[:-1]:
            node = getattr(node, p)
        if not hasattr(node, parts[-1]):
            raise KeyError(f"unknown override key {key}")
        setattr(node, parts[-1], yaml.safe_load(val))
    return cfg


def from_dict(d: dict) -> Config:
    return _fill(Config(), copy.deepcopy(d))
