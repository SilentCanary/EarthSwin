import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pmgst.config import load_config  # noqa: E402
from pmgst.data.synth import write_synthetic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    write_synthetic(d, seed=0, chips_x=6, chips_y=6, chip=64, ctx=32, n_dates=10, log=lambda *a: None)
    return d


@pytest.fixture()
def cfg(synth_dir, tmp_path):
    c = load_config(ROOT / "configs" / "smoke.yaml")
    c.data.root = str(synth_dir)
    c.train.out_dir = str(tmp_path / "run")
    return c
