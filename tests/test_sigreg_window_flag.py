"""The trainer passes --sigreg-legacy-window to the SIGReg statistic.

Default off: the fixed bound W = 6/σ. On: the bound 6/√K of the runs before
the window fix (src.loss.sigreg_loss).
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "experiments" / "2026-04-27_freq-embedding" / "scripts" / "train.py"
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def train():
    spec = importlib.util.spec_from_file_location("train_sigreg_window", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def recorded_legacy_window(train, monkeypatch, argv):
    seen = {}

    def record(z, **kwargs):
        seen.update(kwargs)
        return z.new_zeros(())

    monkeypatch.setattr(train, "sigreg_loss", record)
    args = train.parse_args(["--weight-decay", "0.1", *argv])
    train.sigreg_term(torch.randn(2, 4, 1, 8), args, None)
    return seen["legacy_window"]


def test_fixed_window_by_default(train, monkeypatch):
    assert recorded_legacy_window(train, monkeypatch, []) is False


def test_flag_restores_the_legacy_window(train, monkeypatch):
    assert recorded_legacy_window(train, monkeypatch, ["--sigreg-legacy-window"]) is True
