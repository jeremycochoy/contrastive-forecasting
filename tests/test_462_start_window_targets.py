"""--keep-start-window-targets (#462): the value objective of MPE.

MPE (#421ew) trained before 289b2fc8. That commit drops from the value
objective each target in the first window of the EWMA (start_window_anchors).
The flag keeps these targets, so a run trains the objective of MPE. Without
the flag the objective drops them, as before this card.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
FLAG = "--keep-start-window-targets"


@pytest.fixture(scope="module")
def train_py():
    spec = importlib.util.spec_from_file_location("train_py_462", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def target_mask_given(train_py, monkeypatch, argv, start_mask):
    """The target mask that value_objective gives the value-space loss."""
    seen = {}

    def objective(model, x_norm, **kw):
        seen["target_mask"] = kw["target_mask"]
        return "loss"

    monkeypatch.setattr(train_py, "value_space_objective", objective)
    args = train_py.parse_args(["--weight-decay", "0.1",
                                "--value-space-objective", *argv])
    x = torch.zeros(2, 32, 1)
    inputs = dict(x_norm=x, target_mask=None, pad_mask=None,
                  start_mask=start_mask, sample_sizes=None, freq_ids=None,
                  freq_embs=None, seasonality_ids=None,
                  seasonality_embs=None)
    assert train_py.value_objective(None, inputs, args, ()) == "loss"
    return seen["target_mask"]


def start_window():
    start = torch.zeros(2, 32, 1, dtype=torch.bool)
    start[:, :16] = True
    return start


def test_the_flag_keeps_the_start_window_targets(train_py, monkeypatch):
    assert target_mask_given(train_py, monkeypatch, [FLAG],
                             start_window()) is None


def test_without_the_flag_no_target_is_in_the_start_window(train_py,
                                                          monkeypatch):
    start = start_window()
    target = target_mask_given(train_py, monkeypatch, [], start)
    assert torch.equal(target, ~start)


def test_the_flag_is_off_by_default(train_py):
    args = train_py.parse_args(["--weight-decay", "0.1"])
    assert args.keep_start_window_targets is False


def test_the_flag_is_on_the_value_space_allowlist(train_py):
    assert "keep_start_window_targets" in train_py.VALUE_SPACE_FLAGS
    assert train_py.value_space_conflicts(
        ["--value-space-objective", "--weight-decay", "0.1", FLAG]) == []


def test_the_trainer_refuses_the_flag_without_the_value_objective(tmp_path):
    never = tmp_path / "never"
    r = subprocess.run(
        [sys.executable, str(TRAIN_PY), "--device", "cpu", "--weight-decay",
         "0.1", FLAG, "--total-steps", "1", "--save-dir", str(never)],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
    assert r.returncode != 0
    assert "Add --value-space-objective" in r.stdout + r.stderr
    assert not never.exists()
