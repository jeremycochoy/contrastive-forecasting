"""The context that the official eval gives the backbone (--context-len).

load_models sets the module value T_RAW from --t-raw after the module is
imported. The predictor took `t_raw: int = T_RAW` as a default of its
signature, bound at import to 1024: every score read the last 1024 values,
while the backbones and their heads train on 4096.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
EVAL_PY = (REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
           / "eval_gift_eval_official.py")
CPU = torch.device("cpu")


def eval_module(tag):
    pytest.importorskip("gift_eval")
    spec = importlib.util.spec_from_file_location(f"eval_context_{tag}", EVAL_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse(module, monkeypatch, *flags):
    monkeypatch.setattr(sys, "argv", ["eval", "--backbone-path", "bb.pth",
                                      "--head-path", "head.pth", *flags])
    return module.parse_args()


def dataset(horizon=48):
    return SimpleNamespace(prediction_length=horizon,
                           test_data=SimpleNamespace(label=[]))


def test_the_predictor_reads_the_window_that_load_models_sets():
    ev = eval_module("window")
    ev.T_RAW = 4096        # what load_models does with --t-raw 4096
    predictor = ev.ContrastiveForecasterPredictor(None, None, 48, CPU)
    history = np.arange(10_000, dtype=np.float32)
    assert predictor.t_raw == 4096
    assert predictor._prepare_context(history).shape == (1, 4096, 1)


def test_an_explicit_window_wins():
    ev = eval_module("explicit")
    ev.T_RAW = 4096
    assert ev.ContrastiveForecasterPredictor(None, None, 48, CPU, t_raw=1024).t_raw == 1024


def test_context_len_defaults_to_the_training_window(monkeypatch):
    ev = eval_module("default")
    args = parse(ev, monkeypatch, "--t-raw", "4096")
    ev.T_RAW = args.t_raw
    assert args.context_len is None
    assert ev.build_predictor(args, None, None, dataset(), CPU).t_raw == 4096


def test_context_len_1024_reproduces_the_scores_before_the_fix(monkeypatch):
    ev = eval_module("legacy")
    args = parse(ev, monkeypatch, "--t-raw", "4096", "--context-len", "1024")
    ev.T_RAW = args.t_raw
    assert ev.build_predictor(args, None, None, dataset(), CPU).t_raw == 1024


def test_strategy_r_reads_the_same_window(monkeypatch):
    ev = eval_module("r")
    args = parse(ev, monkeypatch, "--t-raw", "4096", "--strategy", "R",
                 "--zero-head")
    ev.T_RAW = args.t_raw
    args.reconstruction_latent = "encoder"   # load_models reads it off the checkpoint
    predictor = ev.build_predictor(args, None, None, dataset(), CPU)
    assert isinstance(predictor, ev.ReconstructionPredictor)
    assert predictor.t_raw == 4096
