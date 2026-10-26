"""muP (src/mup.py): the rules on a small model, the attention logit scale,
and the trainer with --mup-base-d-model.

The trainer runs on the CPU on synthetic series, as in
tests/test_414_lr_schedule_resume.py.
"""
import copy
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from src.blocks import DecoderOnlyTransformerLayer
from src.mup import (FIXED, HIDDEN, INPUT, OUTPUT, mup_init_, mup_param_groups,
                     parameter_rules)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "experiments" / "2026-04-27_freq-embedding" / "scripts" / "train.py"


class Toy(nn.Module):
    """One parameter of each class, at the width ``d``."""

    def __init__(self, d):
        super().__init__()
        self.emb = nn.Embedding(7, d)
        self.inp = nn.Linear(5, d)
        self.hid = nn.Linear(d, 2 * d)
        self.out = nn.Linear(2 * d, 3)
        self.norm = nn.LayerNorm(d)
        self.gru = nn.GRU(4, 6)
        self.mix = nn.Parameter(torch.zeros(d, d))
        self.scalar = nn.Parameter(torch.tensor(1.0))


def toy_rules(d, base_d=8):
    torch.manual_seed(0)
    model = Toy(d)
    with torch.device("meta"):
        base, delta = Toy(base_d), Toy(2 * base_d)
    return model, parameter_rules(model, base, delta)


def kinds_of(rules):
    return {name: kind for name, (kind, _, _) in rules.items()}


def test_the_classes_are_the_same_at_the_base_width():
    _, at_base = toy_rules(8)
    _, wider = toy_rules(16)
    assert kinds_of(at_base) == kinds_of(wider)
    assert {m_in for _, m_in, _ in at_base.values()} == {1.0}


def test_at_the_base_width_every_tensor_keeps_its_init_and_rate():
    model, rules = toy_rules(8)
    before = copy.deepcopy(model.state_dict())
    mup_init_(model, rules)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name
    groups = mup_param_groups(model, rules, lr=1e-3, weight_decay=0.1)
    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-3 and groups[0]["weight_decay"] == 0.1
    assert groups[0]["lr_mult"] == 1.0


def test_the_init_factors_multiply_the_weights_of_their_class():
    model, rules = toy_rules(8)
    before = copy.deepcopy(model.state_dict())
    mup_init_(model, rules, input_mult=0.5, hidden_mult=2.0, output_mult=3.0)
    after = model.state_dict()
    factor = {"emb.weight": 0.5, "inp.weight": 0.5, "hid.weight": 2.0, "mix": 2.0,
              "out.weight": 3.0}
    for name, value in after.items():
        torch.testing.assert_close(value, before[name] * factor.get(name, 1.0))


def test_at_twice_the_width_each_class_gets_its_rule():
    model, rules = toy_rules(16)
    kinds = kinds_of(rules)
    assert kinds["emb.weight"] == INPUT
    assert kinds["inp.weight"] == INPUT and kinds["inp.bias"] == INPUT
    assert kinds["hid.weight"] == HIDDEN and kinds["hid.bias"] == INPUT
    assert kinds["out.weight"] == OUTPUT and kinds["out.bias"] == FIXED
    assert kinds["norm.weight"] == INPUT and kinds["mix"] == HIDDEN
    assert kinds["scalar"] == FIXED and kinds["gru.weight_hh_l0"] == FIXED
    assert rules["hid.weight"][1] == 2.0 and rules["out.weight"][1] == 2.0

    before = copy.deepcopy(model.state_dict())
    mup_init_(model, rules)
    after = model.state_dict()
    torch.testing.assert_close(after["out.weight"], before["out.weight"] / math.sqrt(2))
    torch.testing.assert_close(after["hid.bias"], before["hid.bias"] * math.sqrt(2))
    torch.testing.assert_close(after["out.bias"], before["out.bias"] * math.sqrt(2))
    changed = {"out.weight", "hid.bias", "out.bias"}
    for name in set(after) - changed:
        assert torch.equal(after[name], before[name]), name

    groups = mup_param_groups(model, rules, lr=1e-3, weight_decay=0.1)
    names = {id(p): n for n, p in model.named_parameters()}
    by_mult = {g["lr_mult"]: {names[id(p)] for p in g["params"]} for g in groups}
    assert by_mult[0.5] == {"hid.weight", "out.weight", "mix"}
    slow = next(g for g in groups if g["lr_mult"] == 0.5)
    assert slow["lr"] == pytest.approx(5e-4) and slow["weight_decay"] == pytest.approx(0.2)


def test_a_recurrent_layer_that_scales_is_refused():
    with torch.device("meta"):
        base, delta = nn.GRU(4, 6), nn.GRU(4, 12)
    with pytest.raises(NotImplementedError, match="recurrent"):
        parameter_rules(nn.GRU(4, 6), base, delta)


def attention_layer(qk_norm=True):
    return DecoderOnlyTransformerLayer(16, 2, dim_feedforward=32, dropout=0.0,
                                       batch_first=True, bias=False, qk_norm=qk_norm)


def test_the_logit_scale_reaches_a_layer_through_its_state_dict():
    torch.manual_seed(0)
    layer = attention_layer().eval()
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        default = layer(x)
        same = copy.deepcopy(layer)
        same.set_attn_logit_scale(1.0 / math.sqrt(8))
        torch.testing.assert_close(same(x), default)

        scaled = copy.deepcopy(layer)
        scaled.set_attn_logit_scale(0.25)
        want = scaled(x)
        assert not torch.allclose(want, default)

        fresh = attention_layer().eval()
        fresh.load_state_dict(scaled.state_dict())
        assert fresh._attn_scale == 0.25
        assert torch.equal(fresh(x), want)


def test_a_layer_without_qk_norm_refuses_a_scale():
    with pytest.raises(ValueError, match="qk-norm"):
        attention_layer(qk_norm=False).set_attn_logit_scale(0.25)


def trainer(save_dir, run_name, d_model, mup_base=None, extra=(), ok=True):
    cmd = [
        sys.executable, "-u", str(TRAIN_PY), "--device", "cpu",
        "--total-steps", "20", "--save-every", "20", "--traj-save-every", "20",
        "--batch-size", "2", "--lr", "1e-2", "--weight-decay", "0.1",
        "--lr-final", "1e-4", "--lr-cosine-steps", "40",
        "--save-dir", str(save_dir), "--run-name", run_name,
        "--mix-ratio", "1.0", "--synth-kind", "periodic", "--t-raw", "64",
        "--n-channels", "1", "--d-model", str(d_model), "--n-heads", "2",
        "--num-encoder-layers", "1", "--num-layers", "1", "--qk-norm",
        "--log-every", "5", "--seed", "20260728",
        "--loss-shape", "cosine_similarity_batch_full_hh_negs_xshh_allt",
        "--tau", "0.10", "--hf-repo", "none", "--hf-path", "none",
    ]
    if mup_base is not None:
        cmd += ["--mup-base-d-model", str(mup_base)]
    cmd += list(extra)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    run = subprocess.run(cmd, env=env, capture_output=True, text=True,
                         cwd=str(REPO_ROOT), timeout=600)
    assert (run.returncode == 0) == ok, run.stderr[-3000:]
    return run


def test_the_base_width_trains_as_the_run_without_mup(tmp_path):
    trainer(tmp_path, "plain", 8)
    run = trainer(tmp_path, "mup", 8, mup_base=8)
    assert "muP: base d_model 8 with 2 heads" in run.stdout
    plain = (tmp_path / "plain_losses.csv").read_text().splitlines()
    mup = (tmp_path / "mup_losses.csv").read_text().splitlines()
    assert len(plain) > 10 and plain == mup


def test_a_wider_run_scales_the_hidden_rates_and_the_logits(tmp_path):
    run = trainer(tmp_path, "wide", 16, mup_base=8)
    assert "attention layers with the muP logit scale" in run.stdout
    state = torch.load(tmp_path / "wide_step20_optimizer.pth", map_location="cpu",
                       weights_only=False)
    groups = state["optimizer_state_dict"]["param_groups"]
    assert sorted(g["lr_mult"] for g in groups) == [0.5, 1.0]
    for g in groups:
        assert g["lr"] == pytest.approx(groups[0]["lr"] / groups[0]["lr_mult"] * g["lr_mult"])
        assert g["weight_decay"] == pytest.approx(0.1 / g["lr_mult"])
    weights = torch.load(tmp_path / "wide_step20.pth", map_location="cpu", weights_only=False)
    weights = weights.get("model_state_dict", weights)
    scales = {k: float(v) for k, v in weights.items() if k.endswith("attn_logit_scale")}
    assert scales and all(v == pytest.approx(math.sqrt(4) / 8) for v in scales.values())


def test_an_init_factor_changes_the_run_at_the_base_width(tmp_path):
    trainer(tmp_path, "plain", 8)
    run = trainer(tmp_path, "half", 8, mup_base=8, extra=["--mup-init-hidden", "0.5"])
    assert "init x input 1, hidden 0.5, output 1" in run.stdout
    plain = (tmp_path / "plain_losses.csv").read_text().splitlines()
    half = (tmp_path / "half_losses.csv").read_text().splitlines()
    assert plain[0] == half[0] and plain[1] != half[1]


def test_an_init_factor_needs_a_base_width(tmp_path):
    run = trainer(tmp_path, "nobase", 8, extra=["--mup-init-hidden", "0.5"], ok=False)
    assert "needs --mup-base-d-model" in run.stderr
