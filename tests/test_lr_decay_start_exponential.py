"""--lr-decay-start and the exponential anneal: the cooldown of a
constant-rate run, the decay of the WSD schedule of MiniCPM (arXiv
2404.06395, section 6.2).

The rate holds at --lr until --lr-decay-start. From there the exponential
shape halves it every --lr-half-life steps, down to --lr-final at the lowest.
The sqrt shape is the 1-sqrt anneal (Hägele and others 2024, arXiv
2405.18392): it gets to --lr-final at the end of the anneal and holds there.
Without the new flags, `scheduled_lr` gives the same rate as before, bit for
bit. A run resumed from a constant-rate save anneals from its own rate, and
the checkpoint carries the decay start and the half-life.

The resume test runs `train.py` on the smallest CPU arch with `--mix-ratio
1.0` (pure synth), as test_414_lr_schedule_resume.py does.
"""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
LR = 1e-2


@pytest.fixture(scope="module")
def train_py():
    spec = importlib.util.spec_from_file_location("train_py_decay", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def old_scheduled_lr(step, lr_start, lr_final, total, warmup=0,
                     shape="cosine"):
    """`scheduled_lr` of 9e3a0b75, before the decay start."""
    if step < warmup:
        return lr_start * step / warmup
    if shape == "linear":
        t = min(step - warmup, total - warmup)
        return lr_start + (lr_final - lr_start) * t / (total - warmup)
    t = min(max(step - warmup, 0), total - warmup)
    return lr_final + 0.5 * (lr_start - lr_final) * (
        1.0 + math.cos(math.pi * t / (total - warmup)))


def scheduled_lr_before_sqrt(step, lr_start, lr_final, total, warmup=0,
                             shape="cosine", start=0, half_life=0):
    """`scheduled_lr` of 3c1e4701, before the sqrt shape."""
    if step < warmup:
        return lr_start * step / warmup
    begin = max(warmup, start)
    if step < begin:
        return lr_start
    if shape == "exponential":
        return max(lr_final, lr_start * 0.5 ** ((step - begin) / half_life))
    if shape == "linear":
        t = min(step - begin, total - begin)
        return lr_start + (lr_final - lr_start) * t / (total - begin)
    t = min(max(step - begin, 0), total - begin)
    return lr_final + 0.5 * (lr_start - lr_final) * (
        1.0 + math.cos(math.pi * t / (total - begin)))


def test_exponential_halves_every_half_life(train_py):
    def lr(step):
        return train_py.scheduled_lr(step, 1e-3, 0.0, 166000, 0,
                                     "exponential", 135000, 5000)
    for step in (1, 100000, 134999, 135000):
        assert lr(step) == 1e-3
    assert lr(140000) == pytest.approx(5e-4)
    assert lr(145000) == pytest.approx(2.5e-4)
    assert lr(166000) == pytest.approx(1e-3 * 0.5 ** 6.2)


def test_exponential_holds_at_the_floor(train_py):
    def lr(step):
        return train_py.scheduled_lr(step, 1e-3, 1e-4, 0, 0, "exponential",
                                     100, 10)
    assert lr(130) == pytest.approx(1.25e-4)
    assert lr(140) == 1e-4
    assert lr(10 ** 6) == 1e-4


def test_decay_start_holds_then_anneals(train_py):
    def lr(step, shape):
        return train_py.scheduled_lr(step, 1e-3, 0.0, 200, 0, shape, 100)
    for shape in ("linear", "cosine"):
        assert lr(50, shape) == 1e-3
        assert lr(100, shape) == pytest.approx(1e-3)
        assert lr(150, shape) == pytest.approx(5e-4)
        assert lr(200, shape) == pytest.approx(0.0)
        assert lr(250, shape) == pytest.approx(0.0)


def test_warmup_then_stable_then_decay(train_py):
    def lr(step):
        return train_py.scheduled_lr(step, 1e-3, 0.0, 0, 100, "exponential",
                                     300, 50)
    assert lr(50) == pytest.approx(5e-4)
    assert lr(200) == 1e-3
    assert lr(350) == pytest.approx(5e-4)


@pytest.mark.parametrize("shape", ["cosine", "linear"])
@pytest.mark.parametrize("warmup", [0, 10000])
def test_no_decay_start_is_bit_identical(train_py, shape, warmup):
    for step in range(0, 260001, 997):
        assert (train_py.scheduled_lr(step, 1e-3, 5.6e-5, 200000, warmup,
                                      shape)
                == old_scheduled_lr(step, 1e-3, 5.6e-5, 200000, warmup,
                                    shape))


@pytest.mark.parametrize("warmup, start", [(0, 100), (100, 0)])
def test_sqrt_anneal_from_its_start_to_its_end(train_py, warmup, start):
    def lr(step):
        return train_py.scheduled_lr(step, 1e-3, 1e-4, 200, warmup, "sqrt",
                                     start)
    assert lr(100) == pytest.approx(1e-3)
    assert lr(150) == pytest.approx(
        1e-4 + (1e-3 - 1e-4) * (1 - math.sqrt(0.5)))
    assert lr(200) == 1e-4
    assert lr(250) == 1e-4
    assert lr(10 ** 6) == 1e-4


def test_sqrt_cooldown_of_a_constant_rate_run(train_py):
    def lr(step):
        return train_py.scheduled_lr(step, 1e-3, 0.0, 166000, 0, "sqrt",
                                     135000)
    for step in (1, 100000, 134999, 135000):
        assert lr(step) == 1e-3
    assert lr(150000) == pytest.approx(1e-3 * (1 - math.sqrt(15000 / 31000)))
    assert lr(166000) == 0.0
    assert lr(200000) == 0.0


@pytest.mark.parametrize("shape", ["cosine", "linear", "exponential"])
@pytest.mark.parametrize("warmup, start",
                         [(0, 0), (10000, 0), (10000, 135000)])
def test_the_other_shapes_are_bit_identical(train_py, shape, warmup, start):
    half_life = 5000 if shape == "exponential" else 0
    for step in range(0, 200001, 997):
        args = (step, 1e-3, 5.6e-5, 166000, warmup, shape, start, half_life)
        assert (train_py.scheduled_lr(*args)
                == scheduled_lr_before_sqrt(*args))


def _args(**changes):
    args = dict(lr_decay_shape="cosine", lr_half_life=0, lr_decay_start=0,
                lr_final=None, lr_warmup_steps=0, lr_cosine_steps=0,
                total_steps=166000)
    args.update(changes)
    return SimpleNamespace(**args)


@pytest.mark.parametrize("changes", [
    dict(lr_decay_shape="exponential", lr_final=0.0),
    dict(lr_half_life=5000, lr_final=0.0),
    dict(lr_decay_shape="exponential", lr_half_life=5000),
    dict(lr_decay_start=135000),
    dict(lr_decay_start=135000, lr_final=0.0, lr_warmup_steps=140000),
    dict(lr_decay_start=170000, lr_final=0.0),
    dict(lr_decay_start=-5, lr_final=0.0),
    dict(lr_decay_shape="sqrt", lr_half_life=5000, lr_final=0.0,
         lr_decay_start=135000),
    dict(lr_decay_shape="sqrt", lr_decay_start=135000),
    dict(lr_decay_shape="sqrt", lr_final=0.0, lr_decay_start=170000),
])
def test_refuses_a_schedule_that_cannot_run(train_py, changes):
    with pytest.raises(SystemExit):
        train_py.check_lr_schedule(_args(**changes))


@pytest.mark.parametrize("changes", [
    dict(),
    dict(lr_final=0.0),
    dict(lr_decay_shape="exponential", lr_half_life=5000, lr_final=0.0,
         lr_decay_start=135000),
    dict(lr_decay_shape="exponential", lr_half_life=5000, lr_final=0.0,
         lr_decay_start=170000),
    dict(lr_decay_shape="linear", lr_final=0.0, lr_decay_start=135000),
    dict(lr_decay_shape="sqrt", lr_final=0.0, lr_decay_start=135000),
])
def test_accepts_a_schedule_that_runs(train_py, changes):
    train_py.check_lr_schedule(_args(**changes))


def _cmd(save_dir, total_steps, resume=None, cooldown=True):
    cmd = [
        sys.executable, "-u", str(TRAIN_PY),
        "--device", "cpu", "--total-steps", str(total_steps),
        "--save-every", "50", "--traj-save-every", "50",
        "--batch-size", "2", "--lr", str(LR), "--weight-decay", "0.1",
        "--save-dir", str(save_dir), "--run-name", "cd",
        "--mix-ratio", "1.0", "--synth-kind", "periodic",
        "--t-raw", "64", "--n-channels", "1", "--d-model", "8",
        "--n-heads", "2", "--num-encoder-layers", "1", "--num-layers", "1",
        "--log-every", "50", "--seed", "20260728",
        "--loss-shape", "cosine_similarity_batch_full_hh_negs_xshh_allt",
        "--tau", "0.10", "--hf-repo", "none", "--hf-path", "none",
    ]
    if cooldown:
        cmd += ["--lr-final", "0", "--lr-decay-shape", "exponential",
                "--lr-decay-start", "100", "--lr-half-life", "50"]
    if resume is not None:
        cmd += ["--resume", resume]
    return cmd


def _run(cmd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(cmd, env=env, capture_output=True, text=True,
                          cwd=str(REPO_ROOT), timeout=900)


def _state(save_dir, step):
    path = next(save_dir.glob(f"cd*_step{step}_optimizer.pth"))
    return torch.load(path, map_location="cpu", weights_only=False)


def _lr(state):
    return state["optimizer_state_dict"]["param_groups"][0]["lr"]


def test_a_cooldown_resumes_from_a_constant_rate_save(tmp_path):
    save_dir = tmp_path / "runs"
    save_dir.mkdir()
    flat = _run(_cmd(save_dir, 100, cooldown=False))
    assert flat.returncode == 0, flat.stderr[-3000:]
    assert _lr(_state(save_dir, 100)) == pytest.approx(LR)

    resume = str(next(save_dir.glob("cd_step100.pth")))
    cool = _run(_cmd(save_dir, 200, resume=resume))
    assert cool.returncode == 0, cool.stderr[-3000:]
    assert _lr(_state(save_dir, 150)) == pytest.approx(LR / 2, rel=1e-6)
    assert _lr(_state(save_dir, 200)) == pytest.approx(LR / 4, rel=1e-6)
    held = _state(save_dir, 200)["lr_schedule"]
    assert held["decay_shape"] == "exponential"
    assert (held["decay_start"], held["half_life"]) == (100, 50)


def test_a_resume_reads_the_cooldown_back(tmp_path):
    save_dir = tmp_path / "runs"
    save_dir.mkdir()
    first = _run(_cmd(save_dir, 150))
    assert first.returncode == 0, first.stderr[-3000:]
    resume = str(next(save_dir.glob("cd_step150.pth")))
    second = _run(_cmd(save_dir, 200, resume=resume, cooldown=False))
    assert second.returncode == 0, second.stderr[-3000:]
    assert "resumed schedule from the checkpoint" in second.stdout
    assert _lr(_state(save_dir, 200)) == pytest.approx(LR / 4, rel=1e-6)


def test_a_resume_reads_the_sqrt_anneal_back(tmp_path):
    save_dir = tmp_path / "runs"
    save_dir.mkdir()
    sqrt = ["--lr-final", "0", "--lr-decay-shape", "sqrt",
            "--lr-decay-start", "100", "--lr-cosine-steps", "300"]
    first = _run(_cmd(save_dir, 150, cooldown=False) + sqrt)
    assert first.returncode == 0, first.stderr[-3000:]
    assert _lr(_state(save_dir, 150)) == pytest.approx(LR / 2, rel=1e-6)
    resume = str(next(save_dir.glob("cd_step150.pth")))
    second = _run(_cmd(save_dir, 200, resume=resume, cooldown=False))
    assert second.returncode == 0, second.stderr[-3000:]
    assert "shape sqrt" in second.stdout
    assert _lr(_state(save_dir, 200)) == pytest.approx(
        LR * (1 - math.sqrt(0.5)), rel=1e-6)
