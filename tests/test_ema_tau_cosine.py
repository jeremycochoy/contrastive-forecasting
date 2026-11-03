"""Tests for the cosine shape of the EMA momentum schedule.

#388 moves the EMA momentum α of the teacher in a straight line from
``--ema-tau`` to ``--ema-tau-end``, and #393 anchors that ramp to
``--ema-tau-ramp-steps``. ``--ema-tau-shape cosine`` keeps the two ends and
the span, and makes the ramp a half cosine:

    α(step) = end − (end − start)·½·(1 + cos(π·t/span))
    t = step clamped to [0, span], span = ramp_steps, else total_steps

α holds at the end value after the span. The default shape, ``linear``,
gives each run the α it had before the flag, bit for bit.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.models import (SCHEDULE_SHAPES, cosine_schedule_at_step,
                        ema_tau_at_step, linear_schedule_at_step)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")

# The run that asks for the shape: α from 0.9 to 1.0 over the first 200,000
# steps (the span of the rate anneal of BLK), in a leg of 665,000 steps.
START, END, SPAN, BUDGET = 0.9, 1.0, 200_000, 665_000


def cosine(step, span=SPAN, start=START, end=END):
    """The formula of the shape, written out."""
    t = min(max(step, 0), span)
    return end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * t / span))


# --- 1. the cosine ramp ---------------------------------------------------


class TestCosineSchedule:

    def at(self, step, total=BUDGET, end=END, ramp=SPAN):
        return cosine_schedule_at_step(step, total, START, end, ramp)

    def test_start_value_at_step_zero(self):
        assert self.at(0) == pytest.approx(START)

    def test_midpoint_at_half_the_span(self):
        assert self.at(SPAN // 2) == pytest.approx(0.95)

    def test_end_value_at_the_span(self):
        assert self.at(SPAN) == END

    def test_held_at_the_end_value_after_the_span(self):
        """α = 1.0 exactly: the teacher stays frozen to the end of the leg."""
        for step in (SPAN + 1, 300_000, BUDGET, 2 * BUDGET):
            assert self.at(step) == END

    def test_the_formula_between_the_ends(self):
        for step in (1, 25_000, 50_000, 150_000, SPAN - 1):
            assert self.at(step) == pytest.approx(cosine(step), rel=1e-12)

    def test_without_ramp_steps_the_span_is_the_budget(self):
        assert self.at(50_000, total=100_000, ramp=None) == pytest.approx(0.95)
        assert self.at(100_000, total=100_000, ramp=None) == END

    def test_the_curve_ignores_the_budget(self):
        """Legs that stop at 40k and at 665k apply one α at one step."""
        for step in (1, 40_000, 100_000, SPAN - 1, SPAN):
            got = {self.at(step, total=budget)
                   for budget in (40_000, 200_000, BUDGET)}
            assert len(got) == 1, f"step {step} gave {got}"

    def test_monotone_and_bounded(self):
        vals = [self.at(s) for s in range(0, BUDGET + 1, 5_000)]
        assert vals == sorted(vals)
        assert min(vals) >= START and max(vals) <= END

    def test_no_end_value_is_constant(self):
        """The contract of the linear ramp: no end value, no schedule."""
        for step in (0, 100_000, BUDGET):
            assert self.at(step, end=None) == START

    def test_negative_step_is_clamped(self):
        assert self.at(-5) == pytest.approx(START)

    def test_zero_span_returns_the_end_value(self):
        assert self.at(0, total=0, ramp=None) == END


# --- 2. ema_tau_at_step: the shape ----------------------------------------


class TestEmaTauShape:

    @pytest.mark.parametrize("total,start,end,ramp", [
        (100_000, 0.9, 1.0, None),     # #388: the ramp spans the budget
        (400_000, 0.9, 1.0, 100_000),  # #393: the ramp has an anchor
        (665_000, 0.9, 1.0, 665_000),
        (100, 0.99, 0.9, None),        # a ramp that goes down
        (100_000, 0.9, None, None),    # a constant α
        (100_000, 0.9, 1.0, 0),        # a zero anchor takes the budget
    ])
    def test_the_default_is_the_linear_ramp_bit_for_bit(self, total, start,
                                                         end, ramp):
        for step in (-5, 0, 1, 7, 50, 333, 25_000, 50_000, 99_999, 100_000,
                     120_000, 665_000):
            old = linear_schedule_at_step(step, total, start, end, ramp)
            assert ema_tau_at_step(step, total, start, end, ramp) == old
            assert ema_tau_at_step(step, total, start, end, ramp,
                                   shape="linear") == old

    def test_the_cosine_shape(self):
        for step in (0, 1, 50_000, SPAN // 2, SPAN, BUDGET):
            assert (ema_tau_at_step(step, BUDGET, START, END, SPAN,
                                    shape="cosine")
                    == cosine_schedule_at_step(step, BUDGET, START, END, SPAN))

    def test_an_unknown_shape_is_refused(self):
        with pytest.raises(ValueError, match="sine"):
            ema_tau_at_step(0, 100, START, END, shape="sine")


# --- 3. the training flag -------------------------------------------------


@pytest.fixture(scope="module")
def train_py():
    """train.py as a module. Its ``main()`` runs only under ``__main__``."""
    spec = importlib.util.spec_from_file_location("train_py_ema_cosine",
                                                  TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_train(tmp_path, run_name, extra):
    """A tiny run with an EMA teacher on the CPU: the arm of #388."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    save_dir = tmp_path / "runs"
    cmd = [
        sys.executable, "-u", str(TRAIN_PY),
        "--device", "cpu", "--total-steps", "4", "--save-every", "100",
        "--batch-size", "2", "--lr", "1e-3", "--weight-decay", "0.1",
        "--save-dir", str(save_dir), "--run-name", run_name,
        "--mix-ratio", "1.0", "--synth-kind", "periodic",
        "--t-raw", "64", "--n-channels", "1", "--d-model", "32",
        "--n-heads", "2", "--num-layers", "1", "--num-encoder-layers", "1",
        "--log-every", "1", "--seed", "42", "--tau", "0.10",
        "--hf-repo", "none", "--hf-path", "none",
        "--loss-shape", "cosine_similarity_batch_no_time_neg",
        "--no-main-contrastive-loss", "--align-loss-weight", "1.0",
        "--ema-embedding", "--ema-encoder", "--ema-tau", "0.9",
    ] + extra
    res = subprocess.run(cmd, env=env, capture_output=True, text=True,
                         cwd=str(tmp_path), timeout=900)
    return res, save_dir


class TestTrainFlag:

    def test_the_default_shape_is_linear(self, train_py):
        assert train_py.parse_args(["--weight-decay", "0.1"]).ema_tau_shape \
            == "linear"

    def test_the_flag_takes_each_shape_and_no_other(self, train_py):
        for shape in SCHEDULE_SHAPES:
            args = train_py.parse_args(["--weight-decay", "0.1",
                                        "--ema-tau-shape", shape])
            assert args.ema_tau_shape == shape
        with pytest.raises(SystemExit):
            train_py.parse_args(["--weight-decay", "0.1",
                                 "--ema-tau-shape", "sine"])

    def test_cosine_without_an_end_value_is_refused(self, tmp_path):
        res, save_dir = run_train(tmp_path, "cosnoend",
                                  ["--ema-tau-shape", "cosine"])
        out = res.stdout + res.stderr
        assert res.returncode != 0
        assert "--ema-tau-shape" in out and "--ema-tau-end" in out
        assert not (save_dir / "cosnoend_losses.csv").exists()

    def test_the_cosine_alpha_reaches_the_teacher(self, tmp_path):
        """The losses CSV logs the α of each teacher update. A span of 3 in
        4 steps gives the shape of the real run: a cosine, then a hold."""
        res, save_dir = run_train(
            tmp_path, "cos4", ["--ema-tau-end", "1.0",
                               "--ema-tau-ramp-steps", "3",
                               "--ema-tau-shape", "cosine"])
        assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
        rows = list(csv.DictReader(open(save_dir / "cos4_losses.csv")))
        alphas = [float(r["ema_tau"]) for r in rows]
        assert alphas == pytest.approx([cosine(s, span=3) for s in (1, 2, 3, 4)])
        assert alphas == pytest.approx([0.925, 0.975, 1.0, 1.0])
