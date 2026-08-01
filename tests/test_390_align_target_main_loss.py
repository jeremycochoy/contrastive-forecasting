"""Tests for #390 — `align_target` inside the `align_loss_weight` branch.

#388 gave the STANDALONE :func:`src.loss.align_loss` an explicit target, and
that function only runs under ``--no-main-contrastive-loss``. Every #379 run
took the other path: the ``align_loss_weight`` add-on inside
:func:`src.loss.contrastive_latent_loss`, which always targeted the student's
own ``sg(h_{t+1})``. #390 makes that branch honour the same
``student`` / ``teacher`` choice.

Pinned here:

1. VALUE — the add-on is exactly ``λ·(2 − 2·cos(f_t, target_{t+1})).mean()``
   added to the loss, so ``L(λ, target) − L(0)`` identifies the target used.
   Checked against an independent formula for both targets, on the two
   settings the #390 cells train on (``rep_only`` with and without
   ``moco_rep_keys``).
2. DEFAULT — no ``align_target`` (arg or config key) keeps the student
   target byte-for-byte, so #379 stays reproducible.
3. PRECEDENCE — the function arg overrides the config key (same contract as
   ``align_loss_weight`` / ``moco_negatives``).
4. GUARDS — ``teacher`` without a teacher latent, and any unknown value,
   raise rather than silently falling back to the student. That silent
   fallback is the defect #390 exists to remove.
5. GRADIENT — no gradient reaches the teacher through the add-on.
6. TRAIN WIRING — ``--align-target teacher`` reaches the loss with the main
   contrastive loss ON (the #379 cell shape), and the flag combinations that
   cannot do anything abort.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.loss import contrastive_latent_loss

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "experiments" / "2026-04-27_freq-embedding" / "scripts" / "train.py"

# The loss shape of all 10 retrained #390 cells (arm5 / arm6_v2 families).
SHAPE = "cosine_similarity_batch_rep_only"


def _spec(align_weight=1.0, align_target=None, moco_rep=False, tau=0.1):
    tc = {
        "contrastive_divergence_temperature": tau,
        "contrastive_latent_noise": None,
        "loss_shape": SHAPE,
        "contrastive_latent_delay": 0,
        "include_positive_in_denominator": False,
        "stopgrad_positive_h": False,
        "subtract_contrastive_floor": False,
        "moco_negatives": False,
        "moco_rep_keys": moco_rep,
        "align_loss_weight": align_weight,
    }
    if align_target is not None:
        tc["align_target"] = align_target
    return SimpleNamespace(train_configuration=tc)


def _latents(B=3, T=5, C=2, H=8, seed=0):
    """Deterministic fp64 latents (isolates the math from fp32 noise)."""
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    o = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    t = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    return f, o, t


def _align_reference(f, target):
    """L_align longhand: (2 − 2·cos(f_t, target_{t+1})).mean()."""
    fn = F.normalize(f, p=2, dim=-1)[:, :-1]
    tn = F.normalize(target, p=2, dim=-1)[:, 1:]
    return (2.0 - 2.0 * (fn * tn).sum(-1)).mean()


def _base_loss(f, o, t, moco_rep):
    """The same run with the add-on off — the term the align branch adds to."""
    return contrastive_latent_loss(
        (f, o), False, _spec(align_weight=0.0, moco_rep=moco_rep),
        teacher_original_latent=t)


MOCO_REP = [
    pytest.param(False, id="arm5"),        # rep_only + align
    pytest.param(True, id="arm6_v2"),      # rep_only + align + moco_rep_keys
]


class TestAlignTargetValue:
    """L(λ, target) − L(0) must equal λ·L_align(f, target) exactly."""

    @pytest.mark.parametrize("moco_rep", MOCO_REP)
    def test_default_target_is_the_student(self, moco_rep):
        """No `align_target` anywhere ⇒ the #379 objective."""
        f, o, t = _latents(seed=11)
        got = contrastive_latent_loss(
            (f, o), False, _spec(moco_rep=moco_rep), teacher_original_latent=t)
        want = _base_loss(f, o, t, moco_rep) + _align_reference(f, o)
        assert torch.allclose(got, want, atol=1e-12, rtol=0.0)

    @pytest.mark.parametrize("moco_rep", MOCO_REP)
    def test_teacher_target_uses_the_teacher_latent(self, moco_rep):
        f, o, t = _latents(seed=12)
        got = contrastive_latent_loss(
            (f, o), False, _spec(moco_rep=moco_rep),
            teacher_original_latent=t, align_target="teacher")
        want = _base_loss(f, o, t, moco_rep) + _align_reference(f, t)
        assert torch.allclose(got, want, atol=1e-12, rtol=0.0)

    def test_teacher_target_changes_the_value(self):
        """Guard against a parameter that is accepted and then ignored."""
        f, o, t = _latents(seed=13)
        student = contrastive_latent_loss((f, o), False, _spec(),
                                          teacher_original_latent=t)
        teacher = contrastive_latent_loss((f, o), False, _spec(),
                                          teacher_original_latent=t,
                                          align_target="teacher")
        assert not torch.allclose(student, teacher)

    def test_weight_scales_the_teacher_term(self):
        f, o, t = _latents(seed=14)
        base = _base_loss(f, o, t, moco_rep=False)
        one = contrastive_latent_loss(
            (f, o), False, _spec(align_weight=1.0),
            teacher_original_latent=t, align_target="teacher")
        half = contrastive_latent_loss(
            (f, o), False, _spec(align_weight=0.5),
            teacher_original_latent=t, align_target="teacher")
        assert torch.allclose(half - base, 0.5 * (one - base), atol=1e-12,
                              rtol=0.0)

    def test_zero_weight_drops_the_term_whatever_the_target(self):
        f, o, t = _latents(seed=15)
        got = contrastive_latent_loss(
            (f, o), False, _spec(align_weight=0.0),
            teacher_original_latent=t, align_target="teacher")
        assert torch.allclose(got, _base_loss(f, o, t, moco_rep=False),
                              atol=0.0, rtol=0.0)


class TestAlignTargetResolution:
    """Same arg-over-config-key precedence as the other add-on knobs."""

    def test_config_key_selects_the_teacher(self):
        """train.py sets the run-level knob through the config key."""
        f, o, t = _latents(seed=21)
        got = contrastive_latent_loss(
            (f, o), False, _spec(align_target="teacher"),
            teacher_original_latent=t)
        want = _base_loss(f, o, t, moco_rep=False) + _align_reference(f, t)
        assert torch.allclose(got, want, atol=1e-12, rtol=0.0)

    def test_function_arg_overrides_the_config_key(self):
        f, o, t = _latents(seed=22)
        got = contrastive_latent_loss(
            (f, o), False, _spec(align_target="teacher"),
            teacher_original_latent=t, align_target="student")
        want = _base_loss(f, o, t, moco_rep=False) + _align_reference(f, o)
        assert torch.allclose(got, want, atol=1e-12, rtol=0.0)

    def test_explicit_student_matches_the_absent_key(self):
        f, o, t = _latents(seed=23)
        absent = contrastive_latent_loss((f, o), False, _spec(),
                                         teacher_original_latent=t)
        explicit = contrastive_latent_loss((f, o), False,
                                           _spec(align_target="student"),
                                           teacher_original_latent=t)
        assert torch.allclose(absent, explicit, atol=0.0, rtol=0.0)

    def test_student_target_needs_no_teacher(self):
        """#379's own runs also exist without an EMA teacher — the default
        path must not start requiring one."""
        f, o, _ = _latents(seed=24)
        got = contrastive_latent_loss((f, o), False, _spec())
        want = contrastive_latent_loss(
            (f, o), False, _spec(align_weight=0.0)) + _align_reference(f, o)
        assert torch.allclose(got, want, atol=1e-12, rtol=0.0)


class TestAlignTargetGuards:

    def test_teacher_target_without_a_teacher_raises(self):
        """Silently training on the student is the defect being fixed."""
        f, o, _ = _latents(seed=31)
        with pytest.raises(ValueError, match="align_target"):
            contrastive_latent_loss((f, o), False, _spec(),
                                    align_target="teacher")

    def test_unknown_target_raises(self):
        f, o, t = _latents(seed=32)
        with pytest.raises(ValueError, match="align_target"):
            contrastive_latent_loss((f, o), False, _spec(),
                                    teacher_original_latent=t,
                                    align_target="ema")

    def test_unknown_config_key_value_raises(self):
        f, o, t = _latents(seed=33)
        with pytest.raises(ValueError, match="align_target"):
            contrastive_latent_loss((f, o), False, _spec(align_target="ema"),
                                    teacher_original_latent=t)


class TestAlignTargetGradient:

    def test_no_gradient_reaches_the_teacher(self):
        """The EMA path owns the teacher's update; the add-on must not."""
        f, o, t = _latents(seed=41)
        f.requires_grad_(True)
        t.requires_grad_(True)
        contrastive_latent_loss((f, o), False, _spec(align_weight=1.0),
                                teacher_original_latent=t,
                                align_target="teacher").backward()
        assert f.grad is not None and f.grad.abs().sum() > 0
        assert t.grad is None or t.grad.abs().sum() == 0

    def test_forecaster_gradient_differs_between_targets(self):
        """The add-on pulls f toward a different point, so df must change."""
        grads = {}
        for target in ("student", "teacher"):
            f, o, t = _latents(seed=42)
            f.requires_grad_(True)
            contrastive_latent_loss((f, o), False, _spec(),
                                    teacher_original_latent=t,
                                    align_target=target).backward()
            grads[target] = f.grad.clone()
        assert not torch.allclose(grads["student"], grads["teacher"])


# --- train.py wiring ------------------------------------------------------


def _assert_train_deps_available() -> None:
    for mod in ("torch", "numpy", "datasets"):
        try:
            __import__(mod)
        except ImportError as e:
            pytest.fail(f"train.py dep {mod!r} not importable: {e}")


def _run_train(tmp_path, run_name, extra):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    save_dir = tmp_path / "runs"
    save_dir.mkdir(exist_ok=True)
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
    ] + extra
    return subprocess.run(cmd, env=env, capture_output=True, text=True,
                          cwd=str(tmp_path), timeout=900), save_dir


# The #390 cell shape: main contrastive loss ON, rep_only, align add-on on,
# EMA teacher attached — arm5 stripped down to a 4-step CPU run.
CELL_BASE = [
    "--loss-shape", SHAPE, "--align-loss-weight", "1.0",
    "--ema-embedding", "--ema-encoder", "--ema-tau", "0.9",
]
CELL_STUDENT = CELL_BASE + ["--align-target", "student"]
CELL_TEACHER = CELL_BASE + ["--align-target", "teacher"]


class TestTrainWiring:

    def test_align_target_reaches_the_add_on(self, tmp_path):
        """The point of #390: with the main contrastive loss ON,
        `--align-target teacher` has to change what the run trains on.
        Three 4-step runs, same seed, same arm: two differ only in
        `--align-target`, the third repeats the student one and pins the run
        down as deterministic. If the flag stopped at the standalone
        `align_loss()` (the #388 state), all three columns would match."""
        _assert_train_deps_available()
        names = ("a390student", "a390student2", "a390teacher")
        arms = (CELL_STUDENT, CELL_STUDENT, CELL_TEACHER)
        save_dir = None
        for name, arm in zip(names, arms):
            res, save_dir = _run_train(tmp_path, name, arm)
            if res.returncode != 0:
                pytest.fail(f"{name} rc={res.returncode}\n{res.stderr[-3000:]}")

        def losses(name):
            path = save_dir / f"{name}_losses.csv"
            return [float(r["loss"]) for r in csv.DictReader(open(path))]

        student, repeat, teacher = (losses(n) for n in names)
        assert len(student) == len(teacher) == 4
        assert student == repeat, "the run is not deterministic at this seed"
        assert student != teacher, (
            "--align-target teacher trained on the same loss as student: "
            f"{student} vs {teacher}")

    def test_default_is_the_student_target(self, tmp_path):
        """#379 reproducibility: omitting the flag must equal
        `--align-target student`."""
        _assert_train_deps_available()
        names = ("a390default", "a390explicit")
        arms = (CELL_BASE, CELL_STUDENT)
        save_dir = None
        for name, arm in zip(names, arms):
            res, save_dir = _run_train(tmp_path, name, arm)
            if res.returncode != 0:
                pytest.fail(f"{name} rc={res.returncode}\n{res.stderr[-3000:]}")

        def losses(name):
            path = save_dir / f"{name}_losses.csv"
            return [float(r["loss"]) for r in csv.DictReader(open(path))]

        assert losses(names[0]) == losses(names[1])

    def test_teacher_target_without_a_teacher_aborts(self, tmp_path):
        _assert_train_deps_available()
        res, _ = _run_train(
            tmp_path, "a390noteacher",
            ["--loss-shape", SHAPE, "--align-loss-weight", "1.0",
             "--align-target", "teacher"])
        assert res.returncode != 0
        assert "align-target" in (res.stdout + res.stderr)

    def test_teacher_target_without_the_add_on_aborts(self, tmp_path):
        """`--align-target teacher` with no L_align term at all is a
        no-op flag — the silent-no-op class of bug this issue is about."""
        _assert_train_deps_available()
        res, _ = _run_train(
            tmp_path, "a390noweight",
            ["--loss-shape", SHAPE, "--ema-embedding", "--ema-encoder",
             "--ema-tau", "0.9", "--align-target", "teacher"])
        assert res.returncode != 0
        assert "align-target" in (res.stdout + res.stderr)
