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
7. CALL SITES — ``align_target`` is a RUN-LEVEL key on
   ``LOSS_SPEC.train_configuration``, so every call reading that spec sees it,
   not only the training step. The teacher guard raises, so a call site that
   reads a ``'teacher'`` spec without handing over the teacher latent would
   die *mid-run* — hours into a 200k-step wave. The shapes of the trainer's
   other calls are pinned here.
8. NO ASSERT — ``python -O`` strips ``assert``. A stripped guard is exactly
   the silent student fallback this flag exists to remove, so no guard on the
   teacher target may be an ``assert``.
"""

from __future__ import annotations

import ast
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


class TestAlignTargetCallSites:
    """Every call site that reads a shared spec has to stay alive.

    ``align_target`` is a run-level key, so it reaches every call passing
    that spec. Before #390 those calls could omit the teacher latent
    harmlessly; now the guard raises. The trainer makes two calls per step —
    the training one (teacher passed) and the ``loss_tau_ref`` diagnostic
    (weight forced to 0, no teacher) — and the two optional entry points
    (``validation=True``, ``get_history=True``) take the same branch.
    """

    def test_validation_call_honours_the_teacher_target(self):
        """`validation` gates only the latent-noise injection, so the
        add-on must be the same term on both sides."""
        f, o, t = _latents(seed=51)
        got = contrastive_latent_loss(
            (f, o), True, _spec(align_target="teacher"),
            teacher_original_latent=t)
        train = contrastive_latent_loss(
            (f, o), False, _spec(align_target="teacher"),
            teacher_original_latent=t)
        assert torch.allclose(got, train, atol=0.0, rtol=0.0)

    def test_validation_call_without_a_teacher_raises(self):
        """The guard is not gated on `validation` — a validation call
        reading a 'teacher' spec must not silently score the student."""
        f, o, _ = _latents(seed=52)
        with pytest.raises(ValueError, match="align_target"):
            contrastive_latent_loss((f, o), True, _spec(align_target="teacher"))

    def test_loss_tau_ref_call_shape_survives_a_teacher_spec(self):
        """train.py's per-step diagnostic passes `align_loss_weight=0.0`
        and NO teacher while the shared spec says `align_target='teacher'`.
        It must stay a pure contrastive reference, not raise. This is the
        crash that would land hours into a wave."""
        f, o, _ = _latents(seed=53)
        spec = _spec(align_target="teacher")
        got = contrastive_latent_loss(
            (f.detach(), o.detach()), False, spec,
            tau_override=torch.tensor(0.07, dtype=f.dtype),
            align_loss_weight=0.0,
            subtract_contrastive_floor=False,
            moco_negatives=False, moco_rep_keys=False)
        want = contrastive_latent_loss(
            (f.detach(), o.detach()), False, _spec(align_weight=0.0),
            tau_override=torch.tensor(0.07, dtype=f.dtype),
            align_loss_weight=0.0,
            subtract_contrastive_floor=False,
            moco_negatives=False, moco_rep_keys=False)
        assert torch.allclose(got, want, atol=0.0, rtol=0.0)

    def test_zero_weight_in_the_config_needs_no_teacher(self):
        """Same guarantee through the config key rather than the arg: a
        run-level 'teacher' target with the term off costs nothing."""
        f, o, _ = _latents(seed=54)
        got = contrastive_latent_loss(
            (f, o), False, _spec(align_weight=0.0, align_target="teacher"))
        want = contrastive_latent_loss((f, o), False, _spec(align_weight=0.0))
        assert torch.allclose(got, want, atol=0.0, rtol=0.0)

    def test_get_history_call_honours_the_teacher_target(self):
        """The diagnostic history path returns through the same branch."""
        f, o, t = _latents(seed=55)
        got, hist = contrastive_latent_loss(
            (f, o), False, _spec(align_target="teacher"),
            teacher_original_latent=t, get_history=True)
        want = _base_loss(f, o, t, moco_rep=False) + _align_reference(f, t)
        assert torch.allclose(got, want, atol=1e-12, rtol=0.0)
        assert hist == (f, o)

    def test_get_history_without_a_teacher_raises(self):
        f, o, _ = _latents(seed=56)
        with pytest.raises(ValueError, match="align_target"):
            contrastive_latent_loss((f, o), False, _spec(align_target="teacher"),
                                    get_history=True)

    def test_trainer_call_sites_all_stay_on_the_safe_side(self):
        """The two calls train.py makes with the shared LOSS_SPEC, checked at
        the source. Each must EITHER hand over the teacher latent OR force
        the add-on off. Losing either property turns a teacher-target run
        into a crash hours into a wave, which unit tests on synthetic
        tensors would not catch."""
        tree = ast.parse(TRAIN_PY.read_text())
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "contrastive_latent_loss"]
        assert len(calls) == 2, (
            f"expected the training call and the loss_tau_ref diagnostic, "
            f"found {len(calls)} — new call site, re-check this guarantee.")
        for call in calls:
            kw = {k.arg: k for k in call.keywords}
            passes_teacher = "teacher_original_latent" in kw
            forces_off = (
                "align_loss_weight" in kw
                and ast.unparse(kw["align_loss_weight"].value) == "0.0")
            assert passes_teacher or forces_off, (
                f"{TRAIN_PY.name}:{call.lineno} reads LOSS_SPEC (which may "
                "carry align_target='teacher') but neither passes "
                "teacher_original_latent nor forces align_loss_weight=0.0.")


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


def _run_train(tmp_path, run_name, extra, py_flags=()):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    save_dir = tmp_path / "runs"
    save_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable, "-u", *py_flags, str(TRAIN_PY),
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


class TestGuardsSurviveOptimizedMode:
    """`python -O` strips every `assert`. A guard that vanishes under -O
    silently reinstates the student target — the #382 defect."""

    def test_no_assert_guards_the_teacher_target(self):
        """Source-level, because the guard sits on a branch the CLI checks
        already make unreachable: with `--align-target teacher` argparse
        demands an EMA teacher, so `teacher_o_lat` is never None there. It
        is defence in depth, and defence in depth must not evaporate."""
        tree = ast.parse(TRAIN_PY.read_text())
        offenders = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Assert)
            and "teacher" in ast.unparse(node)
            and "align" in ast.unparse(node).lower()]
        assert not offenders, (
            "assert guards the L_align teacher target at "
            f"{TRAIN_PY.name}:{offenders} — `python -O` removes it and the "
            "target falls back to the student. Raise instead.")

    def test_cli_guard_still_aborts_under_dash_O(self, tmp_path):
        """End-to-end -O run of the combination the launcher can never
        produce but a hand-edited command line can."""
        _assert_train_deps_available()
        res, _ = _run_train(
            tmp_path, "a390dashO",
            ["--loss-shape", SHAPE, "--align-loss-weight", "1.0",
             "--align-target", "teacher"], py_flags=("-O",))
        assert res.returncode != 0
        assert "align-target" in (res.stdout + res.stderr)
