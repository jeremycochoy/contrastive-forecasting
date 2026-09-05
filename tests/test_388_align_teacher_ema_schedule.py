"""Tests for #388: teacher-targeted L_align and the linear EMA-momentum schedule.

Three pieces of behaviour, each independently defaulting to the #382 run:

1. :func:`src.loss.align_loss` accepts an explicit ``target_latent``. Passing
   the EMA teacher's ``h`` makes L_align a teacher-target BYOL term. Omitting
   it keeps the student's own ``sg(h_{t+1})`` target, byte-for-byte.
2. :func:`src.models.ema_tau_at_step` interpolates the EMA momentum α linearly
   from a start to an end value across the step budget. ``end=None`` (default)
   returns the start value at every step, so existing runs are unchanged.
3. train.py wires both through CLI flags, writes the live α to
   ``<run>_losses.csv``, and the latent-drift probe records the teacher's
   ``h_t`` next to the student's.
4. Both CSVs refuse to append under a header from another schema: the runs
   auto-resume, and a pre-#388 file would take the new rows silently and
   shift every column after the first.
5. `vs_initial` drift rows are grouped by their reference step, because a
   resumed probe re-seeds that reference.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.forecasting_head import (extract_encoder_latents,
                                  extract_teacher_encoder_latents)
from src.loss import align_loss
from src.models import ConfigurableModel, ema_tau_at_step

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "experiments" / "2026-04-27_freq-embedding" / "scripts" / "train.py"


def _latents(B=3, T=5, C=1, H=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    o = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    t = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    return f, o, t


def _align_reference(f, target):
    """L_align written out longhand: 2 - 2*cos(f_t, target_{t+1}), meaned."""
    fn = F.normalize(f, p=2, dim=-1)[:, :-1]
    tn = F.normalize(target, p=2, dim=-1)[:, 1:]
    return (2.0 - 2.0 * (fn * tn).sum(-1)).mean()


# --- 1. align_loss: explicit target ---------------------------------------


class TestAlignTarget:

    def test_default_target_is_the_student_latent(self):
        """No ``target_latent`` ⇒ the #382 objective, exactly."""
        f, o, _ = _latents(seed=11)
        assert torch.allclose(align_loss(f, o), _align_reference(f, o),
                              atol=0.0, rtol=0.0)

    def test_teacher_target_uses_the_teacher_latent(self):
        f, o, t = _latents(seed=12)
        got = align_loss(f, o, target_latent=t)
        assert torch.allclose(got, _align_reference(f, t), atol=0.0, rtol=0.0)

    def test_teacher_target_changes_the_value(self):
        """Guard against a signature that silently ignores the new argument."""
        f, o, t = _latents(seed=13)
        assert not torch.allclose(align_loss(f, o), align_loss(f, o, target_latent=t))

    def test_weight_scales_the_teacher_target_term(self):
        f, o, t = _latents(seed=14)
        one = align_loss(f, o, weight=1.0, target_latent=t)
        half = align_loss(f, o, weight=0.5, target_latent=t)
        assert torch.allclose(half, 0.5 * one)

    def test_no_gradient_reaches_the_target(self):
        """The teacher is a stop-grad target — the EMA path owns its update."""
        f, o, t = _latents(seed=15)
        f.requires_grad_(True)
        o.requires_grad_(True)
        t.requires_grad_(True)
        align_loss(f, o, target_latent=t).backward()
        assert f.grad is not None and f.grad.abs().sum() > 0
        assert t.grad is None or t.grad.abs().sum() == 0
        assert o.grad is None or o.grad.abs().sum() == 0


# --- 2. ema_tau_at_step: linear schedule ----------------------------------


class TestEmaTauSchedule:

    def test_no_end_value_is_constant(self):
        for step in (0, 1, 500, 100_000):
            assert ema_tau_at_step(step, 100_000, 0.9, None) == 0.9

    def test_endpoints(self):
        assert ema_tau_at_step(0, 100_000, 0.9, 1.0) == pytest.approx(0.9)
        assert ema_tau_at_step(100_000, 100_000, 0.9, 1.0) == pytest.approx(1.0)

    def test_linear_midpoint(self):
        assert ema_tau_at_step(50_000, 100_000, 0.9, 1.0) == pytest.approx(0.95)
        assert ema_tau_at_step(25_000, 100_000, 0.9, 1.0) == pytest.approx(0.925)

    def test_monotone_and_bounded(self):
        vals = [ema_tau_at_step(s, 1000, 0.9, 1.0) for s in range(0, 1001, 50)]
        assert vals == sorted(vals)
        assert min(vals) >= 0.9 and max(vals) <= 1.0

    def test_clamped_beyond_the_budget(self):
        """A resume that overshoots --total-steps must not push α past the end."""
        assert ema_tau_at_step(120_000, 100_000, 0.9, 1.0) == pytest.approx(1.0)
        assert ema_tau_at_step(-5, 100_000, 0.9, 1.0) == pytest.approx(0.9)

    def test_decreasing_schedule_is_allowed(self):
        assert ema_tau_at_step(50, 100, 0.99, 0.9) == pytest.approx(0.945)

    def test_zero_budget_returns_the_end_value(self):
        assert ema_tau_at_step(0, 0, 0.9, 1.0) == pytest.approx(1.0)


# --- 3. teacher latents helper -------------------------------------------


def _tiny_model(**over):
    cfg = dict(C=1, H=16, W=8, nhead=2, num_layers=1, num_encoder_layers=1,
               encoder_type="gru", dropout=0.0, rev_norm_kind="ewma",
               rev_norm_span=8, ema_embedding=True, ema_encoder=True)
    cfg.update(over)
    return ConfigurableModel(**cfg).eval()


class TestTeacherEncoderLatents:

    def test_matches_the_student_at_initialisation(self):
        """The teacher is a deepcopy of the student at step 0, so both paths
        must return the same h. This is what makes drift_cos(student, teacher)
        a meaningful quantity later in training."""
        model = _tiny_model()
        x = torch.randn(2, 64, 1, generator=torch.Generator().manual_seed(3))
        student, _ = extract_encoder_latents(model, x)
        teacher, _ = extract_teacher_encoder_latents(model, x)
        assert teacher.shape == student.shape
        assert torch.allclose(teacher, student, atol=1e-5)

    def test_diverges_after_a_teacher_update_toward_a_moved_student(self):
        model = _tiny_model()
        x = torch.randn(2, 64, 1, generator=torch.Generator().manual_seed(4))
        with torch.no_grad():
            for p in model.transformer.input_to_latent.parameters():
                p.add_(0.1)
        model.update_teacher(0.9)
        student, _ = extract_encoder_latents(model, x)
        teacher, _ = extract_teacher_encoder_latents(model, x)
        assert not torch.allclose(teacher, student, atol=1e-4)

    def test_raises_without_a_teacher(self):
        model = _tiny_model(ema_embedding=False, ema_encoder=False)
        x = torch.randn(2, 64, 1)
        with pytest.raises(RuntimeError):
            extract_teacher_encoder_latents(model, x)


# --- 4. train.py wiring (static) ------------------------------------------


def parse_argparse_defaults() -> dict[str, object]:
    """Return the default= value of every add_argument call in train.py."""
    tree = ast.parse(TRAIN_PY.read_text())
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value.startswith("--")):
            continue
        default = None
        for kw in node.keywords:
            if kw.arg == "default":
                try:
                    default = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    default = "<non-literal>"
                break
        defaults[first.value] = default
    return defaults


class TestTrainFlags:

    def test_align_target_defaults_to_student(self):
        """#382 stays reproducible: the target only moves when asked."""
        defaults = parse_argparse_defaults()
        assert "--align-target" in defaults
        assert defaults["--align-target"] == "student"

    def test_ema_tau_end_defaults_to_none(self):
        """No end value ⇒ α constant ⇒ existing runs unchanged."""
        defaults = parse_argparse_defaults()
        assert "--ema-tau-end" in defaults
        assert defaults["--ema-tau-end"] is None

    def test_ema_tau_default_unchanged(self):
        assert parse_argparse_defaults()["--ema-tau"] == 0.99

    def test_teacher_update_uses_the_schedule(self):
        """`update_teacher(args.ema_tau)` would freeze α at the start value."""
        src = TRAIN_PY.read_text()
        assert "model.update_teacher(args.ema_tau)" not in src
        assert "ema_tau_at_step(" in src


# --- 5. end-to-end on CPU -------------------------------------------------


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


# The standalone L_align arm: main contrastive loss off, align term on.
NO_TEACHER_ARM = [
    "--loss-shape", "cosine_similarity_batch_no_time_neg",
    "--no-main-contrastive-loss", "--align-loss-weight", "1.0",
]
# The same arm with an EMA teacher attached, at #388's starting α.
ARM_BASE = NO_TEACHER_ARM + ["--ema-embedding", "--ema-encoder",
                             "--ema-tau", "0.9"]
TEACHER_ARM = ARM_BASE + ["--align-target", "teacher"]
STUDENT_ARM = ARM_BASE + ["--align-target", "student"]


class TestEndToEnd:

    def test_align_target_reaches_the_loss(self, tmp_path):
        """The whole point of #388: the teacher's h has to arrive at
        ``align_loss``. Three 4-step runs, same seed, same arm, same RNG
        draws. Two of them differ only in ``--align-target``; the third
        repeats the student one and pins the run down as deterministic, so
        the difference can only come from the target. If train.py passed the
        student target in both (the #382 bug), the columns would match."""
        _assert_train_deps_available()
        names = ("a388student", "a388student2", "a388teacher")
        arms = (STUDENT_ARM, STUDENT_ARM, TEACHER_ARM)
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

    def test_ema_tau_is_blank_without_a_teacher(self, tmp_path):
        """The column is promised blank — not 0, not the default α — for a
        run with no teacher, so a reader can tell 'no teacher' from 'α=0.9'."""
        _assert_train_deps_available()
        res, save_dir = _run_train(tmp_path, "a388noema", NO_TEACHER_ARM)
        if res.returncode != 0:
            pytest.fail(f"rc={res.returncode}\n{res.stderr[-3000:]}")
        rows = list(csv.DictReader(open(save_dir / "a388noema_losses.csv")))
        assert rows and "ema_tau" in rows[0]
        assert {r["ema_tau"] for r in rows} == {""}

    def test_scheduled_alpha_is_logged_every_step(self, tmp_path):
        """α must appear in <run>_losses.csv and move from start to end."""
        _assert_train_deps_available()
        res, save_dir = _run_train(
            tmp_path, "a388sched", TEACHER_ARM + ["--ema-tau-end", "1.0",
                                                  "--latent-drift-probe-every", "2"])
        if res.returncode != 0:
            pytest.fail(f"rc={res.returncode}\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")

        rows = list(csv.DictReader(open(save_dir / "a388sched_losses.csv")))
        assert rows, "no rows written"
        assert "ema_tau" in rows[0], f"columns: {list(rows[0])}"
        alphas = [float(r["ema_tau"]) for r in rows]
        assert len(alphas) == 4, f"expected one row per step, got {len(alphas)}"
        assert alphas == sorted(alphas)
        assert alphas[0] == pytest.approx(0.925)     # step 1 of 4, 0.9 -> 1.0
        assert alphas[-1] == pytest.approx(1.0)

    def test_probe_records_student_and_teacher(self, tmp_path):
        """The drift CSV gains a `latent` column with both encoders in it."""
        _assert_train_deps_available()
        res, save_dir = _run_train(
            tmp_path, "a388probe", TEACHER_ARM + ["--latent-drift-probe-every", "2"])
        if res.returncode != 0:
            pytest.fail(f"rc={res.returncode}\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")

        rows = list(csv.DictReader(open(save_dir / "a388probe_latent_drift.csv")))
        assert rows, "probe wrote no comparison rows"
        assert "latent" in rows[0], f"columns: {list(rows[0])}"
        assert {r["latent"] for r in rows} == {"student_h", "teacher_h"}
        for r in rows:
            assert 0.0 <= float(r["drift_cos"]) <= 2.0

    def test_constant_alpha_when_no_end_value(self, tmp_path):
        """Default path: one α for the whole run, as in #382."""
        _assert_train_deps_available()
        res, save_dir = _run_train(tmp_path, "a388const", TEACHER_ARM)
        if res.returncode != 0:
            pytest.fail(f"rc={res.returncode}\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
        rows = list(csv.DictReader(open(save_dir / "a388const_losses.csv")))
        assert {float(r["ema_tau"]) for r in rows} == {0.9}

    def test_schedule_survives_a_resume(self, tmp_path):
        """α is a function of the global step, so a run that dies at step 2
        and resumes must pick the schedule up where it left off — not
        restart it. Also guards the CSV append: the new `ema_tau` column
        must not trip the resume schema check."""
        _assert_train_deps_available()
        res, save_dir = _run_train(
            tmp_path, "a388res",
            TEACHER_ARM + ["--ema-tau-end", "1.0", "--save-every", "2"])
        if res.returncode != 0:
            pytest.fail(f"first leg rc={res.returncode}\n{res.stderr[-3000:]}")
        ckpt = save_dir / "a388res_0k.pth"
        assert ckpt.exists(), sorted(p.name for p in save_dir.iterdir())

        res2, _ = _run_train(
            tmp_path, "a388res",
            TEACHER_ARM + ["--ema-tau-end", "1.0", "--save-every", "2",
                           "--resume", str(ckpt)])
        if res2.returncode != 0:
            pytest.fail(f"resume rc={res2.returncode}\n{res2.stderr[-3000:]}")

        rows = list(csv.DictReader(open(save_dir / "a388res_losses.csv")))
        by_step = {int(r["step"]): float(r["ema_tau"]) for r in rows}
        # Steps 3 and 4 come from the resumed leg. α must be 0.975 / 1.0,
        # the values the schedule gives at those global steps.
        assert by_step[3] == pytest.approx(0.975)
        assert by_step[4] == pytest.approx(1.0)

    def test_align_target_teacher_without_a_teacher_aborts(self, tmp_path):
        """Silently falling back to the student target is what #382 did by
        accident. Fail loudly instead."""
        _assert_train_deps_available()
        res, _ = _run_train(tmp_path, "a388noteacher",
                            NO_TEACHER_ARM + ["--align-target", "teacher"])
        assert res.returncode != 0
        assert "align-target" in (res.stdout + res.stderr)


# --- 6. CSV schema guards and probe teacher detection ---------------------


def load_train_module():
    """Import train.py as a module — its ``main()`` sits behind ``__main__``."""
    _assert_train_deps_available()
    spec = importlib.util.spec_from_file_location("train_py_388", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The losses-CSV header as it stood before #388 appended `ema_tau` — what a
# run started on the #382 code left on disk.
LOSSES_HEADER_PRE_388 = [
    "step", "loss", "loss_tau_ref", "gap", "gap_ratio", "ff", "fp", "tp",
    "cross_batch", "hf_rows_consumed", "synth_rows_consumed", "mixup_applied",
    "r2_random", "r2_naive", "u_temporal", "u_batch", "auc", "top1", "top3",
    "cpc_aux", "sigreg_e", "sigreg_h", "u_temporal_e", "u_batch_e",
    "u_batchtime", "u_batchtime_e",
]

# The four columns #409 appended after `ema_tau`: the live weight on L_rep
# and the three per-term readouts.
LOSSES_COLUMNS_409 = ["rep_w", "l_pred", "l_rep", "l_align"]

# The drift-CSV header as it stood before #388 inserted `latent` (PR #387).
DRIFT_HEADER_PRE_388 = [
    "step", "kind", "step_ref", "delta_step",
    "drift_cos", "drift_cos_aligned", "rot_gap", "cka",
]


def write_csv(path, header, rows=()):
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)
    return str(path)


class TestLossesCsvResumeSchema:

    def test_pre_388_losses_csv_is_refused(self, tmp_path):
        """A #382 CSV has 26 columns, the #388 writer 27. Appending would
        shift every value one column left from the resume step on."""
        train = load_train_module()
        path = write_csv(tmp_path / "old_losses.csv", LOSSES_HEADER_PRE_388,
                         [[0] * len(LOSSES_HEADER_PRE_388)])
        with pytest.raises(SystemExit, match="schema mismatch"):
            train.CSVLogger(path)

    def test_current_losses_csv_is_appended_to(self, tmp_path):
        train = load_train_module()
        path = write_csv(tmp_path / "cur_losses.csv",
                         LOSSES_HEADER_PRE_388 + ["ema_tau"]
                         + LOSSES_COLUMNS_409)
        logger = train.CSVLogger(path)
        logger.close()
        with open(path) as fh:
            assert len(list(csv.reader(fh))) == 1     # no duplicate header


class TestDriftCsvResumeSchema:

    def probe(self, train, path):
        return train.LatentDriftProbe(str(path), torch.zeros(2, 64, 1), "cpu")

    def test_pre_388_drift_csv_is_refused(self, tmp_path):
        """`latent` is column 2, so a pre-#388 file cannot take the new rows:
        the header would say `kind` where the row now says `student_h`."""
        train = load_train_module()
        path = write_csv(tmp_path / "old_latent_drift.csv",
                         DRIFT_HEADER_PRE_388,
                         [[0, "adjacent", 0, 0, 0.0, 0.0, 0.0, 1.0]])
        with pytest.raises(SystemExit, match="schema mismatch"):
            self.probe(train, path)

    def test_current_drift_csv_is_appended_to(self, tmp_path):
        train = load_train_module()
        path = write_csv(tmp_path / "cur_latent_drift.csv",
                         train.LatentDriftProbe.COLUMNS)
        self.probe(train, path).close()
        with open(path) as fh:
            assert len(list(csv.reader(fh))) == 1     # no duplicate header

    def test_column_order_is_the_one_the_runs_write(self):
        """The #388 runs are in flight against this exact order. Moving a
        column would corrupt their CSV on the next resume."""
        train = load_train_module()
        assert train.LatentDriftProbe.COLUMNS == [
            "step", "latent", "kind", "step_ref", "delta_step",
            "drift_cos", "drift_cos_aligned", "rot_gap", "cka"]


class TestProbeTeacherDetection:
    """The probe reads the teacher off the model by attribute. A rename would
    silently drop every teacher_h row, so pin the detection to a real model."""

    def probe_of(self, train, tmp_path, model):
        p = train.LatentDriftProbe(str(tmp_path / "d.csv"),
                                   torch.zeros(2, 64, 1), "cpu")
        try:
            return p.extract_h(model)
        finally:
            p.close()

    def test_teacher_h_present_when_the_model_has_a_teacher(self, tmp_path):
        train = load_train_module()
        h = self.probe_of(train, tmp_path, _tiny_model())
        assert set(h) == {"student_h", "teacher_h"}
        assert h["teacher_h"].shape == h["student_h"].shape

    def test_encoder_only_teacher_is_detected(self, tmp_path):
        """--ema-encoder alone still makes a teacher."""
        train = load_train_module()
        h = self.probe_of(train, tmp_path,
                          _tiny_model(ema_embedding=False, ema_encoder=True))
        assert set(h) == {"student_h", "teacher_h"}

    def test_student_only_without_a_teacher(self, tmp_path):
        train = load_train_module()
        h = self.probe_of(train, tmp_path,
                          _tiny_model(ema_embedding=False, ema_encoder=False))
        assert set(h) == {"student_h"}

    def test_probe_leaves_training_mode_untouched(self, tmp_path):
        train = load_train_module()
        model = _tiny_model().train()
        self.probe_of(train, tmp_path, model)
        assert model.training


# --- 7. drift curves across a resume --------------------------------------


MAKE_PLOTS = (REPO_ROOT / "experiments"
              / "2026-08-01_align_teacher_ema_schedule" / "scripts"
              / "make_plots.py")

DRIFT_COLUMNS = ["run", "arm", "alpha", "latent", "kind", "step", "step_ref",
                 "delta_step", "drift_cos", "drift_cos_aligned", "rot_gap",
                 "cka"]


def load_make_plots():
    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location("make_plots_388", MAKE_PLOTS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def drift_row(kind, step, step_ref, drift):
    return ["align_teacher_a09", "align_teacher", "const_0.9", "student_h",
            kind, step, step_ref, step - step_ref, drift, drift, 0.0, 1.0]


class TestDriftSeriesAcrossAResume:
    """run_arm.sh auto-resumes, and a resumed probe re-seeds its `vs_initial`
    reference to the resume step. Two references are two curves."""

    def test_vs_initial_splits_on_the_reference_step(self, tmp_path):
        mp = load_make_plots()
        path = write_csv(tmp_path / "drift.csv", DRIFT_COLUMNS, [
            drift_row("vs_initial", 10_000, 5_000, 0.20),
            drift_row("vs_initial", 15_000, 5_000, 0.30),
            drift_row("vs_initial", 25_000, 20_000, 0.05),   # after a resume
            drift_row("vs_initial", 30_000, 20_000, 0.09),
        ])
        got = mp.curves(mp.read_drift(path), "align_teacher", "vs_initial")
        assert len(got) == 2, "two baselines were joined into one curve"
        assert sorted(len(pts) for _, _, pts in got) == [2, 2]

    def test_adjacent_rows_stay_one_curve(self, tmp_path):
        """`adjacent` references the previous probe, so its step_ref moves
        every row — keying on it would shatter the curve into single points."""
        mp = load_make_plots()
        path = write_csv(tmp_path / "drift.csv", DRIFT_COLUMNS, [
            drift_row("adjacent", 10_000, 5_000, 0.20),
            drift_row("adjacent", 15_000, 10_000, 0.18),
            drift_row("adjacent", 20_000, 15_000, 0.17),
        ])
        got = mp.curves(mp.read_drift(path), "align_teacher", "adjacent")
        assert len(got) == 1
        assert [s for s, _ in got[0][2]] == [10_000, 15_000, 20_000]
