"""Tests for #409: a linear schedule on the L_rep weight, and per-term logging.

Four pieces of behaviour. Each one defaults to the objective every published
run trained on, so an existing command line is unchanged.

1. :func:`src.models.linear_schedule_at_step` is the ramp
   :func:`src.models.ema_tau_at_step` already used, named and reusable.
   ``value_end=None`` returns the start value at every step.
2. :func:`src.loss.contrastive_latent_loss` takes a ``rep_loss_weight``
   function argument. It scales L_rep in BOTH shapes that carry that term:
   ``..._split_pred_rep`` and ``..._rep_only``. Before #409 the second shape
   read no weight at all, so the best cell of the project could not decay it.
   Weight 0.0 skips the whole term.
3. The same function fills an optional ``term_out`` dict with the UNWEIGHTED
   value of L_pred, L_rep and L_align. Before #409 neither L_rep nor L_align
   reached the CSV, and a report had to read L_rep as the residual of the
   total (see reports/2026-08-19_ema_momentum_k32/scripts/plot_loss_terms.py).
4. train.py wires the schedule through ``--rep-loss-weight-end`` /
   ``--rep-loss-weight-ramp-steps``, writes the live weight and the three
   terms to ``<run>_losses.csv``, and refuses a flag combination that would
   train something other than what it says.
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

from src.loss import contrastive_latent_loss
from src.models import ema_tau_at_step, linear_schedule_at_step

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "experiments" / "2026-04-27_freq-embedding" / "scripts" / "train.py"

REP_ONLY = "cosine_similarity_batch_rep_only"
SPLIT = "cosine_similarity_batch_split_pred_rep"


def _spec(shape, **extra):
    cfg = {
        "contrastive_divergence_temperature": 0.10,
        "contrastive_latent_noise": None,
        "loss_shape": shape,
        "contrastive_latent_delay": 0,
    }
    cfg.update(extra)
    return SimpleNamespace(train_configuration=cfg)


def _latents(B=3, T=5, C=2, H=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    o = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    return f, o


# --- 1. the linear ramp ---------------------------------------------------


class TestLinearSchedule:

    def test_no_end_value_is_constant(self):
        for step in (0, 1, 500, 100_000):
            assert linear_schedule_at_step(step, 100_000, 1.0, None) == 1.0

    def test_endpoints(self):
        assert linear_schedule_at_step(0, 40_000, 1.0, 0.0, 10_000) == pytest.approx(1.0)
        assert linear_schedule_at_step(10_000, 40_000, 1.0, 0.0, 10_000) == pytest.approx(0.0)

    def test_linear_midpoint(self):
        assert linear_schedule_at_step(5_000, 40_000, 1.0, 0.0, 10_000) == pytest.approx(0.5)
        assert linear_schedule_at_step(2_500, 40_000, 1.0, 0.0, 10_000) == pytest.approx(0.75)

    def test_held_at_the_end_value_past_the_ramp(self):
        """#409's runs train to 40,000 and the ramp ends at 10,000."""
        for step in (10_001, 20_000, 40_000):
            assert linear_schedule_at_step(step, 40_000, 1.0, 0.0, 10_000) == 0.0

    def test_ramp_steps_anchors_the_ramp(self):
        """A leg resumes with a new --total-steps. The ramp must not move."""
        assert linear_schedule_at_step(5_000, 40_000, 1.0, 0.0, 10_000) == \
            linear_schedule_at_step(5_000, 200_000, 1.0, 0.0, 10_000)

    def test_without_ramp_steps_the_span_is_the_budget(self):
        assert linear_schedule_at_step(5_000, 10_000, 1.0, 0.0) == pytest.approx(0.5)

    def test_zero_span_returns_the_end_value(self):
        assert linear_schedule_at_step(0, 0, 1.0, 0.0) == 0.0

    def test_negative_step_is_clamped(self):
        assert linear_schedule_at_step(-5, 40_000, 1.0, 0.0, 10_000) == pytest.approx(1.0)

    def test_ema_tau_at_step_is_unchanged(self):
        """#388's schedule keeps its values, whatever it delegates to."""
        assert ema_tau_at_step(0, 100_000, 0.9, None) == 0.9
        assert ema_tau_at_step(50_000, 100_000, 0.9, 1.0) == pytest.approx(0.95)
        assert ema_tau_at_step(120_000, 100_000, 0.9, 1.0) == pytest.approx(1.0)
        assert ema_tau_at_step(40_000, 200_000, 0.9, 1.0, 100_000) == pytest.approx(0.94)


# --- 2. the weight reaches L_rep -----------------------------------------


class TestRepWeightOnRepOnly:
    """The whole main loss of the project's best cell IS L_rep."""

    def test_default_is_the_published_objective(self):
        f, o = _latents(seed=1)
        base = contrastive_latent_loss((f, o), False, _spec(REP_ONLY))
        one = contrastive_latent_loss((f, o), False, _spec(REP_ONLY),
                                      rep_loss_weight=1.0)
        assert torch.equal(base, one)

    def test_config_key_default_is_the_published_objective(self):
        f, o = _latents(seed=1)
        base = contrastive_latent_loss((f, o), False, _spec(REP_ONLY))
        keyed = contrastive_latent_loss(
            (f, o), False, _spec(REP_ONLY, rep_loss_weight=1.0))
        assert torch.equal(base, keyed)

    def test_the_weight_scales_the_term(self):
        f, o = _latents(seed=2)
        one = contrastive_latent_loss((f, o), False, _spec(REP_ONLY),
                                      rep_loss_weight=1.0)
        half = contrastive_latent_loss((f, o), False, _spec(REP_ONLY),
                                       rep_loss_weight=0.5)
        assert half == pytest.approx(0.5 * float(one))

    def test_weight_zero_removes_the_term(self):
        f, o = _latents(seed=3)
        got = contrastive_latent_loss((f, o), False, _spec(REP_ONLY),
                                      rep_loss_weight=0.0)
        assert float(got) == 0.0

    def test_weight_zero_carries_no_gradient_from_the_term(self):
        """At weight 0.0 nothing pushes the representations apart. The
        encoder must take no gradient from L_rep."""
        f, o = _latents(seed=4)
        o.requires_grad_(True)
        loss = contrastive_latent_loss((f, o), False, _spec(REP_ONLY),
                                       rep_loss_weight=0.0)
        assert not loss.requires_grad

    def test_the_function_argument_overrides_the_config_key(self):
        f, o = _latents(seed=5)
        spec = _spec(REP_ONLY, rep_loss_weight=1.0)
        got = contrastive_latent_loss((f, o), False, spec, rep_loss_weight=0.0)
        assert float(got) == 0.0

    def test_the_align_add_on_survives_weight_zero(self):
        """L_align is what the run keeps training on after the decay."""
        f, o = _latents(seed=6)
        f.requires_grad_(True)
        loss = contrastive_latent_loss((f, o), False, _spec(REP_ONLY),
                                       rep_loss_weight=0.0,
                                       align_loss_weight=1.0)
        assert float(loss.detach()) > 0.0
        loss.backward()
        assert f.grad is not None and f.grad.abs().sum() > 0


class TestRepWeightOnSplit:

    def test_default_is_the_published_objective(self):
        f, o = _latents(seed=7)
        base = contrastive_latent_loss((f, o), False, _spec(SPLIT))
        one = contrastive_latent_loss((f, o), False, _spec(SPLIT),
                                      rep_loss_weight=1.0)
        assert torch.equal(base, one)

    def test_the_function_argument_overrides_the_config_key(self):
        f, o = _latents(seed=8)
        spec = _spec(SPLIT, rep_loss_weight=1.0)
        terms = {}
        got = contrastive_latent_loss((f, o), False, spec,
                                      rep_loss_weight=0.0, term_out=terms)
        assert float(got) == pytest.approx(terms["l_pred"])


# --- 3. the terms reach a caller -----------------------------------------


class TestTermOut:

    def test_rep_only_reports_the_unweighted_term(self):
        f, o = _latents(seed=9)
        terms = {}
        got = contrastive_latent_loss((f, o), False, _spec(REP_ONLY),
                                      rep_loss_weight=0.25, term_out=terms)
        assert float(got) == pytest.approx(0.25 * terms["l_rep"])

    def test_split_terms_rebuild_the_total(self):
        f, o = _latents(seed=10)
        terms = {}
        got = contrastive_latent_loss((f, o), False, _spec(SPLIT),
                                      rep_loss_weight=0.5, term_out=terms)
        rebuilt = terms["l_pred"] + 0.5 * terms["l_rep"]
        assert float(got) == pytest.approx(rebuilt)

    def test_align_is_reported_unweighted(self):
        f, o = _latents(seed=11)
        one, three = {}, {}
        contrastive_latent_loss((f, o), False, _spec(REP_ONLY),
                                align_loss_weight=1.0, term_out=one)
        contrastive_latent_loss((f, o), False, _spec(REP_ONLY),
                                align_loss_weight=3.0, term_out=three)
        assert one["l_align"] == pytest.approx(three["l_align"])

    def test_align_rebuilds_the_total(self):
        f, o = _latents(seed=12)
        terms = {}
        got = contrastive_latent_loss((f, o), False, _spec(REP_ONLY),
                                      align_loss_weight=2.0, term_out=terms)
        assert float(got) == pytest.approx(terms["l_rep"] + 2.0 * terms["l_align"])

    def test_a_skipped_term_is_absent_not_zero(self):
        """A reader must tell 'the term was off' from 'the term read 0'."""
        f, o = _latents(seed=13)
        terms = {}
        contrastive_latent_loss((f, o), False, _spec(REP_ONLY),
                                rep_loss_weight=0.0, term_out=terms)
        assert "l_rep" not in terms
        assert "l_align" not in terms

    def test_no_dict_is_the_default(self):
        """term_out=None costs nothing and changes nothing."""
        f, o = _latents(seed=14)
        base = contrastive_latent_loss((f, o), False, _spec(REP_ONLY))
        with_dict = contrastive_latent_loss((f, o), False, _spec(REP_ONLY),
                                            term_out={})
        assert torch.equal(base, with_dict)

    def test_the_reported_terms_carry_no_graph(self):
        f, o = _latents(seed=15)
        o.requires_grad_(True)
        terms = {}
        contrastive_latent_loss((f, o), False, _spec(REP_ONLY), term_out=terms)
        assert isinstance(terms["l_rep"], float)


# --- 4. train.py wiring (static) -----------------------------------------


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

    def test_the_schedule_flags_default_to_off(self):
        defaults = parse_argparse_defaults()
        assert defaults["--rep-loss-weight-end"] is None
        assert defaults["--rep-loss-weight-ramp-steps"] is None

    def test_the_base_weight_default_is_unchanged(self):
        assert parse_argparse_defaults()["--rep-loss-weight"] == 1.0

    def test_the_loss_call_takes_the_scheduled_weight(self):
        """A weight written once into LOSS_SPEC would freeze at step 0."""
        src = TRAIN_PY.read_text()
        assert "rep_loss_weight=rep_w_now" in src
        assert "linear_schedule_at_step(" in src


# --- 5. end to end on CPU -------------------------------------------------


def _assert_train_deps_available() -> None:
    for mod in ("torch", "numpy", "datasets"):
        try:
            __import__(mod)
        except ImportError as e:
            pytest.fail(f"train.py dep {mod!r} not importable: {e}")


def _run_train(tmp_path, run_name, extra, steps=6, t_raw=256):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    save_dir = tmp_path / "runs"
    save_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable, "-u", str(TRAIN_PY),
        "--device", "cpu", "--total-steps", str(steps), "--save-every", "100",
        "--batch-size", "2", "--lr", "1e-3", "--weight-decay", "0.1",
        "--save-dir", str(save_dir), "--run-name", run_name,
        "--mix-ratio", "1.0", "--synth-kind", "periodic",
        "--t-raw", str(t_raw), "--n-channels", "1", "--d-model", "32",
        "--n-heads", "2", "--num-layers", "1", "--num-encoder-layers", "1",
        "--log-every", "1", "--seed", "42", "--tau", "0.10",
        "--hf-repo", "none", "--hf-path", "none",
    ] + extra
    return subprocess.run(cmd, env=env, capture_output=True, text=True,
                          cwd=str(tmp_path), timeout=900), save_dir


# The project's best cell, in miniature: the whole main loss is L_rep, and
# L_align on the EMA teacher is the term that stays when L_rep is decayed out.
CELL = [
    "--loss-shape", REP_ONLY, "--align-loss-weight", "1.0",
    "--moco-rep-keys", "--tau-rep", "1.0",
    "--ema-embedding", "--ema-encoder", "--ema-tau", "0.9",
    "--align-target", "teacher",
]
DECAY = ["--rep-loss-weight-end", "0.0", "--rep-loss-weight-ramp-steps", "4"]


def _rows(save_dir, name):
    with open(save_dir / f"{name}_losses.csv", newline="") as fh:
        return list(csv.DictReader(fh))


class TestEndToEnd:

    def test_the_weight_and_the_terms_are_logged(self, tmp_path):
        _assert_train_deps_available()
        res, save_dir = _run_train(tmp_path, "a409decay", CELL + DECAY)
        if res.returncode != 0:
            pytest.fail(f"rc={res.returncode}\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
        rows = _rows(save_dir, "a409decay")
        assert len(rows) == 6
        for col in ("rep_w", "l_rep", "l_align", "l_pred"):
            assert col in rows[0], f"{col} missing from the CSV header"
        weights = [float(r["rep_w"]) for r in rows]
        assert weights == sorted(weights, reverse=True)
        assert weights[0] == pytest.approx(0.75)
        assert weights[3] == pytest.approx(0.0)
        assert weights[-1] == 0.0

    def test_l_rep_goes_blank_when_the_term_switches_off(self, tmp_path):
        """0.0 in the column would read as 'the term is at its floor'. It is
        not computed at all past the ramp, and the CSV has to say so."""
        _assert_train_deps_available()
        res, save_dir = _run_train(tmp_path, "a409blank", CELL + DECAY)
        if res.returncode != 0:
            pytest.fail(f"rc={res.returncode}\n{res.stderr[-3000:]}")
        rows = _rows(save_dir, "a409blank")
        assert rows[0]["l_rep"] != ""
        assert rows[-1]["l_rep"] == ""
        assert rows[-1]["l_align"] != ""

    def test_the_control_holds_the_weight_and_still_logs_the_terms(self, tmp_path):
        _assert_train_deps_available()
        res, save_dir = _run_train(tmp_path, "a409ctl", CELL)
        if res.returncode != 0:
            pytest.fail(f"rc={res.returncode}\n{res.stderr[-3000:]}")
        rows = _rows(save_dir, "a409ctl")
        assert {r["rep_w"] for r in rows} == {"1.0"}
        assert all(r["l_rep"] != "" for r in rows)

    def test_the_schedule_changes_the_training(self, tmp_path):
        """Same seed, same arm, one flag. A schedule that never reached the
        loss would give two identical loss columns."""
        _assert_train_deps_available()
        out = {}
        for name, extra in (("a409same", CELL), ("a409sched", CELL + DECAY)):
            res, save_dir = _run_train(tmp_path, name, extra)
            if res.returncode != 0:
                pytest.fail(f"{name} rc={res.returncode}\n{res.stderr[-3000:]}")
            out[name] = [r["loss"] for r in _rows(save_dir, name)]
        assert out["a409same"] != out["a409sched"]

    def test_the_columns_rebuild_the_total(self, tmp_path):
        """The three term columns hold the RAW term, as `sigreg_e` and
        `cpc_aux` do. This arm carries L_rep and L_align only, so the total
        is the weighted sum of the two."""
        _assert_train_deps_available()
        res, save_dir = _run_train(tmp_path, "a409sum", CELL + DECAY)
        if res.returncode != 0:
            pytest.fail(f"rc={res.returncode}\n{res.stderr[-3000:]}")
        for r in _rows(save_dir, "a409sum"):
            w = float(r["rep_w"])
            rebuilt = float(r["l_align"])
            if r["l_rep"]:
                rebuilt += w * float(r["l_rep"])
            assert float(r["loss"]) == pytest.approx(rebuilt, rel=1e-5)

    def test_the_reference_column_holds_the_base_weight(self, tmp_path):
        """`loss_tau_ref` is the run's fixed reference curve. The schedule
        travels as a function argument, so the diagnostic keeps reading the
        base weight and does not fall to the align term at the ramp."""
        _assert_train_deps_available()
        res, save_dir = _run_train(tmp_path, "a409ref", CELL + DECAY)
        if res.returncode != 0:
            pytest.fail(f"rc={res.returncode}\n{res.stderr[-3000:]}")
        rows = _rows(save_dir, "a409ref")
        after = [float(r["loss_tau_ref"]) for r in rows if float(r["rep_w"]) == 0.0]
        assert after, "the ramp never reached 0.0"
        assert min(after) > 1.0

    def test_the_depth_run_decays(self, tmp_path):
        """The card's arms train at k = 3. The decay has to reach a run whose
        loss calls itself once per depth."""
        _assert_train_deps_available()
        arm = CELL + DECAY + ["--train-rollout-depth", "3",
                              "--train-rollout-reduce", "sum"]
        res, save_dir = _run_train(tmp_path, "a409k3", arm)
        if res.returncode != 0:
            pytest.fail(f"rc={res.returncode}\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
        rows = _rows(save_dir, "a409k3")
        assert [float(r["rep_w"]) for r in rows][-1] == 0.0
        assert rows[0]["l_rep"] != "" and rows[-1]["l_rep"] == ""
        assert all(r["l_align"] != "" for r in rows)

    def test_l_align_is_the_depth_zero_copy_of_the_student_pair(self, tmp_path):
        """`cos_err_d0` is 1 - cos of the forecast against the STUDENT next
        latent, and the term is 2 - 2*cos of the same pair. So a student-target
        run reads `l_align = 2*cos_err_d0`, and a report can take the other
        depths off the `cos_err_d*` columns.

        The identity is the student target's alone. `plot_loss_terms.py` of
        the EMA momentum sweep applied it to a TEACHER-target run, which is
        the reconstruction #409 replaces with a logged term."""
        _assert_train_deps_available()
        depth = ["--train-rollout-depth", "3", "--train-rollout-reduce", "sum"]
        arm = [a if a != "teacher" else "student" for a in CELL]
        res, save_dir = _run_train(tmp_path, "a409k3s", arm + DECAY + depth)
        if res.returncode != 0:
            pytest.fail(f"rc={res.returncode}\n{res.stderr[-3000:]}")
        for r in _rows(save_dir, "a409k3s"):
            assert float(r["l_align"]) == pytest.approx(
                2.0 * float(r["cos_err_d0"]), rel=1e-6)

        res, save_dir = _run_train(tmp_path, "a409k3t", CELL + DECAY + depth)
        if res.returncode != 0:
            pytest.fail(f"rc={res.returncode}\n{res.stderr[-3000:]}")
        teacher = _rows(save_dir, "a409k3t")
        assert any(float(r["l_align"]) != pytest.approx(
            2.0 * float(r["cos_err_d0"]), rel=1e-6) for r in teacher), (
            "the teacher target read the student latent")

    def test_a_run_without_the_flags_leaves_the_columns_readable(self, tmp_path):
        """A shape that reads no rep weight must not report one."""
        _assert_train_deps_available()
        arm = ["--loss-shape", "cosine_similarity_batch_no_time_neg"]
        res, save_dir = _run_train(tmp_path, "a409other", arm)
        if res.returncode != 0:
            pytest.fail(f"rc={res.returncode}\n{res.stderr[-3000:]}")
        rows = _rows(save_dir, "a409other")
        assert {r["rep_w"] for r in rows} == {""}
        assert {r["l_rep"] for r in rows} == {""}


class TestResumeSchema:
    """The four new columns change the losses-CSV header. A run started on the
    pre-#409 code must not take the new rows: every value after `ema_tau`
    would read four columns left of its name."""

    def test_a_pre_409_losses_csv_is_refused(self, tmp_path):
        _assert_train_deps_available()
        save_dir = tmp_path / "runs"
        save_dir.mkdir()
        path = save_dir / "a409old_losses.csv"
        header = ("step,loss,loss_tau_ref,gap,gap_ratio,ff,fp,tp,cross_batch,"
                  "hf_rows_consumed,synth_rows_consumed,mixup_applied,"
                  "r2_random,r2_naive,u_temporal,u_batch,auc,top1,top3,"
                  "cpc_aux,sigreg_e,sigreg_h,u_temporal_e,u_batch_e,"
                  "u_batchtime,u_batchtime_e,ema_tau")
        path.write_text(header + "\n" + ",".join("0" * 27) + "\n")
        res, _ = _run_train(tmp_path, "a409old", CELL)
        assert res.returncode != 0
        assert "schema mismatch" in res.stdout + res.stderr


class TestGuards:
    """Every refusal below stops a run whose command line says one thing and
    whose objective is another."""

    def _fails(self, tmp_path, name, extra, needle):
        res, _ = _run_train(tmp_path, name, extra)
        assert res.returncode != 0, f"expected a refusal, got rc=0\n{res.stdout[-2000:]}"
        joined = res.stdout + res.stderr
        assert needle in joined, f"{needle!r} not in:\n{joined[-3000:]}"

    def test_ramp_steps_without_an_end_value(self, tmp_path):
        _assert_train_deps_available()
        self._fails(tmp_path, "a409g1",
                    CELL + ["--rep-loss-weight-ramp-steps", "4"],
                    "--rep-loss-weight-end")

    def test_a_shape_that_reads_no_rep_weight(self, tmp_path):
        _assert_train_deps_available()
        arm = ["--loss-shape", "cosine_similarity_batch_no_time_neg",
               "--rep-loss-weight-end", "0.0"]
        self._fails(tmp_path, "a409g2", arm, "loss_shape")

    def test_a_ramp_of_zero_steps(self, tmp_path):
        _assert_train_deps_available()
        self._fails(tmp_path, "a409g3",
                    CELL + ["--rep-loss-weight-end", "0.0",
                            "--rep-loss-weight-ramp-steps", "0"],
                    "--rep-loss-weight-ramp-steps")

    def test_a_negative_end_value(self, tmp_path):
        _assert_train_deps_available()
        self._fails(tmp_path, "a409g4",
                    CELL + ["--rep-loss-weight-end", "-1.0"],
                    "--rep-loss-weight-end")

    def test_decay_to_zero_with_no_other_term(self, tmp_path):
        """L_rep is the whole loss of this arm. At weight 0.0 the backward
        pass has nothing to differentiate, and the run would die at the ramp
        instead of at step 0."""
        _assert_train_deps_available()
        arm = ["--loss-shape", REP_ONLY, "--align-loss-weight", "0.0",
               "--cpc-infonce-weight", "0.0",
               "--rep-loss-weight-end", "0.0",
               "--rep-loss-weight-ramp-steps", "4"]
        self._fails(tmp_path, "a409g5", arm, "no other term")
