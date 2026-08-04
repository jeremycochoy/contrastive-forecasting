"""Tests for #393: the fixed-step α anchor and the backbone ladder.

#388 ramps the EMA momentum α across ``--total-steps``. #393 stops each of
its ten runs at a different step, so a budget-relative ramp would give
every run a different α curve and the ten would not be comparable. The
ramp is therefore anchored to a fixed step: α rises linearly 0.9 → 1.0
over steps 0..100k and holds at 1.0 after.

The ladder itself is the second half. Each run trains to 40k, then to
100k — both unconditional — then extends 100k at a time for as long as
the extend rule allows and the dataset lasts. The rule and the step cap
are pure functions here so the driver's decisions are checkable without
a GPU.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

from src.models import ema_tau_at_step

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
EXP_DIR = REPO_ROOT / "experiments" / "2026-08-04_ema_sched_ladder"
LADDER_PY = EXP_DIR / "scripts" / "ladder.py"
RUN_LEG = EXP_DIR / "scripts" / "run_leg.sh"
EVAL_STOP = EXP_DIR / "scripts" / "eval_stop.sh"
SMOKE = EXP_DIR / "scripts" / "smoke_e2e.sh"
GIFT = REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
HEAD_TRAIN_PY = GIFT / "train_forecasting_head.py"
EVAL_PY = GIFT / "eval_gift_eval_official.py"

# Confirmed from small_v1/manifest.json on the HF repo (4274 shards).
SMALL_V1_ROWS = 42_571_692


def load_ladder():
    spec = importlib.util.spec_from_file_location("ladder_393", LADDER_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- 1. α anchored to a fixed step ----------------------------------------


class TestFixedStepAlphaAnchor:

    def test_ramp_steps_none_is_the_388_behaviour(self):
        """Non-regression: the #388 runs are on disk against this curve."""
        for step in (0, 25_000, 50_000, 100_000):
            assert (ema_tau_at_step(step, 100_000, 0.9, 1.0, None)
                    == pytest.approx(ema_tau_at_step(step, 100_000, 0.9, 1.0)))

    def test_endpoints(self):
        assert ema_tau_at_step(0, 400_000, 0.9, 1.0, 100_000) == pytest.approx(0.9)
        assert ema_tau_at_step(100_000, 400_000, 0.9, 1.0, 100_000) == pytest.approx(1.0)

    def test_the_curve_ignores_the_budget(self):
        """The reason the anchor exists: a 40k run and a 400k run must apply
        the same α at the same global step."""
        for step in (1, 20_000, 40_000, 99_999, 100_000):
            got = {ema_tau_at_step(step, budget, 0.9, 1.0, 100_000)
                   for budget in (40_000, 100_000, 200_000, 400_000)}
            assert len(got) == 1, f"step {step} gave {got}"

    def test_holds_at_one_after_the_anchor(self):
        """From 100k on the teacher stops moving, whatever the budget."""
        for step in (100_001, 200_000, 665_000):
            assert ema_tau_at_step(step, 700_000, 0.9, 1.0, 100_000) == pytest.approx(1.0)

    def test_linear_between_the_endpoints(self):
        assert ema_tau_at_step(40_000, 400_000, 0.9, 1.0, 100_000) == pytest.approx(0.94)
        assert ema_tau_at_step(50_000, 400_000, 0.9, 1.0, 100_000) == pytest.approx(0.95)

    def test_alpha_at_the_first_stop_is_still_moving(self):
        """The issue calls this out: at bb40k α ≈ 0.94, the teacher is live."""
        assert ema_tau_at_step(40_000, 400_000, 0.9, 1.0, 100_000) < 1.0

    def test_no_end_value_stays_constant(self):
        """An anchor without a schedule is still a constant α."""
        assert ema_tau_at_step(50_000, 400_000, 0.9, None, 100_000) == 0.9

    def test_monotone_and_bounded(self):
        vals = [ema_tau_at_step(s, 400_000, 0.9, 1.0, 100_000)
                for s in range(0, 400_001, 10_000)]
        assert vals == sorted(vals)
        assert min(vals) >= 0.9 and max(vals) <= 1.0

    def test_zero_anchor_falls_back_to_the_budget(self):
        assert (ema_tau_at_step(50_000, 100_000, 0.9, 1.0, 0)
                == pytest.approx(ema_tau_at_step(50_000, 100_000, 0.9, 1.0)))


def argparse_defaults(path: Path) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value.startswith("--")):
            continue
        value = None
        for kw in node.keywords:
            if kw.arg == "default":
                try:
                    value = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    value = "<non-literal>"
                break
        defaults[first.value] = value
    return defaults


def argparse_flags(path: Path) -> set[str]:
    """Every long option a script accepts, aliases included."""
    flags: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        for arg in node.args:
            if (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    and arg.value.startswith("--")):
                flags.add(arg.value)
    return flags


def shell_flags(path: Path) -> set[str]:
    """Long options a launcher passes on a command line.

    Comment lines are dropped: they mention flags of other tools (`git
    worktree remove --force`) that the trainer has never heard of.
    """
    import re
    body = "\n".join(line for line in path.read_text().splitlines()
                     if not line.lstrip().startswith("#"))
    return set(re.findall(r"(?<![\w-])--[a-z0-9][a-z0-9-]*", body))


class TestTrainFlag:

    def test_ramp_steps_defaults_to_none(self):
        """Omitted ⇒ ramp over --total-steps ⇒ #388 runs unchanged."""
        assert argparse_defaults(TRAIN_PY).get("--ema-tau-ramp-steps") is None

    def test_the_flag_reaches_the_schedule(self):
        """A flag parsed but never passed would silently keep the old curve."""
        src = TRAIN_PY.read_text()
        assert "args.ema_tau_ramp_steps" in src
        call = src[src.index("ema_tau_at_step("):]
        assert "ema_tau_ramp_steps" in call[:400]


# --- 2. the ladder ---------------------------------------------------------


class TestStops:

    def test_the_first_two_stops(self):
        ladder = load_ladder()
        assert ladder.next_stop(0) == 40_000
        assert ladder.next_stop(40_000) == 100_000

    def test_then_one_hundred_thousand_at_a_time(self):
        ladder = load_ladder()
        assert ladder.next_stop(100_000) == 200_000
        assert ladder.next_stop(200_000) == 300_000
        assert ladder.next_stop(300_000) == 400_000

    def test_head_budget_changes_at_the_second_stop(self):
        ladder = load_ladder()
        assert ladder.head_steps_for(40_000) == 15_000
        assert ladder.head_steps_for(100_000) == 30_000
        assert ladder.head_steps_for(400_000) == 30_000


class TestExtendRule:
    """Compare each head against its own value at the previous stop. Lower
    GM-Relative MASE is better, so 'down' means the number decreased."""

    def decide(self, stop, previous, current, **kw):
        return load_ladder().ladder_decision(stop, previous, current, **kw)

    def test_forty_k_is_unconditional(self):
        """No previous stop exists; the rule first applies at 100k."""
        got = self.decide(40_000, None, {"student": 1.30, "teacher": 1.40})
        assert got["extend"] is True
        assert got["branch"] == "unconditional"
        assert set(got["heads"]) == {"student", "teacher"}

    def test_both_down_extends_and_keeps_both(self):
        got = self.decide(100_000, {"student": 1.30, "teacher": 1.40},
                          {"student": 1.25, "teacher": 1.35})
        assert got["extend"] is True
        assert got["branch"] == "both_down"
        assert set(got["heads"]) == {"student", "teacher"}

    def test_one_down_extends_and_drops_the_other(self):
        got = self.decide(100_000, {"student": 1.30, "teacher": 1.40},
                          {"student": 1.25, "teacher": 1.45})
        assert got["extend"] is True
        assert got["branch"] == "one_down"
        assert tuple(got["heads"]) == ("student",)

    def test_one_down_keeps_the_teacher_when_the_teacher_is_the_one(self):
        got = self.decide(100_000, {"student": 1.30, "teacher": 1.40},
                          {"student": 1.31, "teacher": 1.35})
        assert got["extend"] is True
        assert tuple(got["heads"]) == ("teacher",)

    def test_neither_down_stops_the_run(self):
        got = self.decide(100_000, {"student": 1.30, "teacher": 1.40},
                          {"student": 1.31, "teacher": 1.45})
        assert got["extend"] is False
        assert got["branch"] == "none_down"

    def test_a_flat_head_has_not_gone_down(self):
        got = self.decide(100_000, {"student": 1.30}, {"student": 1.30})
        assert got["extend"] is False

    def test_a_single_surviving_head_still_decides(self):
        """Once a head is dropped the run continues on the other one alone."""
        got = self.decide(200_000, {"student": 1.25}, {"student": 1.20})
        assert got["extend"] is True
        assert tuple(got["heads"]) == ("student",)
        assert got["branch"] == "one_down"

    def test_a_dropped_head_is_not_resurrected(self):
        """The teacher was dropped at 100k, so its 40k value must not re-enter
        the comparison at 200k."""
        got = self.decide(200_000, {"student": 1.25, "teacher": 1.40},
                          {"student": 1.20})
        assert tuple(got["heads"]) == ("student",)

    def test_the_data_cap_stops_the_run(self):
        """Extending past one pass over the dataset is not allowed, however
        good the numbers look."""
        got = self.decide(600_000, {"student": 1.30, "teacher": 1.40},
                          {"student": 1.20, "teacher": 1.30},
                          step_cap=665_182)
        assert got["extend"] is False
        assert got["branch"] == "data_exhausted"

    def test_the_cap_allows_a_stop_that_fits(self):
        got = self.decide(400_000, {"student": 1.30}, {"student": 1.20},
                          step_cap=665_182)
        assert got["extend"] is True


class TestBatchComposition:
    """What the step cap rests on: how many of a step's rows are real.

    `--mix-ratio 0.0078125` reads like 1/128 of the batch is synthetic,
    which would put fewer than 64 real rows in a step and push the cap
    above 665,182. It does not, and the composition has to say why in the
    file the report quotes.
    """

    def test_the_mix_ratio_rounds_away_at_batch_64(self):
        """64 x 1/128 = 0.5, and train.py takes ``int(round(...))``, which
        Python rounds half to EVEN. synth_bs is 0: the batch is all real."""
        comp = load_ladder().batch_composition(64, 0.0078125, 0.0, 1, 1)
        assert comp["synth_bs"] == 0
        assert comp["hf_bs"] == 64
        assert comp["hf_rows_per_step"] == 64

    def test_crossfade_triplets_are_additive_and_free(self):
        """The 3 triplet rows are blended from the real sub-batch, so they
        widen the batch without consuming another HF row."""
        comp = load_ladder().batch_composition(64, 0.0078125, 0.0, 1, 1)
        assert comp["triplet_rows"] == 3
        assert comp["total_batch_rows"] == 67
        assert comp["hf_rows_per_step"] == 64

    def test_a_mix_ratio_that_does_not_round_away_takes_rows(self):
        """The guard on the above: a real synthetic fraction does shrink the
        real sub-batch, and then the cap would rise."""
        comp = load_ladder().batch_composition(64, 0.5, 0.0, 0, 1)
        assert (comp["synth_bs"], comp["hf_bs"]) == (32, 32)
        assert comp["hf_rows_per_step"] == 32

    def test_composition_agrees_with_rows_per_step(self):
        for bs, mix, cross, chan in [(64, 0.0078125, 0.0, 1), (64, 0.5, 0.0, 1),
                                     (64, 0.25, 0.25, 1), (24, 0.0, 0.0, 4)]:
            ladder = load_ladder()
            comp = ladder.batch_composition(bs, mix, cross, 1, chan)
            assert comp["hf_rows_per_step"] == ladder.hf_rows_per_step(
                bs, mix, cross, chan)

    def test_the_experiment_batch_is_all_real(self):
        ladder = load_ladder()
        comp = ladder.experiment_batch_composition()
        assert comp["synth_bs"] == 0 and comp["cross_bs"] == 0
        assert comp["hf_rows_per_step"] == ladder.BATCH_SIZE


class TestRecordedDatasetRows:
    """`results/dataset_rows.json` is what the report quotes. It must carry
    the basis, not just the answer."""

    def record(self):
        import json
        return json.loads((EXP_DIR / "results" / "dataset_rows.json").read_text())

    def test_the_row_count_and_the_cap(self):
        rec = self.record()
        assert rec["total_rows"] == SMALL_V1_ROWS
        assert rec["step_cap"] == 665_182

    def test_the_cap_follows_from_the_recorded_basis(self):
        """No hand-typed number: the file's own fields have to produce it."""
        rec = self.record()
        assert rec["hf_rows_per_step"] == rec["batch_composition"]["hf_rows_per_step"]
        assert rec["step_cap"] == rec["total_rows"] // rec["hf_rows_per_step"]

    def test_the_basis_is_spelled_out(self):
        rec = self.record()
        comp = rec["batch_composition"]
        assert comp["hf_bs"] == 64 and comp["synth_bs"] == 0
        assert comp["triplet_rows"] == 3
        assert rec["basis"]

    def test_the_2026_05_03_cross_check_is_recorded(self):
        """The issue asks whether the derived cap agrees with that run's
        167,000-step epoch at batch 256. Same dataset, all-real batches, so
        it must, and the file has to show the comparison."""
        rec = self.record()
        check = rec["cross_check_2026_05_03"]
        assert check["reported_steps"] == 167_000
        assert check["batch_size"] == 256
        assert check["derived_steps"] == SMALL_V1_ROWS // 256 == 166_295
        assert abs(check["relative_gap"]) < 0.01
        assert check["agrees"] is True


class TestStepCap:
    """One pass over the dataset. Rows per step mirrors train.py's
    ``hf_rows_per_step = (batch_size - synth_bs - cross_bs) * C``."""

    def test_rows_per_step_matches_the_trainer(self):
        ladder = load_ladder()
        assert ladder.hf_rows_per_step(64, 0.0078125, 0.0, 1) == 64

    def test_mix_ratio_takes_rows_off_the_real_stream(self):
        ladder = load_ladder()
        assert ladder.hf_rows_per_step(64, 0.5, 0.0, 1) == 32
        assert ladder.hf_rows_per_step(64, 0.25, 0.25, 1) == 32

    def test_channels_multiply_the_rows(self):
        ladder = load_ladder()
        assert ladder.hf_rows_per_step(64, 0.0, 0.0, 2) == 128

    def test_the_cap_is_a_whole_number_of_steps(self):
        ladder = load_ladder()
        assert ladder.step_cap(SMALL_V1_ROWS, 64) == 665_182

    def test_the_experiment_cap(self):
        """Recorded so the report's number and the driver's cannot drift."""
        ladder = load_ladder()
        rows_per_step = ladder.hf_rows_per_step(
            ladder.BATCH_SIZE, ladder.MIX_RATIO, ladder.CROSSFADE_RATIO,
            ladder.N_CHANNELS)
        assert ladder.step_cap(SMALL_V1_ROWS, rows_per_step) == 665_182

    def test_the_row_count_is_the_confirmed_one(self):
        assert load_ladder().SMALL_V1_ROWS == SMALL_V1_ROWS


class TestCells:
    """Ten runs, one per row of the issue's union table."""

    def test_ten_cells(self):
        assert len(load_ladder().CELLS) == 10

    def test_slugs_are_unique(self):
        cells = load_ladder().CELLS
        assert len({c["slug"] for c in cells}) == 10

    def test_the_head_to_head_pair_runs_first(self):
        """`arm6_v2 combab` leads both parent reports; the issue asks for its
        student/teacher pair at the front of the queue."""
        cells = load_ladder().CELLS
        assert [c["slug"] for c in cells[:2]] == ["arm6_v2_combab_alignS",
                                                  "arm6_v2_combab_alignT"]

    def test_align_targets(self):
        cells = {c["slug"]: c for c in load_ladder().CELLS}
        assert cells["arm6_v2_combab_alignS"]["align"] == "student"
        assert cells["arm6_v2_combab_alignT"]["align"] == "teacher"
        assert cells["arm4_combab"]["align"] is None
        assert cells["arm1_nse"]["align"] is None

    def test_every_cell_names_a_379_arm(self):
        arms = {"arm6_v2_combab", "arm5_combab", "arm6_v2_ncpc",
                "arm6_v2_nse", "arm4_combab", "arm1_nse"}
        assert {c["arm"] for c in load_ladder().CELLS} == arms


# --- 3. launcher shape -----------------------------------------------------


class TestRunLeg:

    def body(self):
        return RUN_LEG.read_text()

    def test_every_cell_has_a_case_block(self):
        body = self.body()
        for cell in load_ladder().CELLS:
            assert f"  {cell['slug']})" in body, cell["slug"]

    def test_the_alpha_schedule_is_on_every_cell(self):
        """One shared block, so no cell can miss the schedule."""
        body = self.body()
        assert body.count("--ema-tau-end 1.0") == 1
        assert body.count("--ema-tau-ramp-steps 100000") == 1
        assert body.count("--ema-tau 0.9") == 1

    def test_the_backbone_recipe_is_the_379_one(self):
        body = self.body()
        for literal in ("--batch-size 64", "--d-model 64", "--n-heads 8",
                        "--num-encoder-layers 3", "--num-layers 3",
                        "--seed \"$SEED\"", "--t-raw 4096",
                        "--encoder-type gru", "--rev-norm-span 128",
                        "--hf-path small_v1"):
            assert body.count(literal) == 1, literal
        assert "SEED=20260520" in body

    def test_align_target_only_on_the_align_cells(self):
        body = self.body()
        assert body.count("--align-target teacher") == 4
        assert body.count("--align-target student") == 4

    def test_no_fresh_start_from_a_388_or_379_checkpoint(self):
        """The α curve differs from step 1, so those checkpoints do not belong
        to this schedule. Resume is only ever from this experiment's own run."""
        assert "RUNS" in self.body()
        assert "2026-07-21_split_pred_rep_small/runs" not in self.body()

    def test_every_flag_is_one_train_py_accepts(self):
        """A typo'd flag costs a GPU-hours-long launch to discover."""
        unknown = shell_flags(RUN_LEG) - argparse_flags(TRAIN_PY)
        assert not unknown, unknown


class TestSmokeScript:
    """`smoke_e2e.sh` is the pre-merge end-to-end check. It has to keep
    exercising the two new mechanisms through the real launchers."""

    def test_it_drives_all_three_scripts(self):
        body = SMOKE.read_text()
        for script in ("train.py", "train_forecasting_head.py",
                       "eval_gift_eval_official.py"):
            assert script in body, script

    def test_it_exercises_both_encoders_and_the_ramp(self):
        body = SMOKE.read_text()
        assert "--ema-tau-ramp-steps" in body
        for source in ("student", "teacher"):
            assert f"--encoder-source {source}" in body

    def test_every_flag_is_one_of_the_three_scripts_accepts(self):
        known = (argparse_flags(TRAIN_PY) | argparse_flags(HEAD_TRAIN_PY)
                 | argparse_flags(EVAL_PY))
        unknown = shell_flags(SMOKE) - known
        assert not unknown, unknown


class TestEvalStop:

    def body(self):
        return EVAL_STOP.read_text()

    def test_both_encoders_are_reachable(self):
        assert "--encoder-source" in self.body()

    def test_the_head_and_the_eval_share_one_encoder_source(self):
        """A teacher head evaluated through the student is the failure this
        experiment cannot afford. One variable feeds both calls."""
        body = self.body()
        assert body.count("--encoder-source \"$ENC\"") == 2

    def test_the_protocol_constants(self):
        body = self.body()
        for literal in ("--seed 20260722", "--forecast-len 16",
                        "--strategy B4", "--grad-clip 1.0"):
            assert literal in body, literal

    def test_every_flag_is_one_of_the_two_scripts_accepts(self):
        known = argparse_flags(HEAD_TRAIN_PY) | argparse_flags(EVAL_PY)
        unknown = shell_flags(EVAL_STOP) - known
        assert not unknown, unknown
