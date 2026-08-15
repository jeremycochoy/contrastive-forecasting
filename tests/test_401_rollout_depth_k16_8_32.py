"""Tests for #401: rollout depth k = 16, 8, 32, and heads as long as the backbone.

#401 runs ONE configuration — #373's cell A4, `arm6_v2 combab` with `L_align`
on the student and the scheduled EMA — at three rollout depths. The study
therefore adds no trainer flag and no new pipeline. It reuses #373's runner,
head trainer and GIFT-Eval, and supplies only the depth, the stops and the
head budgets.

That is the contract these tests hold:

  * the study's constants are the card's: depths 16, 8, 32 IN THAT ORDER,
    stops 40k / 100k / 200k, phase-1 head 30,000 steps, student encoder;
  * the configuration is #373's A4 cell, flag for flag;
  * the runner is #373's `run_leg_k.sh` — no second trainer invocation
    exists in this study;
  * the head and the eval are #373's `head_eval_bb.sh` — same protocol,
    same 97 configs;
  * phase 2 matches the head budget to the backbone stop;
  * every artefact lands in THIS study's directory and on a durable root,
    and no run name can collide with a published #373 one.

The two path overrides the reuse needs (`CF_STUDY_DIR` on #373's two
scripts) are checked to default to #373's own directory, so #373's numbers
stay reproducible from its own commands.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP = REPO_ROOT / "reports" / "2026-08-15_rollout_depth_k16_8_32"
PARENT = REPO_ROOT / "reports" / "2026-08-08_rollout_depth"

STUDY_SH = EXP / "scripts" / "study.sh"
RUN_ARM = EXP / "scripts" / "run_arm_k.sh"
HEAD_EVAL = EXP / "scripts" / "head_eval.sh"
PHASE1 = EXP / "scripts" / "phase1.sh"
PHASE2 = EXP / "scripts" / "phase2.sh"
COLLECT = EXP / "scripts" / "collect.sh"
SMOKE = EXP / "scripts" / "smoke_k16.sh"
PICK_PY = EXP / "scripts" / "pick_phase2_arms.py"
RUN_SH = EXP / "run.sh"
LAUNCH_SYNC = EXP / "sync" / "launch_sync.sh"

PARENT_LEG = PARENT / "scripts" / "run_leg_k.sh"
PARENT_HEAD = PARENT / "scripts" / "head_eval_bb.sh"

PEAK_PROBE = REPO_ROOT / "scripts" / "gpu_peak_probe.py"

# The card's three depths, in the order the card runs them.
DEPTHS = (16, 8, 32)
STOPS = (40_000, 100_000, 200_000)
HEAD_STEPS_PHASE1 = 30_000
CELL = "arm6_v2_combab_alignS"

# The issue's configuration block, flag for flag.
ISSUE_FLAGS = (
    ("--loss-shape", "cosine_similarity_batch_rep_only"),
    ("--align-loss-weight", "1.0"),
    ("--moco-rep-keys", None),
    ("--tau-rep", "1.0"),
    ("--align-target", "student"),
    ("--cpc-infonce-weight", "0.0"),
)


def strip_comments(text: str) -> str:
    """Drop full-line bash comments so a token search sees only code."""
    return "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_sh(script: Path, *args, env=None, cwd=None):
    """Run a script and return the completed process, never raising."""
    full = dict(os.environ)
    full.update(env or {})
    return subprocess.run(
        ["bash", str(script), *[str(a) for a in args]],
        capture_output=True, text=True, env=full, cwd=str(cwd or REPO_ROOT),
        timeout=180,
    )


def study_value(name: str) -> str:
    """One exported value of study.sh, read by sourcing it."""
    out = subprocess.run(
        ["bash", "-c", f'. "{STUDY_SH}" >/dev/null && printf "%s" "${{{name}}}"'],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def study_call(snippet: str, env=None) -> subprocess.CompletedProcess:
    """Source study.sh and run one snippet against its functions."""
    full = dict(os.environ)
    full.update(env or {})
    return subprocess.run(
        ["bash", "-c", f'. "{STUDY_SH}" >/dev/null && {snippet}'],
        capture_output=True, text=True, env=full, timeout=60,
    )


# --- 1. The layout -----------------------------------------------------------


class TestLayout:
    """REPORT_STANDARD: results/, plots/, scripts/, sync/, one report file."""

    @pytest.mark.parametrize("sub", ["results", "plots", "scripts", "sync"])
    def test_subdirectory_exists(self, sub):
        assert (EXP / sub).is_dir(), f"{EXP.name}/{sub} missing"

    @pytest.mark.parametrize(
        "script", [STUDY_SH, RUN_ARM, HEAD_EVAL, PHASE1, PHASE2, COLLECT,
                   SMOKE, PICK_PY, RUN_SH, LAUNCH_SYNC])
    def test_script_exists(self, script):
        assert script.is_file(), f"{script} missing"

    @pytest.mark.parametrize(
        "script",
        sorted(EXP.glob("*.sh")) + sorted(EXP.glob("scripts/*.sh"))
        + sorted(EXP.glob("sync/*.sh")))
    def test_shell_script_parses(self, script):
        """A launcher that does not parse fails after the GPU time is spent."""
        out = subprocess.run(["bash", "-n", str(script)],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr

    def test_run_sh_covers_the_three_stages(self):
        """CLAUDE.md: the experiment's launcher is `run.sh` in its own dir."""
        code = strip_comments(RUN_SH.read_text())
        for stage in ("smoke_k16.sh", "phase1.sh", "phase2.sh"):
            assert stage in code, f"run.sh does not run {stage}"

    def test_the_sync_loop_is_373s(self):
        """One set of measured per-class size floors, not a second copy."""
        code = strip_comments(LAUNCH_SYNC.read_text())
        assert "sync_loop.sh" in code
        assert "safe_pull.sh" in code, "raw scp corrupts the prior good copy"


# --- 2. The study's constants ------------------------------------------------


class TestStudyConstants:

    def test_depths_are_16_8_32_in_that_order(self):
        """The card names the order, because the cheapest arm answers first."""
        assert study_value("CF401_DEPTHS").split() == [str(k) for k in DEPTHS]

    def test_stops_are_40k_100k_200k(self):
        assert study_value("CF401_STOPS").split() == [str(s) for s in STOPS]

    def test_phase1_head_is_30000_steps(self):
        assert study_value("CF401_HEAD_STEPS_P1") == str(HEAD_STEPS_PHASE1)

    def test_head_reads_the_student_encoder(self):
        assert study_value("CF401_ENC") == "student"

    def test_cell_is_373s_a4(self):
        assert study_value("CF401_CELL") == CELL

    def test_durable_root_is_not_373s(self):
        """Two studies on one root cannot be told apart by a sync loop."""
        root = study_value("CF401_ROOT")
        assert root and not root.rstrip("/").endswith("cf-373")

    def test_durable_root_is_not_ephemeral(self):
        """CLAUDE.md checkpoint safety rule 4: never /tmp, never the checkout."""
        root = study_value("CF401_ROOT")
        assert not root.startswith("/tmp")
        assert not root.startswith(str(REPO_ROOT))


# --- 3. The configuration is #373's A4 cell ----------------------------------


class TestConfiguration:
    """The issue pins one configuration. It is already in #373's launcher."""

    @pytest.fixture(scope="class")
    def cell_block(self) -> str:
        code = strip_comments(PARENT_LEG.read_text())
        m = re.search(rf"^  {CELL}\)$(.*?);;", code, re.M | re.S)
        assert m, f"no '{CELL}' case in {PARENT_LEG}"
        return m.group(1)

    @pytest.mark.parametrize("flag,value", ISSUE_FLAGS)
    def test_issue_flag_is_in_the_cell(self, cell_block, flag, value):
        assert flag in cell_block, f"{flag} missing from the {CELL} case"
        if value is not None:
            assert re.search(rf"{re.escape(flag)}\s+{re.escape(value)}",
                             cell_block), f"{flag} is not {value}"

    def test_scheduled_ema_is_in_the_shared_block(self):
        code = strip_comments(PARENT_LEG.read_text())
        assert re.search(r"--ema-tau\s+0\.9\s+--ema-tau-end\s+1\.0\s+"
                         r"--ema-tau-ramp-steps\s+100000", code)
        assert "--ema-embedding" in code and "--ema-encoder" in code

    def test_sigreg_stays_at_the_shared_default(self):
        code = strip_comments(PARENT_LEG.read_text())
        for flag in ("--sigreg-embedding", "--sigreg-encoding",
                     "--sigreg-n-chunk 2048",
                     "--sigreg-embedding-weight 1.0",
                     "--sigreg-encoding-weight 1.0"):
            assert flag in code, f"{flag} missing"

    def test_backbone_shape_and_seed_are_the_cards(self):
        code = strip_comments(PARENT_LEG.read_text())
        for flag in ("--d-model 64", "--n-heads 8", "--num-encoder-layers 3",
                     "--num-layers 3", "--batch-size 64",
                     "--hf-repo jeremycochoy/gift-pretrain-full-4096",
                     "--hf-path small_v1"):
            assert flag in code, f"{flag} missing"
        assert 'SEED="${SEED:-20260520}"' in code

    def test_the_depth_guard_does_not_fire_on_this_configuration(self):
        """`rep_only` adds zero per depth. `L_align` is what consumes it."""
        train = load_module(
            REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py", "train_401")

        class Args:
            loss_shape = "cosine_similarity_batch_rep_only"
            align_loss_weight = 1.0
            cpc_infonce_weight = 0.0
            no_main_contrastive_loss = False
            pred_loss_weight = 1.0

        assert train.main_term_depth_gap(Args()) is not None
        assert not train.rollout_depth_has_no_consumer(Args())


# --- 4. The runner is #373's --------------------------------------------------


class TestRunnerReuse:

    @pytest.fixture(scope="class")
    def run_arm_code(self) -> str:
        return strip_comments(RUN_ARM.read_text())

    def test_it_calls_373s_leg_runner(self, run_arm_code):
        assert "run_leg_k.sh" in run_arm_code

    @pytest.mark.parametrize("script", [RUN_ARM, HEAD_EVAL, PHASE1, PHASE2])
    def test_no_second_trainer_invocation(self, script):
        """One pipeline. A second `train.py` call is a second protocol."""
        assert "train.py" not in strip_comments(script.read_text())

    def test_it_passes_the_depth_and_the_durable_root(self, run_arm_code):
        for var in ("K=", "RUNS=", "CF_STUDY_DIR="):
            assert var in run_arm_code, f"{var} not passed to the runner"

    def test_it_refuses_a_depth_outside_the_study(self):
        out = run_sh(RUN_ARM, 3, 40_000, env={"CF401_DRY_RUN": "1"})
        assert out.returncode != 0
        assert "3" in out.stderr

    def test_it_refuses_a_stop_outside_the_study(self):
        out = run_sh(RUN_ARM, 16, 55_000, env={"CF401_DRY_RUN": "1"})
        assert out.returncode != 0

    @pytest.mark.parametrize("k", DEPTHS)
    @pytest.mark.parametrize("stop", STOPS)
    def test_dry_run_names_the_depth_and_the_target(self, k, stop):
        out = run_sh(RUN_ARM, k, stop, env={"CF401_DRY_RUN": "1"})
        assert out.returncode == 0, out.stderr
        assert f"K={k}" in out.stdout
        assert str(stop) in out.stdout
        assert CELL in out.stdout


class TestStudyDirOverride:
    """#373's two scripts take this study's directory, and default to their own."""

    @pytest.mark.parametrize("script", [PARENT_LEG, PARENT_HEAD])
    def test_override_exists(self, script):
        assert "CF_STUDY_DIR" in script.read_text(), f"{script.name}"

    @pytest.mark.parametrize("script", [PARENT_LEG, PARENT_HEAD])
    def test_default_is_373s_own_directory(self, script):
        """Non-regression: #373's numbers stay reproducible from its commands."""
        code = strip_comments(script.read_text())
        m = re.search(r'\$\{CF_STUDY_DIR:-([^}]*)\}', code)
        assert m, f"{script.name} has no defaulted CF_STUDY_DIR"
        assert "2026-08-08_rollout_depth" in m.group(1)


# --- 5. The head and the eval are #373's -------------------------------------


class TestHeadAndEval:

    @pytest.fixture(scope="class")
    def head_code(self) -> str:
        return strip_comments(HEAD_EVAL.read_text())

    def test_it_calls_373s_head_and_eval_script(self, head_code):
        assert "head_eval_bb.sh" in head_code

    def test_the_parent_evaluates_97_configs(self):
        """The 97-config count is asserted by #373's own eval, not re-derived."""
        eval_local = strip_comments(
            (PARENT / "scripts" / "eval_local.sh").read_text())
        assert "-ne 97" in eval_local

    @pytest.mark.parametrize("k", DEPTHS)
    @pytest.mark.parametrize("stop", STOPS)
    def test_phase1_head_is_30000_steps_on_the_student(self, k, stop):
        out = run_sh(HEAD_EVAL, k, stop, env={"CF401_DRY_RUN": "1"})
        assert out.returncode == 0, out.stderr
        assert str(HEAD_STEPS_PHASE1) in out.stdout
        assert "student" in out.stdout

    @pytest.mark.parametrize("stop", STOPS)
    def test_phase2_head_steps_match_the_backbone_steps(self, stop):
        out = run_sh(HEAD_EVAL, 16, stop, stop, env={"CF401_DRY_RUN": "1"})
        assert out.returncode == 0, out.stderr
        assert f"HEAD_STEPS={stop}" in out.stdout

    def test_tags_of_the_two_phases_differ(self):
        """A shared tag would let phase 2 read phase 1's score file."""
        p1 = run_sh(HEAD_EVAL, 16, 40_000, env={"CF401_DRY_RUN": "1"}).stdout
        p2 = run_sh(HEAD_EVAL, 16, 40_000, 40_000,
                    env={"CF401_DRY_RUN": "1"}).stdout
        tag1 = re.search(r"TAG=(\S+)", p1).group(1)
        tag2 = re.search(r"TAG=(\S+)", p2).group(1)
        assert tag1 != tag2

    def test_it_refuses_a_head_budget_that_is_not_the_cards(self):
        out = run_sh(HEAD_EVAL, 16, 40_000, 12_345,
                     env={"CF401_DRY_RUN": "1"})
        assert out.returncode != 0


# --- 6. The two phases --------------------------------------------------------


class TestPhase1Plan:

    @pytest.fixture(scope="class")
    def plan(self) -> list[str]:
        out = run_sh(PHASE1, env={"CF401_DRY_RUN": "1"})
        assert out.returncode == 0, out.stderr
        return [ln for ln in out.stdout.splitlines() if ln.startswith(("arm ", "head "))]

    def test_nine_legs_and_nine_heads(self, plan):
        assert sum(1 for ln in plan if ln.startswith("arm ")) == len(DEPTHS) * len(STOPS)
        assert sum(1 for ln in plan if ln.startswith("head ")) == len(DEPTHS) * len(STOPS)

    def test_arms_run_in_the_cards_order(self, plan):
        order = [int(re.search(r"k=(\d+)", ln).group(1))
                 for ln in plan if ln.startswith("arm ")]
        assert order == [k for k in DEPTHS for _ in STOPS]

    def test_stops_climb_within_an_arm(self, plan):
        legs = [ln for ln in plan if ln.startswith("arm ")]
        for i in range(0, len(legs), len(STOPS)):
            steps = [int(re.search(r"steps=(\d+)", ln).group(1))
                     for ln in legs[i:i + len(STOPS)]]
            assert steps == list(STOPS)

    def test_every_leg_is_followed_by_its_head(self, plan):
        assert [ln.split()[0] for ln in plan] == ["arm", "head"] * (len(DEPTHS) * len(STOPS))

    @pytest.mark.parametrize("script", [PHASE1, PHASE2])
    def test_a_waited_head_is_not_launched_through_setsid(self, script):
        """setsid forks, and then `wait` returns on a PID already gone."""
        code = strip_comments(script.read_text())
        assert "wait " in code, f"{script.name} never waits for its heads"
        assert "setsid" not in code, (
            f"{script.name} backgrounds a head through setsid and waits on it")


class TestPhase2Plan:

    def test_it_runs_matched_heads_on_two_arms(self, tmp_path):
        scores = tmp_path / "scores.csv"
        write_scores(scores, {16: 1.05, 8: 1.09, 32: 1.20})
        out = run_sh(PHASE2, env={"CF401_DRY_RUN": "1",
                                  "CF401_SCORES": str(scores)})
        assert out.returncode == 0, out.stderr
        heads = [ln for ln in out.stdout.splitlines() if ln.startswith("head ")]
        assert len(heads) == 2 * len(STOPS)
        for ln in heads:
            k = int(re.search(r"k=(\d+)", ln).group(1))
            assert k in (16, 8)
            stop = int(re.search(r"stop=(\d+)", ln).group(1))
            steps = int(re.search(r"steps=(\d+)", ln).group(1))
            assert steps == stop

    def test_it_refuses_an_incomplete_phase_1(self, tmp_path):
        """The card: wait until every stop at every k is scored."""
        scores = tmp_path / "scores.csv"
        write_scores(scores, {16: 1.05, 8: 1.09, 32: 1.20}, drop=(32, 200_000))
        out = run_sh(PHASE2, env={"CF401_DRY_RUN": "1",
                                  "CF401_SCORES": str(scores)})
        assert out.returncode != 0


def write_scores(path: Path, best_by_k: dict[int, float], drop=None):
    """A phase-1 scores.csv: every (k, stop), `best_by_k` at the 100k stop."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phase", "k", "stop", "head_steps", "encoder", "score"])
        for k, best in best_by_k.items():
            for stop in STOPS:
                if drop == (k, stop):
                    continue
                score = best if stop == 100_000 else best + 0.05
                w.writerow([1, k, stop, HEAD_STEPS_PHASE1, "student",
                            f"{score:.4f}"])


# --- 7. The phase-2 pick rule -------------------------------------------------


class TestPickPhase2Arms:

    @pytest.fixture(scope="class")
    def pick(self):
        return load_module(PICK_PY, "pick_401")

    def rows(self, best_by_k, drop=None):
        out = []
        for k, best in best_by_k.items():
            for stop in STOPS:
                if drop == (k, stop):
                    continue
                out.append({"k": k, "stop": stop,
                            "score": best if stop == 100_000 else best + 0.05})
        return out

    def test_it_takes_the_two_lowest_best_scores(self, pick):
        got = pick.pick_arms(self.rows({16: 1.05, 8: 1.09, 32: 1.20}))
        assert set(got) == {16, 8}

    def test_an_arm_is_scored_by_its_best_stop(self, pick):
        """A late stop that improves must be able to carry its arm."""
        rows = [{"k": 16, "stop": 40_000, "score": 1.30},
                {"k": 16, "stop": 100_000, "score": 1.30},
                {"k": 16, "stop": 200_000, "score": 1.01},
                {"k": 8, "stop": 40_000, "score": 1.20},
                {"k": 8, "stop": 100_000, "score": 1.20},
                {"k": 8, "stop": 200_000, "score": 1.20},
                {"k": 32, "stop": 40_000, "score": 1.25},
                {"k": 32, "stop": 100_000, "score": 1.25},
                {"k": 32, "stop": 200_000, "score": 1.25}]
        assert set(pick.pick_arms(rows)) == {16, 8}

    def test_the_result_is_in_the_studys_run_order(self, pick):
        got = pick.pick_arms(self.rows({16: 1.20, 8: 1.09, 32: 1.05}))
        assert got == [8, 32], "run order is 16, 8, 32"

    def test_a_tie_is_broken_by_the_run_order(self, pick):
        got = pick.pick_arms(self.rows({16: 1.10, 8: 1.10, 32: 1.10}))
        assert got == [16, 8]

    def test_it_refuses_a_missing_stop(self, pick):
        with pytest.raises(ValueError):
            pick.pick_arms(self.rows({16: 1.05, 8: 1.09, 32: 1.20},
                                     drop=(32, 200_000)))

    def test_it_refuses_a_missing_arm(self, pick):
        with pytest.raises(ValueError):
            pick.pick_arms(self.rows({16: 1.05, 8: 1.09}))


# --- 8. Collecting the scores -------------------------------------------------


class TestCollect:

    def test_it_reads_every_score_file_into_one_csv(self, tmp_path):
        res = tmp_path / "results"
        res.mkdir()
        (res / "score_k16_bb40k_h30k_student.txt").write_text("1.0731\n")
        (res / "score_k8_bb200k_h200k_student.txt").write_text("1.0512\n")
        (res / "stops.log").write_text("not a score\n")
        out = run_sh(COLLECT, env={"CF401_RESULTS": str(res)})
        assert out.returncode == 0, out.stderr
        rows = list(csv.DictReader(open(res / "scores.csv")))
        assert len(rows) == 2
        by_k = {int(r["k"]): r for r in rows}
        assert by_k[16]["stop"] == "40000"
        assert by_k[16]["head_steps"] == "30000"
        assert by_k[16]["phase"] == "1"
        assert by_k[16]["score"] == "1.0731"
        assert by_k[8]["head_steps"] == "200000"
        assert by_k[8]["phase"] == "2"

    def test_it_skips_an_empty_score_file(self, tmp_path):
        """An eval killed mid-write leaves one, and it is not a 0.0 score."""
        res = tmp_path / "results"
        res.mkdir()
        (res / "score_k16_bb40k_h30k_student.txt").write_text("")
        out = run_sh(COLLECT, env={"CF401_RESULTS": str(res)})
        assert out.returncode == 0, out.stderr
        rows = list(csv.DictReader(open(res / "scores.csv")))
        assert rows == []


# --- 9. The k = 16 smoke test -------------------------------------------------


class TestSmokeK16:
    """Proves the flag runs at this depth, and says what it costs."""

    @pytest.fixture(scope="class")
    def smoke_code(self) -> str:
        return strip_comments(SMOKE.read_text())

    def test_it_measures_step_time_and_peak_memory(self, smoke_code):
        assert "gpu_peak_probe.py" in smoke_code
        assert "timing:" in SMOKE.read_text()

    def test_it_writes_every_number_to_results(self, smoke_code):
        header = re.search(r'echo "(cell,[^"]*)"', smoke_code)
        assert header, "no CSV header in the smoke script"
        cols = header.group(1).split(",")
        for col in ("k", "total_ms", "peak_mib", "depth_cols", "rc"):
            assert col in cols, f"{col} missing from {cols}"

    def test_it_runs_the_studys_own_runner(self, smoke_code):
        assert "run_arm_k.sh" in smoke_code or "run_leg_k.sh" in smoke_code

    def test_its_scratch_root_is_durable_and_not_the_studys(self, smoke_code):
        m = re.search(r'SCRATCH="\$\{[A-Z0-9_]+:-([^}]*)\}"', smoke_code)
        assert m, "no defaulted SCRATCH in the smoke script"
        assert not m.group(1).startswith("/tmp")

    def test_the_default_depths_include_16_and_a_k0_reference(self, smoke_code):
        m = re.search(r'DEPTHS="\$\{DEPTHS:-([^}]*)\}"', smoke_code)
        assert m, "no defaulted DEPTHS"
        assert m.group(1).split() == ["0", "16"]

    @pytest.fixture(scope="class")
    def measured(self) -> dict[int, dict]:
        csv_path = EXP / "results" / "smoke_k16.csv"
        assert csv_path.is_file(), "no results/smoke_k16.csv"
        rows = list(csv.DictReader(open(csv_path)))
        assert rows, "smoke_k16.csv holds no measurement"
        return {int(r["k"]): r for r in rows}

    def test_the_measurement_is_recorded(self, measured):
        """The run plan depends on it, so the number is committed."""
        assert 16 in measured, "k = 16 was not measured"
        for k, row in measured.items():
            assert row["total_ms"], f"no step time at k = {k}"
            assert row["peak_mib"], f"no peak memory at k = {k}"

    def test_the_measurement_ran_clean(self, measured):
        for k, row in measured.items():
            assert row["rc"] == "0", f"k = {k} exited rc={row['rc']}"

    def test_the_flag_reached_every_measured_depth(self, measured):
        """k + 1 `cos_err_dj` columns at depth k. That is what proves it ran."""
        for k, row in measured.items():
            want = k + 1 if k > 0 else 0
            assert int(row["depth_cols"]) == want, (
                f"k = {k} wrote {row['depth_cols']} cos_err_dj column(s), "
                f"want {want}")


# --- 10. The per-process GPU peak probe --------------------------------------


class TestGpuPeakProbe:
    """Reusable, so it lives in the top-level scripts/ directory."""

    @pytest.fixture(scope="class")
    def probe(self):
        assert PEAK_PROBE.is_file(), f"{PEAK_PROBE} missing"
        return load_module(PEAK_PROBE, "gpu_peak_probe_401")

    def test_descendants_walks_the_whole_tree(self, probe):
        ppid = {100: 1, 200: 100, 300: 200, 400: 1}
        assert probe.descendants(100, ppid) == {100, 200, 300}

    def test_descendants_of_a_leaf_is_itself(self, probe):
        assert probe.descendants(400, {400: 1}) == {400}

    def test_a_cycle_does_not_hang(self, probe):
        assert probe.descendants(1, {1: 2, 2: 1}) == {1, 2}

    def test_peak_ignores_a_neighbours_process(self, probe):
        """Elisa is shared. The card total is not this run's memory."""
        samples = [[(200, 4000), (999, 17000)], [(200, 9000), (999, 17000)]]
        assert probe.peak_used_mib(samples, {100, 200}) == 9000

    def test_peak_sums_the_processes_of_one_run(self, probe):
        samples = [[(200, 4000), (300, 1000)]]
        assert probe.peak_used_mib(samples, {200, 300}) == 5000

    def test_peak_of_nothing_is_zero(self, probe):
        assert probe.peak_used_mib([], {200}) == 0

    def test_it_parses_nvidia_smi_output(self, probe):
        text = "200, 4096\n999, 17000\n"
        assert probe.parse_compute_apps(text) == [(200, 4096), (999, 17000)]

    def test_it_ignores_a_not_supported_row(self, probe):
        text = "200, [N/A]\n201, 512\n"
        assert probe.parse_compute_apps(text) == [(201, 512)]


# --- 11. No artefact of this study can be taken for #373's -------------------


class TestNoCollision:

    def test_the_run_name_carries_this_studys_depth(self):
        for k in DEPTHS:
            out = study_call(f'cf401_run_name {k}')
            assert out.returncode == 0, out.stderr
            assert out.stdout.strip().endswith(f"cf373k{k}")

    def test_no_run_name_collides_with_a_published_373_one(self):
        published = {p.name for p in (PARENT / "results").glob("score_*.txt")}
        for k in DEPTHS:
            for stop in STOPS:
                out = run_sh(HEAD_EVAL, k, stop, env={"CF401_DRY_RUN": "1"})
                tag = re.search(r"TAG=(\S+)", out.stdout).group(1)
                assert f"score_{tag}.txt" not in published

    def test_the_checkpoint_path_is_under_this_studys_root(self):
        root = study_value("CF401_ROOT")
        out = study_call('cf401_leg_dir 16 40000')
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip().startswith(root)
        assert out.stdout.strip().endswith("leg_40k")

    def test_two_arms_never_share_a_save_directory(self):
        """One cell at three depths. Rule 3: no overlapping save path."""
        dirs = set()
        for k in DEPTHS:
            for stop in STOPS:
                out = study_call(f'cf401_leg_dir {k} {stop}')
                assert out.returncode == 0, out.stderr
                dirs.add(out.stdout.strip())
        assert len(dirs) == len(DEPTHS) * len(STOPS)

    def test_the_runner_is_given_the_arms_own_root(self):
        out = run_sh(RUN_ARM, 16, 40_000, env={"CF401_DRY_RUN": "1"})
        runs = re.search(r"RUNS=(\S+)", out.stdout).group(1)
        assert runs.endswith("/k16"), runs
