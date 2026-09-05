"""#407 — A4's continuation to one full pass over ``small_v1``.

#373 stopped A4 at 200,000 steps, which is 30% of the training data. The
card gives the same run more steps — 300,000, 450,000 and 665,000 — with a
quantile head and a GIFT-Eval at each stop. Nothing else changes.

"Nothing else changes" is the whole contract, and a driver can break it
without an error. So the tests here cover five risks, and each one is a
failure that still produces a plausible curve.

  * Every leg must CONTINUE the run. The card pins two md5 sums on the
    200,000-step pair, and that covers the first leg only. The 450,000 and
    the 665,000 legs resume what the leg before them wrote, so each leg is
    gated on the step it starts at, before it trains and again after. That
    gate reads four strings out of two scripts this study does not own, so
    each one is rendered here from the source that writes it.
  * The recipe must live in one place. Every training flag comes from
    #373's ``run_leg_k.sh``. A copy of the flags in this study's driver is
    a second place for them to drift.
  * 665,000 is not a multiple of ``--save-every``, so the last stop needs
    ``--extra-save-steps``. Without it the leg trains for 18 hours and
    writes no checkpoint at the stop.
  * A head that never scores must fail the driver. A missing point on the
    curve with a zero exit code reads as a finished study. The head's own
    exit code does not answer this. The score file does.
  * The figure's grey rule is #373's 1.0660. Typed a second time, it is a
    number that no longer tracks the file it came from.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDY = REPO_ROOT / "reports" / "2026-08-20_a4_full_pass"
SCRIPTS = STUDY / "scripts"
FULL_PASS_PY = SCRIPTS / "full_pass.py"
RUN_PASS_SH = SCRIPTS / "run_pass.sh"
COLLECT_SH = SCRIPTS / "collect.sh"
PLOT_PY = SCRIPTS / "plot_full_pass.py"

PARENT = REPO_ROOT / "reports" / "2026-08-08_rollout_depth"
RUN_LEG_K = PARENT / "scripts" / "run_leg_k.sh"
STOP_K = PARENT / "scripts" / "stop_k.sh"
EVAL_LOCAL = PARENT / "scripts" / "eval_local.sh"
PARENT_RESULTS = PARENT / "results"

TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")

# Confirmed from small_v1/manifest.json on the HF repo (4274 shards). The
# same constant guards #393's ladder in tests/test_393_ladder.py. Two other
# counts of the same dataset are in play — see full_pass.ROW_COUNTS.
SMALL_V1_ROWS = 42_571_692

CELL = "arm6_v2_combab_alignS"
RUN_NAME = f"cf393_{CELL}_cf373k3"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def strip_comments(text: str) -> str:
    """Remove full-line bash comments so a token search sees only code."""
    return "\n".join(ln for ln in text.splitlines()
                     if not ln.lstrip().startswith("#"))


@pytest.fixture(scope="module")
def fp():
    return load(FULL_PASS_PY, "cf407_full_pass")


@pytest.fixture(scope="module")
def driver_code() -> str:
    return strip_comments(RUN_PASS_SH.read_text())


# --- 1. the card's stops ---------------------------------------------------


class TestStops:

    def test_the_three_stops(self, fp):
        assert fp.STOPS == [300_000, 450_000, 665_000]

    def test_the_run_continues_and_does_not_restart(self, fp):
        """Every stop is above the checkpoint the card names."""
        assert fp.RESUME_STEP == 200_000
        assert min(fp.STOPS) > fp.RESUME_STEP

    def test_the_parent_stops_are_373s(self, fp):
        assert fp.PARENT_STOPS == [40_000, 100_000, 200_000]

    def test_the_module_records_every_row_count_in_play(self, fp):
        """Three counts of small_v1 disagree, so all three are written down."""
        assert fp.ROW_COUNTS == {
            "manifest": 42_571_692,
            "shard_arithmetic": 42_740_000,
        }
        assert fp.CARD_PASS_STEPS == 665_156
        assert fp.ROWS == fp.ROW_COUNTS["manifest"]

    def test_the_module_and_this_file_hold_one_manifest_count(self, fp):
        assert fp.ROW_COUNTS["manifest"] == SMALL_V1_ROWS

    def test_the_three_sources_give_three_step_counts(self, fp):
        assert fp.pass_steps() == {
            "manifest": 665_182,
            "shard_arithmetic": 667_812,
            "card": 665_156,
        }

    @pytest.mark.parametrize("source",
                             ["manifest", "shard_arithmetic", "card"])
    def test_the_last_stop_is_one_pass_under_every_count(self, fp, source):
        """665,000 covers 99.5% to 100% of one pass, whichever count is right.

        The choice between the three does not change the card's question at
        this resolution. The lowest of the three ratios is 0.9958, under the
        shard arithmetic.
        """
        ratio = fp.STOPS[-1] / fp.pass_steps()[source]
        assert 0.995 <= ratio <= 1.0

    def test_one_pass_arithmetic(self, fp):
        assert fp.steps_for_one_pass(SMALL_V1_ROWS, 64) == 665_182
        assert fp.steps_for_one_pass(640, 64) == 10

    @pytest.mark.parametrize("stop", [300_000, 450_000, 665_000])
    def test_every_stop_of_the_card_is_a_whole_thousand(self, fp, stop):
        assert fp.check_stop(stop) == []
        assert fp.stop_k(stop) == stop // 1000

    def test_a_stop_off_the_thousand_grid_is_refused(self, fp):
        """665,500 and 665,000 would both name themselves `bb665k`."""
        assert fp.check_stop(665_500) != []
        with pytest.raises(ValueError):
            fp.stop_k(665_500)

    def test_a_stop_that_is_not_positive_is_refused(self, fp):
        assert fp.check_stop(0) != []
        assert fp.check_stop(-1000) != []

    def test_the_driver_refuses_a_stop_off_the_thousand_grid(self,
                                                             driver_code):
        assert "% 1000" in driver_code


# --- 2. the checkpoint the continuation resumes ----------------------------


class TestResumeCheckpoint:

    def test_the_card_pins_both_files(self, fp):
        assert fp.RESUME_MD5 == {
            "cf393_arm6_v2_combab_alignS_cf373k3_r2_200k.pth":
                "f477c03525bf5e169704715511f1c6d7",
            "cf393_arm6_v2_combab_alignS_cf373k3_r2_200k_optimizer.pth":
                "740891276637ff7bce744b1d9109d57a",
        }

    def test_the_optimizer_state_comes_too(self, fp):
        """A resume without it loses the step counter and AdamW momentum."""
        backbone = [n for n in fp.RESUME_MD5
                    if not n.endswith("_optimizer.pth")]
        assert len(backbone) == 1
        sidecar = backbone[0].replace(".pth", "_optimizer.pth")
        assert sidecar in fp.RESUME_MD5

    def test_check_resume_accepts_a_matching_pair(self, fp, tmp_path):
        leg = tmp_path / fp.CELL / f"leg_{fp.RESUME_STEP // 1000}k"
        leg.mkdir(parents=True)
        digests = {}
        for i, name in enumerate(sorted(fp.RESUME_MD5)):
            body = f"payload {i}".encode()
            (leg / name).write_bytes(body)
            digests[name] = hashlib.md5(body).hexdigest()
        assert fp.check_resume(tmp_path, expect=digests) == []

    def test_check_resume_reports_a_wrong_digest(self, fp, tmp_path):
        leg = tmp_path / fp.CELL / f"leg_{fp.RESUME_STEP // 1000}k"
        leg.mkdir(parents=True)
        for name in fp.RESUME_MD5:
            (leg / name).write_bytes(b"wrong")
        problems = fp.check_resume(tmp_path)
        assert len(problems) == 2
        assert all("md5" in p for p in problems)

    def test_check_resume_reports_a_missing_file(self, fp, tmp_path):
        problems = fp.check_resume(tmp_path)
        assert len(problems) == 2
        assert all("missing" in p for p in problems)

    def test_the_driver_runs_the_check_before_it_trains(self, driver_code):
        assert "--check-resume" in driver_code

    @pytest.mark.skipif(not Path("/home/jupyter/cf373_r3/sync").is_dir(),
                        reason="the round-3 root is on elisa only")
    def test_the_checkpoint_on_this_machine_matches_the_card(self, fp):
        assert fp.check_resume("/home/jupyter/cf373_r3/sync") == []


class TestResumeDigestOverride:
    """The test seam that lets the driver run off elisa.

    The stub launcher needs no real checkpoint. Only the md5 gate does, and
    a 5 MB pair lives on one machine. So the gate reads its digests from
    `CF407_RESUME_MD5` when that variable is set.
    """

    def test_with_no_variable_the_gate_uses_the_cards_digests(self, fp):
        assert fp.resume_md5(env={}) == fp.RESUME_MD5

    def test_an_empty_variable_uses_the_cards_digests(self, fp):
        assert fp.resume_md5(env={fp.ENV_MD5: "  "}) == fp.RESUME_MD5

    def test_the_variable_replaces_the_digests(self, fp):
        names = sorted(fp.RESUME_MD5)
        raw = f"{names[0]}=aaa,{names[1]}=bbb"
        assert fp.resume_md5(env={fp.ENV_MD5: raw}) == {names[0]: "aaa",
                                                        names[1]: "bbb"}

    def test_the_variable_cannot_change_which_files_are_checked(self, fp):
        """The seam changes the digests. It never drops a file."""
        name = sorted(fp.RESUME_MD5)[0]
        with pytest.raises(ValueError):
            fp.resume_md5(env={fp.ENV_MD5: f"{name}=aaa"})
        with pytest.raises(ValueError):
            fp.resume_md5(env={fp.ENV_MD5: "other.pth=aaa,third.pth=bbb"})

    def test_a_malformed_variable_is_refused(self, fp):
        with pytest.raises(ValueError):
            fp.resume_md5(env={fp.ENV_MD5: "no-equals-sign"})

    def test_check_resume_reads_the_variable(self, fp, tmp_path,
                                             monkeypatch):
        leg = tmp_path / fp.CELL / f"leg_{fp.RESUME_STEP // 1000}k"
        leg.mkdir(parents=True)
        pairs = []
        for i, name in enumerate(sorted(fp.RESUME_MD5)):
            body = f"stub {i}".encode()
            (leg / name).write_bytes(body)
            pairs.append(f"{name}={hashlib.md5(body).hexdigest()}")
        monkeypatch.setenv(fp.ENV_MD5, ",".join(pairs))
        assert fp.check_resume(tmp_path) == []


# --- 3. every leg continues the run ----------------------------------------


class TestLegContinuity:
    """The failure the card must not have: a leg that starts at step 0.

    `run_leg_k.sh` resumes the furthest checkpoint it finds, and it starts
    fresh when it finds none. Either way it trains to the target, writes a
    checkpoint and scores. `train.py` does the same when the optimizer
    sidecar is missing: it loads the weights, prints "starting fresh" and
    counts from 0.
    """

    def test_the_first_leg_starts_at_the_checkpoint_the_card_pins(self, fp):
        assert fp.prior_stop(300_000) == fp.RESUME_STEP == 200_000

    def test_each_later_leg_starts_at_the_stop_before_it(self, fp):
        assert fp.prior_stop(450_000) == 300_000
        assert fp.prior_stop(665_000) == 450_000

    def test_a_step_that_is_not_a_stop_has_no_prior(self, fp):
        with pytest.raises(ValueError):
            fp.prior_stop(500_000)

    def test_the_step_a_checkpoint_name_carries(self, fp):
        assert fp.ckpt_step("x/cf393_a_cf373k3_200k.pth") == 200_000
        assert fp.ckpt_step("x/cf393_a_cf373k3_r2_200k.pth") == 200_000
        assert fp.ckpt_step("x/cf393_a_cf373k3_best_gap.pth") is None

    def test_the_sidecar_name_matches_the_checkpoint_module(self, fp):
        sys.path.insert(0, str(REPO_ROOT))
        from src.checkpoint import get_optimizer_state_path
        path = "/tmp/run_300k.pth"
        assert fp.sidecar(path) == get_optimizer_state_path(path)

    def test_the_resume_source_is_the_furthest_leg(self, fp, tmp_path):
        """Never by mtime. A copied checkpoint set comes back in copy order."""
        for step_k, leg_k in ((100, 100), (200, 200), (140, 200)):
            leg = tmp_path / CELL / f"leg_{leg_k}k"
            leg.mkdir(parents=True, exist_ok=True)
            (leg / f"{RUN_NAME}_{step_k}k.pth").write_bytes(b"x")
        newest = tmp_path / CELL / "leg_100k" / f"{RUN_NAME}_100k.pth"
        os.utime(newest, (2 ** 31, 2 ** 31))
        assert fp.ckpt_step(fp.resume_source(tmp_path)) == 200_000

    def test_the_resume_source_tolerates_the_rn_infix(self, fp, tmp_path):
        leg = tmp_path / CELL / "leg_200k"
        leg.mkdir(parents=True)
        (leg / f"{RUN_NAME}_r2_200k.pth").write_bytes(b"x")
        assert fp.resume_source(tmp_path).endswith("_r2_200k.pth")

    def test_an_empty_root_has_no_resume_source(self, fp, tmp_path):
        assert fp.resume_source(tmp_path) is None

    # ---- before the leg ---------------------------------------------------

    def test_the_gate_refuses_a_root_with_no_checkpoint(self, fp, tmp_path):
        problems = fp.check_leg_start(tmp_path, 300_000)
        assert problems and "step 0" in problems[0]

    def test_the_gate_accepts_the_prior_stop(self, fp, tmp_path):
        self._land(tmp_path, 200_000)
        assert fp.check_leg_start(tmp_path, 300_000) == []

    def test_the_gate_refuses_a_checkpoint_below_the_prior_stop(self, fp,
                                                                tmp_path):
        """The 450k leg must not resume the 200k checkpoint."""
        self._land(tmp_path, 200_000)
        problems = fp.check_leg_start(tmp_path, 450_000)
        assert problems and "200000" in problems[0]

    def test_the_gate_accepts_a_crash_resume_inside_the_leg(self, fp,
                                                            tmp_path):
        """A leg that died at 380k resumes 380k, not 300k. That is legal."""
        self._land(tmp_path, 300_000)
        self._land(tmp_path, 380_000, leg=450_000)
        assert fp.check_leg_start(tmp_path, 450_000) == []

    def test_the_gate_refuses_a_checkpoint_with_no_optimizer_state(
            self, fp, tmp_path):
        self._land(tmp_path, 200_000, with_sidecar=False)
        problems = fp.check_leg_start(tmp_path, 300_000)
        assert any("optimizer" in p for p in problems)

    def test_the_gate_passes_a_leg_that_is_already_done(self, fp, tmp_path):
        """`run_leg_k.sh` skips it, so there is no resume to check.

        The chain behind it is checked instead. The watchdog re-fires the
        driver onto checkpoints no process of its own watched land.
        """
        self._land(tmp_path, 200_000)
        self._land(tmp_path, 300_000)
        assert fp.check_leg_start(tmp_path, 300_000) == []

    def test_a_finished_leg_with_no_trajectory_behind_it_is_refused(
            self, fp, tmp_path):
        """A 300k checkpoint alone did not come from the card's 200k."""
        self._land(tmp_path, 300_000)
        problems = fp.check_leg_start(tmp_path, 300_000)
        assert problems and "200000" in problems[0]

    def test_a_finished_leg_wants_every_stop_behind_it(self, fp, tmp_path):
        """The 450k leg needs 200k AND 300k on disk, not just one."""
        self._land(tmp_path, 200_000)
        self._land(tmp_path, 450_000)
        problems = fp.check_leg_start(tmp_path, 450_000)
        assert problems and "300000" in problems[0]
        self._land(tmp_path, 300_000)
        assert fp.check_leg_start(tmp_path, 450_000) == []

    def test_a_finished_leg_wants_the_sidecars_behind_it(self, fp, tmp_path):
        self._land(tmp_path, 200_000, with_sidecar=False)
        self._land(tmp_path, 300_000)
        problems = fp.check_leg_start(tmp_path, 300_000)
        assert any("optimizer" in p for p in problems)

    # ---- after the leg ----------------------------------------------------

    def test_the_done_check_accepts_a_real_continuation(self, fp, tmp_path):
        self._land(tmp_path, 300_000)
        assert fp.check_leg_done(
            tmp_path, 300_000,
            "[08-20] [c] RESUME from run_r2_200k.pth (step 200k)",
            "Resumed from /x/run_r2_200k.pth at step 200000") == []

    def test_the_done_check_refuses_a_fresh_start(self, fp, tmp_path):
        self._land(tmp_path, 300_000)
        problems = fp.check_leg_done(tmp_path, 300_000,
                                     "[08-20] [c] FRESH start at step 0", "")
        assert problems and "step 0" in problems[0]

    def test_the_done_check_refuses_a_resume_at_step_zero(self, fp, tmp_path):
        """The launcher resumed. train.py found no sidecar and counted from 0."""
        self._land(tmp_path, 300_000)
        problems = fp.check_leg_done(
            tmp_path, 300_000,
            "[08-20] [c] RESUME from run_r2_200k.pth (step 200k)",
            "  [checkpoint] No optimizer state, starting fresh.\n"
            "Resumed from /x/run_r2_200k.pth at step 0")
        assert problems and "step 0" in problems[0]

    def test_the_done_check_refuses_a_resume_below_the_prior_stop(
            self, fp, tmp_path):
        self._land(tmp_path, 450_000)
        problems = fp.check_leg_done(
            tmp_path, 450_000,
            "[08-20] [c] RESUME from run_200k.pth (step 200k)",
            "Resumed from /x/run_200k.pth at step 200000")
        assert problems and "200000" in problems[0]

    def test_the_done_check_accepts_a_skipped_leg(self, fp, tmp_path):
        self._land(tmp_path, 300_000)
        assert fp.check_leg_done(tmp_path, 300_000,
                                 "[08-20] [c] SKIP: run_300k.pth already "
                                 "on disk", "") == []

    def test_the_done_check_refuses_a_leg_that_wrote_no_checkpoint(
            self, fp, tmp_path):
        problems = fp.check_leg_done(
            tmp_path, 300_000,
            "[08-20] [c] RESUME from run_r2_200k.pth (step 200k)",
            "Resumed from /x/run_r2_200k.pth at step 200000")
        assert any("no checkpoint" in p for p in problems)

    def test_the_done_check_refuses_a_leg_with_no_outcome_line(self, fp,
                                                               tmp_path):
        self._land(tmp_path, 300_000)
        problems = fp.check_leg_done(tmp_path, 300_000, "", "")
        assert problems

    def test_the_done_check_refuses_a_silent_train_log(self, fp, tmp_path):
        self._land(tmp_path, 300_000)
        problems = fp.check_leg_done(
            tmp_path, 300_000,
            "[08-20] [c] RESUME from run_r2_200k.pth (step 200k)", "")
        assert problems and "Resumed from" in problems[0]

    # ---- where the two logs live ------------------------------------------

    def test_the_log_paths_come_off_the_checkout(self, fp):
        wt = "/w"
        res = "/w/reports/2026-08-08_rollout_depth/results"
        assert fp.parent_results(wt) == res
        assert fp.cell_log(wt) == f"{res}/leg_{CELL}.log"
        assert fp.train_log(wt) == f"{res}/run_{RUN_NAME}.log"

    def test_the_module_prints_the_two_log_paths(self, tmp_path):
        """The driver reads them rather than spelling the run name again."""
        proc = subprocess.run(
            [sys.executable, str(FULL_PASS_PY), "--log-paths", "--wt", "/w"],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert proc.stdout.split() == [
            f"/w/reports/2026-08-08_rollout_depth/results/leg_{CELL}.log",
            f"/w/reports/2026-08-08_rollout_depth/results/run_{RUN_NAME}.log",
        ]

    def test_reading_a_log_from_an_offset(self, fp, tmp_path):
        path = tmp_path / "run.log"
        path.write_text("old line\nnew line\n")
        assert fp.read_since(path, len("old line\n")) == "new line\n"
        assert fp.read_since(tmp_path / "absent", 0) == ""

    # ---- helper -----------------------------------------------------------

    @staticmethod
    def _land(root: Path, step: int, leg: int | None = None,
              with_sidecar: bool = True):
        """Put a checkpoint pair at `step` into the leg dir for `leg`."""
        leg_k = (leg if leg is not None else step) // 1000
        d = Path(root) / CELL / f"leg_{leg_k}k"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{RUN_NAME}_{step // 1000}k.pth").write_bytes(b"x")
        if with_sidecar:
            (d / f"{RUN_NAME}_{step // 1000}k_optimizer.pth").write_bytes(b"x")


# --- 4. the recipe stays in #373's launcher --------------------------------


# Flags that decide what the backbone learns. Every one of them belongs to
# `run_leg_k.sh`. This study's driver must name none of them.
RECIPE_FLAGS = ("--loss-shape", "--align-loss-weight", "--align-target",
                "--moco-rep-keys", "--tau-rep", "--cpc-infonce-weight",
                "--train-rollout-depth", "--ema-tau", "--batch-size",
                "--lr", "--seed", "--hf-path", "--total-steps")


class TestOneRecipe:

    def test_the_driver_calls_373s_launcher(self, driver_code):
        assert "run_leg_k.sh" in driver_code

    def test_the_driver_calls_373s_stop_script(self, driver_code):
        assert "stop_k.sh" in driver_code

    @pytest.mark.parametrize("flag", RECIPE_FLAGS)
    def test_the_driver_repeats_no_training_flag(self, driver_code, flag):
        assert flag not in driver_code

    def test_the_driver_names_the_cell_the_card_names(self, driver_code):
        assert "arm6_v2_combab_alignS" in driver_code

    def test_the_a4_case_block_is_unchanged(self):
        """Non-regression: A4's four loss flags, in #373's launcher."""
        code = strip_comments(RUN_LEG_K.read_text())
        m = re.search(r'(?m)^\s*arm6_v2_combab_alignS\)\s*\n(.*?)\n\s*;;',
                      code, re.DOTALL)
        assert m, "no arm6_v2_combab_alignS case block in run_leg_k.sh"
        body = " ".join(m.group(1).split())
        assert "--loss-shape cosine_similarity_batch_rep_only" in body
        assert "--align-loss-weight 1.0" in body
        assert "--moco-rep-keys" in body
        assert "--tau-rep 1.0" in body
        assert "--cpc-infonce-weight 0.0" in body
        assert "--align-target student" in body

    def test_the_shared_block_still_carries_k_and_the_seed(self):
        code = strip_comments(RUN_LEG_K.read_text())
        assert '--train-rollout-depth "$K"' in code
        assert 'K="${K:-3}"' in code
        assert 'SEED="${SEED:-20260520}"' in code
        assert "--batch-size 64" in code

    def test_the_driver_holds_k_at_three(self, fp):
        assert fp.K == 3

    def test_the_launcher_still_names_the_run_the_module_names(self, fp):
        """`RUN_NAME` is how this study finds every checkpoint of the run."""
        code = strip_comments(RUN_LEG_K.read_text())
        assert 'NAME="cf393_${CELL}_cf373k${K}${RUN_SUFFIX:-}"' in code
        assert fp.RUN_NAME == RUN_NAME

    def test_the_launcher_still_saves_one_dir_per_leg(self, fp):
        """`leg_dir` mirrors `leg_paths.sh`. A shared dir branches the name."""
        code = strip_comments(
            (PARENT / "scripts" / "leg_paths.sh").read_text())
        assert "printf '%s/leg_%dk\\n'" in code
        assert fp.leg_dir("/r", 300_000) == f"/r/{CELL}/leg_300k"

    def test_the_driver_and_the_module_agree(self, fp, driver_code):
        """Bash cannot import the module, so the two are checked instead."""
        assert f'CELL="{fp.CELL}"' in driver_code
        assert f'CELL_ID="{fp.CELL_ID}"' in driver_code
        assert f"DEPTH={fp.K}" in driver_code
        assert f"HEAD_STEPS={fp.HEAD_STEPS}" in driver_code
        assert f"HEAD_SEED={fp.HEAD_SEED}" in driver_code
        default = " ".join(str(s) for s in fp.STOPS)
        assert f"STOPS=({default})" in driver_code


# --- 5. 665,000 is off the save cadence ------------------------------------


@pytest.fixture(scope="module")
def train_mod():
    """train.py's own save-cadence functions, without running its main()."""
    spec = importlib.util.spec_from_file_location("cf407_train", TRAIN_PY)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


class TestSaveCadence:

    def test_the_launcher_defaults_extra_saves_to_the_target(self):
        code = strip_comments(RUN_LEG_K.read_text())
        assert 'EXTRA_SAVES="${EXTRA_SAVES:-$TARGET_STEPS}"' in code
        assert 'SAVE_EVERY="${SAVE_EVERY:-20000}"' in code

    def test_the_driver_does_not_override_extra_saves(self, driver_code):
        assert "EXTRA_SAVES" not in driver_code

    @pytest.mark.parametrize("stop", [300_000, 450_000, 665_000])
    def test_every_stop_writes_its_checkpoint(self, train_mod, fp, stop):
        extras = train_mod.parse_extra_save_steps(str(stop))
        assert train_mod.should_snapshot(stop, 20_000, extras)

    def test_665k_would_be_lost_without_the_extra_save(self, train_mod):
        """The reason the default matters: 665,000 is not on the grid."""
        assert not train_mod.should_snapshot(665_000, 20_000, set())

    @pytest.mark.parametrize("stop", [300_000, 450_000, 665_000])
    def test_the_checkpoint_name_a_fresh_leg_writes(self, fp, stop):
        assert fp.ckpt_name(stop) == f"{RUN_NAME}_{stop // 1000}k.pth"

    def test_the_checkpoint_lookup_tolerates_the_rn_infix(self, fp, tmp_path):
        """`ckpt_at_step` globs, so a re-fired leg's file is still found."""
        leg = tmp_path / CELL / "leg_300k"
        leg.mkdir(parents=True)
        (leg / f"{RUN_NAME}_r2_300k.pth").write_bytes(b"x")
        assert fp.ckpt_path(tmp_path, 300_000).endswith("_r2_300k.pth")

    def test_the_checkpoint_lookup_ignores_the_optimizer_file(self, fp,
                                                              tmp_path):
        leg = tmp_path / CELL / "leg_300k"
        leg.mkdir(parents=True)
        (leg / f"{RUN_NAME}_300k_optimizer.pth").write_bytes(b"x")
        assert fp.ckpt_path(tmp_path, 300_000) is None


# --- 6. the head and the eval protocol -------------------------------------


class TestHeadProtocol:

    def test_the_card_s_head_budget_and_seed(self, fp):
        assert fp.HEAD_STEPS == 30_000
        assert fp.HEAD_SEED == 20260722

    def test_both_encoders(self, fp):
        assert fp.HEADS == ["student", "teacher"]

    def test_the_driver_pins_the_budget_and_the_seed(self, driver_code):
        assert "HEAD_STEPS=30000" in driver_code
        assert "HEAD_SEED=20260722" in driver_code

    def test_the_driver_trains_both_heads(self, driver_code):
        assert "student" in driver_code and "teacher" in driver_code

    def test_the_stop_script_still_clips_the_head(self):
        """The card asks for --grad-clip 1.0 on the head, and only there."""
        code = strip_comments(STOP_K.read_text())
        assert "--quantile-head --grad-clip 1.0" in code

    def test_the_backbone_is_never_clipped(self):
        """CLAUDE.md: never grad-clip this project's backbone."""
        assert "--grad-clip" not in strip_comments(RUN_LEG_K.read_text())


class TestEvalProtocol:

    def test_the_official_b4_strategy_at_horizon_16(self):
        code = strip_comments(EVAL_LOCAL.read_text())
        assert "--strategy B4 --forecast-len 16" in code

    def test_the_merge_insists_on_97_configs(self):
        """The gate reads its count from EVAL_EXPECT_CONFIGS, which is 97.

        #404 moved the literal into that variable. The test pins the
        comparison and the default, so a change to either one fails here.
        """
        code = strip_comments(EVAL_LOCAL.read_text())
        assert 'EVAL_EXPECT_CONFIGS="${EVAL_EXPECT_CONFIGS:-97}"' in code
        assert '"$n_rows" -ne "$EVAL_EXPECT_CONFIGS"' in code
        assert '"$n_uniq" -ne "$EVAL_EXPECT_CONFIGS"' in code

    def test_a_head_that_wrote_a_score_passes_the_gate(self, fp, tmp_path):
        results = Path(fp.parent_results(tmp_path))
        results.mkdir(parents=True)
        (results / "score_A4_k3_bb300k_student.txt").write_text("1.0500\n")
        assert fp.check_score(tmp_path, 300_000, "student") == []

    def test_a_head_that_wrote_no_score_fails_the_gate(self, fp, tmp_path):
        """The outcome of the check above: the eval stops before the score."""
        assert fp.check_score(tmp_path, 300_000, "student") != []

    def test_a_score_file_with_no_number_fails_the_gate(self, fp, tmp_path):
        results = Path(fp.parent_results(tmp_path))
        results.mkdir(parents=True)
        (results / "score_A4_k3_bb300k_teacher.txt").write_text("\n")
        assert fp.check_score(tmp_path, 300_000, "teacher") != []


# --- 7. what the collector copies ------------------------------------------


CSV_HEADER = "dataset,model,eval_metrics/MASE[0.5]"


def gift_csv(path: Path, n: int, final_newline: bool = True):
    """A merged all_results.csv with `n` distinct configs."""
    rows = [CSV_HEADER] + [f"cfg{i}/H/short,contrastive_tiny,1.0"
                           for i in range(n)]
    text = "\n".join(rows) + ("\n" if final_newline else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def collect(tmp_path: Path, stops="300", **files) -> subprocess.CompletedProcess:
    """Run collect.sh over a tree built for the test."""
    env = os.environ.copy()
    env.update({
        "WT": str(tmp_path / "wt"),
        "RUNS": str(tmp_path / "runs"),
        "CF407_RESULTS": str(tmp_path / "res"),
        "STOPS": stops,
    })
    return subprocess.run(["bash", str(COLLECT_SH)], capture_output=True,
                          text=True, env=env, timeout=300)


class TestCollect:

    def _study(self, tmp_path, tag, n_configs, final_newline=True):
        parent = tmp_path / "wt" / "reports" / "2026-08-08_rollout_depth" \
            / "results"
        parent.mkdir(parents=True, exist_ok=True)
        (parent / f"score_{tag}.txt").write_text("1.0500\n")
        gift = tmp_path / "runs" / "eval" / tag / "gift"
        gift_csv(gift / "all_results.csv", n_configs, final_newline)
        (gift / "summary.txt").write_text(
            "Aggregate GM-Relative MASE (97 configs): 1.0500\n")

    def test_a_complete_eval_brings_its_score_across(self, tmp_path):
        self._study(tmp_path, "A4_k3_bb300k_student", 97)
        proc = collect(tmp_path)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert (tmp_path / "res" / "score_A4_k3_bb300k_student.txt").is_file()
        assert (tmp_path / "res" / "eval" / "A4_k3_bb300k_student"
                / "all_results.csv").is_file()

    def test_a_short_eval_brings_nothing_across(self, tmp_path):
        """A score over 40 configs is not this study's metric."""
        self._study(tmp_path, "A4_k3_bb300k_student", 40)
        proc = collect(tmp_path)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert not (tmp_path / "res"
                    / "score_A4_k3_bb300k_student.txt").exists()
        assert not (tmp_path / "res" / "eval"
                    / "A4_k3_bb300k_student").exists()
        assert "skip" in proc.stdout

    def test_a_csv_with_no_final_newline_still_counts_97(self, tmp_path):
        """`wc -l` minus one loses the last row of an unterminated file."""
        self._study(tmp_path, "A4_k3_bb300k_student", 97, final_newline=False)
        proc = collect(tmp_path)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert (tmp_path / "res" / "score_A4_k3_bb300k_student.txt").is_file()

    def test_a_score_with_no_eval_beside_it_stays_behind(self, tmp_path):
        parent = tmp_path / "wt" / "reports" / "2026-08-08_rollout_depth" \
            / "results"
        parent.mkdir(parents=True)
        (parent / "score_A4_k3_bb300k_student.txt").write_text("1.0500\n")
        proc = collect(tmp_path)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert not (tmp_path / "res"
                    / "score_A4_k3_bb300k_student.txt").exists()

    def test_it_counts_what_it_copied(self, tmp_path):
        self._study(tmp_path, "A4_k3_bb300k_student", 97)
        self._study(tmp_path, "A4_k3_bb300k_teacher", 12)
        proc = collect(tmp_path)
        assert "scores 1" in proc.stdout
        assert "evals 1" in proc.stdout
        assert "skipped 1" in proc.stdout


# --- 8. the deliverable figure ---------------------------------------------


class TestFigureData:

    def test_the_module_reads_no_file_at_import(self):
        """`BEST_BEFORE = best_before()` made every importer read #373's disk.

        The plot, the driver and the collector all import this module. One
        of them runs on a box that holds no copy of #373's results.
        """
        tree = ast.parse(FULL_PASS_PY.read_text())
        top = [n for node in tree.body
               if not isinstance(node, (ast.FunctionDef, ast.ClassDef))
               for n in ast.walk(node) if isinstance(n, ast.Call)]
        named = {n.func.id for n in top if isinstance(n.func, ast.Name)}
        assert "open" not in named
        assert "best_before" not in named

    def test_the_grey_rule_is_373s_committed_score(self, fp):
        """1.0660 is a file, not a number someone typed twice."""
        published = float(
            (PARENT_RESULTS / "score_A4_k3_bb200k_student.txt").read_text())
        assert fp.best_before(PARENT_RESULTS) == published

    def test_tags_match_the_score_files_373_wrote(self, fp):
        assert fp.tag(200_000, "student") == "A4_k3_bb200k_student"
        assert fp.tag(665_000, "teacher") == "A4_k3_bb665k_teacher"

    def test_the_student_points_come_off_disk(self, fp):
        got = fp.curve("student", results=None, parent=PARENT_RESULTS)
        assert got[40_000] == 1.0862
        assert got[100_000] == 1.0801
        assert got[200_000] == 1.0660

    def test_the_teacher_points_come_off_disk(self, fp):
        """The card puts both heads on the axis, so both are pinned."""
        got = fp.curve("teacher", results=None, parent=PARENT_RESULTS)
        assert got[40_000] == 1.0855
        assert got[100_000] == 1.0874
        assert got[200_000] == 1.0828

    def test_both_heads_have_all_three_parent_points(self, fp):
        for head in fp.HEADS:
            got = fp.curve(head, results=None, parent=PARENT_RESULTS)
            assert sorted(got) == fp.PARENT_STOPS

    def test_this_study_extends_the_same_curve(self, fp, tmp_path):
        (tmp_path / "score_A4_k3_bb300k_student.txt").write_text("1.0500\n")
        got = fp.curve("student", results=tmp_path, parent=PARENT_RESULTS)
        assert sorted(got) == [40_000, 100_000, 200_000, 300_000]
        assert got[300_000] == 1.05

    def test_a_stop_with_no_score_is_absent_not_zero(self, fp, tmp_path):
        got = fp.curve("teacher", results=tmp_path, parent=PARENT_RESULTS)
        assert 300_000 not in got

    def test_an_unreadable_score_file_is_absent(self, fp, tmp_path):
        (tmp_path / "score_A4_k3_bb300k_student.txt").write_text("")
        got = fp.curve("student", results=tmp_path, parent=PARENT_RESULTS)
        assert 300_000 not in got


# --- 9. the driver, end to end, with the two child scripts stubbed --------


# A stand-in for #373's `run_leg_k.sh`. It resolves the resume the way
# `leg_paths.sh` does, writes the two lines the real launcher and train.py
# write, and lands the target checkpoint pair. So the driver's own
# continuity gates run against a leg that behaves like the real one.
LEG_STUB = r"""#!/bin/bash
cell="$1"; target="$2"
res="$WT/reports/2026-08-08_rollout_depth/results"
name="cf393_${cell}_cf373k3"
tk=$(( target / 1000 ))
mkdir -p "$res"
echo "leg cell=$cell target=$target RUNS=${RUNS:-} BB_GPU=${BB_GPU:-}" \
  >>"$CF407_LOG"
done_ckpt=$(ls "$RUNS/$cell"/leg_"$tk"k/"$name"*_"$tk"k.pth 2>/dev/null \
  | grep -v optimizer | head -1)
if [ -n "$done_ckpt" ]; then
  echo "[stub] [$cell] SKIP: $(basename "$done_ckpt") already on disk" \
    >>"$res/leg_$cell.log"
  exit 0
fi
latest=$(ls "$RUNS/$cell"/leg_*/"$name"*_[0-9]*k.pth 2>/dev/null \
  | sed -E 's|.*_([0-9]+)k\.pth$|\1 &|' | sort -k1,1n | tail -1 | cut -d' ' -f2-)
leg="$RUNS/$cell/leg_${tk}k"
mkdir -p "$leg"
if [ -n "$latest" ] && [ -z "${CF407_STUB_FRESH:-}" ]; then
  k=$(printf '%s\n' "$latest" | sed -E 's|.*_([0-9]+)k\.pth$|\1|')
  echo "resume $(basename "$latest")" >>"$CF407_LOG"
  echo "[stub] [$cell] RESUME from $(basename "$latest") (step ${k}k)" \
    >>"$res/leg_$cell.log"
  echo "Resumed from $latest at step ${CF407_STUB_START:-$(( k * 1000 ))}" \
    >>"$res/run_$name.log"
else
  echo "resume none" >>"$CF407_LOG"
  echo "[stub] [$cell] FRESH start at step 0" >>"$res/leg_$cell.log"
fi
: >"$leg/${name}_${tk}k.pth"
: >"$leg/${name}_${tk}k_optimizer.pth"
exit 0
"""

STOP_STUB = """#!/bin/bash
echo "stop cell=$1 k=$2 target=$3 head=$4 ROOT=${CF373_ROOT:-}\
 HEAD_STEPS=${HEAD_STEPS:-} HEAD_SEED=${HEAD_SEED:-} GPU=${BB_GPU:-}" \\
  >>"$CF407_LOG"
n=$(grep -c "head=$4 " "$CF407_LOG")
[ -n "${CF407_FAIL_ONCE:-}" ] && [ "$n" -eq 1 ] && exit 9
[ -n "${CF407_FAIL_HEAD:-}" ] && [ "$4" = "$CF407_FAIL_HEAD" ] && exit 9
# The score, where `stop_k.sh` points `eval_local.sh`. CF407_NO_SCORE holds
# it back for one head, which is what an eval short of the 97 configs does:
# the script still exits 0, and no number lands.
res="$WT/reports/2026-08-08_rollout_depth/results"
tag="${1}_k${2}_bb$(( $3 / 1000 ))k_${4}"
mkdir -p "$res"
[ "${CF407_NO_SCORE:-}" = "$4" ] || \
  echo "${CF407_STUB_SCORE:-1.0500}" >"$res/score_${tag}.txt"
exit 0
"""

REAL_ROOT = Path("/home/jupyter/cf373_r3/sync")
needs_checkpoint = pytest.mark.skipif(
    not (REAL_ROOT / CELL / "leg_200k").is_dir(),
    reason="the checkpoint the card pins is on elisa only")


def stub_checkout(tmp_path: Path, leg_body: str = LEG_STUB) -> Path:
    """A checkout whose #373 launcher and stop script only take notes."""
    scripts = tmp_path / "wt" / "reports" / "2026-08-08_rollout_depth" \
        / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for name, body in (("run_leg_k.sh", leg_body), ("stop_k.sh", STOP_STUB)):
        (scripts / name).write_text(body)
        (scripts / name).chmod(0o755)
    return tmp_path / "wt"


def stub_root(tmp_path: Path) -> tuple[Path, str]:
    """The card's checkpoint pair, as two small files.

    Returns the root and the `CF407_RESUME_MD5` value that matches it, so
    the whole driver runs on a machine that holds no real checkpoint.
    """
    leg = tmp_path / "runs" / CELL / "leg_200k"
    leg.mkdir(parents=True)
    pairs = []
    for i, name in enumerate(sorted(
            ("cf393_arm6_v2_combab_alignS_cf373k3_r2_200k.pth",
             "cf393_arm6_v2_combab_alignS_cf373k3_r2_200k_optimizer.pth"))):
        body = f"stub checkpoint {i}".encode()
        (leg / name).write_bytes(body)
        pairs.append(f"{name}={hashlib.md5(body).hexdigest()}")
    return tmp_path / "runs", ",".join(pairs)


def linked_root(tmp_path: Path) -> Path:
    """The card's checkpoint pair, linked rather than copied (5 MB each)."""
    leg = tmp_path / "runs" / CELL / "leg_200k"
    leg.mkdir(parents=True)
    src = REAL_ROOT / CELL / "leg_200k"
    for name in ("cf393_arm6_v2_combab_alignS_cf373k3_r2_200k.pth",
                 "cf393_arm6_v2_combab_alignS_cf373k3_r2_200k_optimizer.pth"):
        (leg / name).symlink_to(src / name)
    return tmp_path / "runs"


def drive(tmp_path, runs=None, extra_env=None, stops=(), leg_body=LEG_STUB,
          digests=None):
    env = os.environ.copy()
    if runs is None:
        runs, digests = stub_root(tmp_path)
    env.update({
        "WT": str(stub_checkout(tmp_path, leg_body)),
        "RUNS": str(runs),
        "CF407_RESULTS": str(tmp_path / "res"),
        "CF407_LOG": str(tmp_path / "calls.log"),
        # The gate would otherwise wait a day for a card another session
        # filled, which is what it is for outside a test.
        "BB_VRAM_MIB": "0",
        "BB_GPU": "1",
    })
    if digests:
        env["CF407_RESUME_MD5"] = digests
    env.update(extra_env or {})
    proc = subprocess.run(["bash", str(RUN_PASS_SH), *[str(s) for s in stops]],
                          capture_output=True, text=True, env=env, timeout=300)
    log = tmp_path / "calls.log"
    return proc, (log.read_text().splitlines() if log.exists() else [])


class TestDriver:

    def test_it_refuses_a_root_that_is_not_the_cards(self, tmp_path):
        """No checkpoint, no training. The gate runs before the first leg."""
        _, digests = stub_root(tmp_path)
        proc, calls = drive(tmp_path, tmp_path / "empty", digests=digests)
        assert proc.returncode == 3
        assert calls == []

    def test_it_refuses_a_checkpoint_with_the_wrong_digest(self, tmp_path):
        runs, _ = stub_root(tmp_path)
        wrong = ",".join(f"{n}=00000000000000000000000000000000"
                         for n in sorted(
                             ("cf393_arm6_v2_combab_alignS_cf373k3_r2_200k"
                              ".pth",
                              "cf393_arm6_v2_combab_alignS_cf373k3_r2_200k"
                              "_optimizer.pth")))
        proc, calls = drive(tmp_path, runs, digests=wrong)
        assert proc.returncode == 3
        assert calls == []

    def test_it_walks_the_three_stops_in_order(self, tmp_path):
        proc, calls = drive(tmp_path)
        assert proc.returncode == 0, proc.stderr[-2000:]
        legs = [ln for ln in calls if ln.startswith("leg ")]
        assert [ln.split("target=")[1].split()[0] for ln in legs] == \
            ["300000", "450000", "665000"]

    def test_each_leg_resumes_the_stop_before_it(self, tmp_path):
        """The card's contract, leg by leg. This is what a restart breaks."""
        _, calls = drive(tmp_path)
        resumed = [ln.split(None, 1)[1] for ln in calls
                   if ln.startswith("resume ")]
        assert resumed == [
            f"{RUN_NAME}_r2_200k.pth",
            f"{RUN_NAME}_300k.pth",
            f"{RUN_NAME}_450k.pth",
        ]

    def test_a_leg_that_starts_fresh_stops_the_driver(self, tmp_path):
        """A restart at step 0 still scores. It must never reach a head."""
        proc, calls = drive(tmp_path, extra_env={"CF407_STUB_FRESH": "1"})
        assert proc.returncode == 3
        assert [ln for ln in calls if ln.startswith("stop ")] == []
        assert len([ln for ln in calls if ln.startswith("leg ")]) == 1

    def test_a_resume_that_lands_at_step_zero_stops_the_driver(self, tmp_path):
        """train.py with no optimizer sidecar: it resumes, then counts from 0."""
        proc, calls = drive(tmp_path, extra_env={"CF407_STUB_START": "0"})
        assert proc.returncode == 3
        assert [ln for ln in calls if ln.startswith("stop ")] == []

    def test_a_second_run_skips_the_legs_it_already_did(self, tmp_path):
        """The driver is re-fireable. A skipped leg is not a restart."""
        runs, digests = stub_root(tmp_path)
        first, _ = drive(tmp_path, runs, digests=digests)
        assert first.returncode == 0, first.stderr[-2000:]
        (tmp_path / "calls.log").unlink()
        second, calls = drive(tmp_path, runs, digests=digests)
        assert second.returncode == 0, second.stderr[-2000:]
        assert [ln for ln in calls if ln.startswith("resume ")] == []

    def test_each_leg_is_followed_by_both_heads(self, tmp_path):
        _, calls = drive(tmp_path)
        shape = [ln.split()[0] + ":" + ln.split("head=")[1].split()[0]
                 if ln.startswith("stop ") else ln.split()[0]
                 for ln in calls]
        assert [s for s in shape if s != "resume"] == \
            ["leg", "stop:student", "stop:teacher"] * 3

    def test_it_hands_the_launcher_the_cell_and_the_target(self, tmp_path):
        _, calls = drive(tmp_path, stops=(300_000,))
        leg = [ln for ln in calls if ln.startswith("leg ")][0]
        assert "cell=arm6_v2_combab_alignS" in leg
        assert "target=300000" in leg

    def test_the_two_scripts_read_one_root(self, tmp_path):
        """`run_leg_k.sh` takes RUNS, `stop_k.sh` takes CF373_ROOT."""
        runs, digests = stub_root(tmp_path)
        _, calls = drive(tmp_path, runs, stops=(300_000,), digests=digests)
        assert f"RUNS={runs}" in calls[0]
        assert f"ROOT={runs}" in [ln for ln in calls
                                  if ln.startswith("stop ")][0]

    def test_it_pins_the_cards_head_protocol(self, tmp_path):
        _, calls = drive(tmp_path, stops=(300_000,))
        for line in [ln for ln in calls if ln.startswith("stop ")]:
            assert "cell=A4 k=3 target=300000" in line
            assert "HEAD_STEPS=30000" in line
            assert "HEAD_SEED=20260722" in line

    def test_a_failed_head_is_retried_once(self, tmp_path):
        """A transient must not cost 30,000 GPU steps that already ran."""
        _, calls = drive(tmp_path, stops=(300_000,),
                         extra_env={"CF407_FAIL_ONCE": "1"})
        heads = [ln for ln in calls if ln.startswith("stop ")]
        assert len(heads) == 4          # two heads, each attempted twice

    def test_a_head_that_never_scores_fails_the_driver(self, tmp_path):
        """A point missing off the curve must not read as a finished study."""
        proc, calls = drive(tmp_path, stops=(300_000,),
                            extra_env={"CF407_FAIL_HEAD": "teacher"})
        assert proc.returncode == 4
        assert "300k/teacher" in proc.stdout + proc.stderr

    def test_a_head_that_exits_0_and_scores_nothing_is_missing(self,
                                                               tmp_path):
        """A clean exit code is not a score.

        `eval_local.sh` writes `score_<tag>.txt` last, and it stops before
        that line when the merged CSV is short of the 97 configs. The pair
        then reaches `collect.sh`, which drops it, and the figure draws a
        shorter line that reads as a finished study.
        """
        proc, calls = drive(tmp_path, stops=(300_000,),
                            extra_env={"CF407_NO_SCORE": "teacher"})
        assert proc.returncode == 4
        out = proc.stdout + proc.stderr
        assert "300k/teacher" in out
        assert "300k/student" not in out
        assert len([ln for ln in calls if ln.startswith("stop ")]) == 2

    def test_a_score_file_that_holds_no_number_is_missing(self, tmp_path):
        """An interrupted writer leaves a file, not a number."""
        proc, _ = drive(tmp_path, stops=(300_000,),
                        extra_env={"CF407_STUB_SCORE": "not a number"})
        assert proc.returncode == 4
        assert "300k/student" in proc.stdout + proc.stderr

    def test_a_dead_head_does_not_cost_the_other_stops(self, tmp_path):
        """One lost point, not three. The driver finishes, then reports."""
        proc, calls = drive(tmp_path, extra_env={"CF407_FAIL_HEAD": "teacher"})
        assert proc.returncode == 4
        legs = [ln for ln in calls if ln.startswith("leg ")]
        assert len(legs) == 3

    def test_a_failed_leg_stops_the_driver(self, tmp_path):
        """The 450k leg resumes the 300k checkpoint. Never skip a leg."""
        proc, calls = drive(
            tmp_path,
            leg_body='#!/bin/bash\necho "leg $2" >>"$CF407_LOG"\nexit 1\n')
        assert proc.returncode == 1
        assert len([ln for ln in calls if ln.startswith("leg ")]) == 1

    def test_it_refuses_a_stop_off_the_thousand_grid(self, tmp_path):
        proc, calls = drive(tmp_path, stops=(300_500,))
        assert proc.returncode == 2
        assert calls == []

    @needs_checkpoint
    def test_the_real_checkpoint_drives_the_first_leg(self, tmp_path):
        """One end-to-end pass over the card's own 5 MB pair, on elisa."""
        proc, calls = drive(tmp_path, linked_root(tmp_path), stops=(300_000,))
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert f"resume {RUN_NAME}_r2_200k.pth" in calls


class TestPlot:

    def test_it_writes_a_figure(self, tmp_path):
        out = tmp_path / "full_pass.png"
        (tmp_path / "score_A4_k3_bb300k_student.txt").write_text("1.0500\n")
        (tmp_path / "score_A4_k3_bb300k_teacher.txt").write_text("1.0700\n")
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        proc = subprocess.run(
            [sys.executable, str(PLOT_PY), "--results", str(tmp_path),
             "--parent", str(PARENT_RESULTS), "--out", str(out)],
            capture_output=True, text=True, env=env, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert out.is_file() and out.stat().st_size > 10_000

    def test_it_draws_the_parent_points_alone(self, tmp_path):
        """Before the first new stop lands, the figure is still #373's."""
        out = tmp_path / "empty.png"
        proc = subprocess.run(
            [sys.executable, str(PLOT_PY), "--results", str(tmp_path),
             "--parent", str(PARENT_RESULTS), "--out", str(out)],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert out.is_file()

    def test_one_point_on_the_rule_is_not_a_zero_height_axis(self, tmp_path):
        """The degenerate case: the only value is the rule itself."""
        parent = tmp_path / "parent"
        parent.mkdir()
        (parent / "score_A4_k3_bb200k_student.txt").write_text("1.0660\n")
        out = tmp_path / "one.png"
        proc = subprocess.run(
            [sys.executable, str(PLOT_PY), "--results", str(tmp_path),
             "--parent", str(parent), "--out", str(out)],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert out.is_file()

    def test_the_rule_label_and_the_legend_do_not_share_a_corner(self):
        """The rule label is anchored to the RIGHT edge, at y = 1.0660.

        So the legend belongs on the left. Round 7 moved it up: the two
        heads now dip to their lowest point at 200,000 steps, which puts
        the deepest part of the curve and the rule under it in the bottom
        left, and leaves the top left empty.
        """
        code = PLOT_PY.read_text()
        assert '(0.995, best)' in code
        assert 'loc="upper left"' in code


# --- 10. what the gates read out of #373's scripts -------------------------
#
# `check_leg_done` reads four strings and two file names, and `check_score`
# reads a third file name. Every one of them belongs to a script this study
# does not own. A rename inside #373's launcher, inside `stop_k.sh` or
# inside train.py leaves a gate with nothing to read, and the driver stops
# after the first leg. The gates fail closed, so the cost is time and not a
# wrong number — 40 GPU-hours of it. The tests here render each line from
# the source that writes it.

LEG_PATHS = PARENT / "scripts" / "leg_paths.sh"

# What train.py prints when it resumes. `check_leg_done` reads the step out
# of it. The launcher says which checkpoint it DECIDED to resume, train.py
# says which step it really started at, and the two disagree when the
# optimizer sidecar is missing.
TRAIN_RESUME_PRINT = 'print(f"Resumed from {args.resume} at step {start_step}")'

# The checkpoints the rendered lines carry. A 450k leg resumes what the
# 300k leg wrote, and a re-fired 450k leg finds its own file already there.
SAMPLE_RESUME = f"/runs/{CELL}/leg_300k/{RUN_NAME}_300k.pth"
SAMPLE_DONE = f"/runs/{CELL}/leg_450k/{RUN_NAME}_450k.pth"


def launcher_log_statement(marker: str) -> str:
    """The one `log "..."` of #373's launcher that carries `marker`.

    Greedy up to the last quote on the line, because the SKIP statement
    shares its line with the `exit` that follows it.
    """
    found = []
    for line in RUN_LEG_K.read_text().splitlines():
        match = re.search(r'log ".*"', line)
        if match and marker in match.group(0):
            found.append(match.group(0))
    assert len(found) == 1, f"want one log statement for {marker!r}: {found}"
    return found[0]


def launcher_assignment(name: str) -> str:
    """The one line of #373's launcher that sets `name`."""
    found = [ln.strip() for ln
             in strip_comments(RUN_LEG_K.read_text()).splitlines()
             if ln.strip().startswith(f"{name}=")]
    assert len(found) == 1, f"want one `{name}=` line: {found}"
    return found[0]


def launcher_log_function() -> str:
    """#373's `log()`. It names the file the outcome lines land in."""
    match = re.search(r"(?m)^log\(\)\{.*$", RUN_LEG_K.read_text())
    assert match, "no log() definition in run_leg_k.sh"
    return match.group(0)


def run_launcher(wt: Path, body: str) -> subprocess.CompletedProcess:
    """Run `body` with the variables #373's launcher holds around its logs.

    The two directory assignments, the run name, the train log, `log()` and
    the statement itself all come out of `run_leg_k.sh`, and `ckpt_step_k`
    out of `leg_paths.sh`. Nothing about the two logs is retyped here, so an
    edit to either script reaches these tests.
    """
    script = "\n".join([
        "set -uo pipefail",
        f'. "{LEG_PATHS}"',
        f'WT="{wt}"',
        f'CELL="{CELL}"',
        "K=3",
        launcher_assignment("OUT"),
        launcher_assignment("RES"),
        'mkdir -p "$RES"',
        launcher_assignment("NAME"),
        launcher_assignment("tlog"),
        f'latest="{SAMPLE_RESUME}"',
        f'done_ckpt="{SAMPLE_DONE}"',
        launcher_log_function(),
        body,
    ])
    return subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True, timeout=60)


def train_resume_line(start_step: int, ckpt: str = SAMPLE_RESUME) -> str:
    """The line train.py prints on a resume, from train.py's own text."""
    return (TRAIN_RESUME_PRINT
            .removeprefix('print(f"').removesuffix('")')
            .replace("{args.resume}", ckpt)
            .replace("{start_step}", str(start_step)))


class TestGateInputs:

    def test_the_launcher_still_writes_the_resume_line(self, fp, tmp_path):
        proc = run_launcher(tmp_path, launcher_log_statement("RESUME from"))
        assert proc.returncode == 0, proc.stderr
        match = fp.LEG_RESUME.search(Path(fp.cell_log(tmp_path)).read_text())
        assert match, proc.stdout
        assert int(match.group(1)) == 300

    def test_the_launcher_still_writes_the_fresh_line(self, fp, tmp_path):
        proc = run_launcher(tmp_path, launcher_log_statement("FRESH"))
        assert proc.returncode == 0, proc.stderr
        assert fp.LEG_FRESH in Path(fp.cell_log(tmp_path)).read_text()

    def test_the_launcher_still_writes_the_skip_line(self, fp, tmp_path):
        proc = run_launcher(tmp_path, launcher_log_statement("SKIP"))
        assert proc.returncode == 0, proc.stderr
        assert fp.LEG_SKIP in Path(fp.cell_log(tmp_path)).read_text()

    def test_the_launcher_log_is_the_file_the_gate_reads(self, fp, tmp_path):
        """The three lines above land in `cell_log`, or nothing reads them."""
        run_launcher(tmp_path, launcher_log_statement("FRESH"))
        assert Path(fp.cell_log(tmp_path)).is_file()

    def test_the_train_log_is_the_file_the_gate_reads(self, fp, tmp_path):
        proc = run_launcher(tmp_path, 'printf "%s\\n" "$tlog"')
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == fp.train_log(tmp_path)

    def test_the_launcher_sends_train_py_to_that_log(self):
        """train.py's stdout is what puts its resume line in the train log."""
        assert '>>"$tlog" 2>&1' in strip_comments(RUN_LEG_K.read_text())

    def test_train_py_still_prints_the_step_it_started_at(self, fp):
        assert TRAIN_RESUME_PRINT in TRAIN_PY.read_text()
        match = fp.TRAIN_RESUME.search(train_resume_line(300_000))
        assert match and int(match.group(1)) == 300_000

    def test_the_gate_accepts_the_lines_the_two_scripts_write(self, fp,
                                                              tmp_path):
        """End to end: #373's own lines, through this study's gate.

        This is the 450k leg, which is what every leg but the first looks
        like. The gate reads a start step of 300,000 and lets the heads run.
        """
        root = tmp_path / "runs"
        leg = root / CELL / "leg_450k"
        leg.mkdir(parents=True)
        (leg / f"{RUN_NAME}_450k.pth").write_bytes(b"x")
        (leg / f"{RUN_NAME}_450k_optimizer.pth").write_bytes(b"x")
        wt = tmp_path / "wt"
        run_launcher(wt, launcher_log_statement("RESUME from"))
        assert fp.check_leg_done(root, 450_000,
                                 Path(fp.cell_log(wt)).read_text(),
                                 train_resume_line(300_000)) == []

    def test_the_score_file_is_the_one_the_stop_script_writes(self, fp):
        """`check_score` reads what `eval_local.sh` wrote, where it wrote it."""
        code = strip_comments(STOP_K.read_text())
        assert 'RES="$WT/reports/2026-08-08_rollout_depth/results"' in code
        assert 'TAG="${CELL_ID}_k${K}_bb${STOP_K}k_${ENC}"' in code
        assert 'SCORE_OUT="$RES/score_${TAG}.txt"' in code
        assert fp.parent_results("/w") == \
            "/w/reports/2026-08-08_rollout_depth/results"
        assert fp.score_path(fp.parent_results("/w"), 300_000, "student") == \
            "/w/reports/2026-08-08_rollout_depth/results/" \
            "score_A4_k3_bb300k_student.txt"

    def test_the_stop_script_hands_that_path_to_the_eval(self):
        """`eval_local.sh` takes it as its last argument, and writes it last."""
        code = strip_comments(STOP_K.read_text())
        assert '"$OUT" "$SCORE_OUT"' in code
        assert 'mv "$SCORE_OUT.tmp" "$SCORE_OUT"' in \
            strip_comments(EVAL_LOCAL.read_text())


# --- 12. round 3 of the review --------------------------------------------

BAND_QUEUE_SH = SCRIPTS / "band_queue.sh"
READ_BACK_SH = SCRIPTS / "read_back.sh"
AWAIT_BAND_SH = SCRIPTS / "await_band.sh"
TEACHER_HEAD_INPUTS_PY = SCRIPTS / "teacher_head_inputs.py"
TEACHER_TRACK_PY = SCRIPTS / "teacher_frozen_track.py"
SHARD_ORDER_PY = SCRIPTS / "shard_order.py"
HEAD_BAND_PY = SCRIPTS / "head_band.py"
TEACHER_CHECK_SH = SCRIPTS / "teacher_check.sh"
STUDY_RESULTS = STUDY / "results"


@pytest.fixture(scope="module")
def hb():
    return load(HEAD_BAND_PY, "cf407_head_band")


@pytest.fixture(scope="module")
def so():
    return load(SHARD_ORDER_PY, "cf407_shard_order")


class TestRedrawSeparation:
    """Item 2. The re-draw must not overwrite the card's own number."""

    def test_redraw_tag_carries_the_seed(self, hb, tmp_path):
        anchor = hb.draw_path(tmp_path, 200_000, "student",
                              hb.PROTOCOL_SEED)
        again = hb.draw_path(tmp_path, 200_000, "student",
                             hb.PROTOCOL_SEED, redraw=True)
        assert anchor != again
        assert again.endswith(f"_s{hb.PROTOCOL_SEED}.txt")

    def test_local_draws_prefer_the_redraw(self, hb, tmp_path):
        for name, value in (("score_A4_k3_bb200k_student.txt", "1.0660"),
                            ("score_A4_k3_bb200k_student_s20260722.txt",
                             "1.0672"),
                            ("score_A4_k3_bb200k_student_s20260723.txt",
                             "1.0691")):
            (tmp_path / name).write_text(value + "\n")
        published = hb.draws(200_000, "student", tmp_path, None)
        local = hb.local_draws(200_000, "student", tmp_path, None)
        assert published[hb.PROTOCOL_SEED] == 1.0660
        assert local[hb.PROTOCOL_SEED] == 1.0672
        assert local[20260723] == 1.0691

    def test_redraw_table_reports_the_delta(self, hb, tmp_path):
        (tmp_path / "score_A4_k3_bb200k_student.txt").write_text("1.0660\n")
        (tmp_path / "score_A4_k3_bb200k_student_s20260722.txt").write_text(
            "1.0672\n")
        rows = hb.redraw_table([200_000], tmp_path, None)
        assert len(rows) == 1
        stop, head, anchor, again, delta = rows[0]
        assert (stop, head) == (200_000, "student")
        assert anchor == 1.0660 and again == 1.0672
        assert delta == pytest.approx(0.0012, abs=1e-9)

    def test_collect_replicates_sweeps_the_protocol_seed(self):
        code = strip_comments(
            (SCRIPTS / "collect_replicates.sh").read_text())
        assert "20260722" in code, "the re-draw would never be collected"


class TestBandComparison:
    """Item 6. This card's band, read against the published one."""

    def test_published_band_is_a_range_not_a_std(self, hb):
        pooled, cell_rows = hb.published_band()
        assert pooled == pytest.approx(0.0384, abs=5e-4)
        # The same cell's own rows are the closer comparison.
        assert cell_rows, "noise_band.py gave no row for this cell"
        assert all(r[0] == hb.THIS_CELL for r in cell_rows)

    def test_selection_gap_comes_from_its_own_file(self, hb):
        gap, ctx = hb.selection_gap(STUDY_RESULTS)
        assert gap == pytest.approx(0.0141, abs=1e-4)
        assert ctx["rank"] == 1

    def test_compare_reads_a_wide_band_as_unresolvable(self, hb):
        wide = [(200_000, "student", {20260722: 1.0660, 20260723: 1.1000})]
        text = "\n".join(hb.compare(wide, STUDY_RESULTS))
        assert "cannot resolve" in text
        narrow = [(200_000, "student", {20260722: 1.0660, 20260723: 1.0665})]
        text = "\n".join(hb.compare(narrow, STUDY_RESULTS))
        assert "narrower than" in text

    def test_largest_range_ignores_single_draw_rows(self, hb):
        table = [(200_000, "student", {20260722: 1.0660}),
                 (200_000, "teacher", {20260722: 1.08, 20260723: 1.09})]
        assert hb.largest_range(table) == pytest.approx(0.01, abs=1e-9)


class TestTeacherHeadInputs:
    """Item 3. The teacher head does not read teacher tensors only."""

    def test_the_answer_is_on_disk(self):
        path = STUDY_RESULTS / "teacher_head_inputs_100k_200k.json"
        assert path.exists(), "run teacher_head_inputs.py on 100k and 200k"
        import json
        got = json.loads(path.read_text())
        assert got["moved_from_teacher"] == 0
        assert got["moved_from_student"] > 0
        assert got["verdict"].startswith("NOT A NULL")

    def test_the_control_moves_the_teacher(self):
        path = STUDY_RESULTS / "teacher_head_inputs_40k_100k.json"
        assert path.exists()
        import json
        got = json.loads(path.read_text())
        assert got["moved_from_teacher"] == got["from_teacher"] > 0

    def test_promotion_map_matches_the_loader(self):
        """The script must read the loader's own map, never a copy."""
        code = TEACHER_HEAD_INPUTS_PY.read_text()
        assert "_TEACHER_PROMOTIONS" in code
        assert "from src.checkpoint import" in code

    def test_teacher_check_runs_the_head_input_script(self):
        code = strip_comments(TEACHER_CHECK_SH.read_text())
        assert "teacher_head_inputs.py" in code
        assert "teacher_frozen_track.py" in code


class TestTeacherTrack:
    """Item 4. The teacher points are models. Track them, do not pool."""

    def test_track_is_not_called_a_null_when_the_input_moves(self):
        mod = load(TEACHER_TRACK_PY, "cf407_teacher_track")
        moves = [{"file": "teacher_head_inputs_100k_200k.json",
                  "moved_from_teacher": 0, "moved_from_student": 32}]
        assert mod.one_encoder(moves) is True
        assert mod.head_input_constant(moves) is False

    def test_the_points_are_draws_only_when_nothing_moves(self):
        mod = load(TEACHER_TRACK_PY, "cf407_teacher_track")
        moves = [{"file": "teacher_head_inputs_100k_200k.json",
                  "moved_from_teacher": 0, "moved_from_student": 0}]
        assert mod.head_input_constant(moves) is True

    def test_frozen_stops_start_at_the_end_of_the_ramp(self):
        mod = load(TEACHER_TRACK_PY, "cf407_teacher_track")
        assert mod.FROZEN_FROM == 100_000
        assert min(mod.frozen_stops()) == 100_000
        assert 40_000 not in mod.frozen_stops()

    def test_the_stops_are_never_pooled(self):
        """No mean, no standard deviation, no range over the stops.

        The five teacher points are five models. A pooled statistic over
        them reads as a draw statistic, and a reader takes it for a noise
        band. This test is the guard on that mistake.
        """
        mod = load(TEACHER_TRACK_PY, "cf407_teacher_track")
        assert not hasattr(mod, "statistics")
        code = TEACHER_TRACK_PY.read_text()
        for banned in ("statistics.fmean", "statistics.stdev",
                       "max(values) - min(values)"):
            assert banned not in code, banned

    def test_neighbouring_stops_give_a_change_not_a_spread(self):
        mod = load(TEACHER_TRACK_PY, "cf407_teacher_track")
        got = mod.steps({100_000: 1.0874, 200_000: 1.0828, 300_000: 1.1030})
        assert [(a, b) for a, b, _ in got] == [(100_000, 200_000),
                                               (200_000, 300_000)]
        assert got[0][2] == pytest.approx(-0.0046, abs=1e-6)
        assert got[1][2] == pytest.approx(+0.0202, abs=1e-6)

    def test_the_promotion_line_is_quoted_verbatim(self):
        """The artefact must carry the reason, checked against the source."""
        mod = load(TEACHER_TRACK_PY, "cf407_teacher_track")
        path, lineno = mod.PROMOTION_SITE.split(":")
        line = (REPO_ROOT / path).read_text().splitlines()[int(lineno) - 1]
        assert line.strip() == mod.PROMOTION_LINE
        assert mod.PROMOTION_LINE == "out = dict(state_dict)"


class TestShardSample:
    """Item 9. 40 shards, and the two halves pooled."""

    def test_forty_shards_by_default(self, so):
        assert len(so.DEFAULT_SHARDS) == 40
        assert 1279 in so.DEFAULT_SHARDS and 1280 in so.DEFAULT_SHARDS
        assert max(so.DEFAULT_SHARDS) == so.N_SHARDS - 1

    def test_halves_pool_by_rows_not_by_shard(self, so):
        out = {"shards": [
            {"shard": 0, "rows": 10_000, "mix": {"a": 1.0}},
            {"shard": 2000, "rows": 100, "mix": {"b": 1.0}},
            {"shard": 3000, "rows": 9_900, "mix": {"a": 1.0}},
        ]}
        got = so.halves(out)
        assert got["n_rows_before"] == 10_000
        assert got["n_rows_after"] == 10_000
        # The 100-row shard must not carry the same weight as a full one.
        assert got["tv_between_halves"] == pytest.approx(0.01, abs=1e-9)

    def test_verdict_quotes_the_sample_size(self, so):
        path = STUDY_RESULTS / "shard_order.json"
        assert path.exists()
        import json
        got = json.loads(path.read_text())
        assert got["n_sampled"] == 40
        assert "40 of 4274 shards" in got["verdict"]
        assert got["halves"]["tv_between_halves"] < so.GROUPED_TV


class TestTheFigure:
    """The deliverable. Every mark on the axes is a measurement.

    Round 7 dropped the pooled ribbon. One number over both heads and every
    stop understated the widest measured range by 2.4 times, and the line
    ran through the protocol draw rather than the mean, so the picture and
    the tables disagreed by 0.0052 at 450,000 steps.
    """

    # The report review cut the caption to a label: the legend, the y-axis
    # label and the rule annotation already carry what the long form said.
    WORDS = ("GM-Relative MASE against backbone train step, student and "
             "teacher heads.")

    def test_the_caption_is_the_agreed_words(self):
        mod = load(PLOT_PY, "cf407_plot")
        assert mod.CAPTION == self.WORDS

    def test_the_figure_carries_no_band(self):
        """No ribbon and no shaded band. The dots are the spread."""
        code = PLOT_PY.read_text()
        assert "fill_between" not in code
        assert "Patch" not in code

    def test_the_line_joins_the_means(self, tmp_path):
        mod = load(PLOT_PY, "cf407_plot")
        results = tmp_path / "results"
        results.mkdir()
        for seed, value in ((20260722, 1.0660), (20260723, 1.0652),
                            (20260724, 1.0642)):
            (results /
             f"score_A4_k3_bb300k_student_s{seed}.txt").write_text(f"{value}\n")
        got = mod.means("student", results, tmp_path / "none")
        assert got[300_000] == pytest.approx(1.06513, abs=5e-6)

    def test_one_draw_is_its_own_mean(self, tmp_path):
        """665k carries one draw. It must not drop off the line."""
        mod = load(PLOT_PY, "cf407_plot")
        results = tmp_path / "results"
        results.mkdir()
        (results / "score_A4_k3_bb665k_student.txt").write_text("1.0783\n")
        got = mod.means("student", results, tmp_path / "none")
        assert got == {665_000: pytest.approx(1.0783)}

    def test_a_stop_with_no_draw_is_absent(self, tmp_path):
        """A head that has not scored draws a shorter line, not a guess."""
        mod = load(PLOT_PY, "cf407_plot")
        results = tmp_path / "results"
        results.mkdir()
        assert mod.means("teacher", results, tmp_path / "none") == {}

    def test_a_label_never_lands_on_the_rule(self):
        """The student runs under the teacher, so its label goes below it.

        At 450,000 steps its lowest draw sits just above 1.0660, and a
        number printed on the rule reads as the rule's own label.
        """
        mod = load(PLOT_PY, "cf407_plot")
        curves = {"student": {450_000: 1.0743}, "teacher": {450_000: 1.0952}}
        draws = {"student": {450_000: {1: 1.0691, 2: 1.0761, 3: 1.0778}},
                 "teacher": {450_000: {1: 1.0986, 2: 1.0924, 3: 1.0945}}}
        sides = mod.label_side(curves, draws, 1.0660, 0.0512)
        assert sides[("student", 450_000)] == 1
        assert sides[("teacher", 450_000)] == 1

    def test_a_label_keeps_its_side_when_the_rule_is_clear(self):
        """200k is the deepest point. Its label belongs under it."""
        mod = load(PLOT_PY, "cf407_plot")
        curves = {"student": {200_000: 1.0651}, "teacher": {200_000: 1.0800}}
        draws = {"student": {200_000: {1: 1.0660, 2: 1.0652, 3: 1.0642}},
                 "teacher": {200_000: {1: 1.0828, 2: 1.0809, 3: 1.0764}}}
        sides = mod.label_side(curves, draws, 1.0660, 0.0512)
        assert sides[("student", 200_000)] == -1

    def test_the_provenance_stays_off_the_axes(self):
        """The two checkpoint trees are recorded, not drawn."""
        mod = load(PLOT_PY, "cf407_plot")
        note = mod.provenance()
        assert "cf373_r2" in note and "cf373_r3" in note and "_r2_" in note
        assert note not in mod.CAPTION
        code = PLOT_PY.read_text()
        assert "fig.text" not in code

    def test_no_card_number_reaches_the_axes(self):
        """A cold reader cannot read an issue number."""
        mod = load(PLOT_PY, "cf407_plot")
        drawn = [mod.CAPTION, "prior best", "rollout-depth study point",
                 "one head-seed draw", "student head", "teacher head"]
        for text in drawn:
            assert "#" not in text
        code = PLOT_PY.read_text()
        for token in ('"best before #', "published by #"):
            assert token not in code


class TestBandQueue:
    """Items 2 and 7. The queue must not fight the driver or the watchdog."""

    def test_queue_runs_on_the_other_card(self):
        code = strip_comments(BAND_QUEUE_SH.read_text())
        assert 'BAND_GPU="${BAND_GPU:-1}"' in code

    def test_queue_holds_the_three_stages_card_one_owns(self):
        code = strip_comments(BAND_QUEUE_SH.read_text())
        assert '"200000|20260722|now"' in code
        assert '"300000|20260723 20260724|ckpt"' in code
        assert '"450000|20260723 20260724|ckpt"' in code
        # 665k belongs to the watchdog. Two firers would race.
        assert "665000" not in code

    def test_the_checkpoint_gate_demands_the_sidecar(self):
        """A backbone alone can be one the driver is still writing.

        `save_snapshot` writes the backbone first and the optimizer file
        second, so the sidecar is what proves the write finished.
        """
        code = strip_comments(BAND_QUEUE_SH.read_text())
        assert '_optimizer.pth' in code

    def test_queue_waits_on_the_running_band(self):
        code = strip_comments(BAND_QUEUE_SH.read_text())
        assert "replicate_alive" in code
        assert "/proc/$p/cmdline" in code, "pgrep alone matches the launcher"

    def test_queue_runs_one_band_at_a_time(self):
        """Card 1 holds one flock, so a second band would only queue."""
        code = strip_comments(BAND_QUEUE_SH.read_text())
        assert "any_replicate_alive" in code


# --- 13. the band queue, end to end ---------------------------------------
#
# Items 2 and 7 are armed rather than finished: both wait on card 1. So the
# firing path itself is the risk, and these run it. A stub stands in for
# `replicate_heads.sh` and records the arguments it was called with.

QUEUE_STUB = """#!/bin/bash
echo "STUB $*" >> "$CF407_RESULTS/fired.txt"
[ -n "${STUB_SLEEP:-}" ] && sleep "$STUB_SLEEP"
exit 0
"""

CKPT_STUB = """import os, sys
# The sandbox stands in for `full_pass.py --ckpt-at <stop> --root <root>`.
# The real script PRINTS the checkpoint it found, and the queue then tests
# the file and its optimizer sidecar. So this must print a path too.
stop = int(sys.argv[sys.argv.index("--ckpt-at") + 1])
root = os.environ.get("STUB_CKPT_DIR", "")
path = os.path.join(root, "bb_%dk.pth" % (stop // 1000))
if not root or not os.path.isfile(path):
    sys.exit(3)
print(path)
"""


@pytest.fixture
def queue(tmp_path):
    """A sandbox copy of `band_queue.sh`, with stubs for what it launches.

    `replicate_alive` resolves `argv[1]` against the launching process's own
    working directory and demands THIS copy of the script. So a sandbox run
    never sees the band running on the real machine, and a real band never
    sees the sandbox's stub.
    """
    import subprocess

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "band_queue.sh").write_text(BAND_QUEUE_SH.read_text())
    stub = scripts / "replicate_heads.sh"
    stub.write_text(QUEUE_STUB)
    stub.chmod(0o755)
    (scripts / "full_pass.py").write_text(CKPT_STUB)

    res = tmp_path / "res"
    res.mkdir()
    parent = tmp_path / "wt" / "reports" / "2026-08-08_rollout_depth" / "results"
    parent.mkdir(parents=True)

    class Queue:
        results = res
        parent_results = parent
        alive = []

        def score(self, head, seed, stop_k=200, value="1.0672"):
            (parent / f"score_A4_k3_bb{stop_k}k_{head}_s{seed}.txt").write_text(
                value + "\n")

        def land(self, stop_k, sidecar=True):
            """That stop's backbone is on disk.

            `sidecar=False` gives a backbone with no optimizer file beside
            it, which is what a checkpoint the driver is still writing
            looks like.
            """
            ck = tmp_path / "ckpt"
            ck.mkdir(exist_ok=True)
            (ck / f"bb_{stop_k}k.pth").write_text("x")
            opt = ck / f"bb_{stop_k}k_optimizer.pth"
            if sidecar:
                opt.write_text("x")
            elif opt.exists():
                opt.unlink()

        def start_band(self, stop):
            """A real process that reads as a band of this sandbox."""
            env = dict(os.environ, CF407_RESULTS=str(res), STUB_SLEEP="30")
            proc = subprocess.Popen(["bash", str(stub), str(stop)], env=env,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            self.alive.append(proc)
            # The queue reads /proc, so the process must be up before it runs.
            for _ in range(50):
                if (Path("/proc") / str(proc.pid) / "cmdline").exists():
                    break
            return proc

        def run(self, ckpt=False, once=True, period=300, max_fires=4,
                timeout=30):
            # `ckpt=True` lands every checkpoint a `ckpt` stage waits on.
            if ckpt:
                self.land(300)
                self.land(450)
            env = dict(os.environ,
                       CF407_RESULTS=str(res), WT=str(tmp_path / "wt"),
                       RUNS=str(tmp_path / "runs"),
                       STUB_CKPT_DIR=str(tmp_path / "ckpt"),
                       QUEUE_PERIOD=str(period),
                       QUEUE_MAX_FIRES=str(max_fires))
            if once:
                env["QUEUE_ONCE"] = "1"
            out = subprocess.run(
                ["bash", str(scripts / "band_queue.sh")], env=env,
                capture_output=True, text=True, timeout=timeout)
            return out.stdout + out.stderr

        def fired(self, at_least=0, timeout=10.0, settle=0.8):
            """The launches the stub recorded.

            `fire` backgrounds the launch, so the queue can exit before the
            stub has written its line. `at_least` waits for that many lines.
            `at_least = 0` means "expect none", so it settles once instead.
            """
            import time as _time
            path = res / "fired.txt"

            def read():
                if not path.exists():
                    return []
                return [ln.split(None, 1)[1]
                        for ln in path.read_text().splitlines() if ln.strip()]

            if at_least <= 0:
                _time.sleep(settle)
                return read()
            deadline = _time.time() + timeout
            while True:
                lines = read()
                if len(lines) >= at_least or _time.time() >= deadline:
                    return lines
                _time.sleep(0.05)

        def stop(self):
            for proc in self.alive:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()

    q = Queue()
    yield q
    q.stop()


class TestBandQueueFires:
    """Item 2 and item 7, run rather than read."""

    def test_stage_one_fires_the_protocol_seed(self, queue):
        queue.run()
        assert queue.fired(1) == ["200000 20260722"]

    def test_stage_one_waits_while_a_band_holds_card_one(self, queue):
        queue.start_band(200_000)
        queue.run()
        # Only the stub's own line. The queue added nothing.
        assert queue.fired(1) == ["200000"]

    def test_a_later_stage_waits_while_a_band_holds_card_one(self, queue):
        queue.start_band(200_000)
        queue.run(ckpt=True)
        fired = " ".join(queue.fired(1))
        assert "300000" not in fired and "450000" not in fired

    def test_the_300k_band_fires_when_its_checkpoint_lands(self, queue):
        for head in ("student", "teacher"):
            queue.score(head, 20260722)
        queue.run(ckpt=True)
        # 450k is ready too, and it must WAIT. One band at a time.
        assert queue.fired(1) == ["300000 20260723 20260724"]

    def test_the_450k_band_fires_when_the_300k_band_drains(self, queue):
        for head in ("student", "teacher"):
            queue.score(head, 20260722)
            for seed in (20260723, 20260724):
                queue.score(head, seed, stop_k=300)
        queue.run(ckpt=True)
        assert queue.fired(1) == ["450000 20260723 20260724"]

    def test_a_stage_waits_for_its_checkpoint(self, queue):
        for head in ("student", "teacher"):
            queue.score(head, 20260722)
        queue.run(ckpt=False)
        assert queue.fired() == []

    def test_a_checkpoint_without_its_sidecar_does_not_fire(self, queue):
        """The driver may still be writing that backbone."""
        for head in ("student", "teacher"):
            queue.score(head, 20260722)
        queue.land(300, sidecar=False)
        queue.run(ckpt=False)
        assert queue.fired() == []

    def test_a_half_scored_redraw_does_not_count_as_done(self, queue):
        queue.score("student", 20260722)          # teacher missing
        queue.run(ckpt=True)
        assert queue.fired(1) == ["200000 20260722"]

    def test_the_fire_cap_stops_a_runaway(self, queue):
        """The stub scores nothing, so every stage burns its cap and stops."""
        for head in ("student", "teacher"):
            queue.score(head, 20260722)
        log = queue.run(ckpt=True, once=False, period=1, max_fires=2)
        fired = queue.fired(4)
        assert sum("300000" in ln for ln in fired) == 2
        assert sum("450000" in ln for ln in fired) == 2
        assert "GIVING UP" in log
        assert "the queue stops" in log

    def test_a_drained_queue_ends(self, queue):
        for head in ("student", "teacher"):
            queue.score(head, 20260722)
            for seed in (20260723, 20260724):
                queue.score(head, seed, stop_k=300)
                queue.score(head, seed, stop_k=450)
        log = queue.run(ckpt=True, once=False, period=1)
        assert queue.fired() == []
        assert "the queue stops" in log


class TestReadBackSurvivesTheAgent:
    """The read-back must not depend on an agent being alive.

    Round 3 put it behind `await_redraw.sh`, a harness background task. That
    task died with its session and its read-back never ran, so the checkout
    kept stale numbers while the draws sat scored on disk.
    """

    def test_read_back_runs_every_step(self):
        code = strip_comments(READ_BACK_SH.read_text())
        for name in ("collect.sh", "collect_replicates.sh", "head_band.py",
                     "teacher_frozen_track.py", "plot_full_pass.py",
                     "mirror_durable.sh"):
            assert name in code, f"read_back.sh skips {name}"

    def test_the_driver_pairs_cross_without_an_agent(self):
        """`collect_replicates.sh` carries the `_s<seed>` draws only.

        The driver's own six pairs carry no seed in their tag, and
        `collect.sh` is the only script that copies them. Round 7 found it
        missing from the read-back, so the last stop reached the study by
        hand.
        """
        code = strip_comments(READ_BACK_SH.read_text())
        assert 'bash "$HERE/collect.sh"' in code

    def test_the_watchdog_reads_back_every_tick(self):
        code = strip_comments((SCRIPTS / "watchdog.sh").read_text())
        assert 'bash "$HERE/read_back.sh"' in code

    def test_a_band_reads_back_when_it_drains(self):
        code = strip_comments((SCRIPTS / "replicate_heads.sh").read_text())
        assert 'bash "$HERE/read_back.sh"' in code

    def test_neither_firer_needs_a_harness_task(self):
        """`await_redraw.sh` is round 3's task. Nothing may depend on it."""
        for name in ("watchdog.sh", "replicate_heads.sh", "band_queue.sh"):
            code = strip_comments((SCRIPTS / name).read_text())
            assert "await_redraw" not in code, f"{name} still calls the task"

    def test_the_mirror_still_runs_hourly(self):
        """`read_back.sh` replaced the watchdog's bare mirror call."""
        watch = strip_comments((SCRIPTS / "watchdog.sh").read_text())
        read_back = strip_comments(READ_BACK_SH.read_text())
        assert "mirror_durable.sh" not in watch
        assert "mirror_durable.sh" in read_back

    def test_read_back_reports_a_failed_step(self, tmp_path):
        """A step that fails must not read as a clean read-back."""
        import subprocess
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (tmp_path / "results").mkdir()
        (tmp_path / "plots").mkdir()
        (scripts / "read_back.sh").write_text(READ_BACK_SH.read_text())
        for name in ("collect.sh", "collect_replicates.sh",
                     "mirror_durable.sh"):
            (scripts / name).write_text("#!/bin/bash\nexit 0\n")
        for name in ("head_band.py", "teacher_frozen_track.py",
                     "plot_full_pass.py"):
            (scripts / name).write_text("import sys\nsys.exit(0)\n")
        out = subprocess.run(["bash", str(scripts / "read_back.sh")],
                             capture_output=True, text=True, timeout=60)
        assert out.returncode == 0 and "fail=0" in out.stdout
        # One broken step, and the exit code carries it.
        (scripts / "plot_full_pass.py").write_text("import sys\nsys.exit(4)\n")
        out = subprocess.run(["bash", str(scripts / "read_back.sh")],
                             capture_output=True, text=True, timeout=60)
        assert out.returncode == 1 and "fail=1" in out.stdout


class TestAwaitCarriesNoWork:
    """The wake-up must lose nothing when its session ends."""

    def test_await_band_does_no_read_back(self):
        """Round 3's task did the read-back and died with its session."""
        code = strip_comments(AWAIT_BAND_SH.read_text())
        for name in ("read_back.sh", "collect_replicates.sh", "head_band.py",
                     "teacher_frozen_track.py", "plot_full_pass.py",
                     "mirror_durable.sh"):
            assert name not in code, f"await_band.sh still runs {name}"

    def test_round_three_task_is_gone(self):
        assert not (SCRIPTS / "await_redraw.sh").exists()

    def test_await_band_exits_when_the_band_is_gone(self, tmp_path):
        """Exit 3, so a caller never waits out the clock on a dead band."""
        import subprocess
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (tmp_path / "results").mkdir()
        (scripts / "await_band.sh").write_text(AWAIT_BAND_SH.read_text())
        (scripts / "replicate_heads.sh").write_text("#!/bin/bash\nexit 0\n")
        env = dict(os.environ, CF407_RESULTS=str(tmp_path / "results"),
                   WT=str(tmp_path / "wt"), AWAIT_TIMEOUT="30",
                   AWAIT_POLL="1")
        (tmp_path / "wt" / "reports" / "2026-08-08_rollout_depth"
         / "results").mkdir(parents=True)
        out = subprocess.run(["bash", str(scripts / "await_band.sh"), "300"],
                             env=env, capture_output=True, text=True,
                             timeout=60)
        assert out.returncode == 3

    def test_await_band_exits_zero_when_every_draw_scored(self, tmp_path):
        import subprocess
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (tmp_path / "results").mkdir()
        (scripts / "await_band.sh").write_text(AWAIT_BAND_SH.read_text())
        parent = (tmp_path / "wt" / "reports" / "2026-08-08_rollout_depth"
                  / "results")
        parent.mkdir(parents=True)
        for seed in (20260723, 20260724):
            for head in ("student", "teacher"):
                (parent / f"score_A4_k3_bb300k_{head}_s{seed}.txt").write_text(
                    "1.0700\n")
        env = dict(os.environ, CF407_RESULTS=str(tmp_path / "results"),
                   WT=str(tmp_path / "wt"), AWAIT_TIMEOUT="30",
                   AWAIT_POLL="1")
        out = subprocess.run(["bash", str(scripts / "await_band.sh"), "300"],
                             env=env, capture_output=True, text=True,
                             timeout=60)
        assert out.returncode == 0 and "SCORED" in out.stdout


class TestProcessGuardIsShared:
    """One process test, in both firers. Two would drift apart."""

    def test_both_resolve_argv1_against_the_process_cwd(self):
        for path in (BAND_QUEUE_SH, SCRIPTS / "watchdog.sh"):
            code = path.read_text()
            assert "/proc/$p/cwd" in code, f"{path.name} keeps a basename test"
            assert "readlink -f" in code

    def test_both_demand_this_checkout(self):
        queue = BAND_QUEUE_SH.read_text()
        watch = (SCRIPTS / "watchdog.sh").read_text()
        assert '[ "$full" = "$SCRIPT" ]' in queue
        assert '[ "$full" = "$REPLICATE_SCRIPT" ]' in watch


# --- 14. the band at the last stop is conditional --------------------------
#
# A compute audit on 2026-08-22 disarmed the 665,000-step band. It used to
# fire on the checkpoint, whatever the stop then scored, and it cost about
# 8 GPU-hours. It now fires on the SCORE, and only inside a window around
# the number the card compares against.
#
# The rule lives in `band_decision.py`, not in a comment, so these tests can
# run it.

BAND_DECISION_PY = SCRIPTS / "band_decision.py"
WATCHDOG_SH = SCRIPTS / "watchdog.sh"
AWAIT_STOP_SH = SCRIPTS / "await_stop.sh"


@pytest.fixture(scope="module")
def bd():
    return load(BAND_DECISION_PY, "cf407_band_decision")


class TestBandDecisionRule:
    """The rule itself: two constants and one comparison."""

    def test_the_center_is_the_measured_200k_mean(self, bd):
        """1.0651 must track `head_band.csv`, not a number typed twice."""
        import csv as _csv
        path = STUDY / "results" / "head_band.csv"
        with open(path, newline="") as fh:
            rows = {(r["stop"], r["head"]): r for r in _csv.DictReader(fh)}
        row = rows[("200000", "student")]
        assert float(row["mean"]) == pytest.approx(bd.BAND_CENTER, abs=5e-5)
        assert int(row["n_draws"]) == 3

    def test_the_window_is_the_audited_one(self, bd):
        assert bd.BAND_RADIUS == 0.01
        assert bd.BAND_STOP == 665_000
        assert bd.BAND_HEAD == "student"

    def test_inside_the_window_fires(self, bd):
        for score in (1.0651, 1.0700, 1.0600, 1.0751, 1.0551):
            verdict, _ = bd.decide(score)
            assert verdict == bd.FIRE, score

    def test_outside_the_window_does_not_fire(self, bd):
        for score in (1.0450, 1.0540, 1.0800, 1.1000):
            verdict, _ = bd.decide(score)
            assert verdict == bd.SKIP, score

    def test_no_score_waits(self, bd):
        assert bd.decide(None) == (bd.WAIT, None)

    def test_the_boundary_belongs_to_fire(self, bd):
        """A point exactly on the edge buys the band. The cheap error is to
        measure a band the card did not need."""
        assert bd.decide(bd.BAND_CENTER + bd.BAND_RADIUS)[0] == bd.FIRE
        assert bd.decide(bd.BAND_CENTER - bd.BAND_RADIUS)[0] == bd.FIRE
        assert bd.decide(bd.BAND_CENTER + bd.BAND_RADIUS + 1e-4)[0] == bd.SKIP
        assert bd.decide(bd.BAND_CENTER - bd.BAND_RADIUS - 1e-4)[0] == bd.SKIP

    def test_the_pooled_sd_the_audit_quotes(self, bd):
        """0.0029, over both heads and every stop that carries a band."""
        sigma = bd.pooled_std(STUDY / "results" / "head_band.csv")
        assert sigma == pytest.approx(0.0029, abs=5e-5)
        # The window is wide against that spread, so a point outside it is
        # not a draw away from the center.
        assert bd.BAND_RADIUS / sigma > 3.0

    def test_pooled_sd_ignores_a_stop_with_one_draw(self, bd, tmp_path):
        path = tmp_path / "head_band.csv"
        path.write_text(
            "stop,head,n_draws,mean,std,range,seeds,"
            "redraw_anchor,redraw_here,redraw_delta\n"
            "200000,student,3,1.0651,0.0009,0.0018,"
            "20260722=1.0660 20260723=1.0652 20260724=1.0642,,,\n"
            "665000,student,1,1.0700,0.0000,0.0000,20260722=1.0700,,,\n")
        sigma = bd.pooled_std(path)
        assert sigma == pytest.approx(0.000902, abs=5e-6)

    def test_pooled_sd_survives_a_missing_file(self, bd, tmp_path):
        assert bd.pooled_std(tmp_path / "gone.csv") is None

    def test_the_offsets_come_off_the_banded_rows(self, bd):
        """One row per banded stop, and the offset is draw minus mean."""
        rows = bd.protocol_offsets(STUDY / "results" / "head_band.csv")
        got = {stop: round(off, 4) for stop, _, _, off in rows}
        assert got == {200_000: 0.0009, 300_000: 0.0003, 450_000: -0.0052}

    def test_the_skip_bounds_the_mean_it_did_not_measure(self, bd):
        """The widest offset each way, applied to the one draw."""
        ctx = bd.skip_context(STUDY / "results" / "head_band.csv",
                              1.0783, 665_000, "student", bd.BAND_CENTER)
        assert ctx["mean_lo"] == pytest.approx(1.0774, abs=5e-5)
        assert ctx["mean_hi"] == pytest.approx(1.0835, abs=5e-5)
        # Even the lowest bound sits far above the 200k band mean.
        assert ctx["rise_lo"] == pytest.approx(0.0123, abs=5e-5)
        assert ctx["sd_lo"] > 4.0

    def test_the_skip_bound_needs_a_banded_row(self, bd, tmp_path):
        assert bd.skip_context(tmp_path / "gone.csv", 1.0783, 665_000,
                               "student", bd.BAND_CENTER) is None
        assert "no bound" in bd.offsets_text(None)


class TestBandDecisionCli:
    """The exit codes the watchdog branches on."""

    def run(self, *args):
        return subprocess.run(
            [sys.executable, str(BAND_DECISION_PY), *args],
            capture_output=True, text=True, timeout=60)

    def test_fire_exits_zero(self):
        out = self.run("--score", "1.0700")
        assert out.returncode == 0 and out.stdout.startswith("FIRE")

    def test_skip_exits_ten(self):
        out = self.run("--score", "1.0450")
        assert out.returncode == 10 and out.stdout.startswith("SKIP")

    def test_wait_exits_twenty(self, tmp_path):
        """An empty results directory is a stop with no score."""
        out = self.run("--results", str(tmp_path))
        assert out.returncode == 20 and out.stdout.startswith("WAIT")

    def test_it_reads_the_score_off_disk(self, tmp_path):
        (tmp_path / "score_A4_k3_bb665k_student.txt").write_text("1.0662\n")
        out = self.run("--results", str(tmp_path))
        assert out.returncode == 0
        assert "1.0662" in out.stdout

    def test_write_records_the_verdict(self, tmp_path):
        rec = tmp_path / "decision.txt"
        out = self.run("--score", "1.0900", "--write", str(rec))
        assert out.returncode == 10
        assert rec.read_text().startswith("SKIP")

    def test_wait_records_nothing(self, tmp_path):
        """A latch file written before the score exists would freeze the
        decision at WAIT."""
        rec = tmp_path / "decision.txt"
        out = self.run("--results", str(tmp_path), "--write", str(rec))
        assert out.returncode == 20 and not rec.exists()

    def test_explain_prints_the_numbers(self):
        out = self.run("--explain", "--score", "1.0700")
        assert "pooled sd" in out.stdout and "window" in out.stdout

    def test_offsets_bound_the_mean_the_skip_did_not_measure(self):
        """A SKIP leaves one draw. The measured offsets bound its mean."""
        out = self.run("--offsets", "--score", "1.0783")
        assert out.returncode == 10
        assert "1.0774" in out.stdout and "1.0835" in out.stdout
        assert "+0.0123" in out.stdout

    def test_offsets_write_a_file(self, tmp_path):
        rec = tmp_path / "offsets.txt"
        out = self.run("--offsets-out", str(rec), "--score", "1.0783")
        assert out.returncode == 10
        assert "band mean lands between" in rec.read_text()


# The sandbox for the watchdog's own tick. Stubs stand in for everything the
# tick calls, and the band stub records the arguments it was called with.
WATCH_STUBS = {
    "read_back.sh": "#!/bin/bash\nexit 0\n",
    "teacher_check.sh": "#!/bin/bash\nexit 0\n",
    "run_pass.sh": "#!/bin/bash\nexit 0\n",
    "replicate_heads.sh":
        '#!/bin/bash\necho "STUB $*" >> "$CF407_RESULTS/fired.txt"\nexit 0\n',
}


def watchdog_sandbox(tmp_path, scores: dict):
    """A checkout with the watchdog, the rule, and `scores` on disk."""
    scripts = tmp_path / "study" / "scripts"
    scripts.mkdir(parents=True)
    results = tmp_path / "study" / "results"
    results.mkdir()
    for name in ("watchdog.sh", "band_decision.py", "full_pass.py"):
        (scripts / name).write_text((SCRIPTS / name).read_text())
    for name, body in WATCH_STUBS.items():
        (scripts / name).write_text(body)
    (STUDY / "results" / "head_band.csv").read_text()
    (results / "head_band.csv").write_text(
        (STUDY / "results" / "head_band.csv").read_text())

    wt = tmp_path / "wt"
    parent = wt / "reports" / "2026-08-08_rollout_depth" / "results"
    parent.mkdir(parents=True)
    (parent / f"run_{RUN_NAME}.log").write_text("[ 665000] loss 0.2\n")
    for (stop, head), value in scores.items():
        (parent / f"score_A4_k3_bb{stop // 1000}k_{head}.txt").write_text(
            f"{value}\n")
    return scripts, results, wt


def watchdog_tick(scripts, results, wt):
    env = dict(os.environ, CF407_RESULTS=str(results), WT=str(wt),
               RUNS=str(wt / "runs"), BB_GPU="1", BAND_GPU="1",
               WATCHDOG_ONCE="1", WATCHDOG_PERIOD="1")
    return subprocess.run(["bash", str(scripts / "watchdog.sh")],
                          env=env, capture_output=True, text=True, timeout=120)


class TestBandAtTheLastStopIsConditional:
    """The firing path, run rather than read."""

    def test_no_score_yet_fires_nothing(self, tmp_path):
        scripts, results, wt = watchdog_sandbox(tmp_path, {})
        watchdog_tick(scripts, results, wt)
        assert not (results / "fired.txt").exists()
        assert not (results / "band_665k_decision.txt").exists()

    def test_a_score_inside_the_window_fires_the_band(self, tmp_path):
        scripts, results, wt = watchdog_sandbox(
            tmp_path, {(665_000, "student"): "1.0660"})
        out = watchdog_tick(scripts, results, wt)
        assert (results / "fired.txt").read_text().strip() == "STUB 665000"
        assert "FIRE" in (results / "band_665k_decision.txt").read_text()
        assert "FIRE" in out.stdout

    def test_a_score_outside_the_window_fires_nothing(self, tmp_path):
        scripts, results, wt = watchdog_sandbox(
            tmp_path, {(665_000, "student"): "1.0450"})
        out = watchdog_tick(scripts, results, wt)
        assert not (results / "fired.txt").exists()
        assert "SKIP" in (results / "band_665k_decision.txt").read_text()
        assert "SKIP" in out.stdout

    def test_a_skip_decides_once(self, tmp_path):
        """The second tick must not re-read the rule or re-log the verdict."""
        scripts, results, wt = watchdog_sandbox(
            tmp_path, {(665_000, "student"): "1.0450"})
        watchdog_tick(scripts, results, wt)
        watchdog_tick(scripts, results, wt)
        rec = (results / "band_665k_decision.txt").read_text().splitlines()
        assert len(rec) == 1
        assert not (results / "fired.txt").exists()

    def test_a_fired_band_does_not_fire_twice(self, tmp_path):
        scripts, results, wt = watchdog_sandbox(
            tmp_path, {(665_000, "student"): "1.0660"})
        watchdog_tick(scripts, results, wt)
        (results / "replicate_665k.log").write_text("[rep665k] backbone ...\n")
        watchdog_tick(scripts, results, wt)
        assert (results / "fired.txt").read_text().count("STUB") == 1

    def test_the_last_tick_decides_before_the_watchdog_exits(self, tmp_path):
        """Both scores land, so `open_stops` is empty and the tick is the
        last one. The decision must still happen."""
        scripts, results, wt = watchdog_sandbox(
            tmp_path, {(300_000, "student"): "1.0867",
                       (300_000, "teacher"): "1.1030",
                       (450_000, "student"): "1.0691",
                       (450_000, "teacher"): "1.0986",
                       (665_000, "student"): "1.0660",
                       (665_000, "teacher"): "1.0800"})
        out = watchdog_tick(scripts, results, wt)
        assert "watchdog stops" in out.stdout
        assert (results / "fired.txt").read_text().strip() == "STUB 665000"


class TestTheTeacherVerdictAtTheLastStop:
    """Item 5. The same rule, read against the teacher score.

    It records a verdict. It fires no band: two more teacher head seeds
    cost about 8 GPU-hours and the card did not buy them.
    """

    REC = "band_665k_teacher_decision.txt"

    def test_the_teacher_window_is_its_own_band_mean(self, bd):
        """1.0800 plus and minus 0.0100, off `head_band.csv`."""
        center = bd.band_center(STUDY / "results" / "head_band.csv",
                                "teacher")
        assert center == pytest.approx(1.0800, abs=5e-5)
        assert center - bd.BAND_RADIUS == pytest.approx(1.0700, abs=5e-5)
        assert center + bd.BAND_RADIUS == pytest.approx(1.0900, abs=5e-5)

    def test_the_student_center_still_comes_off_the_csv(self, bd):
        got = bd.band_center(STUDY / "results" / "head_band.csv", "student")
        assert got == pytest.approx(bd.BAND_CENTER, abs=5e-5)

    def test_a_missing_row_has_no_center(self, bd, tmp_path):
        assert bd.band_center(tmp_path / "gone.csv", "teacher") is None

    def test_no_teacher_score_records_nothing(self, tmp_path):
        scripts, results, wt = watchdog_sandbox(
            tmp_path, {(665_000, "student"): "1.0450"})
        watchdog_tick(scripts, results, wt)
        assert not (results / self.REC).exists()

    def test_outside_the_window_records_a_skip(self, tmp_path):
        scripts, results, wt = watchdog_sandbox(
            tmp_path, {(665_000, "student"): "1.0450",
                       (665_000, "teacher"): "1.0952"})
        out = watchdog_tick(scripts, results, wt)
        assert "SKIP" in (results / self.REC).read_text()
        assert "teacher: SKIP" in out.stdout
        assert not (results / "fired.txt").exists()

    def test_inside_the_window_records_that_it_is_undecided(self, tmp_path):
        """A teacher score inside its window must not fire a band."""
        scripts, results, wt = watchdog_sandbox(
            tmp_path, {(665_000, "student"): "1.0450",
                       (665_000, "teacher"): "1.0800"})
        out = watchdog_tick(scripts, results, wt)
        assert "FIRE" in (results / self.REC).read_text()
        assert "cannot decide the teacher comparison" in out.stdout
        assert not (results / "fired.txt").exists()

    def test_it_decides_once(self, tmp_path):
        scripts, results, wt = watchdog_sandbox(
            tmp_path, {(665_000, "student"): "1.0450",
                       (665_000, "teacher"): "1.0952"})
        watchdog_tick(scripts, results, wt)
        watchdog_tick(scripts, results, wt)
        rec = (results / self.REC).read_text().splitlines()
        assert len(rec) == 1

    def test_it_writes_the_offsets_beside_the_verdict(self, tmp_path):
        scripts, results, wt = watchdog_sandbox(
            tmp_path, {(665_000, "student"): "1.0450",
                       (665_000, "teacher"): "1.0952"})
        watchdog_tick(scripts, results, wt)
        block = (results / "band_665k_teacher_offsets.txt").read_text()
        assert "teacher head" in block
        assert "band mean lands between" in block

    def test_the_last_tick_decides_the_teacher_too(self, tmp_path):
        """`open_stops` is empty on the tick the teacher score lands."""
        scripts, results, wt = watchdog_sandbox(
            tmp_path, {(300_000, "student"): "1.0867",
                       (300_000, "teacher"): "1.1030",
                       (450_000, "student"): "1.0691",
                       (450_000, "teacher"): "1.0986",
                       (665_000, "student"): "1.0450",
                       (665_000, "teacher"): "1.0952"})
        out = watchdog_tick(scripts, results, wt)
        assert "watchdog stops" in out.stdout
        assert "SKIP" in (results / self.REC).read_text()


class TestTheBandIsNoLongerArmed:
    """The checkpoint gate is gone. Two firers must not race either."""

    def test_the_watchdog_no_longer_gates_on_the_checkpoint(self):
        code = strip_comments(WATCHDOG_SH.read_text())
        body = code[code.index("band_at_last_stop()"):]
        body = body[:body.index("\n}")]
        assert "--ckpt-at" not in body, "the band still fires on a checkpoint"
        assert "band_decision.py" in body

    def test_the_rule_is_in_a_script_not_a_comment(self):
        assert BAND_DECISION_PY.exists()
        code = strip_comments(WATCHDOG_SH.read_text())
        # The two constants live in one place only.
        assert "1.0651" not in code
        assert "0.01" not in code

    def test_the_queue_still_owns_no_665k_stage(self):
        code = strip_comments(BAND_QUEUE_SH.read_text())
        assert "665000" not in code


class TestAwaitStopCarriesNoWork:
    """`await_stop.sh` wakes an agent. It must lose nothing when it dies."""

    def test_it_does_no_read_back(self):
        code = strip_comments(AWAIT_STOP_SH.read_text())
        for name in ("read_back.sh", "collect_replicates.sh", "head_band.py",
                     "teacher_frozen_track.py", "plot_full_pass.py",
                     "mirror_durable.sh", "replicate_heads.sh"):
            assert name not in code, f"await_stop.sh runs {name}"

    def test_it_exits_zero_when_both_heads_scored(self, tmp_path):
        scripts, parent = self.sandbox(tmp_path)
        for head in ("student", "teacher"):
            (parent / f"score_A4_k3_bb665k_{head}.txt").write_text("1.0662\n")
        out = self.run(scripts, tmp_path)
        assert out.returncode == 0 and "SCORED" in out.stdout
        assert "1.0662" in out.stdout

    def test_one_head_short_is_not_scored(self, tmp_path):
        """A stop with a student number and no teacher one is not done."""
        scripts, parent = self.sandbox(tmp_path)
        (parent / "score_A4_k3_bb665k_student.txt").write_text("1.0662\n")
        out = self.run(scripts, tmp_path)
        assert out.returncode == 3

    def test_it_gives_up_when_both_keepers_are_gone(self, tmp_path):
        """No driver and no watchdog means nothing will produce the score."""
        scripts, _ = self.sandbox(tmp_path)
        out = self.run(scripts, tmp_path)
        assert out.returncode == 3
        assert "both gone" in out.stdout or "are both gone" in out.stdout

    # --- sandbox helpers ---
    def sandbox(self, tmp_path):
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (tmp_path / "results").mkdir()
        (scripts / "await_stop.sh").write_text(AWAIT_STOP_SH.read_text())
        parent = (tmp_path / "wt" / "reports" / "2026-08-08_rollout_depth"
                  / "results")
        parent.mkdir(parents=True)
        return scripts, parent

    def run(self, scripts, tmp_path):
        env = dict(os.environ, CF407_RESULTS=str(tmp_path / "results"),
                   WT=str(tmp_path / "wt"), AWAIT_TIMEOUT="20",
                   AWAIT_POLL="1", AWAIT_HEARTBEAT="1")
        return subprocess.run(["bash", str(scripts / "await_stop.sh"), "665"],
                              env=env, capture_output=True, text=True,
                              timeout=90)
