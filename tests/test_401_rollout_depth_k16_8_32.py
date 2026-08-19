"""Tests for #401: rollout depth k = 8 and 32, and heads as long as the backbone.

#401 runs ONE configuration — #373's cell A4, `arm6_v2 combab` with `L_align`
on the student and the scheduled EMA — at two rollout depths, under the MEAN
over the k + 1 depth copies. It adds ONE trainer flag,
`--train-rollout-reduce`, and no new pipeline. It reuses #373's runner, head
trainer and GIFT-Eval, and supplies the depth, the reduction, the stops and
the head budgets.

The first run of this card summed the copies. Its 8 scored cells stay in
`results/` as the comparison arm, and this protocol writes nowhere near them:
the reduction picks the run name, the checkpoint root, the results directory
and the plots directory. `tests/test_401_rollout_reduce_mean.py` holds the
objective itself.

That is the contract these tests hold:

  * the study's constants are the card's: depths 8 and 32 IN THAT ORDER,
    stops 40k / 100k / 200k, phase-1 head 30,000 steps, student encoder,
    and the mean over the depth copies.
  * the two arms of this card never share a file.
  * four GPUs across two machines: the box trains backbones, elisa trains
    every head and runs every 97-config GIFT-Eval.
  * the configuration is #373's A4 cell, flag for flag.
  * the runner is #373's `run_leg_k.sh` — no second trainer invocation
    exists in this study.
  * the head and the eval are #373's `head_eval_bb.sh` — same protocol,
    same 97 configs.
  * phase 2 matches the head budget to the backbone stop.
  * every artefact lands in THIS study's directory and on a durable root,
    and no run name can collide with a published #373 one.

The two path overrides the reuse needs (`CF_STUDY_DIR` on #373's two
scripts) are checked to default to #373's own directory, so #373's numbers
stay reproducible from its own commands.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import re
import shutil
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
SMOKE = EXP / "scripts" / "smoke_depth.sh"
PICK_PY = EXP / "scripts" / "pick_phase2_arms.py"
RUN_SH = EXP / "run.sh"
LAUNCH_SYNC = EXP / "sync" / "launch_sync.sh"
LAUNCH_BOX = EXP / "scripts" / "launch_box.sh"
LAUNCH_ELISA = EXP / "scripts" / "launch_elisa.sh"
HEADS_WATCH = EXP / "scripts" / "heads_watch.sh"
PROVISION_BOX = EXP / "scripts" / "provision_box.sh"
BOOTSTRAP_BOX = EXP / "scripts" / "bootstrap_box.sh"

PARENT_LEG = PARENT / "scripts" / "run_leg_k.sh"
PARENT_HEAD = PARENT / "scripts" / "head_eval_bb.sh"

PEAK_PROBE = REPO_ROOT / "scripts" / "gpu_peak_probe.py"

# The arms this protocol runs, in the order it runs them.
DEPTHS = (8, 32)
# Every depth a figure of this card draws. The summed comparison arm ran
# k = 16, so the plots and the palette still carry it.
DEPTHS_DRAWN = (8, 16, 32)
# The reduction over the k + 1 depth copies. `mean` is this protocol.
REDUCE = "mean"
# What a test that exercises a mechanic, and not the objective, sets so its
# paths are the ones it passed in.
SUM = {"CF401_REDUCE": "sum"}
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


def study_value(name: str, env=None) -> str:
    """One exported value of study.sh, read by sourcing it."""
    full = dict(os.environ)
    full.update(env or {})
    out = subprocess.run(
        ["bash", "-c", f'. "{STUDY_SH}" >/dev/null && printf "%s" "${{{name}}}"'],
        capture_output=True, text=True, timeout=60, env=full,
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
                   SMOKE, PICK_PY, RUN_SH, LAUNCH_SYNC, LAUNCH_BOX,
                   LAUNCH_ELISA, HEADS_WATCH, PROVISION_BOX, BOOTSTRAP_BOX])
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
        for stage in ("smoke_depth.sh", "phase1.sh", "phase2.sh"):
            assert stage in code, f"run.sh does not run {stage}"

    def test_the_sync_loop_is_373s(self):
        """One set of measured per-class size floors, not a second copy."""
        code = strip_comments(LAUNCH_SYNC.read_text())
        assert "sync_loop.sh" in code
        assert "safe_pull.sh" in code, "raw scp corrupts the prior good copy"

    def test_a_file_with_a_shebang_is_executable(self):
        """One rule for the whole study: a `#!` line means mode 755.

        Half the scripts were 755 and the newer half 644, so `./scripts/x.sh`
        worked for some and not for others.
        """
        listed = subprocess.run(
            ["git", "ls-files", "-s", str(EXP.relative_to(REPO_ROOT))],
            capture_output=True, text=True, cwd=str(REPO_ROOT)).stdout
        wrong = []
        for line in listed.splitlines():
            mode = line.split()[0]
            path = line.split("\t", 1)[1]
            if (REPO_ROOT / path).read_bytes()[:2] == b"#!" and mode != "100755":
                wrong.append(f"{path} is {mode}")
        assert not wrong, wrong


# --- 2. The study's constants ------------------------------------------------


class TestStudyConstants:

    def test_depths_are_8_and_32(self):
        """Two arms, not three. k = 8 and k = 32 bracket the range, and
        dropping k = 16 buys the answer sooner."""
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

    def test_no_constant_holds_the_phase_2_head_budgets(self):
        """The phase-2 budget is the stop, and it is read from the stop.

        `CF401_HEAD_STEPS_P2` held the three stops outside a trial, and one
        log line was the only reader. A second place to keep the rule can
        drift from phase2.sh and from cf401_require_head_steps, which both
        take the budget from the stop they are given.
        """
        assert study_value("CF401_HEAD_STEPS_P2") == ""
        for path in (STUDY_SH, PHASE2, EXP / "scripts" / "trial_head.sh"):
            assert "CF401_HEAD_STEPS_P2" not in path.read_text(), path.name


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
        """The 97-config count is asserted by #373's own eval, not re-derived.

        The count is now a defaulted variable, so a caller can run a subset —
        #401's trial runs one config. The DEFAULT is the protocol, and the
        merge is still counted against it.
        """
        eval_local = strip_comments(
            (PARENT / "scripts" / "eval_local.sh").read_text())
        assert 'EVAL_EXPECT_CONFIGS:-97' in eval_local
        assert '-ne "$EVAL_EXPECT_CONFIGS"' in eval_local

    def test_a_config_subset_is_not_split_across_shards(self):
        """Each shard would run the same regex and the merge would double."""
        eval_local = strip_comments(
            (PARENT / "scripts" / "eval_local.sh").read_text())
        assert re.search(r'if \[ -n "\$EVAL_CONFIG_FILTER" \]; then\s*\n\s*'
                         r'EVAL_SHARDS=1\s*\nfi', eval_local), (
            "a filtered eval still shards")

    def test_the_eval_does_not_run_under_set_e(self):
        """The subset lines are tests, and a test on an unset value returns 1.

        `set -e` would stop every 97-config eval of both studies at the first
        of them. The trial ran with the filter SET, so only the other branch
        is exercised by a run.
        """
        for line in (PARENT / "scripts" / "eval_local.sh").read_text().splitlines():
            if line.strip().startswith("set "):
                flags = line.split()[1].lstrip("-")
                assert "e" not in flags, line
                assert "errexit" not in line, line

    def test_no_subset_line_decides_the_scripts_exit_status(self):
        """A bare `[ -n "$F" ] && ...` returns 1 when F is unset.

        The `if` form returns 0 on both branches, so the guard holds whatever
        shell options a later edit of this script turns on.
        """
        code = strip_comments(
            (PARENT / "scripts" / "eval_local.sh").read_text())
        bare = re.findall(r'^\s*\[[^]]*\$EVAL_CONFIG_FILTER[^]]*\]\s*&&.*$',
                          code, re.M)
        assert not bare, bare

    def test_the_filter_reaches_the_aggregate_pass(self):
        """Without it a one-config run evaluates the other 96 at the end."""
        eval_local = strip_comments(
            (PARENT / "scripts" / "eval_local.sh").read_text())
        assert "AGG_FILTER" in eval_local
        assert '"${AGG_FILTER[@]}"' in eval_local

    @pytest.mark.parametrize("k", DEPTHS)
    @pytest.mark.parametrize("stop", STOPS)
    def test_phase1_head_is_30000_steps_on_the_student(self, k, stop):
        out = run_sh(HEAD_EVAL, k, stop, env={"CF401_DRY_RUN": "1"})
        assert out.returncode == 0, out.stderr
        assert str(HEAD_STEPS_PHASE1) in out.stdout
        assert "student" in out.stdout

    @pytest.mark.parametrize("stop", STOPS)
    def test_phase2_head_steps_match_the_backbone_steps(self, stop):
        out = run_sh(HEAD_EVAL, 8, stop, stop, env={"CF401_DRY_RUN": "1"})
        assert out.returncode == 0, out.stderr
        assert f"HEAD_STEPS={stop}" in out.stdout

    def test_tags_of_the_two_phases_differ(self):
        """A shared tag would let phase 2 read phase 1's score file."""
        p1 = run_sh(HEAD_EVAL, 8, 40_000, env={"CF401_DRY_RUN": "1"}).stdout
        p2 = run_sh(HEAD_EVAL, 8, 40_000, 40_000,
                    env={"CF401_DRY_RUN": "1"}).stdout
        tag1 = re.search(r"TAG=(\S+)", p1).group(1)
        tag2 = re.search(r"TAG=(\S+)", p2).group(1)
        assert tag1 != tag2

    def test_it_refuses_a_head_budget_that_is_not_the_cards(self):
        out = run_sh(HEAD_EVAL, 8, 40_000, 12_345,
                     env={"CF401_DRY_RUN": "1"})
        assert out.returncode != 0

    @pytest.mark.parametrize("stop,head", [(100_000, 40_000),
                                           (200_000, 100_000),
                                           (40_000, 200_000)])
    def test_it_refuses_another_stops_phase2_budget(self, stop, head):
        """The card's phase-2 rule is head steps = THIS backbone's steps.

        A list test over all four budgets accepted `head_eval.sh 8 100000
        40000`, which writes a tag neither phase defines. collect.sh then
        reads it as phase 1.
        """
        out = run_sh(HEAD_EVAL, 8, stop, head, env={"CF401_DRY_RUN": "1"})
        assert out.returncode != 0, out.stdout
        assert "phase-2" in out.stderr

    @pytest.mark.parametrize("k", DEPTHS)
    def test_the_head_takes_its_own_arms_root(self, k):
        """Rule 3 again: one root per depth, for the heads as for the arms."""
        out = run_sh(HEAD_EVAL, k, 40_000, env={"CF401_DRY_RUN": "1"})
        assert out.returncode == 0, out.stderr
        root = re.search(r"root=(\S+)", out.stdout).group(1)
        assert root.endswith(f"/k{k}"), root

    def test_no_two_depths_share_an_eval_directory(self):
        dirs = set()
        for k in DEPTHS:
            for stop in STOPS:
                out = run_sh(HEAD_EVAL, k, stop, env={"CF401_DRY_RUN": "1"})
                dirs.add(re.search(r"eval=(\S+)", out.stdout).group(1))
        assert len(dirs) == len(DEPTHS) * len(STOPS)


# --- 6. The two phases --------------------------------------------------------


class TestPhase1Plan:

    @pytest.fixture(scope="class")
    def plan(self) -> list[str]:
        out = run_sh(PHASE1, env={"CF401_DRY_RUN": "1"})
        assert out.returncode == 0, out.stderr
        return [ln for ln in out.stdout.splitlines() if ln.startswith(("arm ", "head "))]

    def test_six_legs_and_six_heads(self, plan):
        assert sum(1 for ln in plan if ln.startswith("arm ")) == len(DEPTHS) * len(STOPS)
        assert sum(1 for ln in plan if ln.startswith("head ")) == len(DEPTHS) * len(STOPS)

    def test_arms_run_in_the_studys_order(self, plan):
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


def stub_script(path: Path, body: str):
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(0o755)


@pytest.fixture
def stub_study(tmp_path):
    """A copy of the study's scripts, where the costly two can be replaced.

    phase1.sh runs `$HERE/run_arm_k.sh` and `$HERE/head_eval.sh` out of its
    own directory, so a copy is the only way to make a leg fail without a
    GPU. Every other line, study.sh included, is the study's own code. The
    empty sibling directory is #373's, which study.sh resolves relative to
    itself.
    """
    study = tmp_path / "reports" / EXP.name
    shutil.copytree(EXP / "scripts", study / "scripts")
    (tmp_path / "reports" / PARENT.name / "scripts").mkdir(parents=True)
    return study


class TestPhase1LegFailure:
    """A failed leg is a failed phase.

    The next stop resumes the one below it, so a dead leg makes every stop
    above it meaningless. Phase 1 exited 0 anyway, and the failure then
    surfaced hours later as the picker's "incomplete phase 1" abort.
    """

    def run_phase1(self, study, tmp_path, leg_body):
        stub_script(study / "scripts" / "run_arm_k.sh", leg_body)
        stub_script(study / "scripts" / "head_eval.sh", "exit 0")
        out = run_sh(study / "scripts" / "phase1.sh",
                     env={**SUM,
                          "CF401_RESULTS": str(tmp_path / "res"),
                          "CF401_ROOT": str(tmp_path / "runs"),
                          "HEAD_BG": "0"})
        return out, (tmp_path / "res" / "phase1.log").read_text()

    def test_a_dead_leg_makes_the_phase_fail(self, stub_study, tmp_path):
        out, log = self.run_phase1(stub_study, tmp_path, "exit 9")
        assert out.returncode != 0, "phase 1 reported success after a dead leg"
        assert "rc=9" in log, log

    def test_the_drained_line_counts_the_dead_legs(self, stub_study, tmp_path):
        """The count is what a reader of the log looks at."""
        _, log = self.run_phase1(stub_study, tmp_path, "exit 9")
        drained = [ln for ln in log.splitlines() if "drained" in ln]
        assert drained, log
        assert re.search(r"2 leg\(s\)", drained[-1]), drained[-1]

    def test_a_dead_leg_stops_its_own_arm_only(self, stub_study, tmp_path):
        """k = 32 still runs. The card wants every arm it can get."""
        out, log = self.run_phase1(
            stub_study, tmp_path, '[ "$1" = 8 ] && exit 9\nexit 0')
        assert out.returncode != 0, log
        started = re.findall(r"arm k=(\d+) -> (\d+)", log)
        assert started.count(("8", "40000")) == 1
        assert ("8", "100000") not in started, "it resumed a dead leg"
        assert len([s for s in started if s[0] == "32"]) == len(STOPS)
        assert re.search(r"1 leg\(s\)", log), log

    def test_every_leg_alive_still_exits_zero(self, stub_study, tmp_path):
        out, log = self.run_phase1(stub_study, tmp_path, "exit 0")
        assert out.returncode == 0, out.stdout + out.stderr
        assert re.search(r"0 leg\(s\)", log), log


class TestPhase2Plan:

    def test_it_runs_matched_heads_on_two_arms(self, tmp_path):
        scores = tmp_path / "scores.csv"
        write_scores(scores, {8: 1.09, 32: 1.20})
        out = run_sh(PHASE2, env={**SUM, "CF401_DRY_RUN": "1",
                                  "CF401_SCORES": str(scores)})
        assert out.returncode == 0, out.stderr
        heads = [ln for ln in out.stdout.splitlines() if ln.startswith("head ")]
        assert len(heads) == 2 * len(STOPS)
        for ln in heads:
            k = int(re.search(r"k=(\d+)", ln).group(1))
            assert k in DEPTHS
            stop = int(re.search(r"stop=(\d+)", ln).group(1))
            steps = int(re.search(r"steps=(\d+)", ln).group(1))
            assert steps == stop

    def test_it_refuses_an_incomplete_phase_1(self, tmp_path):
        """The card: wait until every stop at every k is scored."""
        scores = tmp_path / "scores.csv"
        write_scores(scores, {8: 1.09, 32: 1.20}, drop=(32, 200_000))
        out = run_sh(PHASE2, env={**SUM, "CF401_DRY_RUN": "1",
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

    def test_it_takes_every_arm_this_protocol_runs(self, pick):
        """Two arms and a pair to pick: phase 2 repeats both."""
        got = pick.pick_arms(self.rows({8: 1.09, 32: 1.20}))
        assert got == [8, 32]

    def test_it_ranks_by_the_best_score(self, pick):
        """The rank still decides, and `--count 1` is where it shows."""
        got = pick.pick_arms(self.rows({8: 1.20, 32: 1.05}), n=1)
        assert got == [32]

    def test_an_arm_is_scored_by_its_best_stop(self, pick):
        """A late stop that improves must be able to carry its arm."""
        rows = [{"k": 8, "stop": 40_000, "score": 1.30},
                {"k": 8, "stop": 100_000, "score": 1.30},
                {"k": 8, "stop": 200_000, "score": 1.01},
                {"k": 32, "stop": 40_000, "score": 1.25},
                {"k": 32, "stop": 100_000, "score": 1.25},
                {"k": 32, "stop": 200_000, "score": 1.25}]
        assert pick.pick_arms(rows, n=1) == [8]

    def test_the_result_is_in_the_studys_run_order(self, pick):
        got = pick.pick_arms(self.rows({8: 1.20, 32: 1.05}))
        assert got == [8, 32], "run order is 8 then 32"

    def test_a_tie_is_broken_by_the_run_order(self, pick):
        got = pick.pick_arms(self.rows({8: 1.10, 32: 1.10}), n=1)
        assert got == [8]

    def test_it_refuses_a_missing_stop(self, pick):
        with pytest.raises(ValueError):
            pick.pick_arms(self.rows({8: 1.09, 32: 1.20},
                                     drop=(32, 200_000)))

    def test_it_refuses_a_missing_arm(self, pick):
        with pytest.raises(ValueError):
            pick.pick_arms(self.rows({8: 1.09}))


# --- 8. Collecting the scores -------------------------------------------------


class TestCollect:

    def test_it_reads_every_score_file_into_one_csv(self, tmp_path):
        res = tmp_path / "results"
        res.mkdir()
        (res / "score_k16_bb40k_h30k_student.txt").write_text("1.0731\n")
        (res / "score_k8_bb200k_h200k_student.txt").write_text("1.0512\n")
        (res / "stops.log").write_text("not a score\n")
        out = run_sh(COLLECT, env={**SUM, "CF401_RESULTS": str(res)})
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
        out = run_sh(COLLECT, env={**SUM, "CF401_RESULTS": str(res)})
        assert out.returncode == 0, out.stderr
        rows = list(csv.DictReader(open(res / "scores.csv")))
        assert rows == []

    def test_it_reads_a_sub_1000_step_tag(self, tmp_path):
        """A trial budget is not a multiple of 1000, so its tag carries steps."""
        res = tmp_path / "results"
        res.mkdir()
        (res / "score_k16_bb400_h200_student.txt").write_text("1.4000\n")
        (res / "score_k16_bb400_h400_student.txt").write_text("1.3900\n")
        out = run_sh(COLLECT, env={**SUM, "CF401_RESULTS": str(res)})
        assert out.returncode == 0, out.stderr
        rows = list(csv.DictReader(open(res / "scores.csv")))
        assert {int(r["head_steps"]) for r in rows} == {200, 400}
        by_head = {int(r["head_steps"]): r for r in rows}
        assert by_head[200]["phase"] == "1"
        assert by_head[400]["phase"] == "2"

    def test_a_variant_cell_lands_beside_its_base_cell(self, tmp_path):
        """`k32_ema30k_bb40k_h30k_student` is the card's cell at k = 32 with
        the EMA ramp shortened. It holds the same depth, stop and head budget
        as the base cell, so only the `variant` column keeps the two apart.

        Before that column the collector refused the tag and printed one WARN
        line. The score was then in no row of either table, and both
        deliverable figures read those tables.
        """
        res = tmp_path / "results"
        res.mkdir()
        (res / "score_k32_bb40k_h30k_student.txt").write_text("1.2082\n")
        (res / "score_k32_ema30k_bb40k_h30k_student.txt").write_text("1.2385\n")
        out = run_sh(COLLECT, env={**SUM, "CF401_RESULTS": str(res)})
        assert out.returncode == 0, out.stderr
        assert "unparsed" not in out.stderr, out.stderr
        rows = list(csv.DictReader(open(res / "scores.csv")))
        assert len(rows) == 2, rows
        by_variant = {r["variant"]: r for r in rows}
        assert set(by_variant) == {"base", "ema30k"}
        assert by_variant["base"]["score"] == "1.2082"
        assert by_variant["ema30k"]["score"] == "1.2385"
        # Same three numbers on both rows: the variant is a schedule, not a
        # fourth stop or a second head budget.
        for row in rows:
            assert (row["k"], row["stop"], row["head_steps"]) == \
                ("32", "40000", "30000"), row

    def test_a_base_cell_reads_the_base_variant(self, tmp_path):
        """The optional part must not shift the fields that follow it."""
        res = tmp_path / "results"
        res.mkdir()
        (res / "score_k8_bb100k_h30k_student.txt").write_text("1.2857\n")
        out = run_sh(COLLECT, env={**SUM, "CF401_RESULTS": str(res)})
        assert out.returncode == 0, out.stderr
        row, = list(csv.DictReader(open(res / "scores.csv")))
        assert row == {"phase": "1", "k": "8", "variant": "base",
                       "stop": "100000", "head_steps": "30000",
                       "encoder": "student", "score": "1.2857"}


# --- 8b. The per-domain table, which deliverable 1 draws from ----------------


SN_REF = (REPO_ROOT / "reports" / "2026-07-21_split_pred_rep_small" / "results"
          / "seasonal_naive_all_results.csv")
MASE_COL = "eval_metrics/MASE[0.5]"


def write_eval_csv(path: Path, factor: float, n: int = 12):
    """A GIFT-Eval `all_results.csv`, `factor` times the seasonal-naive MASE.

    Built from the committed seasonal-naive reference, so the dataset names
    and the domains are the eval's own and the join in `split_scores.py` is
    the real one.
    """
    with open(SN_REF) as fh:
        ref = [r for r in csv.DictReader(fh)][:n]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", MASE_COL, "domain"])
        w.writeheader()
        for r in ref:
            w.writerow({"dataset": r["dataset"],
                        MASE_COL: float(r[MASE_COL]) * factor,
                        "domain": r["domain"]})
    return {r["domain"] for r in ref}


class TestCollectSplits:
    """Deliverable 1 needs per-domain numbers. The eval publishes none."""

    def build(self, tmp_path, tags):
        """A results dir and a runs root holding one eval per tag."""
        res = tmp_path / "results"
        res.mkdir()
        root = tmp_path / "runs"
        domains = set()
        for k, tag, factor in tags:
            (res / f"score_{tag}.txt").write_text("1.0700\n")
            domains |= write_eval_csv(
                root / f"k{k}" / "eval" / tag / "gift" / "all_results.csv",
                factor)
        return res, root, domains

    def test_it_writes_a_per_domain_row_for_every_eval(self, tmp_path):
        res, root, domains = self.build(tmp_path, [
            (16, "k16_bb40k_h30k_student", 1.05),
            (8, "k8_bb40k_h30k_student", 1.10)])
        out = run_sh(COLLECT, env={**SUM, "CF401_RESULTS": str(res),
                                   "CF401_ROOT": str(root)})
        assert out.returncode == 0, out.stderr + out.stdout
        rows = list(csv.DictReader(open(res / "splits.csv")))
        assert rows, "collect.sh wrote no per-domain table"
        by_tag = {}
        for r in rows:
            if r["split"] == "domain":
                by_tag.setdefault(r["stop"], set()).add(r["name"])
        assert set(by_tag) == {"k16_bb40k_h30k_student",
                               "k8_bb40k_h30k_student"}
        for tag, got in by_tag.items():
            assert got == domains, tag

    def test_the_per_domain_values_are_the_eval_ratio(self, tmp_path):
        """1.05 x seasonal naive on every config is 1.05 on every family."""
        res, root, _ = self.build(
            tmp_path, [(16, "k16_bb40k_h30k_student", 1.05)])
        out = run_sh(COLLECT, env={**SUM, "CF401_RESULTS": str(res),
                                   "CF401_ROOT": str(root)})
        assert out.returncode == 0, out.stderr
        for r in csv.DictReader(open(res / "splits.csv")):
            assert abs(float(r["gm_rel_mase"]) - 1.05) < 1e-4, r

    def test_a_missing_eval_csv_does_not_fail_the_collect(self, tmp_path):
        """A sync loop can still be pulling it. scores.csv is what blocks."""
        res = tmp_path / "results"
        res.mkdir()
        (res / "score_k16_bb40k_h30k_student.txt").write_text("1.0700\n")
        out = run_sh(COLLECT, env={**SUM, "CF401_RESULTS": str(res),
                                   "CF401_ROOT": str(tmp_path / "nothing")})
        assert out.returncode == 0, out.stderr
        assert list(csv.DictReader(open(res / "scores.csv")))

    def test_a_pinned_root_is_the_only_tree_read(self, tmp_path):
        """`CF401_ROOT=<tree>` means read that tree and no other.

        The collector searches a few roots for a tag's eval, because the head
        that produced it may have run under the sync tree or under the study
        default. That search must stop at a root the caller pinned. Without
        the guard, a run pointed at an empty directory found the real study's
        evals through the sync root and reported another run's per-domain
        numbers as its own.
        """
        res = tmp_path / "results"
        res.mkdir()
        (res / "score_k8_bb40k_h30k_student.txt").write_text("1.0700\n")
        out = run_sh(COLLECT, env={**SUM, "CF401_RESULTS": str(res),
                                   "CF401_ROOT": str(tmp_path / "nothing"),
                                   "CF401_SYNC_ROOT": str(tmp_path / "sync")})
        assert out.returncode == 0, out.stderr
        assert list(csv.DictReader(open(res / "scores.csv")))
        # No table at all, or an empty one. Never another tree's rows.
        splits = res / "splits.csv"
        rows = list(csv.DictReader(open(splits))) if splits.is_file() else []
        assert rows == [], rows

    def test_an_unpinned_root_also_looks_in_the_sync_tree(self, tmp_path):
        """The heads ran on elisa, against the tree the sync loop lands in.

        A `collect.sh` run by hand holds the study default root, which is not
        that tree. Before this, such a run wrote a per-domain table with the
        variant cell alone and dropped all eight grid cells.
        """
        res = tmp_path / "results"
        res.mkdir()
        sync = tmp_path / "sync"
        (res / "score_k8_bb40k_h30k_student.txt").write_text("1.0700\n")
        write_eval_csv(
            sync / "k8" / "eval" / "k8_bb40k_h30k_student" / "gift"
            / "all_results.csv", 1.05)
        out = run_sh(COLLECT, env={**SUM, "CF401_RESULTS": str(res),
                                   "CF401_SYNC_ROOT": str(sync)})
        assert out.returncode == 0, out.stderr + out.stdout
        rows = list(csv.DictReader(open(res / "splits.csv")))
        assert {r["stop"] for r in rows} == {"k8_bb40k_h30k_student"}, rows

    def test_a_variant_eval_is_found_under_its_own_root(self, tmp_path):
        """A variant cell trains under `<root>-<variant>`, and #373's runner
        lays its eval one level deeper than this study's own layout."""
        res = tmp_path / "results"
        res.mkdir()
        root = tmp_path / "runs"
        (res / "score_k32_ema30k_bb40k_h30k_student.txt").write_text("1.2385\n")
        write_eval_csv(
            tmp_path / "runs-ema30k" / "k32" / CELL / "eval"
            / "k32_ema30k_bb40k_h30k_student" / "gift" / "all_results.csv",
            1.05)
        out = run_sh(COLLECT, env={**SUM, "CF401_RESULTS": str(res),
                                   "CF401_ROOT": str(root)})
        assert out.returncode == 0, out.stderr + out.stdout
        rows = list(csv.DictReader(open(res / "splits.csv")))
        assert {r["stop"] for r in rows} == \
            {"k32_ema30k_bb40k_h30k_student"}, rows


# --- 9. The k = 16 smoke test -------------------------------------------------


class TestSmokeDepth:
    """Proves the flag runs at this depth, and says what it costs."""

    @pytest.fixture(scope="class")
    def smoke_code(self) -> str:
        return strip_comments(SMOKE.read_text())

    def test_it_measures_step_time_and_peak_memory(self, smoke_code):
        assert "gpu_peak_probe.py" in smoke_code
        assert "timing:" in SMOKE.read_text()

    def test_it_writes_every_number_to_results(self, smoke_code):
        header = re.search(r'"(cell,reduce,k,[^"]*)"', smoke_code)
        assert header, "no CSV header in the smoke script"
        cols = header.group(1).split(",")
        for col in ("reduce", "k", "total_ms", "peak_mib", "depth_cols", "rc"):
            assert col in cols, f"{col} missing from {cols}"

    def test_it_measures_the_objective_the_study_trains(self, smoke_code):
        """The mean adds one pass over the f-bearing terms, so its step time
        is its own number."""
        assert "--train-rollout-reduce $CF401_REDUCE" in smoke_code

    def test_it_runs_the_studys_own_runner(self, smoke_code):
        assert "run_arm_k.sh" in smoke_code or "run_leg_k.sh" in smoke_code

    def test_its_scratch_root_is_durable_and_not_the_studys(self, smoke_code):
        m = re.search(r'SCRATCH="\$\{[A-Z0-9_]+:-([^}]*)\}"', smoke_code)
        assert m, "no defaulted SCRATCH in the smoke script"
        assert not m.group(1).startswith("/tmp")

    def test_the_default_depths_are_the_arms_and_a_k0_reference(
            self, smoke_code):
        m = re.search(r'DEPTHS="\$\{DEPTHS:-([^}]*)\}"', smoke_code)
        assert m, "no defaulted DEPTHS"
        assert m.group(1).split() == ["0"] + [str(k) for k in DEPTHS]

    @pytest.fixture(scope="class")
    def measured(self) -> dict[int, dict]:
        """The summed arm's committed table, on four depths.

        It is what the run plan sizes this card against: 169.7 ms at k = 0,
        257.1 ms at k = 8 and 530.5 ms at k = 32 on one RTX 4090, at a peak
        that does not grow with the depth. The mean re-measures into
        `results/mean/smoke_depth.csv`, and the schema there carries the
        reduction as its own column.
        """
        csv_path = EXP / "results" / "smoke_k16.csv"
        assert csv_path.is_file(), "no results/smoke_k16.csv"
        rows = list(csv.DictReader(open(csv_path)))
        assert rows, "smoke_k16.csv holds no measurement"
        return {int(r["k"]): r for r in rows}

    def test_the_measurement_is_recorded(self, measured):
        """The run plan depends on it, so the number is committed."""
        for k in DEPTHS:
            assert k in measured, f"k = {k} was not measured"
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

    def test_all_four_depths_are_in_the_committed_table(self, measured):
        """The run plan reads the ratio, so it needs the k = 0 reference."""
        assert sorted(measured) == [0, 8, 16, 32], sorted(measured)

    def test_the_two_arms_write_two_tables(self):
        """The mean must not append its rows to the summed arm's table: the
        two objectives have different step times at the same depth."""
        out = study_call('printf "%s" "$CF401_RESULTS"')
        assert out.stdout.strip().endswith("/results/mean"), out.stdout
        out = study_call('printf "%s" "$CF401_RESULTS"', env=SUM)
        assert out.stdout.strip().endswith("/results"), out.stdout


class TestSmokeTableIsKept:
    """The table a run of this stage has already measured is kept.

    `bash run.sh` runs the smoke stage again. A truncating write would
    replace every measured row with the rows of the newest list.
    """

    HEADER = ("cell,reduce,k,steps,windows,data_ms,fwd_ms,bwd_ms,total_ms,"
              "sps,peak_mib,depth_cols,card_free_mib,rc")

    def table(self, results, depths=(0, 8), reduce="sum"):
        """A measured table in this stage's own schema.

        The reduction is the run's own: it is half the key of a row, because
        one depth costs two different numbers under the two objectives.
        """
        results.mkdir(parents=True, exist_ok=True)
        path = results / "smoke_depth.csv"
        rows = [self.HEADER]
        for k in depths:
            rows.append(f"{CELL},{reduce},{k},300,2,1,2,3,{100 + k},5,5400,"
                        f"{k + 1 if k else 0},19000,0")
        path.write_text("\n".join(rows) + "\n")
        return path

    def smoke(self, results, tmp_path, env=None):
        full = {"CF401_RESULTS": str(results),
                "CF401_REDUCE": "sum",
                "CF401_SMOKE_ROOT": str(tmp_path / "cf-401-smoke-scratch")}
        full.update(env or {})
        return run_sh(SMOKE, 10, env=full)

    def test_a_second_run_keeps_the_measured_rows(self, tmp_path):
        results = tmp_path / "results"
        table = self.table(results)
        before = table.read_text()
        out = self.smoke(results, tmp_path, env={"DEPTHS": "0 8"})
        assert out.returncode == 0, out.stdout + out.stderr
        assert table.read_text() == before, "the second run replaced the table"

    def test_a_measured_depth_is_skipped_and_says_so(self, tmp_path):
        results = tmp_path / "results"
        self.table(results)
        out = self.smoke(results, tmp_path, env={"DEPTHS": "0 8"})
        log = (results / "smoke_depth.log").read_text()
        for k in (0, 8):
            assert re.search(rf"SKIP k={k}\b", log), log
        assert "CF401_SMOKE_FORCE" in log, "the log does not say how to redo it"

    def test_a_fresh_directory_gets_the_header(self, tmp_path):
        results = tmp_path / "results"
        out = self.smoke(results, tmp_path, env={"DEPTHS": " "})
        assert out.returncode == 0, out.stdout + out.stderr
        header = (results / "smoke_depth.csv").read_text().splitlines()
        assert header and header[0] == self.HEADER, header

    def test_a_table_from_another_header_is_refused(self, tmp_path):
        """Appending to it would put two formats in one file. The summed
        arm's `smoke_k16.csv` is one such format."""
        results = tmp_path / "results"
        results.mkdir()
        (results / "smoke_depth.csv").write_text("cell,k,rc\nx,8,0\n")
        out = self.smoke(results, tmp_path)
        assert out.returncode != 0
        assert "header" in out.stderr, out.stderr


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
            out = study_call(f'cf401_run_name {k}', env=SUM)
            assert out.returncode == 0, out.stderr
            assert out.stdout.strip().endswith(f"cf373k{k}")

    def test_the_run_name_carries_the_reduction(self):
        """Two objectives at one depth write two checkpoint sets and two
        losses CSVs, and a name that reads them apart is what keeps the
        comparison arm readable."""
        for k in DEPTHS:
            out = study_call(f'cf401_run_name {k}')
            assert out.returncode == 0, out.stderr
            assert out.stdout.strip().endswith(f"cf373k{k}_{REDUCE}")

    def test_no_run_name_collides_with_a_published_373_one(self):
        published = {p.name for p in (PARENT / "results").glob("score_*.txt")}
        for k in DEPTHS:
            for stop in STOPS:
                out = run_sh(HEAD_EVAL, k, stop, env={"CF401_DRY_RUN": "1"})
                tag = re.search(r"TAG=(\S+)", out.stdout).group(1)
                assert f"score_{tag}.txt" not in published

    def test_the_checkpoint_path_is_under_this_studys_root(self):
        root = study_value("CF401_ROOT")
        out = study_call('cf401_leg_dir 8 40000')
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip().startswith(root)
        assert out.stdout.strip().endswith("leg_40k")

    def test_two_arms_never_share_a_save_directory(self):
        """One cell at two depths. Rule 3: no overlapping save path."""
        dirs = set()
        for k in DEPTHS:
            for stop in STOPS:
                out = study_call(f'cf401_leg_dir {k} {stop}')
                assert out.returncode == 0, out.stderr
                dirs.add(out.stdout.strip())
        assert len(dirs) == len(DEPTHS) * len(STOPS)

    def test_the_runner_is_given_the_arms_own_root(self):
        out = run_sh(RUN_ARM, 8, 40_000, env={"CF401_DRY_RUN": "1"})
        runs = re.search(r"RUNS=(\S+)", out.stdout).group(1)
        assert runs.endswith("/k8"), runs


# --- 12. The card's two deliverables ------------------------------------------


PLOT_LADDER = EXP / "scripts" / "plot_depth_ladder.py"
PLOT_RADAR = EXP / "scripts" / "plot_domain_radar.py"
MAKE_PLOTS = EXP / "scripts" / "make_plots.sh"
DEPTH_COLOURS = EXP / "scripts" / "depth_colours.py"
PLOTS_DIR = EXP / "plots"

DOMAINS = ["Econ/Fin", "Energy", "Healthcare", "Nature", "Sales",
           "Transport", "Web/CloudOps"]


def weighted_gm(pairs):
    """The `all,all` value split_scores.py writes: one geometric mean."""
    total = sum(n for n, _ in pairs)
    return math.exp(sum(n * math.log(v) for n, v in pairs) / total)


def write_splits(path: Path, rows, aggregates=None):
    """A `split_scores.py` table: (label, domain, n, value) tuples.

    Every label also carries the `all,all` row the real table carries, which
    is the aggregate the radar picks a panel's stop by. `aggregates` sets
    that row for a label, so a table can hold a stop whose best FAMILY and
    whose best AGGREGATE are two different stops.
    """
    per_label = {}
    for label, _, n, value in rows:
        per_label.setdefault(label, []).append((n, value))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stop", "split", "name", "n", "gm_rel_mase"])
        for label, pairs in per_label.items():
            value = (aggregates or {}).get(label, weighted_gm(pairs))
            w.writerow([label, "all", "all", sum(n for n, _ in pairs),
                        f"{value:.6f}"])
        for label, domain, n, value in rows:
            w.writerow([label, "domain", domain, n, f"{value:.6f}"])


def study_splits(path: Path, phase: int = 1):
    """Every depth at every stop, one phase, on all seven families."""
    rows = []
    for i, k in enumerate(DEPTHS):
        for stop in STOPS:
            head = stop if phase == 2 else HEAD_STEPS_PHASE1
            label = (f"k{k}_bb{stop // 1000}k_h{head // 1000}k_student")
            for j, d in enumerate(DOMAINS):
                rows.append((label, d, 5 + j, 1.0 + 0.02 * i + 0.01 * j))
    write_splits(path, rows)


class TestDeliverables:
    """The card asks for two figures. This is the code that draws them."""

    def test_both_plot_scripts_exist(self):
        assert PLOT_RADAR.is_file(), "deliverable 1 has no script"
        assert PLOT_LADDER.is_file(), "deliverable 2 has no script"
        assert MAKE_PLOTS.is_file(), "no launcher for the two figures"

    def test_plots_holds_a_tracked_file(self):
        """git tracks no empty directory, and TestLayout asserts plots/ is one."""
        tracked = subprocess.run(
            ["git", "ls-files", str(PLOTS_DIR.relative_to(REPO_ROOT))],
            capture_output=True, text=True, cwd=str(REPO_ROOT)).stdout.split()
        assert tracked, "plots/ has no tracked file — it is gone on a clone"

    def test_the_palette_comes_from_373s_validated_theme(self):
        """One colour system across both reports, and it was validated once."""
        mod = load_module(DEPTH_COLOURS, "depth_colours_401")
        assert set(mod.COLOUR) == set(DEPTHS_DRAWN)
        assert list(mod.DEPTHS) == list(DEPTHS), "the arms are the two run now"
        assert len(set(mod.COLOUR.values())) == len(DEPTHS_DRAWN), \
            "two depths share a hue"
        parent = load_module(PARENT / "scripts" / "cell_colours.py",
                             "cell_colours_401")
        for col in mod.COLOUR.values():
            assert col in parent.PALETTE, f"{col} is not in #373's theme"

    def test_the_ladder_draws_every_depth_at_every_stop(self, tmp_path):
        scores = tmp_path / "scores.csv"
        write_scores(scores, {8: 1.09, 32: 1.20})
        out = tmp_path / "depth_ladder.png"
        r = subprocess.run([sys.executable, str(PLOT_LADDER),
                            "--scores", str(scores), "--out", str(out)],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr
        assert out.is_file() and out.stat().st_size > 10_000
        assert "2 line(s)" in r.stdout, r.stdout

    def test_the_ladder_draws_both_phases(self, tmp_path):
        """The card's second question IS the difference between the panels."""
        scores = tmp_path / "scores.csv"
        with open(scores, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["phase", "k", "stop", "head_steps", "encoder", "score"])
            for k in DEPTHS:
                for stop in STOPS:
                    w.writerow([1, k, stop, HEAD_STEPS_PHASE1, "student", 1.10])
                    w.writerow([2, k, stop, stop, "student", 1.08])
        out = tmp_path / "both.png"
        r = subprocess.run([sys.executable, str(PLOT_LADDER),
                            "--scores", str(scores), "--out", str(out)],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr
        assert "4 line(s)" in r.stdout, r.stdout

    def test_the_ladder_refuses_an_empty_table(self, tmp_path):
        scores = tmp_path / "scores.csv"
        scores.write_text("phase,k,stop,head_steps,encoder,score\n")
        r = subprocess.run([sys.executable, str(PLOT_LADDER),
                            "--scores", str(scores),
                            "--out", str(tmp_path / "x.png")],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode != 0
        assert "ABORT" in r.stderr

    def test_the_radar_draws_one_panel_per_depth(self, tmp_path):
        splits = tmp_path / "splits.csv"
        study_splits(splits, phase=1)
        out = tmp_path / "radar.png"
        r = subprocess.run([sys.executable, str(PLOT_RADAR),
                            "--splits", str(splits), "--phase", "1",
                            "--out", str(out)],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr
        assert out.is_file() and out.stat().st_size > 10_000
        assert "2 panel(s)" in r.stdout, r.stdout

    def test_the_radar_separates_the_two_phases(self, tmp_path):
        """A phase-2 row carries head steps = stop. Phase 1 has none of them."""
        splits = tmp_path / "splits.csv"
        study_splits(splits, phase=2)
        r = subprocess.run([sys.executable, str(PLOT_RADAR),
                            "--splits", str(splits), "--phase", "1",
                            "--out", str(tmp_path / "x.png")],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode != 0, "a phase-2 table drew a phase-1 figure"
        r = subprocess.run([sys.executable, str(PLOT_RADAR),
                            "--splits", str(splits), "--phase", "2",
                            "--out", str(tmp_path / "p2.png")],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr

    def test_the_radar_reads_373s_k3_reference(self, tmp_path):
        """The dashed polygon is the same cell at #373's depth, not a constant."""
        parent_splits = PARENT / "results" / "splits.csv"
        assert parent_splits.is_file()
        keys = {r["stop"] for r in csv.DictReader(open(parent_splits))
                if r["split"] == "domain"}
        for stop in STOPS:
            assert f"A4_k3_bb{stop // 1000}k_student" in keys, stop

    def test_the_radar_names_the_stop_each_panel_drew(self, tmp_path):
        """The two deliverables must name one stop per depth, so it is read."""
        splits = tmp_path / "splits.csv"
        study_splits(splits, phase=1)
        r = subprocess.run([sys.executable, str(PLOT_RADAR),
                            "--splits", str(splits), "--phase", "1",
                            "--out", str(tmp_path / "named.png")],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr
        for k in DEPTHS:
            assert f"k{k}@bb" in r.stdout, r.stdout

    def test_make_plots_draws_what_it_can_and_skips_the_rest(self, tmp_path):
        """It runs at any point in the study, including before phase 2."""
        res = tmp_path / "results"
        res.mkdir()
        for k in DEPTHS:
            for stop in STOPS:
                (res / f"score_k{k}_bb{stop // 1000}k_h30k_student.txt"
                 ).write_text("1.0700\n")
        plots = tmp_path / "plots"
        out = run_sh(MAKE_PLOTS, env={**SUM, "CF401_RESULTS": str(res),
                                      "CF401_ROOT": str(tmp_path / "runs"),
                                      "CF401_PLOTS": str(plots)})
        assert out.returncode == 0, out.stderr
        assert (plots / "depth_ladder.png").is_file(), out.stdout
        # No eval CSV, so no per-domain table and no radar. It says so.
        assert "SKIP domain_radar_phase1" in out.stdout, out.stdout


class TestRadarPanelPick:
    """Which stop a depth's panel draws.

    `--stop best` gives each depth its own best stop. The two deliverables
    read one table, so "best" has to be the aggregate deliverable 2 draws. By
    the best single FAMILY, a depth whose aggregate is best at 200k draws its
    40k panel, and the two figures then disagree about one depth.
    """

    @pytest.fixture(scope="class")
    def radar(self):
        return load_module(PLOT_RADAR, "plot_domain_radar_401")

    def panels(self, radar, path, phase=1, stop="best"):
        vals, aggs, _ = radar.load_domains(path)
        return radar.pick_panels(vals, aggs, phase, stop)

    def two_stops(self, path, aggregates):
        """k = 16 at 40k and at 200k. 40k holds the one low family."""
        rows = []
        for j, d in enumerate(DOMAINS):
            rows.append(("k16_bb40k_h30k_student", d, 5 + j,
                         0.50 if j == 0 else 1.60))
            rows.append(("k16_bb200k_h30k_student", d, 5 + j, 1.05))
        write_splits(path, rows, aggregates=aggregates)

    def test_the_panel_is_the_best_aggregate_not_the_best_family(
            self, radar, tmp_path):
        splits = tmp_path / "splits.csv"
        self.two_stops(splits, {"k16_bb40k_h30k_student": 1.30,
                                "k16_bb200k_h30k_student": 1.05})
        assert self.panels(radar, splits) == [
            (16, 200_000, "base", "k16_bb200k_h30k_student")]

    def test_the_worse_aggregate_loses_even_with_six_better_families(
            self, radar, tmp_path):
        """The same table, with the aggregates the other way around."""
        splits = tmp_path / "splits.csv"
        self.two_stops(splits, {"k16_bb40k_h30k_student": 1.02,
                                "k16_bb200k_h30k_student": 1.05})
        assert self.panels(radar, splits) == [
            (16, 40_000, "base", "k16_bb40k_h30k_student")]

    def test_a_tie_goes_to_the_earlier_stop(self, radar, tmp_path):
        splits = tmp_path / "splits.csv"
        self.two_stops(splits, {"k16_bb40k_h30k_student": 1.05,
                                "k16_bb200k_h30k_student": 1.05})
        assert self.panels(radar, splits)[0][1] == 40_000

    def test_a_table_with_no_aggregate_row_is_refused(self, radar, tmp_path):
        """Without the row there is no rule, and no figure is better than the
        wrong stop under a title that names it."""
        splits = tmp_path / "splits.csv"
        splits.write_text(
            "stop,split,name,n,gm_rel_mase\n"
            + "".join(f"k16_bb40k_h30k_student,domain,{d},7,1.05\n"
                      for d in DOMAINS))
        with pytest.raises(SystemExit) as err:
            self.panels(radar, splits)
        assert "ABORT" in str(err.value), err.value

    def test_a_pinned_stop_reads_no_aggregate(self, radar, tmp_path):
        """`--stop 200000` names the stop itself, so nothing is picked."""
        splits = tmp_path / "splits.csv"
        splits.write_text(
            "stop,split,name,n,gm_rel_mase\n"
            + "".join(f"k16_bb200k_h30k_student,domain,{d},7,1.05\n"
                      for d in DOMAINS))
        assert self.panels(radar, splits, stop="200000") == [
            (16, 200_000, "base", "k16_bb200k_h30k_student")]

    def test_a_variant_takes_its_own_panel(self, radar, tmp_path):
        """The tag carries a fifth part for a second training schedule at the
        same three numbers. Keyed on (depth, stop) alone the two rows are one
        panel, and the figure draws whichever the table lists second."""
        splits = tmp_path / "splits.csv"
        rows = []
        for label, base in (("k32_bb40k_h30k_student", 1.20),
                            ("k32_ema30k_bb40k_h30k_student", 1.24)):
            for j, d in enumerate(DOMAINS):
                rows.append((label, d, 5 + j, base + 0.01 * j))
        write_splits(splits, rows)
        got = self.panels(radar, splits)
        assert [(k, s, v) for k, s, v, _ in got] == [
            (32, 40_000, "base"), (32, 40_000, "ema30k")], got

    def test_a_variant_never_wins_a_depths_own_panel(self, radar, tmp_path):
        """Even when its aggregate is the better one. The base panel picks a
        stop over the CARD's schedule, so a side run cannot take that slot."""
        splits = tmp_path / "splits.csv"
        rows = []
        for label, base in (("k32_bb40k_h30k_student", 1.30),
                            ("k32_ema30k_bb40k_h30k_student", 1.00)):
            for j, d in enumerate(DOMAINS):
                rows.append((label, d, 5 + j, base))
        write_splits(splits, rows)
        first, *_ = self.panels(radar, splits)
        assert first[2] == "base", first

    def test_the_panels_are_in_the_studys_run_order(self, radar, tmp_path):
        splits = tmp_path / "splits.csv"
        study_splits(splits, phase=1)
        assert [k for k, _, _, _ in self.panels(radar, splits)] == list(DEPTHS)


# --- 13. The trial: the head half of the pipeline, before phase 1 -------------


TRIAL = EXP / "scripts" / "trial_head.sh"


class TestTrialMode:
    """The pre-merge checklist wants an end-to-end run on representative input.

    Phase 1 spends 3.2 hours of backbone time before its first head runs, so
    the study carries a documented budget override and a script that drives
    the whole pipeline through it.
    """

    def test_the_trial_script_exists(self):
        assert TRIAL.is_file()

    def test_it_runs_the_studys_own_scripts(self):
        code = strip_comments(TRIAL.read_text())
        for script in ("run_arm_k.sh", "phase1.sh", "collect.sh", "phase2.sh"):
            assert script in code, f"the trial does not exercise {script}"

    def test_the_override_is_off_by_default(self):
        assert study_value("CF401_STOPS").split() == [str(s) for s in STOPS]
        assert study_value("CF401_HEAD_STEPS_P1") == str(HEAD_STEPS_PHASE1)

    def test_the_override_replaces_the_two_step_counts(self):
        env = {"CF401_TRIAL": "400"}
        assert study_value_env("CF401_STOPS", env) == "400"
        assert study_value_env("CF401_HEAD_STEPS_P1", env) == "200"

    def test_a_trial_writes_nowhere_the_study_writes(self):
        env = {"CF401_TRIAL": "400"}
        assert study_value("CF401_ROOT") != study_value_env("CF401_ROOT", env)
        assert study_value_env("CF401_ROOT", env).endswith("-trial")
        assert study_value_env("CF401_RESULTS", env).endswith("/trial")

    def test_the_two_trial_phases_write_two_tags(self):
        """Both budgets round to 0k, so a `%dk` tag would give them one name."""
        env = {"CF401_TRIAL": "400"}
        p1 = study_call('cf401_tag 16 400 200', env=env).stdout.strip()
        p2 = study_call('cf401_tag 16 400 400', env=env).stdout.strip()
        assert p1 and p2 and p1 != p2, (p1, p2)

    def test_the_phase2_rule_still_holds_in_a_trial(self):
        env = {"CF401_TRIAL": "400"}
        ok = study_call('cf401_require_head_steps 400 400', env=env)
        assert ok.returncode == 0
        bad = study_call('cf401_require_head_steps 12345 400', env=env)
        assert bad.returncode != 0

    def test_the_trial_refuses_a_root_that_is_not_a_trial_root(self):
        code = strip_comments(TRIAL.read_text())
        assert "*-trial)" in code and "*/trial)" in code

    def test_it_runs_one_eval_config_not_97(self):
        code = strip_comments(TRIAL.read_text())
        assert "EVAL_CONFIG_FILTER" in code
        assert "EVAL_EXPECT_CONFIGS=1" in code

    def test_the_log_names_the_trials_phase_2_budget(self):
        """The trial has one stop, and the phase-2 budget is that stop."""
        code = strip_comments(TRIAL.read_text())
        assert re.search(r'phase 2 \$TRIAL_STEPS', code), (
            "the log line does not name the budget phase 2 runs at")


def study_value_env(name: str, env: dict) -> str:
    """One value of study.sh, sourced with `env` set."""
    full = dict(os.environ)
    full.update(env)
    out = subprocess.run(
        ["bash", "-c", f'. "{STUDY_SH}" >/dev/null && printf "%s" "${{{name}}}"'],
        capture_output=True, text=True, timeout=60, env=full)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


# --- 14. The step-count label, which the tags and collect.sh both read -------


class TestStepLabels:

    @pytest.mark.parametrize("steps,label", [(40_000, "40k"), (200_000, "200k"),
                                             (30_000, "30k"), (400, "400"),
                                             (1, "1"), (1000, "1k")])
    def test_a_label_is_thousands_only_when_it_divides(self, steps, label):
        out = study_call(f'cf401_steps_label {steps}')
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == label

    @pytest.mark.parametrize("steps", [40_000, 200_000, 30_000, 400, 1, 1000])
    def test_the_label_round_trips(self, steps):
        out = study_call(f'cf401_steps_of "$(cf401_steps_label {steps})"')
        assert out.returncode == 0, out.stderr
        assert int(out.stdout.strip()) == steps


# --- 15. The smoke script's two review gaps ----------------------------------


class TestSmokeGuards:

    # `run_one` does `rm -rf "$SCRATCH/k<k>"`. Every case below is checked
    # before the script trains anything, and every one of them points
    # CF401_RESULTS at a temporary directory: a test must never write into
    # the study's own results.
    @pytest.mark.parametrize("root", ["/", "/tmp", "relative/path", "/x"])
    def test_it_refuses_a_scratch_root_it_must_not_remove_from(
            self, root, tmp_path):
        out = run_sh(SMOKE, 10, env={"CF401_SMOKE_ROOT": root,
                                     "CF401_RESULTS": str(tmp_path)})
        assert out.returncode != 0, root
        assert "CF401_SMOKE_ROOT" in out.stderr
        assert not list(tmp_path.iterdir()), "it wrote before it refused"

    def test_an_empty_override_falls_back_to_the_default_root(self, tmp_path):
        """`${VAR:-default}` substitutes for empty as well as for unset.

        So an empty override cannot make the target `/k0`. The value it does
        take is logged, because a scratch root nothing names is a root nobody
        checks.  `DEPTHS=" "` runs no depth, so this trains nothing.
        """
        out = run_sh(SMOKE, 10, env={**SUM, "CF401_SMOKE_ROOT": "",
                                     "DEPTHS": " ",
                                     "CF401_RESULTS": str(tmp_path)})
        assert out.returncode == 0, out.stderr
        log = (tmp_path / "smoke_depth.log").read_text()
        m = re.search(r"scratch root: (\S+)", log)
        assert m, log
        assert m.group(1).startswith("/home/"), m.group(1)

    def test_the_probe_rc_is_recorded(self):
        """An empty peak column with no cause was the review's gap 8."""
        code = strip_comments(SMOKE.read_text())
        assert "probe_rc" in code, "the probe's exit status is discarded"
        assert re.search(r'\[ -n "\$peak" \] \|\| log', code), (
            "a missing peak writes no line")
        assert "2>/dev/null" not in code.split("run_one()")[-1], (
            "the probe still sends an error to /dev/null")


class TestDepthCols:
    """`cos_err_dj` counting, and the CRLF header that made it off by one."""

    def build(self, tmp_path, header: str, eol: str = "\r\n"):
        leg = tmp_path / "cell" / "leg_0k"
        leg.mkdir(parents=True)
        csv_path = leg / "run_losses.csv"
        csv_path.write_bytes((header + eol + "0" + eol).encode())
        return tmp_path

    def cols(self, root) -> str:
        out = study_call(f'cf401_depth_cols "{root}"')
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()

    @pytest.mark.parametrize("k", [0, 1, 3, 8, 16, 32])
    def test_it_counts_k_plus_1_columns_at_depth_k(self, tmp_path, k):
        cols = ",".join(["step", "loss"]
                        + [f"cos_err_d{j}" for j in range(k + 1 if k else 0)])
        root = self.build(tmp_path, cols)
        assert self.cols(root) == str(k + 1 if k else 0)

    def test_the_last_column_is_counted_through_a_crlf_header(self, tmp_path):
        """The writer ends every line CRLF, so the last field carries a \\r.

        Without `tr -d '\\r'` the count reads k, which is off by one and still
        plausible — a k = 16 run would report 16 columns and look right.
        """
        cols = ",".join(["step"] + [f"cos_err_d{j}" for j in range(17)])
        assert self.cols(self.build(tmp_path, cols, eol="\r\n")) == "17"

    def test_it_counts_the_same_through_a_bare_lf_header(self, tmp_path):
        cols = ",".join(["step"] + [f"cos_err_d{j}" for j in range(17)])
        assert self.cols(self.build(tmp_path, cols, eol="\n")) == "17"

    def test_a_column_that_only_starts_with_the_prefix_is_not_counted(
            self, tmp_path):
        cols = "step,cos_err_d0,cos_err_d0_ema,cos_err_dj_mean"
        assert self.cols(self.build(tmp_path, cols)) == "1"

    def test_no_csv_under_the_root_prints_nothing(self, tmp_path):
        assert self.cols(tmp_path) == ""


# --- 16. Neither phase drops a head's exit status ----------------------------


class TestHeadExitStatus:
    """A dead head left no line in phase1.log, and the failure surfaced hours
    later as an "incomplete phase 1" abort from the picker."""

    @pytest.mark.parametrize("script", [PHASE1, PHASE2])
    def test_the_wait_result_is_read(self, script):
        code = strip_comments(script.read_text())
        assert re.search(r'wait "\$\{heads\[\$i\]\}"; rc=\$\?', code), (
            f"{script.name} discards the status of every head")

    @pytest.mark.parametrize("script", [PHASE1, PHASE2])
    def test_a_failed_head_is_named_in_the_log(self, script):
        code = strip_comments(script.read_text())
        assert "head_names" in code, f"{script.name} logs no (k, stop)"
        assert re.search(r'log "head \$\{head_names\[\$i\]\} rc=\$rc', code)

    @pytest.mark.parametrize("script", [PHASE1, PHASE2])
    def test_the_phase_exits_non_zero_when_a_head_failed(self, script):
        code = strip_comments(script.read_text())
        assert re.search(r'\[ "\$failed" -eq 0 \] \|\| exit 1', code), (
            f"{script.name} reports success after a head died")


# --- 17. The sync loop reaches one level deeper than #373's --------------


VERIFY_DEPTH = EXP / "sync" / "verify_glob_depth.sh"


class TestSyncDepth:

    def test_the_loop_reaches_this_studys_layout(self):
        """The arms save at <root>/k<K>/<cell>/leg_<N>k, one level deeper."""
        out = run_sh(VERIFY_DEPTH)
        assert out.returncode == 0, out.stdout + out.stderr
        assert "PASS" in out.stdout

    def test_the_loop_does_not_bound_its_walk(self):
        loop = (PARENT / "sync" / "sync_loop.sh").read_text()
        assert "find '$1' -type f" in loop
        assert "-maxdepth" not in loop

    def test_launch_sync_records_the_verification(self):
        """A reader must not have to re-derive it from the loop's source."""
        code = LAUNCH_SYNC.read_text()
        assert "verify_glob_depth.sh" in code
        assert "maxdepth" in code


# --- 18. The trial actually ran ----------------------------------------------


TRIAL_LOG = EXP / "results" / "trial" / "trial.log"


class TestTrialEvidence:
    """The pre-merge checklist asks for an end-to-end run on real input.

    The log is committed and the rest of the trial's output is not: a
    one-config score file beside the study's own is one glob away from being
    read as a study number.
    """

    def test_the_log_of_a_passing_trial_is_committed(self):
        assert TRIAL_LOG.is_file(), "no record that the pipeline ever ran"
        text = TRIAL_LOG.read_text()
        assert "TRIAL PASSED" in text, text[-400:]

    def test_the_log_shows_both_phases_scored(self):
        text = TRIAL_LOG.read_text()
        assert re.search(r"^1,16,400,200,student,", text, re.M), text
        assert re.search(r"^2,16,400,400,student,", text, re.M), text

    def test_the_log_shows_a_per_domain_row(self):
        """Deliverable 1's input, produced by the real collect.sh."""
        assert re.search(r",domain,\w", TRIAL_LOG.read_text())

    def test_the_trial_scores_are_not_committed(self):
        tracked = subprocess.run(
            ["git", "ls-files", "reports/2026-08-15_rollout_depth_k16_8_32/"
                                "results/trial/"],
            capture_output=True, text=True, cwd=str(REPO_ROOT)).stdout.split()
        assert tracked == ["reports/2026-08-15_rollout_depth_k16_8_32/"
                           "results/trial/trial.log"], tracked


# --- 19. The objective: the mean over the k + 1 depth copies -----------------


class TestTheMeanObjective:
    """This protocol trains the MEAN. The stopped arm trained the sum.

    `tests/test_401_rollout_reduce_mean.py` holds what the two reductions do
    to the loss. What is pinned here is that the study asks for the mean, and
    that the two arms never write one another's files.
    """

    def test_the_study_runs_the_mean(self):
        assert study_value("CF401_REDUCE") == REDUCE

    def test_the_flag_reaches_373s_runner(self):
        """#373's `run_leg_k.sh` appends GAP_ARGS last to the trainer command
        line, so the reduction rides in with no second trainer call."""
        out = run_sh(RUN_ARM, 8, 40_000, env={"CF401_DRY_RUN": "1"})
        assert out.returncode == 0, out.stderr
        assert f"GAP_ARGS=--train-rollout-reduce {REDUCE}" in out.stdout
        assert f"RUN_SUFFIX=_{REDUCE}" in out.stdout

    def test_the_runner_appends_gap_args_to_the_trainer(self):
        """The flag is worth nothing if the runner drops it."""
        leg = strip_comments(PARENT_LEG.read_text())
        assert 'read -r -a GAP_ARGS_ARR <<<"${GAP_ARGS:-}"' in leg
        assert '"${GAP_ARGS_ARR[@]}"' in leg

    def test_the_trainer_has_the_flag(self):
        """A checkout without it trains the summed objective and says
        nothing."""
        train = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
                 / "scripts" / "train.py").read_text()
        assert '"--train-rollout-reduce"' in train
        assert ('LOSS_SPEC.train_configuration["train_rollout_reduce"]'
                in train)

    @pytest.mark.parametrize("name", ["CF401_RESULTS", "CF401_PLOTS"])
    def test_every_output_directory_carries_the_reduction(self, name):
        assert study_value(name).endswith(f"/{REDUCE}")
        assert not study_value(name, env=SUM).endswith(f"/{REDUCE}")

    def test_the_checkpoint_root_carries_the_reduction(self):
        """A suffix on the root, not a subdirectory: a sync loop pulls the
        whole root, and two objectives under one root are one pull."""
        assert study_value("CF401_ROOT").endswith(f"-{REDUCE}")
        assert not study_value("CF401_ROOT", env=SUM).endswith(f"-{REDUCE}")

    def test_the_summed_arm_keeps_the_paths_it_wrote(self):
        """Its 8 scored cells are the comparison, and every earlier comment
        points at `results/` and `plots/`."""
        assert study_value("CF401_RESULTS", env=SUM) == str(EXP / "results")
        assert study_value("CF401_PLOTS", env=SUM) == str(EXP / "plots")
        assert (EXP / "results" / "score_k8_bb40k_h30k_student.txt").is_file()

    def test_the_two_arms_never_share_a_score_file(self):
        """One tag, two objectives, one file — the mean would overwrite the
        number this card is measured against."""
        mean = run_sh(HEAD_EVAL, 8, 40_000, env={"CF401_DRY_RUN": "1"}).stdout
        summed = run_sh(HEAD_EVAL, 8, 40_000,
                        env={**SUM, "CF401_DRY_RUN": "1"}).stdout
        got = [re.search(r"score=(\S+)", t).group(1) for t in (mean, summed)]
        assert got[0] != got[1], got
        assert got[1] == str(EXP / "results" / "score_k8_bb40k_h30k_student.txt")

    def test_the_two_arms_never_share_a_checkpoint_root(self):
        assert study_value("CF401_ROOT") != study_value("CF401_ROOT", env=SUM)

    @pytest.mark.parametrize("bad", ["median", "avg", "SUM"])
    def test_a_reduction_that_is_not_sum_or_mean_is_refused(self, bad):
        out = study_call('printf ok', env={"CF401_REDUCE": bad})
        assert out.returncode != 0, out.stdout
        assert "CF401_REDUCE" in out.stderr, out.stderr

    def test_an_empty_override_falls_back_to_the_mean(self):
        """`${VAR:-default}` substitutes for empty as well as for unset."""
        assert study_value("CF401_REDUCE", env={"CF401_REDUCE": ""}) == REDUCE

    @pytest.mark.parametrize(
        "name", ["CF401_ROOT", "CF401_RESULTS", "CF401_PLOTS"])
    def test_a_path_given_on_the_command_line_is_taken_as_it_is(self, name):
        """Both machines hand one in and neither can rename it: the box saves
        to /root/cf401_runs, and on elisa the root IS where the sync loop
        lands the box's tree. A suffix on top would point the heads at a
        directory that holds nothing."""
        assert study_value(name, env={name: "/x/given"}) == "/x/given"

    @pytest.mark.parametrize(
        "name", ["CF401_ROOT", "CF401_RESULTS", "CF401_PLOTS"])
    def test_sourcing_it_twice_resolves_the_same_path(self, name):
        """A launcher sets these on its command line, which EXPORTS them, and
        every script it spawns sources study.sh again. A suffix applied on
        top of an inherited value would stack, and each level would read a
        different tree."""
        once = study_value(name)
        twice = subprocess.run(
            ["bash", "-c",
             f'. "{STUDY_SH}" >/dev/null && export {name} && '
             f'bash -c \'. "{STUDY_SH}" >/dev/null && printf "%s" "${name}"\''],
            capture_output=True, text=True, timeout=60).stdout.strip()
        assert twice == once, (once, twice)

    def test_a_trial_of_the_mean_writes_beside_the_mean(self):
        """The trial suffix rides on top of the reduction, not instead."""
        env = {"CF401_TRIAL": "400"}
        assert study_value("CF401_RESULTS", env=env).endswith(
            f"/{REDUCE}/trial")
        assert study_value("CF401_ROOT", env=env).endswith(
            f"-{REDUCE}-trial")


# --- 20. Four GPUs, two machines --------------------------------------------


class TestTwoMachines:
    """The box trains backbones. elisa trains every head and scores it.

    The eval reads gift-eval-data and the gift_eval package, and neither is
    on a rented card, so nothing there can produce a GM-Relative MASE.
    """

    def test_the_box_launcher_trains_backbones_only(self):
        code = strip_comments(LAUNCH_BOX.read_text())
        assert "CF401_HEADS=0" in code, "the box would train heads it cannot score"
        assert "phase1.sh" in code

    def test_the_box_takes_one_card_per_arm(self, tmp_path):
        out = run_sh(LAUNCH_BOX, env={"CF401_DRY_RUN": "1",
                                      "CF401_ROOT": str(tmp_path / "runs"),
                                      "CF401_RESULTS": str(tmp_path / "res"),
                                      "GPUS": "0 1"})
        assert out.returncode == 0, out.stderr
        arms = [ln for ln in out.stdout.splitlines() if ln.startswith("arm ")]
        assert len(arms) == len(DEPTHS), out.stdout
        gpus = [re.search(r"gpu=(\d+)", ln).group(1) for ln in arms]
        assert len(set(gpus)) == len(DEPTHS), "two arms on one card"
        assert all("heads=0" in ln for ln in arms), out.stdout

    def test_the_box_refuses_fewer_cards_than_arms(self, tmp_path):
        """One arm per card. Two arms on one card halves both."""
        out = run_sh(LAUNCH_BOX, env={"CF401_DRY_RUN": "1",
                                      "CF401_ROOT": str(tmp_path / "runs"),
                                      "CF401_RESULTS": str(tmp_path / "res"),
                                      "GPUS": "0"})
        assert out.returncode != 0
        assert "GPU" in out.stderr, out.stderr

    def test_phase1_plans_no_head_when_the_heads_run_elsewhere(self):
        out = run_sh(PHASE1, env={"CF401_DRY_RUN": "1", "CF401_HEADS": "0"})
        assert out.returncode == 0, out.stderr
        lines = out.stdout.split()
        assert out.stdout.count("arm ") == len(DEPTHS) * len(STOPS)
        assert "head" not in lines, out.stdout

    # ---- elisa ----

    def stops_on_disk(self, root, pairs, run_suffix=""):
        """Backbone checkpoints where `cf401_bb_ckpt` looks for them.

        The suffix is the reduction's, and it is not decoration: `sum`'s run
        name is a PREFIX of `mean`'s, so a summed arm must not find a `_mean`
        checkpoint here either.
        """
        for k, stop in pairs:
            leg = (root / f"k{k}" / CELL / f"leg_{stop // 1000}k")
            leg.mkdir(parents=True, exist_ok=True)
            (leg / f"cf393_{CELL}_cf373k{k}{run_suffix}_"
                   f"{stop // 1000}k.pth").write_text("x")
        return root

    def watch(self, tmp_path, pairs, scored=(), env=None):
        root = self.stops_on_disk(tmp_path / "runs", pairs)
        res = tmp_path / "res"
        res.mkdir(parents=True, exist_ok=True)
        for tag in scored:
            (res / f"score_{tag}.txt").write_text("1.0700\n")
        full = {"CF401_DRY_RUN": "1", "CF401_REDUCE": "sum",
                "CF401_ALLOW_LOCAL_ARMS": "1",
                "CF401_ROOT": str(root), "CF401_RESULTS": str(res)}
        full.update(env or {})
        out = run_sh(HEADS_WATCH, env=full)
        assert out.returncode == 0, out.stderr
        return [ln for ln in out.stdout.splitlines() if ln.startswith("head ")]

    def test_a_head_is_planned_for_every_stop_on_disk(self, tmp_path):
        heads = self.watch(tmp_path, [(8, 40_000), (32, 40_000)])
        assert len(heads) == 2, heads
        for ln in heads:
            assert f"steps={HEAD_STEPS_PHASE1}" in ln, ln
            assert "enc=student" in ln, ln

    def test_a_stop_that_has_not_arrived_is_not_planned(self, tmp_path):
        """The box climbs for hours between stops. The watcher waits."""
        assert self.watch(tmp_path, []) == []

    def test_a_scored_stop_is_not_planned_again(self, tmp_path):
        heads = self.watch(tmp_path, [(8, 40_000), (32, 40_000)],
                           scored=("k8_bb40k_h30k_student",))
        assert len(heads) == 1, heads
        assert "k=32" in heads[0], heads

    def test_phase_2_waits_for_a_complete_phase_1(self, tmp_path):
        """The card's own rule, and the picker's. A phase-2 head costs up to
        seven times a phase-1 one."""
        every = [(k, s) for k in DEPTHS for s in STOPS]
        heads = self.watch(tmp_path, every, scored=(
            "k8_bb40k_h30k_student", "k8_bb100k_h30k_student"))
        assert all(f"steps={HEAD_STEPS_PHASE1}" in ln for ln in heads), heads

    def test_phase_2_starts_once_every_phase_1_cell_is_scored(self, tmp_path):
        every = [(k, s) for k in DEPTHS for s in STOPS]
        scored = tuple(f"k{k}_bb{s // 1000}k_h30k_student"
                       for k in DEPTHS for s in STOPS)
        heads = self.watch(tmp_path, every, scored=scored)
        assert len(heads) == len(DEPTHS) * len(STOPS), heads
        for ln in heads:
            stop = int(re.search(r"stop=(\d+)", ln).group(1))
            assert f"steps={stop}" in ln, ln

    def test_elisa_runs_the_watcher_and_checks_the_sync_loop(self):
        code = strip_comments(LAUNCH_ELISA.read_text())
        assert "heads_watch.sh" in code
        assert "make_plots.sh" in code
        assert "sync_loop.sh" in code, "nothing checks that the pull runs"

    def test_the_sync_check_reads_the_argument_list(self):
        """`pgrep -f` also matches a process that only NAMES the file, the
        check itself among them, and `pgrep -c` prints 0 AND exits 1 when it
        matches nothing. Both would report a loop that is not there."""
        code = strip_comments(STUDY_SH.read_text())
        assert "/proc/$p/cmdline" in code, "the check trusts the pattern alone"
        for path in (STUDY_SH, LAUNCH_ELISA):
            body = strip_comments(path.read_text())
            assert "pgrep -fc" not in body and "pgrep -c" not in body

    # ---- the box, before it trains ----

    def test_provisioning_reuses_373s_provisioner(self):
        code = strip_comments(PROVISION_BOX.read_text())
        assert "provision_box.sh" in code
        assert "VAST_SEARCH_ARGS" in code

    def test_provisioning_asks_for_the_cards_the_card_names(self):
        out = run_sh(PROVISION_BOX, env={"CF401_DRY_RUN": "1"})
        assert out.returncode == 0, out.stderr
        assert "--num-gpus 2" in out.stdout, out.stdout
        assert "RTX_4090" in out.stdout and "RTX_5090" in out.stdout
        assert "--min-reliability 0.99" in out.stdout, out.stdout
        args = re.search(r"VAST_SEARCH_ARGS=(.*)", out.stdout).group(1)
        assert "--prosumer" not in args, "datacenter hosts only"

    def test_the_cpu_filter_is_a_preference_and_not_a_gate(self):
        """The 2-GPU pool at this reliability is small. A hard CPU gate over
        a pool that size returns nothing, and the study waits for a machine
        instead of training."""
        out = run_sh(PROVISION_BOX, env={"CF401_DRY_RUN": "1"})
        assert out.returncode == 0, out.stderr
        assert re.search(r"pass 1 VAST_CPU_RE=\S*Ryzen", out.stdout), out.stdout
        assert re.search(r"pass 2 VAST_CPU_RE=\.", out.stdout), out.stdout

    def test_bootstrap_packs_this_studys_scripts_and_gates_the_flag(self):
        code = strip_comments(BOOTSTRAP_BOX.read_text())
        assert "EXTRA_PACK=" in code
        assert "--train-rollout-reduce" in code, (
            "a box without the flag trains the summed objective")

    def test_the_parent_bootstrap_takes_the_extra_paths(self):
        """One remote pipeline, with a hook, not two."""
        boot = strip_comments(
            (PARENT / "scripts" / "bootstrap_remote.sh").read_text())
        assert 'read -r -a EXTRA_PACK_ARR <<<"${EXTRA_PACK:-}"' in boot
        assert '"${EXTRA_PACK_ARR[@]}"' in boot

    @pytest.mark.parametrize(
        "script", sorted(EXP.glob("scripts/*.sh")) + sorted(EXP.glob("*.sh")))
    def test_no_script_calls_vastai_directly(self, script):
        """CLAUDE.md: vastrun-kit only. The account is shared."""
        code = strip_comments(script.read_text())
        assert not re.search(r"(^|[\s;|&(])vastai\b", code), script.name


# --- 17. Which machine owns which arm ----------------------------------------


COST_TABLE = EXP / "results" / REDUCE / "leg_cost.csv"
COST_FROM_LOG = EXP / "scripts" / "cost_from_log.py"
COMPARE_PY = EXP / "scripts" / "compare_arms.py"
PLOT_COMPARE = EXP / "scripts" / "plot_arm_compare.py"

# A trainer command line, as `/proc` and the leg log both hold it.
TRAINER_ARGS = ["python3", "-u", "train.py", "--batch-size", "64",
                "--save-dir", "", "--run-name", "x",
                "--train-rollout-depth", "8"]


def trainer_cmdline(save_dir, reduce=REDUCE, equals=False):
    """The argv of one leg, with or without the reduction flag."""
    args = list(TRAINER_ARGS)
    args[args.index("--save-dir") + 1] = str(save_dir)
    if reduce is None:
        return args
    if equals:
        return args + [f"--train-rollout-reduce={reduce}"]
    return args + ["--train-rollout-reduce", reduce]


def save_dir_of(root, k=8, stop=40_000, cell=CELL):
    return f"{root}/k{k}/{cell}/leg_{stop // 1000}k"


class TestTheLayoutIsWrittenIn:
    """The box owns both backbone arms. elisa owns every head and every eval.

    Both machines source study.sh, so the decision cannot be one default.
    What it can be: elisa's root is where the sync loop LANDS the box's tree,
    and elisa refuses to score one root while a backbone arm of the same
    reduction climbs under another one.
    """

    def test_the_box_runs_root_is_a_remote_path(self):
        """The box saves to its own disk. A local path here would make the
        sync loop pull elisa's own directory onto elisa."""
        root = study_value("CF401_BOX_RUNS")
        assert root.startswith("/root/"), root

    def test_the_box_saves_where_the_sync_loop_pulls_from(self):
        """The loop pulls CF401_BOX_RUNS. A box launched without the variable
        used to save under elisa's own default, which nothing pulls."""
        out = run_sh(LAUNCH_BOX, env={"CF401_DRY_RUN": "1"})
        assert out.returncode == 0, out.stderr
        root = re.search(r"root=(\S+)", out.stdout).group(1)
        assert root == study_value("CF401_BOX_RUNS"), out.stdout

    def test_elisa_reads_the_boxs_tree_and_not_a_local_root(self):
        """The sync loop keeps the relative tree under `<LOCAL_DIR>/sync`, so
        that directory IS the root on elisa's side."""
        sync_root = study_value("CF401_SYNC_ROOT")
        assert sync_root.endswith("/sync"), sync_root
        out = run_sh(LAUNCH_ELISA, env={"CF401_DRY_RUN": "1"})
        assert out.returncode == 0, out.stderr
        root = re.search(r"root=(\S+)", out.stdout).group(1)
        assert root == sync_root, out.stdout
        assert "checkpoints_backup" not in root, (
            "elisa would read a local checkpoints root, not the box's tree")

    def test_a_root_given_on_the_command_line_still_wins(self, tmp_path):
        out = run_sh(LAUNCH_ELISA, env={"CF401_DRY_RUN": "1",
                                       "CF401_ROOT": str(tmp_path / "given")})
        assert out.returncode == 0, out.stderr
        assert f"root={tmp_path / 'given'}" in out.stdout, out.stdout

    def test_the_watcher_reads_the_boxs_tree_too(self):
        """`heads_watch.sh` run on its own has to land on the same root as
        `launch_elisa.sh`, or a session that starts the watcher directly
        scores a different tree."""
        out = run_sh(HEADS_WATCH, env={"CF401_DRY_RUN": "1",
                                      "CF401_ALLOW_LOCAL_ARMS": "1"})
        assert out.returncode == 0, out.stderr
        root = re.search(r"root=(\S+)", out.stdout).group(1)
        assert root == study_value("CF401_SYNC_ROOT"), out.stdout

    def test_the_heads_take_gpu_0_only(self):
        """elisa GPU 1 belongs to another session."""
        code = HEADS_WATCH.read_text()
        assert re.search(r'HEAD_GPUS="\$\{HEAD_GPUS:-0\}"', code)
        for path in (LAUNCH_ELISA, HEADS_WATCH):
            assert "another session" in path.read_text(), path.name

    def test_the_box_launcher_names_the_arms_it_owns(self):
        """The header is where the next reader learns the layout."""
        head = LAUNCH_BOX.read_text()
        assert "k = 8" in head and "k = 32" in head


class TestWhatTheTrainerOfALegRuns:
    """The reduction and the root, read off the trainer's own command line.

    Neither is in the wrapper's argv: the reduction rides in through
    `GAP_ARGS` and the root through `--save-dir`. The command line is the one
    place that names both.
    """

    def reduce_of(self, args):
        joined = " ".join(args)
        out = study_call(f'printf "%s" {joined!r} | cf401_reduce_of_cmdline')
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()

    @pytest.mark.parametrize("reduce", ["sum", "mean"])
    @pytest.mark.parametrize("equals", [False, True])
    def test_it_reads_the_reduction_the_line_names(self, reduce, equals):
        assert self.reduce_of(
            trainer_cmdline("/x/k8/c/leg_40k", reduce, equals)) == reduce

    def test_a_line_without_the_flag_reads_as_the_sum(self):
        """`sum` is train.py's own default, so a leg with no flag trains it."""
        assert self.reduce_of(trainer_cmdline("/x", reduce=None)) == "sum"

    def test_the_root_comes_off_the_save_dir(self):
        out = study_call(
            f'cf401_root_of_save_dir {save_dir_of("/runs/cf-401-mean")!r}')
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "/runs/cf-401-mean"

    @pytest.mark.parametrize("path", [
        "/runs/cf-373/arm6_v2_combab_alignS/leg_40k",     # #373's own layout
        "/runs/cf-401-mean/k8/other_cell/leg_40k",        # another cell
        "/runs/cf-401-mean/k8/arm6_v2_combab_alignS/eval",
        "/runs",
    ])
    def test_a_path_that_is_not_this_studys_layout_is_refused(self, path):
        """The guard must not fire on another study's legs."""
        out = study_call(f'cf401_root_of_save_dir {path!r}')
        assert out.returncode != 0, out.stdout

    def test_a_live_leg_is_listed_with_its_reduction_and_its_root(self, tmp_path):
        """The /proc walk, against a real process."""
        fake = tmp_path / "train.py"
        fake.write_text("import sys, time\ntime.sleep(120)\n")
        save = save_dir_of(str(tmp_path / "runs"))
        proc = subprocess.Popen(
            [sys.executable, str(fake), *trainer_cmdline(save)[3:]],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            out = study_call('cf401_running_legs')
            assert out.returncode == 0, out.stderr
            rows = [ln.split() for ln in out.stdout.splitlines()]
            mine = [r for r in rows if r[0] == str(proc.pid)]
            assert mine, out.stdout
            assert mine[0][1:] == [REDUCE, str(tmp_path / "runs")], mine
        finally:
            proc.kill()
            proc.wait(timeout=30)

    def test_a_leg_of_another_study_is_not_listed(self, tmp_path):
        fake = tmp_path / "train.py"
        fake.write_text("import sys, time\ntime.sleep(120)\n")
        save = f"{tmp_path}/runs/{CELL}/leg_40k"          # #373's layout
        proc = subprocess.Popen(
            [sys.executable, str(fake), *trainer_cmdline(save)[3:]],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            out = study_call('cf401_running_legs')
            assert str(proc.pid) not in out.stdout, out.stdout
        finally:
            proc.kill()
            proc.wait(timeout=30)


class TestTheHeadWatcherRefusesASecondRoot:
    """A box that climbs for 33 h and is then not read is the most expensive
    failure this card has. The watcher reads ONE root, so a backbone arm of
    the same reduction under another root on this machine is refused."""

    def foreign(self, lines, reduce=REDUCE, root="/runs/cf-401-mean"):
        joined = "\n".join(lines)
        return study_call(
            f'printf "%s" {joined!r} | cf401_foreign_arm {reduce} {root}')

    def test_an_arm_under_another_root_is_named(self):
        out = self.foreign(["4242 mean /runs/other"])
        assert out.returncode == 0, out.stdout
        assert "4242" in out.stdout and "/runs/other" in out.stdout

    def test_an_arm_under_this_root_is_not(self):
        """The box's own arms arrive here through the sync loop. On the box
        itself the arms and the root are the same tree."""
        assert self.foreign(["4242 mean /runs/cf-401-mean"]).returncode != 0

    def test_a_trailing_slash_is_the_same_root(self):
        assert self.foreign(["4242 mean /runs/cf-401-mean/"]).returncode != 0

    def test_an_arm_of_the_other_reduction_is_not_refused(self):
        """The summed arm is stopped, and a run of it writes nowhere this
        protocol writes. It is not this watcher's business."""
        assert self.foreign(["4242 sum /runs/cf-401"]).returncode != 0

    def test_nothing_running_is_not_refused(self):
        assert self.foreign([]).returncode != 0

    def test_the_watcher_calls_the_guard(self):
        code = strip_comments(HEADS_WATCH.read_text())
        assert "cf401_foreign_arm" in code, "nothing checks for a second root"
        assert "CF401_ALLOW_LOCAL_ARMS" in code, "the guard cannot be waived"

    def test_the_watcher_refuses_while_a_local_arm_runs(self, tmp_path):
        """End to end: a live leg of this reduction under another root."""
        fake = tmp_path / "train.py"
        fake.write_text("import sys, time\ntime.sleep(120)\n")
        save = save_dir_of(str(tmp_path / "other_runs"))
        proc = subprocess.Popen(
            [sys.executable, str(fake), *trainer_cmdline(save)[3:]],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            env = {"CF401_DRY_RUN": "1", "CF401_ROOT": str(tmp_path / "runs"),
                   "CF401_RESULTS": str(tmp_path / "res")}
            out = run_sh(HEADS_WATCH, env=env)
            assert out.returncode != 0, out.stdout
            assert str(tmp_path / "other_runs") in out.stderr, out.stderr
            waived = run_sh(HEADS_WATCH,
                            env={**env, "CF401_ALLOW_LOCAL_ARMS": "1"})
            assert waived.returncode == 0, waived.stderr
        finally:
            proc.kill()
            proc.wait(timeout=30)


class TestTheLegProvesItsReduction:
    """A mean leg that trained the sum writes the same file names, the same
    columns and the same log lines. The depth has `cf401_depth_cols`; this is
    the reduction's proof, read off the trainer's own command line."""

    def test_the_trainer_writes_its_command_line(self, tmp_path):
        """One line, at startup, so the check reads it in the first minute
        of a 33-hour leg instead of at the end."""
        out = subprocess.run(
            [sys.executable, "-u",
             str(REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
                 / "scripts" / "train.py"),
             "--device", "cpu", "--total-steps", "1", "--batch-size", "2",
             "--lr", "1e-3", "--weight-decay", "0.1",
             "--save-dir", str(tmp_path), "--run-name", "cmdline_probe",
             "--mix-ratio", "1.0", "--synth-kind", "periodic",
             "--t-raw", "64", "--n-channels", "1", "--d-model", "32",
             "--n-heads", "2", "--num-layers", "1", "--num-encoder-layers", "1",
             "--log-every", "1", "--seed", "42",
             "--hf-repo", "none", "--hf-path", "none",
             "--train-rollout-reduce", "mean"],
            capture_output=True, text=True, cwd=str(tmp_path), timeout=900,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
        assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
        lines = [ln for ln in out.stdout.splitlines()
                 if ln.startswith("Command line:")]
        assert len(lines) == 1, out.stdout[:2000]
        assert "--train-rollout-reduce mean" in lines[0], lines[0]

    def log_with(self, results, k, reduce):
        """A leg log as the runner leaves it, with its command line."""
        results.mkdir(parents=True, exist_ok=True)
        name = study_call(f'cf401_run_name {k}').stdout.strip()
        log = results / f"run_{name}.log"
        args = trainer_cmdline(save_dir_of("/runs", k), reduce)
        log.write_text("Command line: " + " ".join(args) + "\n")
        return log

    @pytest.mark.parametrize("reduce", ["sum", "mean"])
    def test_it_reads_the_reduction_out_of_the_leg_log(self, tmp_path, reduce):
        log = self.log_with(tmp_path / "res", 8, reduce)
        out = study_call(f'cf401_leg_reduce {str(log)!r}')
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == reduce

    def test_the_last_command_line_is_the_one_read(self, tmp_path):
        """The runner APPENDS to the log, so a resumed cell's log holds every
        leg's line. This leg's is the last."""
        log = self.log_with(tmp_path / "res", 8, "sum")
        args = trainer_cmdline(save_dir_of("/runs", 8), "mean")
        with open(log, "a") as fh:
            fh.write("  timing: data=1ms  fwd=2ms  bwd=3ms  total=6ms\n")
            fh.write("Command line: " + " ".join(args) + "\n")
        out = study_call(f'cf401_leg_reduce {str(log)!r}')
        assert out.stdout.strip() == "mean", out.stdout

    def test_a_log_with_no_command_line_says_nothing(self, tmp_path):
        (tmp_path / "run.log").write_text("Device: cpu | Params: 1\n")
        out = study_call(f'cf401_leg_reduce {str(tmp_path / "run.log")!r}')
        assert out.returncode != 0, out.stdout
        assert out.stdout.strip() == ""

    # ---- the guard in run_arm_k.sh ----

    def arm(self, study, tmp_path, runner_body, k=8, env=None):
        """`run_arm_k.sh` against a stub of #373's runner."""
        parent = (tmp_path / "reports" / PARENT.name / "scripts")
        stub_script(parent / "run_leg_k.sh", runner_body)
        full = {"CF401_RESULTS": str(tmp_path / "res"),
                "CF401_ROOT": str(tmp_path / "runs"),
                "CF401_REDUCE_CHECK_TIMEOUT": "60"}
        full.update(env or {})
        out = run_sh(study / "scripts" / "run_arm_k.sh", k, 40_000, env=full)
        return out, (tmp_path / "res" / "arms.log").read_text()

    # The stub writes the command line the way run_leg_k.sh's trainer does,
    # into the log this study's own helper names, then holds the leg open.
    STUB = """
log="$CF_RESULTS/run_cf393_${1}_cf373k${K}${RUN_SUFFIX}.log"
mkdir -p "$CF_RESULTS"
echo "Command line: python3 train.py --save-dir $RUNS/$1/leg_40k REDUCE" >>"$log"
sleep 30
"""

    def test_a_leg_that_carries_the_flag_runs(self, stub_study, tmp_path):
        body = self.STUB.replace(
            "REDUCE", f"--train-rollout-reduce {REDUCE}").replace(
            "sleep 30", "exit 0")
        out, log = self.arm(stub_study, tmp_path, body)
        assert out.returncode == 0, out.stdout + out.stderr
        assert f"reduce={REDUCE} OK" in log, log

    def test_a_leg_that_lost_the_flag_is_stopped(self, stub_study, tmp_path):
        out, log = self.arm(stub_study, tmp_path,
                            self.STUB.replace(" REDUCE", ""))
        assert out.returncode != 0, out.stdout
        assert "train-rollout-reduce" in out.stderr, out.stderr
        assert "sum" in out.stderr, "the abort does not name what it read"
        assert "STOPPED" in log, log

    def test_a_leg_that_trains_the_other_reduction_is_stopped(
            self, stub_study, tmp_path):
        out, _ = self.arm(stub_study, tmp_path,
                          self.STUB.replace("REDUCE",
                                            "--train-rollout-reduce sum"))
        assert out.returncode != 0, out.stdout

    def test_a_runner_that_skips_the_leg_is_not_stopped(self, stub_study,
                                                       tmp_path):
        """`run_leg_k.sh` exits 0 without a trainer when the stop is on disk,
        and 9 or 10 on a HOLD or a claim. None of those is a wrong objective."""
        for rc in (0, 9, 10):
            out, _ = self.arm(stub_study, tmp_path, f"exit {rc}")
            assert out.returncode == rc, (rc, out.stdout, out.stderr)

    def test_the_check_reaches_the_summed_arm_too(self, stub_study, tmp_path):
        """`sum` is stated on every leg, so the check holds under both words."""
        out, _ = self.arm(stub_study, tmp_path,
                          self.STUB.replace("REDUCE",
                                            "--train-rollout-reduce mean"),
                          env=SUM)
        assert out.returncode != 0, out.stdout

    def test_the_bootstrap_gates_the_line_on_the_box(self):
        """The box's train.py has to write the command line, or the check
        cannot read it there."""
        code = strip_comments(BOOTSTRAP_BOX.read_text())
        assert "Command line" in code, (
            "a box whose trainer writes no command line cannot be checked")


class TestTheMeanCostTable:
    """The run plan is sized from a step time. `results/smoke_k16.csv` is the
    SUMMED arm's, and the mean adds one pass over the f-bearing terms."""

    @pytest.fixture(scope="class")
    def cost(self) -> dict:
        assert COST_TABLE.is_file(), f"no {COST_TABLE.relative_to(EXP)}"
        rows = list(csv.DictReader(open(COST_TABLE)))
        assert rows, "the cost table holds no measurement"
        return {int(r["k"]): r for r in rows}

    def test_both_arms_are_measured_under_the_mean(self, cost):
        for k in DEPTHS:
            assert k in cost, f"k = {k} was not measured"
            assert cost[k]["reduce"] == REDUCE, cost[k]
            assert float(cost[k]["total_ms"]) > 0, cost[k]

    def test_every_row_names_where_it_came_from(self, cost):
        """A step time measured beside another leg is not the same number as
        one measured alone. The row says which it is."""
        for k, row in cost.items():
            assert row["source"], f"k = {k} names no source"
            assert int(row["concurrent_legs"]) >= 1, row

    def test_the_mean_costs_more_than_the_sum(self, cost):
        """The extra pass over the f-bearing terms is not free. This is the
        fact that makes the summed arm's table the wrong one to plan from."""
        summed = {int(r["k"]): r
                  for r in csv.DictReader(open(EXP / "results"
                                               / "smoke_k16.csv"))}
        for k in DEPTHS:
            assert float(cost[k]["total_ms"]) > float(summed[k]["total_ms"]), k

    def test_the_box_plan_is_sized_from_this_table(self, cost):
        """The hours in `launch_box.sh`'s header are this table's hours."""
        head = LAUNCH_BOX.read_text()
        assert "leg_cost.csv" in head, "the header quotes another table"
        for k in DEPTHS:
            hours = float(cost[k]["hours_200k"])
            assert re.search(rf"{hours:.1f} h", head), (
                f"k = {k} costs {hours:.1f} h, not what the header says")

    def test_the_peak_is_recorded_for_both_arms(self, cost):
        """Two legs on one 24 GiB card is a memory question, and the box
        plan answers it from this column."""
        for k in DEPTHS:
            assert int(cost[k]["used_mib"]) > 0, cost[k]


class TestCostFromLog:
    """The parser behind the cost table. The smoke script's own rule: drop the
    first window, take the median of the rest."""

    def log(self, path, totals, reduce=REDUCE, k=8):
        args = trainer_cmdline(save_dir_of("/runs", k), reduce)
        lines = ["Command line: " + " ".join(args)]
        for i, total in enumerate(totals):
            lines.append(f"[  {200 * (i + 1)}] loss=1.0  {1000 / total:.1f} "
                         f"sps  ETA 1.0h")
            lines.append(f"  timing: data=1.0ms  fwd=2.0ms  bwd=3.0ms  "
                         f"total={total}ms")
        path.write_text("\n".join(lines) + "\n")
        return path

    def run(self, *args):
        return subprocess.run([sys.executable, str(COST_FROM_LOG), *args],
                              capture_output=True, text=True, timeout=120)

    def test_it_drops_the_warm_up_window_and_takes_the_median(self, tmp_path):
        log = self.log(tmp_path / "a.log", [900.0, 300.0, 200.0, 400.0])
        out = self.run(f"--leg=8={log}", "--concurrent-legs", "1")
        assert out.returncode == 0, out.stderr
        row = list(csv.DictReader(out.stdout.splitlines()))[0]
        assert float(row["total_ms"]) == pytest.approx(300.0)
        assert int(row["windows"]) == 3

    def test_it_reports_the_hours_a_200k_leg_costs(self, tmp_path):
        log = self.log(tmp_path / "a.log", [900.0, 360.0, 360.0])
        out = self.run(f"--leg=8={log}", "--concurrent-legs", "1")
        row = list(csv.DictReader(out.stdout.splitlines()))[0]
        assert float(row["hours_200k"]) == pytest.approx(20.0, abs=0.05)

    def test_it_refuses_a_log_that_names_another_reduction(self, tmp_path):
        log = self.log(tmp_path / "a.log", [900.0, 300.0], reduce="sum")
        out = self.run(f"--leg=8={log}", "--reduce", "mean")
        assert out.returncode != 0
        assert "sum" in out.stderr, out.stderr

    def test_it_refuses_a_log_with_no_window_to_measure(self, tmp_path):
        log = self.log(tmp_path / "a.log", [900.0])
        out = self.run(f"--leg=8={log}")
        assert out.returncode != 0
        assert "window" in out.stderr, out.stderr

    def test_one_row_per_log(self, tmp_path):
        a = self.log(tmp_path / "a.log", [900.0, 300.0, 300.0], k=8)
        b = self.log(tmp_path / "b.log", [900.0, 600.0, 600.0], k=32)
        out = self.run("--concurrent-legs", "2", f"--leg=8={a}",
                       f"--leg=32={b}", "--used-mib=32=5532")
        assert out.returncode == 0, out.stderr
        rows = {int(r["k"]): r for r in csv.DictReader(out.stdout.splitlines())}
        assert sorted(rows) == [8, 32]
        assert float(rows[32]["total_ms"]) == pytest.approx(600.0)
        assert rows[32]["used_mib"] == "5532", rows[32]
        assert rows[8]["source"] == "a.log", rows[8]


class TestTheTwoArmsAreJoined:
    """The summed arm is the comparison. A table and a figure hold the pair.

    Both read whatever cells exist: the summed arm has 8 scored cells and the
    mean arm has fewer for a while.
    """

    def scores(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["phase", "k", "stop", "head_steps", "encoder", "score"])
            for k, stop, score in rows:
                w.writerow([1, k, stop, HEAD_STEPS_PHASE1, "student", score])
        return path

    def compare(self, tmp_path, sum_rows, mean_rows, *extra):
        out_csv = tmp_path / "arm_compare.csv"
        args = [sys.executable, str(COMPARE_PY),
                "--sum", str(self.scores(tmp_path / "sum.csv", sum_rows)),
                "--mean", str(self.scores(tmp_path / "mean.csv", mean_rows)),
                "--out", str(out_csv), *extra]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
        return proc, out_csv

    def rows_of(self, path):
        return list(csv.DictReader(open(path)))

    def test_a_cell_both_arms_scored_carries_the_pair(self, tmp_path):
        proc, out = self.compare(tmp_path, [(8, 40_000, 2.0357)],
                                 [(8, 40_000, 1.9000)])
        assert proc.returncode == 0, proc.stderr
        row = self.rows_of(out)[0]
        assert float(row["sum"]) == pytest.approx(2.0357)
        assert float(row["mean"]) == pytest.approx(1.9000)
        assert float(row["delta"]) == pytest.approx(1.9000 - 2.0357, abs=1e-6)
        assert row["better"] == REDUCE

    def test_a_cell_only_the_summed_arm_scored_is_kept(self, tmp_path):
        """k = 16 is the summed arm's alone, and the mean arm has not reached
        most of its own stops yet. A join that dropped those rows would show
        an empty table for days."""
        proc, out = self.compare(tmp_path,
                                 [(8, 40_000, 2.0), (16, 40_000, 4.5)], [])
        assert proc.returncode == 0, proc.stderr
        rows = {int(r["k"]): r for r in self.rows_of(out)}
        assert sorted(rows) == [8, 16], rows
        for row in rows.values():
            assert row["mean"] == "", row
            assert row["delta"] == "" and row["better"] == "", row

    def test_a_cell_only_the_mean_arm_scored_is_kept(self, tmp_path):
        proc, out = self.compare(tmp_path, [], [(32, 200_000, 1.5)])
        assert proc.returncode == 0, proc.stderr
        row = self.rows_of(out)[0]
        assert row["sum"] == "" and float(row["mean"]) == pytest.approx(1.5)

    def test_the_rows_are_in_the_studys_order(self, tmp_path):
        proc, out = self.compare(
            tmp_path, [(32, 40_000, 1.0), (8, 200_000, 1.0), (8, 40_000, 1.0)],
            [])
        seen = [(int(r["k"]), int(r["stop"])) for r in self.rows_of(out)]
        assert seen == [(8, 40_000), (8, 200_000), (32, 40_000)], seen

    def test_it_writes_a_markdown_table_beside_the_csv(self, tmp_path):
        proc, out = self.compare(tmp_path, [(8, 40_000, 2.0)],
                                 [(8, 40_000, 1.9)])
        md = out.with_suffix(".md")
        assert md.is_file(), "no markdown table for the report"
        text = md.read_text()
        assert "1.9" in text and "2.0" in text, text

    def test_a_variant_does_not_merge_into_its_base_cell(self, tmp_path):
        """`k32_ema30k_bb40k` holds the same depth, stop and head budget as
        `k32_bb40k`. Only the schedule differs. Joined on the other four
        fields the two are one key, and the second one read replaces the
        first — so the table would show one cell and one of the two scores,
        with nothing saying which."""
        mean = tmp_path / "mean.csv"
        mean.parent.mkdir(parents=True, exist_ok=True)
        with open(mean, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["phase", "k", "variant", "stop", "head_steps",
                        "encoder", "score"])
            w.writerow([1, 32, "base", 40_000, HEAD_STEPS_PHASE1,
                        "student", 1.2082])
            w.writerow([1, 32, "ema30k", 40_000, HEAD_STEPS_PHASE1,
                        "student", 1.2385])
        out_csv = tmp_path / "arm_compare.csv"
        proc = subprocess.run(
            [sys.executable, str(COMPARE_PY),
             "--sum", str(self.scores(tmp_path / "sum.csv", [])),
             "--mean", str(mean), "--out", str(out_csv)],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        rows = self.rows_of(out_csv)
        assert len(rows) == 2, rows
        by_variant = {r["variant"]: r for r in rows}
        assert set(by_variant) == {"base", "ema30k"}
        assert float(by_variant["base"][REDUCE]) == pytest.approx(1.2082)
        assert float(by_variant["ema30k"][REDUCE]) == pytest.approx(1.2385)
        # The base cell sorts first, so the pair reads together.
        assert [r["variant"] for r in rows] == ["base", "ema30k"]

    def test_a_table_without_the_variant_column_still_joins(self, tmp_path):
        """The summed arm's table was written before the column existed."""
        proc, out = self.compare(tmp_path, [(8, 40_000, 2.0357)],
                                 [(8, 40_000, 1.9000)])
        assert proc.returncode == 0, proc.stderr
        row = self.rows_of(out)[0]
        assert row["variant"] == "base"
        assert row["better"] == REDUCE

    def test_two_empty_tables_are_refused(self, tmp_path):
        proc, _ = self.compare(tmp_path, [], [])
        assert proc.returncode != 0
        assert "no" in proc.stderr.lower(), proc.stderr

    def test_a_missing_arm_table_is_not_an_error(self, tmp_path):
        """The mean arm's `scores.csv` does not exist until its first head."""
        out_csv = tmp_path / "arm_compare.csv"
        proc = subprocess.run(
            [sys.executable, str(COMPARE_PY),
             "--sum", str(self.scores(tmp_path / "sum.csv",
                                      [(8, 40_000, 2.0)])),
             "--mean", str(tmp_path / "absent.csv"), "--out", str(out_csv)],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        assert self.rows_of(out_csv)[0]["mean"] == ""

    # ---- the figure ----

    def draw(self, tmp_path, sum_rows, mean_rows):
        out = tmp_path / "arm_compare.png"
        proc = subprocess.run(
            [sys.executable, str(PLOT_COMPARE),
             "--sum", str(self.scores(tmp_path / "sum.csv", sum_rows)),
             "--mean", str(self.scores(tmp_path / "mean.csv", mean_rows)),
             "--out", str(out)],
            capture_output=True, text=True, timeout=300)
        return proc, out

    def test_the_figure_draws_the_pair(self, tmp_path):
        proc, out = self.draw(
            tmp_path,
            [(k, s, 2.0) for k in DEPTHS for s in STOPS],
            [(k, s, 1.8) for k in DEPTHS for s in STOPS])
        assert proc.returncode == 0, proc.stderr
        assert out.is_file() and out.stat().st_size > 5000

    def test_the_figure_draws_the_summed_arm_alone(self, tmp_path):
        """Day one of the mean arm: nothing of it is scored yet."""
        proc, out = self.draw(tmp_path, [(8, 40_000, 2.0)], [])
        assert proc.returncode == 0, proc.stderr
        assert out.is_file()

    def test_the_figure_refuses_two_empty_tables(self, tmp_path):
        proc, out = self.draw(tmp_path, [], [])
        assert proc.returncode != 0
        assert not out.exists()

    def test_make_plots_draws_the_comparison(self):
        code = strip_comments(MAKE_PLOTS.read_text())
        assert "plot_arm_compare.py" in code
        assert "compare_arms.py" in code


class TestTheMinorGaps:

    def test_the_sync_check_counts_this_studys_loop_only(self, tmp_path):
        """elisa is shared. Another study's sync loop is not this study's."""
        loop = tmp_path / "sync_loop.sh"
        stub_script(loop, "sleep 120")
        mine = tmp_path / "mine"
        theirs = tmp_path / "theirs"
        for d in (mine, theirs):
            d.mkdir()
        proc = subprocess.Popen(["bash", str(loop)], cwd=str(theirs),
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        try:
            assert study_call(
                f'cf401_sync_loops {str(theirs)!r}').stdout.strip() == "1"
            assert study_call(
                f'cf401_sync_loops {str(mine)!r}').stdout.strip() == "0"
        finally:
            proc.kill()
            proc.wait(timeout=30)

    def test_elisa_checks_the_loop_of_its_own_local_root(self):
        code = strip_comments(LAUNCH_ELISA.read_text())
        assert "cf401_sync_loops" in code
        assert "CF401_SYNC_DIR" in code, (
            "the check does not say which local root it wants a loop for")

    def test_the_remote_runs_root_defaults_to_the_boxs_own(self):
        """`REMOTE_RUNS` names a directory on the BOX. A local default would
        make the loop pull elisa's own tree."""
        code = strip_comments(LAUNCH_SYNC.read_text())
        assert 'REMOTE_RUNS="${REMOTE_RUNS:-$CF401_BOX_RUNS}"' in code, code

    @pytest.mark.parametrize("reduce,other", [("sum", "mean"), ("mean", "sum")])
    def test_a_checkpoint_of_the_other_reduction_is_not_matched(
            self, tmp_path, reduce, other):
        """`cf401_bb_ckpt`'s `*` tolerates train.py's `_rN` infix on a
        re-fired leg. It must not tolerate the other arm's suffix."""
        env = {"CF401_ROOT": str(tmp_path), "CF401_REDUCE": other}
        leg = study_call('cf401_leg_dir 8 40000', env=env).stdout.strip()
        name = study_call('cf401_run_name 8', env=env).stdout.strip()
        Path(leg).mkdir(parents=True)
        (Path(leg) / f"{name}_40k.pth").write_text("x")
        got = study_call('cf401_bb_ckpt 8 40000',
                         env={"CF401_ROOT": str(tmp_path),
                              "CF401_REDUCE": reduce})
        assert got.stdout.strip() == "", got.stdout

    def test_a_re_fired_leg_is_still_matched(self, tmp_path):
        """train.py branches the run name to `<name>_r2` when the save dir
        already holds a checkpoint. That one IS this arm's."""
        env = {"CF401_ROOT": str(tmp_path)}
        leg = study_call('cf401_leg_dir 8 40000', env=env).stdout.strip()
        name = study_call('cf401_run_name 8', env=env).stdout.strip()
        Path(leg).mkdir(parents=True)
        (Path(leg) / f"{name}_r2_40k.pth").write_text("x")
        got = study_call('cf401_bb_ckpt 8 40000', env=env)
        assert got.stdout.strip().endswith("_r2_40k.pth"), got.stdout

    def test_the_smoke_skip_reads_the_reduction_too(self, tmp_path):
        """One table can hold both reductions, and the same depth costs two
        different numbers. A skip on `k` alone would report the mean's row as
        the sum's measurement."""
        results = tmp_path / "results"
        results.mkdir()
        header = ("cell,reduce,k,steps,windows,data_ms,fwd_ms,bwd_ms,total_ms,"
                  "sps,peak_mib,depth_cols,card_free_mib,rc")
        (results / "smoke_depth.csv").write_text(
            f"{header}\n{CELL},mean,8,300,2,1,2,3,300,5,5400,9,19000,0\n")
        out = run_sh(SMOKE, 10, env={
            "CF401_RESULTS": str(results), "CF401_REDUCE": "sum",
            "DEPTHS": "8",
            "CF401_SMOKE_ROOT": str(tmp_path / "cf-401-smoke-scratch")})
        log = (results / "smoke_depth.log").read_text()
        assert "SKIP k=8" not in log, log
