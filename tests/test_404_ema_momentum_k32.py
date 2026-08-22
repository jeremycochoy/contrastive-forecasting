"""Tests for #404: an EMA-momentum sweep for L_align on the teacher, at k = 32.

The card started with four arms. It now holds fourteen runs. They share one
configuration — #373's cell `arm6_v2_combab_alignT`, depth k = 32, the mean
over the depth copies — and differ in the EMA momentum α, in the backbone
seed, and on one row in the L_align weight. So every guard here asks one of
two questions:

1. Do the arms differ ONLY in the columns of `arms.tsv`, and does each arm's
   α reach the trainer? Arms that share a configuration also share a failure:
   a command line that carries the wrong α trains arm 2 under arm 1's name,
   and no artefact says so. `run_arm.sh` reads α and the reduction back off
   the trainer's own command line for that reason.
2. Does this study write anywhere #373 or #401 wrote? The card compares its
   arms to published numbers, so one overwritten score file is the
   comparison gone.

The card's own deliverables — the reached-momentum figure, the loss curves,
the domain grid and the table — are covered in section 8.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP = REPO_ROOT / "reports" / "2026-08-19_ema_momentum_k32"
PARENT = REPO_ROOT / "reports" / "2026-08-08_rollout_depth"

STUDY_SH = EXP / "scripts" / "study.sh"
ARMS_TSV = EXP / "scripts" / "arms.tsv"
RUN_ARM = EXP / "scripts" / "run_arm.sh"
HEAD_EVAL = EXP / "scripts" / "head_eval.sh"
PHASE1 = EXP / "scripts" / "phase1.sh"
HEADS_WATCH = EXP / "scripts" / "heads_watch.sh"
COLLECT = EXP / "scripts" / "collect.sh"
SMOKE = EXP / "scripts" / "smoke.sh"
MAKE_PLOTS = EXP / "scripts" / "make_plots.sh"
LAUNCH_BOX = EXP / "scripts" / "launch_box.sh"
LAUNCH_ELISA = EXP / "scripts" / "launch_elisa.sh"
LAUNCH_SYNC = EXP / "sync" / "launch_sync.sh"
RUN_SH = EXP / "run.sh"

REFERENCES_PY = EXP / "scripts" / "references.py"
PLOT_MOMENTUM = EXP / "scripts" / "plot_momentum.py"
PLOT_REACHED = EXP / "scripts" / "plot_reached_two_colours.py"
PLOT_CURVES = EXP / "scripts" / "plot_loss_curves.py"
PLOT_GRID = EXP / "scripts" / "plot_domain_grid.py"
PLOT_RANKING = EXP / "scripts" / "plot_arm_ranking.py"
PLOT_HEALTH = EXP / "scripts" / "plot_backbone_health.py"
MAKE_TABLE = EXP / "scripts" / "make_table.py"

PARENT_LEG = PARENT / "scripts" / "run_leg_k.sh"
PARENT_HEAD = PARENT / "scripts" / "head_eval_bb.sh"
TRAIN_PY = REPO_ROOT / "experiments" / "2026-04-27_freq-embedding" / "scripts" / "train.py"

# The card's configuration. One cell, one depth, one reduction, one stop.
CELL = "arm6_v2_combab_alignT"
K = 32
REDUCE = "mean"
STOP = 40_000
HEAD_STEPS = 30_000
ENC = "student"

# Every arm, exactly as `arms.tsv` gives it, row for row and column for
# column:
#   arm, α at step 0, α at the end of the ramp, the ramp length, the backbone
#   seed, and the L_align weight when the arm moves it.
#
# A `-` is a flag the arm does not pass. An arm that holds α fixed passes no
# `--ema-tau-end` at all — a repeated flag cannot unset one. A row of five
# columns takes the cell's own align weight, 1.0.
#
# Rows 1 to 4 are the card's own four arms. The card says to add an arm when
# the scores show a direction, and the study did that over eight rounds, so
# the table now holds fourteen runs over ten settings. `arms.tsv` states why
# each row exists.
ARM_ROWS = (
    ("a08", "0.8", "-", "-", "20260520"),
    ("a09", "0.9", "-", "-", "20260520"),
    ("s08", "0.8", "1.0", "200000", "20260520"),
    ("s09", "0.9", "1.0", "200000", "20260520"),
    ("a095", "0.95", "-", "-", "20260520"),
    ("s08b", "0.8", "1.0", "200000", "20260521"),
    ("s08c", "0.8", "1.0", "200000", "20260522"),
    ("s08d", "0.8", "1.0", "200000", "20260523"),
    ("r100_09", "0.9", "1.0", "100000", "20260520"),
    ("r100_08", "0.8", "1.0", "100000", "20260520"),
    ("w3_s08", "0.8", "1.0", "200000", "20260520", "3.0"),
    ("r60_09", "0.9", "1.0", "60000", "20260520"),
    ("r100_095", "0.95", "1.0", "100000", "20260520"),
    ("r100_09b", "0.9", "1.0", "100000", "20260524"),
)

# The four EMA columns of every arm. That is what the guards, `study.sh` and
# the trainer's command line carry, and it is what most of this file reads.
ARMS = tuple(row[:4] for row in ARM_ROWS)

# The backbone seed and the L_align weight of each arm, for the tables and the
# figures that carry them.
SEED = {row[0]: row[4] for row in ARM_ROWS}
ALIGN_W = {row[0]: (row[5] if len(row) > 5 else "1.0") for row in ARM_ROWS}

# #373's flag block for this cell, which the card reproduces flag for flag.
ISSUE_FLAGS = (
    ("--loss-shape", "cosine_similarity_batch_rep_only"),
    ("--align-loss-weight", "1.0"),
    ("--moco-rep-keys", None),
    ("--tau-rep", "1.0"),
    ("--align-target", "teacher"),
    ("--cpc-infonce-weight", "0.0"),
    ("--ema-embedding", None),
    ("--ema-encoder", None),
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
    """One value of study.sh, read by sourcing it."""
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


def dry_run(script: Path, *args, env=None):
    """A launcher's plan, with nothing run."""
    full = {"CF404_DRY_RUN": "1"}
    full.update(env or {})
    return run_sh(script, *args, env=full)


# --- 1. The layout -----------------------------------------------------------


class TestLayout:
    """REPORT_STANDARD: results/, plots/, scripts/, sync/ under one dir."""

    @pytest.mark.parametrize("sub", ["results", "plots", "scripts", "sync"])
    def test_subdirectory_exists(self, sub):
        assert (EXP / sub).is_dir(), f"{EXP.name}/{sub} missing"

    @pytest.mark.parametrize(
        "script", [STUDY_SH, ARMS_TSV, RUN_ARM, HEAD_EVAL, PHASE1, HEADS_WATCH,
                   COLLECT, SMOKE, MAKE_PLOTS, LAUNCH_BOX, LAUNCH_ELISA,
                   LAUNCH_SYNC, RUN_SH, REFERENCES_PY, PLOT_MOMENTUM,
                   PLOT_CURVES, PLOT_GRID, MAKE_TABLE])
    def test_file_exists(self, script):
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

    @pytest.mark.parametrize("script", sorted(EXP.glob("scripts/*.py")))
    def test_python_script_compiles(self, script):
        out = subprocess.run([sys.executable, "-m", "py_compile", str(script)],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr

    def test_run_sh_covers_every_stage(self):
        """CLAUDE.md: the experiment's launcher is `run.sh` in its own dir."""
        code = strip_comments(RUN_SH.read_text())
        for stage in ("smoke.sh", "phase1.sh", "make_plots.sh"):
            assert stage in code, f"run.sh does not run {stage}"

    def test_the_sync_loop_is_the_parents(self):
        """One set of measured per-class size floors, not a second copy."""
        code = strip_comments(LAUNCH_SYNC.read_text())
        assert "sync_loop.sh" in code
        assert "safe_pull.sh" in code, "raw scp corrupts the prior good copy"

    def test_a_file_with_a_shebang_is_executable(self):
        """One rule for the whole study: a `#!` line means mode 755."""
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


# --- 2. The four arms --------------------------------------------------------


class TestTheArms:
    """The card's table, and nothing else, decides what runs."""

    def test_arms_tsv_holds_the_cards_rows(self):
        rows = [ln.split("\t") for ln in ARMS_TSV.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
        assert tuple(tuple(r) for r in rows) == ARM_ROWS

    def test_study_lists_the_arms_in_the_cards_order(self):
        assert study_value("CF404_ARMS").split() == [a[0] for a in ARMS]

    @pytest.mark.parametrize("arm,tau,end,ramp", ARMS)
    def test_ema_args_carry_the_arms_momentum(self, arm, tau, end, ramp):
        got = study_call(f'cf404_ema_args {arm}').stdout.split()
        assert got[:2] == ["--ema-tau", tau]
        if end == "-":
            # A fixed arm passes no end value. `--ema-tau-end` cannot be
            # unset by repeating it, so the flag has to be absent.
            assert got == ["--ema-tau", tau], got
        else:
            assert got == ["--ema-tau", tau, "--ema-tau-end", end,
                           "--ema-tau-ramp-steps", ramp], got

    def test_the_two_fixed_arms_carry_no_schedule_flag(self):
        for arm in ("a08", "a09"):
            got = study_call(f'cf404_ema_args {arm}').stdout
            assert "--ema-tau-end" not in got
            assert "--ema-tau-ramp-steps" not in got

    def test_the_ramp_is_anchored_to_200k_not_to_the_stop(self):
        """The arms stop at 40k. Without the anchor each schedule would span
        its own --total-steps and reach α = 1.0 at step 40,000 — five times
        the card's ramp."""
        for arm in ("s08", "s09"):
            got = study_call(f'cf404_ema_args {arm}').stdout.split()
            assert "--ema-tau-ramp-steps" in got
            assert got[got.index("--ema-tau-ramp-steps") + 1] == "200000"

    @pytest.mark.parametrize("arm,tau,_e,_r", ARMS)
    def test_alpha_start_is_readable_for_the_figure(self, arm, tau, _e, _r):
        assert study_call(f'cf404_alpha {arm}').stdout.strip() == tau

    def test_a_missing_arms_table_names_the_file(self):
        """Without the table every guard would refuse every arm, with a
        message about the arm rather than about the missing file."""
        out = study_call('true', env={"CF404_ARMS_TSV": "/nonexistent/arms.tsv"})
        assert out.returncode != 0
        assert "arms table" in out.stderr

    def test_an_unknown_arm_is_refused(self):
        out = study_call('cf404_require_arm a07')
        assert out.returncode != 0
        assert "a07" in out.stderr

    def test_every_card_arm_passes_the_guard(self):
        for arm, *_ in ARMS:
            assert study_call(f'cf404_require_arm {arm}').returncode == 0

    def test_the_arm_flags_parse_under_the_trainer(self):
        """The four α values have to survive train.py's own validation:
        `--ema-tau` must sit in (0, 1) and `--ema-tau-end` in (0, 1]."""
        for arm, *_ in ARMS:
            args = (study_call(f'cf404_ema_args {arm}').stdout.split()
                    + ["--weight-decay", "0.1"])
            out = subprocess.run(
                [sys.executable, "-c",
                 f'import sys; sys.path.insert(0, {str(REPO_ROOT)!r});'
                 f'sys.argv = ["train.py"] + {args!r};'
                 f'import importlib.util as u;'
                 f's = u.spec_from_file_location("t", {str(TRAIN_PY)!r});'
                 f'm = u.module_from_spec(s); s.loader.exec_module(m);'
                 f'a = m.parse_args();'
                 f'print(a.ema_tau, a.ema_tau_end, a.ema_tau_ramp_steps)'],
                capture_output=True, text=True, timeout=300)
            assert out.returncode == 0, f"{arm}: {out.stderr[-2000:]}"


# --- 3. The configuration is #373's teacher cell at k = 32 -------------------


class TestConfiguration:

    def test_the_cell_is_the_teacher_align_arm(self):
        assert study_value("CF404_CELL") == CELL

    def test_the_depth_is_32(self):
        assert study_value("CF404_K") == str(K)

    def test_the_reduction_is_the_mean(self):
        assert study_value("CF404_REDUCE") == REDUCE

    def test_the_only_stop_is_40k(self):
        assert study_value("CF404_STOPS").split() == [str(STOP)]

    def test_the_head_budget_is_30k(self):
        assert study_value("CF404_HEAD_STEPS") == str(HEAD_STEPS)

    def test_the_head_reads_the_student_encoder(self):
        assert study_value("CF404_ENC") == ENC

    @pytest.mark.parametrize("flag,value", ISSUE_FLAGS)
    def test_the_parent_cell_carries_the_cards_flag(self, flag, value):
        """The card's configuration block IS #373's `arm6_v2_combab_alignT`
        plus the depth. Reading the parent runner is what keeps the two from
        drifting: this study writes no trainer command line of its own."""
        code = strip_comments(PARENT_LEG.read_text())
        block = code.split(f"{CELL})", 1)
        assert len(block) == 2, f"{CELL} is not a cell of the parent runner"
        # The cell's own block, plus the shared flag block below the case.
        cell_block = block[1].split(";;", 1)[0]
        shared = code.split("esac", 1)[1]
        haystack = cell_block + shared
        assert flag in haystack, f"{flag} is in neither the cell nor the shared block"
        if value is not None:
            assert f"{flag} {value}" in haystack

    def test_the_depth_is_not_taken_from_the_cells_extra_args(self):
        """#373's rule: every cell takes k from the shared flag block."""
        code = strip_comments(PARENT_LEG.read_text())
        assert '--train-rollout-depth "$K"' in code


# --- 4. The runner is #373's, with the schedule replaceable ------------------


PUBLISHED_EMA = "--ema-tau 0.9 --ema-tau-end 1.0 --ema-tau-ramp-steps 100000"


class TestParentRunnerEmaOverride:
    """#373's runner holds the EMA schedule as a fixed line. This card sweeps
    it, so the line becomes one replaceable unit — `EMA_ARGS`.

    It is a replacement and not an append. `GAP_ARGS` appends, and argparse
    keeps the last value on repeat, which covers an arm that CHANGES a flag.
    It cannot cover an arm that REMOVES one, and two of the four arms hold α
    fixed: they pass no `--ema-tau-end` at all.
    """

    def test_the_trainer_call_expands_the_variable(self):
        code = strip_comments(PARENT_LEG.read_text())
        assert '"${EMA_ARGS_ARR[@]}"' in code, (
            "the trainer call still names the EMA flags inline")

    def test_the_default_is_the_published_schedule(self):
        """Every #393 and #373 leg ran these three flags in this order. A
        caller that sets nothing has to get that command line back."""
        code = strip_comments(PARENT_LEG.read_text())
        assert f"${{EMA_ARGS:-{PUBLISHED_EMA}}}" in code

    def test_the_inline_flags_are_gone_from_the_trainer_call(self):
        """A leftover inline `--ema-tau 0.9` after the override would win or
        lose by position, and either way the arm would not be the arm."""
        code = strip_comments(PARENT_LEG.read_text())
        call = code.split("python3 -u", 1)[1].split("rc=$?", 1)[0]
        assert "--ema-tau" not in call


@pytest.fixture
def fake_checkout(tmp_path):
    """A checkout whose `python3` records the trainer's command line.

    `runs_root` refuses a root under /tmp, which is where pytest's tmp_path
    lives, so the durable root of this fixture sits under the home cache.
    That is the guard doing its job, so the fixture obeys it rather than
    working around it.
    """
    import shutil
    import uuid

    base = Path.home() / ".cache" / "cf404-selftest" / uuid.uuid4().hex
    wt = base / "wt"
    runs = base / "runs"
    bin_dir = base / "bin"
    # The leg's log, its claim file and its arms.log land in CF404_RESULTS.
    # Left at the default they would land in the study's committed results/,
    # and a test run would leave four run logs in the working tree.
    results = base / "results"
    for d in (wt / "experiments", runs, bin_dir, results):
        d.mkdir(parents=True)
    (wt / "experiments" / "2026-04-27_freq-embedding").symlink_to(
        REPO_ROOT / "experiments" / "2026-04-27_freq-embedding")
    (wt / "experiments" / "hf_token.txt").write_text("selftest-token\n")
    capture = base / "argv.txt"

    # Records argv, prints the command line the real trainer prints first,
    # and lays the checkpoint the leg is judged by. Everything after the
    # trainer in run_leg_k.sh then runs as it does in a real leg.
    (bin_dir / "python3").write_text(f"""#!/bin/bash
printf '%s\\n' "$@" >{capture!s}
echo "Command line: python3 $*"
save=""; name=""; steps=""
prev=""
for a in "$@"; do
  case "$prev" in
    --save-dir) save="$a" ;;
    --run-name) name="$a" ;;
    --total-steps) steps="$a" ;;
  esac
  prev="$a"
done
[ -n "$save" ] && [ -n "$name" ] && [ -n "$steps" ] && \\
  mkdir -p "$save" && : >"$save/${{name}}_$(( steps / 1000 ))k.pth"
exit 0
""")
    (bin_dir / "python3").chmod(0o755)
    try:
        yield {"wt": wt, "runs": runs, "bin": bin_dir, "capture": capture,
               "results": results}
    finally:
        shutil.rmtree(base, ignore_errors=True)


def trainer_argv(fake, arm, env=None):
    """Run one leg of `arm` against the recording trainer, return its argv."""
    full = {
        "PATH": f"{fake['bin']}:{os.environ['PATH']}",
        "WT": str(fake["wt"]),
        "CF404_ROOT": str(fake["runs"]),
        "CF404_RESULTS": str(fake["results"]),
        "BB_GPU": "0",
    }
    full.update(env or {})
    out = run_sh(RUN_ARM, arm, STOP, env=full)
    argv = (fake["capture"].read_text().splitlines()
            if fake["capture"].exists() else [])
    return out, argv


def argv_value(argv, flag):
    """The value `argparse` would take for `flag`, or None when it is absent.

    The LAST occurrence wins, because the runner repeats a flag on purpose:
    the shared block states `--cpc-infonce-weight 1.0` and the cell's own
    EXTRA_ARGS overrides it with 0.0 further down the line.
    """
    hits = [i for i, a in enumerate(argv) if a == flag]
    return argv[hits[-1] + 1] if hits else None


class TestTheLegTrainsTheArm:
    """The card's whole result is four numbers that differ only in α. An arm
    whose α did not reach the trainer is a duplicate of another arm, under a
    name that says otherwise, and no artefact of the run says so."""

    @pytest.mark.parametrize("arm,tau,end,ramp", ARMS)
    def test_the_arms_momentum_reaches_the_trainer(self, fake_checkout, arm,
                                                   tau, end, ramp):
        out, argv = trainer_argv(fake_checkout, arm)
        assert argv, f"the trainer never ran: rc={out.returncode}\n{out.stderr[-2000:]}"
        assert argv_value(argv, "--ema-tau") == tau
        assert argv_value(argv, "--ema-tau-end") == (None if end == "-" else end)
        assert argv_value(argv, "--ema-tau-ramp-steps") == (
            None if ramp == "-" else ramp)
        assert out.returncode == 0, out.stderr[-2000:]

    def test_a_fixed_arm_carries_no_ramp_flag_at_all(self, fake_checkout):
        """The published default carries `--ema-tau-end 1.0`. An override
        that appended instead of replacing would leave it on the line, and
        the 'fixed' arm would run the schedule."""
        _, argv = trainer_argv(fake_checkout, "a08")
        assert "--ema-tau-end" not in argv
        assert "--ema-tau-ramp-steps" not in argv

    def test_the_depth_and_the_reduction_reach_the_trainer(self, fake_checkout):
        _, argv = trainer_argv(fake_checkout, "a08")
        assert argv_value(argv, "--train-rollout-depth") == str(K)
        assert argv_value(argv, "--train-rollout-reduce") == REDUCE

    @pytest.mark.parametrize("flag,value", ISSUE_FLAGS)
    def test_the_cards_configuration_reaches_the_trainer(self, fake_checkout,
                                                         flag, value):
        _, argv = trainer_argv(fake_checkout, "a08")
        assert flag in argv, f"{flag} never reached the trainer"
        if value is not None:
            assert argv_value(argv, flag) == value

    def test_the_backbone_shape_is_the_parents(self, fake_checkout):
        """d_model 64, 8 heads, 3 encoder layers, 3 layers, batch 64, seed
        20260520, `gift-pretrain-full-4096 / small_v1` — the card pins each
        one, and each one comes from the parent runner."""
        _, argv = trainer_argv(fake_checkout, "a08")
        for flag, value in (("--d-model", "64"), ("--n-heads", "8"),
                            ("--num-encoder-layers", "3"),
                            ("--num-layers", "3"), ("--batch-size", "64"),
                            ("--seed", "20260520"), ("--tau", "0.10"),
                            ("--sigreg-embedding-weight", "1.0"),
                            ("--sigreg-encoding-weight", "1.0"),
                            ("--hf-repo", "jeremycochoy/gift-pretrain-full-4096"),
                            ("--hf-path", "small_v1")):
            assert argv_value(argv, flag) == value, flag
        for flag in ("--sigreg-embedding", "--sigreg-encoding"):
            assert flag in argv

    def test_the_leg_stops_at_40k(self, fake_checkout):
        _, argv = trainer_argv(fake_checkout, "a08")
        assert argv_value(argv, "--total-steps") == str(STOP)

    def test_every_arm_writes_its_own_run_name(self, fake_checkout):
        names = set()
        for arm, *_ in ARMS:
            _, argv = trainer_argv(fake_checkout, arm)
            names.add(argv_value(argv, "--run-name"))
        assert len(names) == len(ARMS), names

    def test_every_arm_writes_its_own_save_dir(self, fake_checkout):
        dirs = set()
        for arm, *_ in ARMS:
            _, argv = trainer_argv(fake_checkout, arm)
            dirs.add(argv_value(argv, "--save-dir"))
        assert len(dirs) == len(ARMS), dirs

    def test_a_leg_whose_alpha_is_wrong_is_stopped(self, fake_checkout):
        """The verification this study adds: the leg reads α back off the
        trainer's own command line. `CF404_FORCE_EMA` makes the wrapper hand
        the trainer another arm's α, which is what a wiring defect does."""
        out, _ = trainer_argv(fake_checkout, "a08",
                              env={"CF404_FORCE_EMA": "--ema-tau 0.9"})
        assert out.returncode != 0
        assert "0.9" in (out.stdout + out.stderr)

    def test_a_second_leg_of_a_finished_stop_is_a_no_op(self, fake_checkout):
        """Idempotent: a re-fired stop after a crash costs nothing."""
        out, _ = trainer_argv(fake_checkout, "a08")
        assert out.returncode == 0
        fake_checkout["capture"].unlink()
        out2, argv2 = trainer_argv(fake_checkout, "a08")
        assert out2.returncode == 0
        assert argv2 == [], "the leg retrained a stop that is on disk"


class TestParentResultsDirectory:
    """#373 pinned its results/ to its own directory, so a study that reused
    the runner wrote its leg log and its score file into #373's. Every
    experiment keeps its own artefacts in its own directory (REPORT_STANDARD),
    so the path is a variable — with #373's own value as the default."""

    @pytest.mark.parametrize("script", [PARENT_LEG, PARENT_HEAD])
    def test_the_results_directory_is_overridable(self, script):
        code = strip_comments(script.read_text())
        assert "${CF_RESULTS:-" in code, f"{script.name} pins its results/"

    @pytest.mark.parametrize("script", [PARENT_LEG, PARENT_HEAD])
    def test_the_default_is_the_parents_own_results(self, script):
        """A caller that sets nothing writes where every published leg of
        #373 and #393 wrote.

        #401 put a second default inside this one — `CF_STUDY_DIR` names the
        study directory, and `CF_RESULTS` names the results directory
        outright. A string match on the text between `${CF_RESULTS:-` and the
        first `}` reads the inner default, not the path. So bash expands the
        assignments here, with both variables unset, and the assertion reads
        the path that comes out."""
        lines = [ln.strip()
                 for ln in strip_comments(script.read_text()).splitlines()
                 if ln.strip().startswith(("OUT=", "RES="))]
        lines = lines[:[ln.startswith("RES=") for ln in lines].index(True) + 1]
        script_sh = "\n".join(
            ["WT=/wt", "unset CF_RESULTS CF_STUDY_DIR"] + lines + ['echo "$RES"'])
        out = subprocess.run(["bash", "-c", script_sh],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "/wt/reports/2026-08-08_rollout_depth/results"

    def test_this_studys_leg_log_is_in_this_studys_results(self):
        got = study_call('cf404_leg_log a08').stdout.strip()
        assert got.startswith(study_value("CF404_RESULTS"))
        assert "2026-08-08_rollout_depth" not in got

    def test_the_head_wrapper_hands_the_results_directory_over(self):
        code = strip_comments(HEAD_EVAL.read_text())
        assert "CF_RESULTS=" in code

    def test_the_arm_wrapper_hands_the_results_directory_over(self):
        code = strip_comments(RUN_ARM.read_text())
        assert "CF_RESULTS=" in code


class TestRunnerReuse:
    """This study writes no trainer command line of its own."""

    def test_run_arm_calls_the_parent_runner(self):
        code = strip_comments(RUN_ARM.read_text())
        assert "run_leg_k.sh" in code

    def test_run_arm_defines_no_trainer_flags_of_its_own(self):
        """A second copy of the configuration is a second protocol."""
        code = strip_comments(RUN_ARM.read_text())
        assert "--loss-shape" not in code
        assert "train.py" not in code

    def test_head_eval_calls_the_parent_head_script(self):
        code = strip_comments(HEAD_EVAL.read_text())
        assert "head_eval_bb.sh" in code

    def test_the_parent_scripts_this_study_reuses_exist(self):
        for path in (PARENT_LEG, PARENT_HEAD):
            assert path.is_file(), path


# --- 5. No artefact of this card can be taken for a published one ------------


class TestTheCommandLineCounter:
    """`cf404_cmdlines` feeds the wait loop that holds a leg until its trainer
    names its own momentum. It has to answer with an integer for a log that
    does not exist yet, which is every first leg."""

    def test_a_missing_log_counts_zero(self):
        assert study_call('cf404_cmdlines /nonexistent/x.log').stdout.strip() == "0"

    def test_a_log_without_a_command_line_counts_zero(self, tmp_path):
        path = tmp_path / "run.log"
        path.write_text("step 1 loss 3.2\n")
        assert study_call(f'cf404_cmdlines {path}').stdout.strip() == "0"

    def test_each_leg_adds_one(self, tmp_path):
        path = tmp_path / "run.log"
        path.write_text("Command line: python3 a\nnoise\nCommand line: python3 b\n")
        assert study_call(f'cf404_cmdlines {path}').stdout.strip() == "2"

    def test_the_wait_loop_compares_two_integers(self, tmp_path):
        """The defect this guards: `[ "" -le 0 ]` aborts with an error, so the
        loop fell through and the momentum went unchecked in silence."""
        path = tmp_path / "absent.log"
        out = study_call(
            f'n="$(cf404_cmdlines {path})"; [ "$n" -le 0 ] && echo COMPARED')
        assert out.returncode == 0, out.stderr
        assert "COMPARED" in out.stdout

    def test_the_last_command_line_is_the_one_read(self, tmp_path):
        """The runner APPENDS, so a resumed cell's log carries one line per
        leg and the last one is this leg's."""
        path = tmp_path / "run.log"
        path.write_text("Command line: python3 t --ema-tau 0.9\n"
                        "Command line: python3 t --ema-tau 0.8\n")
        got = study_call(f'cf404_last_cmdline {path} | cf404_ema_of_cmdline')
        assert got.stdout.strip() == "0.8 - -"

    def test_a_log_with_no_command_line_returns_non_zero(self, tmp_path):
        path = tmp_path / "run.log"
        path.write_text("nothing here\n")
        assert study_call(f'cf404_last_cmdline {path}').returncode != 0


class TestNoCollision:
    """The card compares its four arms to five published numbers. A path this
    study shares with #373 or #401 overwrites one of them."""

    def test_the_run_name_carries_the_arm(self):
        names = {study_call(f'cf404_run_name {a}').stdout.strip()
                 for a, *_ in ARMS}
        assert len(names) == len(ARMS), names
        for name in names:
            assert f"k{K}" in name and REDUCE in name

    def test_the_run_name_carries_the_depth_and_the_reduction(self):
        """#401 trains the same cell family at k = 32 under `mean`. Only the
        cell and the arm tell the two apart, so both have to be in the name."""
        name = study_call('cf404_run_name a08').stdout.strip()
        assert CELL in name

    def test_no_run_name_is_a_prefix_of_another_in_one_leg_directory(self):
        """`ckpt_at_step` globs `<name>*_<N>k.pth` inside ONE leg directory,
        so a name that prefixes another resolves to the other arm's
        checkpoint when the two share that directory.

        Three names of this card DO prefix another. `a09` prefixes `a095`,
        `s08` prefixes `s08b`, `s08c` and `s08d`, and `r100_09` prefixes
        `r100_095` and `r100_09b`. Each of those names says what the arm is,
        which is worth more than a name the glob could tell apart on its own.
        `cf404_arm_root` gives every arm a root of its own, so no two of them
        ever land in one leg directory and the glob never sees the pair. This
        guard is on the directory for that reason, and it fires the moment
        two arms share one.
        """
        by_dir: dict[str, list[str]] = {}
        for arm, *_ in ARMS:
            leg = study_call(f'cf404_leg_dir {arm} {STOP}').stdout.strip()
            by_dir.setdefault(leg, []).append(
                study_call(f'cf404_run_name {arm}').stdout.strip())
        assert len(by_dir) == len(ARMS), sorted(by_dir)
        for leg, names in by_dir.items():
            names = sorted(names)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    assert not b.startswith(a), f"{a!r} prefixes {b!r} in {leg}"

    def test_each_arm_has_its_own_checkpoint_root(self):
        roots = {study_call(f'cf404_arm_root {a}').stdout.strip()
                 for a, *_ in ARMS}
        assert len(roots) == len(ARMS), roots

    def test_the_root_is_not_the_parents(self):
        """One root for two studies is a sync loop that cannot tell their
        checkpoints apart."""
        root = study_value("CF404_ROOT")
        assert "cf-404" in root
        assert "cf-373" not in root and "cf-401" not in root

    def test_the_root_is_durable(self):
        """CLAUDE.md checkpoint safety rule 4: never /tmp, never inside the
        checkout."""
        root = study_value("CF404_ROOT")
        assert not root.startswith("/tmp")
        assert str(REPO_ROOT) not in root

    def test_the_results_directory_is_this_studys(self):
        assert study_value("CF404_RESULTS").startswith(str(EXP))

    def test_the_tag_carries_the_arm(self):
        tags = {study_call(f'cf404_tag {a} {STOP} {HEAD_STEPS}').stdout.strip()
                for a, *_ in ARMS}
        assert len(tags) == len(ARMS), tags
        for tag in tags:
            assert tag.endswith(f"_bb40k_h30k_{ENC}")


# --- 6. The guards -----------------------------------------------------------


class TestGuards:
    """Every guard prints what it refused. A typo that trains for five hours
    is expensive."""

    def test_an_unknown_stop_is_refused(self):
        out = study_call('cf404_require_stop 100000')
        assert out.returncode != 0 and "100000" in out.stderr

    def test_the_card_stop_passes(self):
        assert study_call(f'cf404_require_stop {STOP}').returncode == 0

    def test_an_unknown_head_budget_is_refused(self):
        out = study_call('cf404_require_head_steps 15000')
        assert out.returncode != 0 and "15000" in out.stderr

    def test_run_arm_refuses_an_unknown_arm(self):
        out = run_sh(RUN_ARM, "a07", STOP, env={"CF404_DRY_RUN": "1"})
        assert out.returncode != 0 and "a07" in out.stderr

    def test_head_eval_refuses_an_unknown_arm(self):
        out = run_sh(HEAD_EVAL, "a07", STOP, env={"CF404_DRY_RUN": "1"})
        assert out.returncode != 0 and "a07" in out.stderr

    def test_run_arm_refuses_an_unknown_stop(self):
        out = run_sh(RUN_ARM, "a08", 100_000, env={"CF404_DRY_RUN": "1"})
        assert out.returncode != 0

    def test_a_dry_run_names_the_arm_and_its_momentum(self):
        out = dry_run(RUN_ARM, "s09", STOP)
        assert out.returncode == 0, out.stderr
        assert "s09" in out.stdout
        assert "0.9" in out.stdout and "200000" in out.stdout


# --- 7. The order, and the scores it collects --------------------------------


class TestPhase1Plan:

    def test_phase1_covers_every_arm_once(self):
        out = dry_run(PHASE1)
        assert out.returncode == 0, out.stderr
        for arm, *_ in ARMS:
            assert len([ln for ln in out.stdout.splitlines()
                        if ln.startswith(f"arm {arm} ")]) == 1, out.stdout

    def test_phase1_plans_one_head_per_arm(self):
        out = dry_run(PHASE1)
        heads = [ln for ln in out.stdout.splitlines() if ln.startswith("head ")]
        assert len(heads) == len(ARMS), out.stdout

    def test_the_box_pairs_the_arms_with_its_cards(self):
        """Two GPUs: two arms run at a time, so a box finishes the arms it
        carries in half as many passes. The deal is round-robin, so the two
        lanes never differ by more than one arm."""
        out = dry_run(LAUNCH_BOX, env={"GPUS": "0 1"})
        assert out.returncode == 0, out.stderr
        gpus = [ln.split("gpu=")[1].split()[0]
                for ln in out.stdout.splitlines() if "gpu=" in ln]
        half = len(ARMS) // 2
        assert sorted(gpus) == ["0"] * (len(ARMS) - half) + ["1"] * half, \
            out.stdout

    def test_the_box_checks_its_checkout_before_it_rents_time(self):
        """A box bootstrapped from a branch without `EMA_ARGS` trains one arm
        four times, and says nothing about it for eleven hours."""
        code = strip_comments(LAUNCH_BOX.read_text())
        assert "cf404_check_checkout" in code

    def test_this_checkout_passes_the_check(self):
        assert study_call('cf404_check_checkout').returncode == 0

    def test_a_checkout_without_the_ema_hook_is_refused(self, tmp_path):
        runner = tmp_path / "reports" / "2026-08-08_rollout_depth" / "scripts"
        runner.mkdir(parents=True)
        (runner / "run_leg_k.sh").write_text("# no hook here\n")
        out = study_call(f'cf404_check_checkout {tmp_path}')
        assert out.returncode != 0
        assert "EMA_ARGS" in out.stderr

    def test_a_checkout_without_the_reduction_is_refused(self, tmp_path):
        """Without it the arms train the SUMMED objective, and #401's k = 32
        number the card compares against came from the MEAN one."""
        runner = tmp_path / "reports" / "2026-08-08_rollout_depth" / "scripts"
        runner.mkdir(parents=True)
        (runner / "run_leg_k.sh").write_text("EMA_ARGS_ARR\n")
        trainer = tmp_path / "experiments" / "2026-04-27_freq-embedding" / "scripts"
        trainer.mkdir(parents=True)
        (trainer / "train.py").write_text("# an old trainer\n")
        out = study_call(f'cf404_check_checkout {tmp_path}')
        assert out.returncode != 0
        assert "--train-rollout-reduce" in out.stderr

    def test_the_box_trains_backbones_only(self):
        """The 97-config eval reads gift-eval-data and the gift_eval package,
        and both live on elisa."""
        out = dry_run(LAUNCH_BOX)
        assert "heads=0" in out.stdout


# --- 7b. The watcher that scores each arm as its backbone lands --------------


def wait_for(predicate, timeout=90.0, step=0.2):
    """True as soon as `predicate` holds, False when `timeout` runs out."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


@pytest.fixture
def watch_study(tmp_path):
    """A copy of this study whose head script is a stub.

    `heads_watch.sh` resolves the study from its own directory, so a copy of
    `scripts/` under a fake repository IS the study. Two scripts are replaced:
    `head_eval.sh`, which would train a head and run 97 GIFT-Eval configs, and
    `collect.sh`, which would call #373's splitter.

    The stub reads `results/stub_rc`. `0` writes the score file the real head
    script writes. Any other value writes nothing and exits with it, which is
    what a bad checkpoint, a missing package or a full disk does.
    """
    import shutil

    repo = tmp_path / "fake"
    study = repo / "reports" / "2026-08-19_ema_momentum_k32"
    (repo / "reports").mkdir(parents=True)
    shutil.copytree(EXP / "scripts", study / "scripts")
    (repo / "reports" / "2026-08-08_rollout_depth").symlink_to(PARENT)

    results = study / "results"
    results.mkdir(parents=True)
    root = tmp_path / "root"
    root.mkdir()
    (results / "stub_rc").write_text("0\n")

    (study / "scripts" / "head_eval.sh").write_text("""#!/bin/bash
. "$(dirname "${BASH_SOURCE[0]}")/study.sh"
echo "$1 $2" >>"$CF404_RESULTS/fired.txt"
rc="$(cat "$CF404_RESULTS/stub_rc")"
[ "$rc" = "0" ] || exit "$rc"
echo 1.2345 >"$(cf404_score_file "$1" "$2")"
""")
    (study / "scripts" / "collect.sh").write_text("#!/bin/bash\nexit 0\n")

    def study_says(snippet):
        out = subprocess.run(
            ["bash", "-c",
             f'. "{study}/scripts/study.sh" >/dev/null && {snippet}'],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "CF404_ROOT": str(root),
                 "CF404_RESULTS": str(results)})
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()

    def backbone(arm):
        """Lay the checkpoint file the watcher polls for, as the sync loop
        does. The path comes from the study itself, never from a second copy
        of the naming rule."""
        path = Path(study_says(
            f'printf "%s/%s_%s.pth" "$(cf404_leg_dir {arm} {STOP})"'
            f' "$(cf404_run_name {arm})" "$(cf404_steps_label {STOP})"'))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return path

    procs = []

    def start(env=None):
        full = dict(os.environ)
        full.update({"CF404_ROOT": str(root), "CF404_RESULTS": str(results),
                     "POLL": "1", "HEAD_GPUS": "0"})
        full.update(env or {})
        out = open(results / "watcher.out", "ab")
        proc = subprocess.Popen(
            ["bash", str(study / "scripts" / "heads_watch.sh")],
            stdout=out, stderr=subprocess.STDOUT, env=full)
        procs.append(proc)
        return proc

    def scored():
        return {f.name[len("score_"):].split("_bb", 1)[0]
                for f in results.glob("score_*.txt") if f.stat().st_size}

    def fired():
        path = results / "fired.txt"
        return [ln.split() for ln in path.read_text().splitlines()] \
            if path.exists() else []

    try:
        yield {"study": study, "root": root, "results": results,
               "backbone": backbone, "start": start, "scored": scored,
               "fired": fired,
               "stub_rc": lambda rc: (results / "stub_rc").write_text(f"{rc}\n"),
               "out": lambda: (results / "watcher.out").read_text()}
    finally:
        for proc in procs:
            proc.kill()
            proc.wait(timeout=30)


class TestTheHeadsAreDone:
    """`cf404_heads_done` is the watcher's exit test. It answers one question:
    is any (arm, stop) pair left to fire?"""

    def state(self, tmp_path):
        return {"CF404_RESULTS": str(tmp_path)}

    def done(self, tmp_path, env=None):
        full = self.state(tmp_path)
        full.update(env or {})
        return study_call("cf404_heads_done", env=full).returncode == 0

    def test_the_pair_count_is_the_arms_times_the_stops(self):
        stops = study_value("CF404_STOPS").split()
        assert study_value("CF404_ARMS").split() == [a for a, *_ in ARMS]
        assert study_call("cf404_pair_count").stdout.strip() == str(
            len(ARMS) * len(stops))

    def test_no_score_is_not_done(self, tmp_path):
        assert not self.done(tmp_path)

    def test_one_scored_arm_is_not_the_study(self, tmp_path):
        """The defect this pins. The box hands the four backbones over about
        five hours apart, so at the moment arm 1 is scored the other three
        are still training and the watcher has to keep waiting."""
        write_score(tmp_path, "a08", 1.19)
        assert not self.done(tmp_path)

    def test_every_arm_scored_is_done(self, tmp_path):
        for arm, *_ in ARMS:
            write_score(tmp_path, arm, 1.19)
        assert self.done(tmp_path)

    def test_an_arm_that_used_its_tries_no_longer_blocks(self, tmp_path):
        """A head that fails for a stable reason would otherwise hold a GPU
        lane and the whole watcher for as long as the session runs."""
        scored = {a for a, *_ in ARMS[:2]}
        for arm in scored:
            write_score(tmp_path, arm, 1.19)
        for arm, *_ in ARMS:
            if arm in scored:
                continue
            tries = study_call(f'cf404_tries_file {arm} {STOP}',
                               env=self.state(tmp_path)).stdout.strip()
            Path(tries).write_text("3\n")
        assert self.done(tmp_path, env={"CF404_HEAD_TRIES": "3"})

    def test_a_pair_with_a_score_is_never_exhausted(self, tmp_path):
        """A pair whose head failed twice and passed on the third try is a
        scored pair, not a dropped one."""
        write_score(tmp_path, "a08", 1.19)
        tries = study_call(f'cf404_tries_file a08 {STOP}',
                           env=self.state(tmp_path)).stdout.strip()
        Path(tries).write_text("9\n")
        out = study_call(f'cf404_exhausted a08 {STOP}', env=self.state(tmp_path))
        assert out.returncode != 0

    def test_a_missing_counter_reads_zero(self, tmp_path):
        out = study_call(f'cf404_tries a08 {STOP}', env=self.state(tmp_path))
        assert out.stdout.strip() == "0"

    def test_each_attempt_adds_one(self, tmp_path):
        env = self.state(tmp_path)
        got = [study_call(f'cf404_bump_tries a08 {STOP}', env=env).stdout.strip()
               for _ in range(3)]
        assert got == ["1", "2", "3"]


class TestHeadsWatch:
    """The watcher runs for the whole study. It fires one head per arm as the
    arm lands, and it exits when no pair is left to fire."""

    def test_it_keeps_waiting_while_an_arm_is_still_on_the_box(self, watch_study):
        """The backbones arrive about five hours apart. A watcher that exits
        at the first scored arm leaves every other arm with no head, and
        `launch_elisa.sh`, which waits on it, stops redrawing the figures."""
        import time
        w = watch_study
        first, rest = ARMS[0][0], [a for a, *_ in ARMS[1:]]
        w["backbone"](first)
        proc = w["start"]()
        assert wait_for(lambda: w["scored"]() == {first}), w["out"]()
        time.sleep(3.0)  # three polls at POLL=1
        assert proc.poll() is None, (
            f"the watcher exited with {len(rest)} arms unscored:\n"
            + w["out"]())

        for arm in rest:
            w["backbone"](arm)
        # The watcher runs the pairs one after the other, so the wait covers
        # every arm of the table and not four of them.
        assert wait_for(lambda: proc.poll() is not None,
                        timeout=20.0 * len(ARMS)), w["out"]()
        assert proc.returncode == 0, w["out"]()
        assert w["scored"] () == {a for a, *_ in ARMS}, w["out"]()

    def test_a_head_that_keeps_failing_is_dropped(self, watch_study):
        """A GIFT-Eval that fails for a stable reason must not re-fire every
        POLL seconds for as long as the session runs."""
        from collections import Counter
        w = watch_study
        w["stub_rc"](7)
        for arm, *_ in ARMS:
            w["backbone"](arm)
        proc = w["start"](env={"CF404_HEAD_TRIES": "2"})
        # Two tries of every arm, in series, so the wait scales with the
        # table.
        assert wait_for(lambda: proc.poll() is not None,
                        timeout=20.0 * len(ARMS)), w["out"]()
        assert Counter(arm for arm, _stop in w["fired"]()) == {
            arm: 2 for arm, *_ in ARMS}, w["fired"]()
        assert "GAVE UP" in w["out"](), w["out"]()

    def test_a_dropped_arm_names_the_file_that_lets_it_run_again(self, watch_study):
        w = watch_study
        w["stub_rc"](7)
        w["backbone"]("a08")
        proc = w["start"](env={"CF404_HEAD_TRIES": "1", "ONCE": "1"})
        proc.wait(timeout=90)
        assert "tries_a08" in w["out"](), w["out"]()

    def test_an_arm_that_scores_on_a_later_try_is_kept(self, watch_study):
        """Two failures then a pass is a scored arm. The watcher must not
        count the failures against an arm that finished."""
        w = watch_study
        w["stub_rc"](7)
        w["backbone"]("a08")
        w["start"](env={"ONCE": "1"}).wait(timeout=90)
        w["stub_rc"](0)
        w["start"](env={"ONCE": "1"}).wait(timeout=90)
        assert w["scored"]() == {"a08"}, w["out"]()

    def test_the_dry_run_states_the_try_budget(self, watch_study):
        out = subprocess.run(
            ["bash", str(watch_study["study"] / "scripts" / "heads_watch.sh")],
            capture_output=True, text=True, timeout=90,
            env={**os.environ, "CF404_DRY_RUN": "1",
                 "CF404_ROOT": str(watch_study["root"]),
                 "CF404_RESULTS": str(watch_study["results"])})
        assert out.returncode == 0, out.stderr
        assert "tries" in out.stdout, out.stdout


def write_score(results: Path, arm: str, value: float):
    results.mkdir(parents=True, exist_ok=True)
    tag = f"{arm}_bb40k_h30k_{ENC}"
    (results / f"score_{tag}.txt").write_text(f"{value}\n")


class TestCollect:

    def collect(self, tmp_path, scores):
        results = tmp_path / "results"
        for arm, value in scores.items():
            write_score(results, arm, value)
        out = run_sh(COLLECT, env={"CF404_RESULTS": str(results)})
        assert out.returncode == 0, out.stderr
        with open(results / "scores.csv") as fh:
            return list(csv.DictReader(fh))

    def test_one_row_per_scored_arm(self, tmp_path):
        rows = self.collect(tmp_path, {"a08": 1.19, "s09": 1.12})
        assert {r["arm"] for r in rows} == {"a08", "s09"}

    def test_the_row_carries_the_arms_momentum(self, tmp_path):
        rows = self.collect(tmp_path, {"s08": 1.15})
        assert rows[0]["alpha"] == "0.8"
        assert rows[0]["schedule"] == "ramp"
        assert rows[0]["score"] == "1.15"

    def test_a_fixed_arm_is_marked_fixed(self, tmp_path):
        rows = self.collect(tmp_path, {"a09": 1.15})
        assert rows[0]["schedule"] == "fixed"

    def test_the_row_carries_the_length_of_the_ramp(self, tmp_path):
        """The momentum figure orders the arms of one alpha by it, so a
        fixed arm, #401's 100,000-step ramp and this card's 200,000-step
        ramp read left to right under one tick."""
        rows = self.collect(tmp_path, {"s08": 1.15, "a09": 1.17})
        by_arm = {r["arm"]: r["ramp"] for r in rows}
        assert by_arm == {"s08": "200000", "a09": "0"}

    def test_an_empty_score_file_is_skipped(self, tmp_path):
        """An eval killed between opening and writing leaves one, and a 0.0
        here would be the best GM-Relative MASE the project ever recorded."""
        results = tmp_path / "results"
        write_score(results, "a08", 1.19)
        (results / f"score_a09_bb40k_h30k_{ENC}.txt").write_text("")
        out = run_sh(COLLECT, env={"CF404_RESULTS": str(results)})
        assert out.returncode == 0, out.stderr
        with open(results / "scores.csv") as fh:
            rows = list(csv.DictReader(fh))
        assert {r["arm"] for r in rows} == {"a08"}

    def test_no_score_yet_is_not_a_failure(self, tmp_path):
        """`make_plots.sh` runs this at any point in the study."""
        out = run_sh(COLLECT, env={"CF404_RESULTS": str(tmp_path / "results")})
        assert out.returncode == 0, out.stderr


# --- 8. The card's four deliverables -----------------------------------------


# The published scores the card compares against, from its own table.
K3_BB200K = 1.0660
K3_BB40K = 1.0862
K32_BB200K = 1.1637
K32_BB40K = 1.2082
K0_PARENT_BB40K = 1.1600

# WHERE EACH ONE COMES FROM. Two trace to the rollout-depth report's
# `results/splits.csv`, rows `A4_k3_bb200k_student` and `A4_k3_bb40k_student`.
# The other three come from the issue card and no run in this repository
# measures them. A figure or a table that prints one prints its source, so a
# reader can tell a measured baseline from a quoted one.
MEASURED = "measured, sibling report"
STATED = "stated by the issue card"
SOURCE = {"K3_BB200K": MEASURED, "K3_BB40K": MEASURED,
          "K32_BB200K": STATED, "K32_BB40K": STATED,
          "K0_PARENT_BB40K": STATED}
# The repeat spread of #373, 0.6% to 1.3%, which the band around K3_BB40K
# holds.
SPREAD = (0.006, 0.013)


def references():
    return load_module(REFERENCES_PY, "cf404_references")


def scores_csv(path: Path, by_arm: dict[str, float]):
    """`collect.sh`'s table, for the arms in `by_arm`.

    The columns are the ones `collect.sh` writes, the seed and the L_align
    weight with them. A figure keys its repeat families on those two, so a
    fixture without them hands every arm one seed and one weight.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha = {a: t for a, t, _e, _r in ARMS}
    sched = {a: ("fixed" if e == "-" else "ramp") for a, _t, e, _r in ARMS}
    ramp = {a: ("0" if r == "-" else r) for a, _t, _e, r in ARMS}
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "alpha", "schedule", "ramp", "seed", "align_w",
                    "stop", "head_steps", "encoder", "score"])
        for arm, score in by_arm.items():
            w.writerow([arm, alpha[arm], sched[arm], ramp[arm], SEED[arm],
                        ALIGN_W[arm], STOP, HEAD_STEPS, ENC, f"{score:.4f}"])
    return path


class TestReferences:
    """The five published numbers live in one file. A figure and a table that
    each carried their own copy would drift."""

    @pytest.mark.parametrize("name,value", [
        ("K3_BB200K", K3_BB200K), ("K3_BB40K", K3_BB40K),
        ("K32_BB200K", K32_BB200K), ("K32_BB40K", K32_BB40K),
        ("K0_PARENT_BB40K", K0_PARENT_BB40K)])
    def test_the_published_number(self, name, value):
        assert getattr(references(), name) == pytest.approx(value)

    @pytest.mark.parametrize("name,source", sorted(SOURCE.items()))
    def test_every_reference_says_where_it_comes_from(self, name, source):
        assert references().SOURCE[name] == source

    def test_the_table_carries_the_source_of_every_row(self):
        rows = references().TABLE
        assert len(rows) == len(SOURCE)
        for _label, _value, src in rows:
            assert src in (MEASURED, STATED), src
        assert {src for _l, _v, src in rows} == {MEASURED, STATED}

    def test_the_k0_line_says_the_card_states_it(self):
        """No run in this repository measures 1.1600, so the dashed line on
        four figures has to carry that. The sibling report's k = 0 aggregates
        at the same stop are 1.2189, 1.2025, 1.1513 and 1.2590."""
        assert STATED in references().K0_LINE
        assert STATED not in references().K3_LINE

    def test_the_band_holds_the_373_repeat_spread(self):
        assert references().SPREAD == pytest.approx(SPREAD)

    def test_the_band_brackets_the_k3_score(self):
        lo, hi = references().band_bounds()
        assert lo < K3_BB40K < hi
        assert lo == pytest.approx(K3_BB40K * (1 - SPREAD[1]))
        assert hi == pytest.approx(K3_BB40K * (1 + SPREAD[1]))

    def test_a_score_inside_the_band_is_reported_inside(self):
        r = references()
        assert r.enters_band(K3_BB40K)
        assert r.enters_band(K3_BB40K * 1.005)
        assert not r.enters_band(K32_BB40K)

    def test_the_two_dotted_lines_are_the_200k_runs(self):
        """Both trained to 200,000 steps and this card's arms stop at 40,000,
        so the figure has to say the lines are a reminder."""
        got = {round(v, 4) for _label, v in references().dotted_lines()}
        assert got == {round(K3_BB200K, 4), round(K32_BB200K, 4)}
        for label, _v in references().dotted_lines():
            assert "200" in label


class TestMomentumFigure:
    """`plot_momentum.py` — the momentum at STEP 0 on the x axis.

    THE REPORT NO LONGER EMBEDS THIS FIGURE. The axis stopped separating the
    arms when the card added a second ramp length: `s08` and `r100_08` both
    start at 0.8. `plot_reached_two_colours.py` replaced it, and the class
    below covers that one.

    The module stays, and every guard here with it, because it holds what
    every score figure of this study draws from: `read_scores`, the reference
    lines, the repeat band and the y range. `plot_reached_two_colours.py` and
    `plot_two_axes.py` both import it.
    """

    def draw(self, tmp_path, by_arm):
        mp = load_module(PLOT_MOMENTUM, "cf404_plot_momentum")
        src = scores_csv(tmp_path / "scores.csv", by_arm)
        out = tmp_path / "momentum.png"
        fig, ax = mp.draw(mp.read_scores(src), out)
        return mp, fig, ax, out

    def test_the_x_axis_is_the_ema_momentum(self, tmp_path):
        _mp, _fig, ax, _out = self.draw(tmp_path, {"a08": 1.19, "s09": 1.11})
        assert "momentum" in ax.get_xlabel().lower()
        assert "mase" in ax.get_ylabel().lower()

    def test_the_401_arm_is_drawn_in_grey(self, tmp_path):
        """The card asks for #401's k = 32 arm at 1.2082, already computed at
        bb40k, so a reader sees the starting point of the sweep."""
        mp, _fig, ax, _out = self.draw(tmp_path, {"a08": 1.19})
        ys = [ln.get_ydata()[0] for ln in ax.get_lines()
              if len(ln.get_ydata()) and ln.get_ydata()[0] == pytest.approx(K32_BB40K)]
        assert ys, f"no artist sits at {K32_BB40K}"

    def test_the_401_arm_sits_at_the_momentum_it_ran(self, tmp_path):
        """#401's k = 32 arm is a cell of this sweep, not only a level: it
        ran alpha = 0.9 raised to 1.0 at step 100,000. So it takes an x
        position, between this card's 0.9 fixed and 0.9 raised at 200,000."""
        r = references()
        assert r.K32_BB40K_ALPHA == pytest.approx(0.9)
        assert r.K32_BB40K_RAMP == 100_000
        mp, _fig, ax, _out = self.draw(tmp_path, {"a08": 1.19})
        placed = [(round(float(ln.get_xdata()[0]), 4),
                   round(float(ln.get_ydata()[0]), 4))
                  for ln in ax.get_lines()
                  if len(ln.get_xdata()) == 1 and len(ln.get_ydata()) == 1]
        assert (round(r.K32_BB40K_ALPHA, 4), round(r.K32_BB40K, 4)) in placed, placed

    def test_the_401_momentum_is_on_the_axis_before_any_arm_reaches_it(self, tmp_path):
        """Only a08 is scored here, at alpha 0.8, so the axis would otherwise
        stop short of 0.9 and hide the point the sweep starts from."""
        mp, _fig, ax, _out = self.draw(tmp_path, {"a08": 1.19})
        lo, hi = ax.get_xlim()
        assert lo < 0.8 and hi > references().K32_BB40K_ALPHA

    def test_the_band_is_drawn_around_the_k3_score(self, tmp_path):
        """The band's own y range is what a reader compares an arm against,
        so check the extent and not the count. `axhspan` lays a Rectangle
        whose y and height are in data coordinates."""
        mp, _fig, ax, _out = self.draw(tmp_path, {"a08": 1.19})
        lo, hi = references().band_bounds()
        drawn = [(round(r.get_y(), 6), round(r.get_y() + r.get_height(), 6))
                 for r in ax.patches if hasattr(r, "get_height")]
        assert (round(lo, 6), round(hi, 6)) in drawn, drawn

    def test_both_dotted_lines_are_drawn(self, tmp_path):
        mp, _fig, ax, _out = self.draw(tmp_path, {"a08": 1.19})
        dotted = {round(float(ln.get_ydata()[0]), 4) for ln in ax.get_lines()
                  if ln.get_linestyle() in (":", "dotted")
                  and len(ln.get_ydata())}
        assert {round(K3_BB200K, 4), round(K32_BB200K, 4)} <= dotted, dotted

    def test_two_arms_at_one_momentum_get_two_label_positions(self, tmp_path):
        """a09 and s09 can score within a hair of one another, and one
        offset for every arm then prints two labels on top of each other."""
        _mp, _fig, ax, _out = self.draw(tmp_path, {"a09": 1.1700,
                                                   "s09": 1.1702})
        offsets = [tuple(t.get_position()) for t in ax.texts]
        assert len(set(offsets)) == len(offsets), offsets

    def test_every_label_stays_inside_the_frame(self, tmp_path):
        """An arm at the right edge takes its label on its left. A label that
        runs past the frame is a score a reader cannot read."""
        _mp, fig, ax, _out = self.draw(
            tmp_path, {"a08": 1.19, "a09": 1.17, "s08": 1.14, "s09": 1.11})
        renderer = fig.canvas.get_renderer()
        frame = ax.get_window_extent(renderer)
        for text in ax.texts:
            box = text.get_window_extent(renderer)
            assert frame.x0 <= box.x0 and box.x1 <= frame.x1, text.get_text()

    def test_it_draws_with_one_arm_scored(self, tmp_path):
        """The figure is redrawn every 30 minutes while the study runs."""
        _mp, _fig, _ax, out = self.draw(tmp_path, {"a08": 1.19})
        assert out.is_file()


class TestTheReachedMomentumFigure:
    """Deliverable 1 — `plot_reached_two_colours.py`, which the report embeds
    as `reached_vertical.png`.

    The x axis of `momentum.png` was the momentum at step 0, and it stopped
    separating the arms: `s08` and `r100_08` both start at 0.8 and reach
    0.840 and 0.880 by the stop. This figure puts the REACHED value on the
    axis, and gives one colour to a momentum that holds its value and another
    to one that rises toward 1.0.

    The seeds of one arm are one point, and a bar joins the lowest and the
    highest of them.
    """

    def reached(self, tmp_path, by_arm, vertical=True):
        """Draw the figure `make_plots.sh` draws. Returns its module, its
        rows, the figure, the axes and the file."""
        rt = load_module(PLOT_REACHED, "cf404_plot_reached")
        src = scores_csv(tmp_path / "scores.csv", by_arm)
        rows = rt.MOM.read_scores(src)
        out = tmp_path / "reached.png"
        drawer = rt.draw_vertical if vertical else rt.draw
        fig, ax = drawer(rows, out)
        return rt, rows, fig, ax, out

    def test_every_scored_arm_reaches_the_figure(self, tmp_path):
        """One point per (schedule, reached momentum) cell, and the seeds of
        one arm inside one cell. The figure prints each cell's mean, so an
        arm the grouping drops is an arm a reader cannot find.

        `s08` and `s08b` are one cell here: same momentum, same ramp, two
        backbone seeds. The other four arms are a cell each.
        """
        by_arm = {"a08": 1.19, "a09": 1.17, "s08": 1.14, "s08b": 1.16,
                  "s09": 1.11, "r100_09": 1.15}
        rt, rows, _fig, ax, out = self.reached(tmp_path, by_arm)
        assert out.is_file() and out.stat().st_size > 0
        grid = rt.cells(rows, STOP)
        assert sum(len(v) for v in grid.values()) == len(by_arm), grid
        assert len(grid) == len(by_arm) - 1, grid
        printed = {t.get_text().strip() for t in ax.texts}
        for scores in grid.values():
            mean = sum(scores) / len(scores)
            assert f"{mean:.4f}" in printed, (mean, printed)

    def test_every_reference_line_gives_the_step_count_it_ran(self, tmp_path):
        """`momentum.png` drew two 200,000-step scores as dotted lines, and
        its caption had to say that they are not a fair comparison. This
        figure draws the two references that ran the arms' OWN 40,000 steps,
        and each label carries that number. A reference whose step count is
        not on it is a comparison a reader cannot check.
        """
        _rt, _rows, _fig, ax, _out = self.reached(
            tmp_path, {"a08": 1.19, "s09": 1.11})
        texts = [t.get_text() for t in ax.texts]
        refs = [t for t in texts if "k = " in t]
        assert len(refs) == 2, texts
        for text in refs:
            assert "40,000 steps" in text, text
        assert not any("200,000" in t for t in texts), texts

    def test_a_schedule_arm_and_a_fixed_arm_are_told_apart(self, tmp_path):
        """This axis is the momentum an arm trains against at the stop, and
        it says nothing about how the arm got there. A held 0.9 and a 0.9
        that rose from a lower value are two different runs, so the colour
        and the marker carry the schedule. Both must differ, and the legend
        has to name each one."""
        rt, _rows, _fig, ax, _out = self.reached(
            tmp_path, {"a09": 1.17, "r60_09": 1.15})
        held, rises = rt.KIND["fixed"], rt.KIND["ramp"]
        assert held["colour"] != rises["colour"]
        assert held["marker"] != rises["marker"]
        drawn = {(ln.get_color(), ln.get_marker()) for ln in ax.get_lines()}
        assert (held["colour"], held["marker"]) in drawn, drawn
        assert (rises["colour"], rises["marker"]) in drawn, drawn
        labels = ax.get_legend_handles_labels()[1]
        assert held["label"] in labels, labels
        assert rises["label"] in labels, labels

    def test_no_score_yet_is_refused_with_a_message_not_a_traceback(self, tmp_path):
        """`make_plots.sh` redraws every 30 minutes, from the first hour of
        the study. It prints the LAST line of a failed draw as its SKIP line,
        so a traceback there tells a reader `ValueError: min() arg is an
        empty sequence` instead of what is missing."""
        rt = load_module(PLOT_REACHED, "cf404_plot_reached")
        rows = rt.MOM.read_scores(scores_csv(tmp_path / "scores.csv", {}))
        for drawer in (rt.draw, rt.draw_vertical):
            with pytest.raises(SystemExit):
                drawer(rows, tmp_path / "x.png")

    def test_it_draws_with_one_arm_scored(self, tmp_path):
        """The figure is redrawn every 30 minutes while the study runs."""
        _rt, _rows, _fig, _ax, out = self.reached(tmp_path, {"a08": 1.19})
        assert out.is_file() and out.stat().st_size > 0


def losses_csv(path: Path, n=40, scale=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "loss", "ema_tau"])
        for i in range(1, n + 1):
            w.writerow([i * 100, f"{scale * (10.0 / i):.6f}", "0.8"])
    return path


class TestLossCurves:
    """Deliverable 2 — one curve per arm, log scale on both axes."""

    def draw(self, tmp_path, arms=("a08", "s09")):
        lc = load_module(PLOT_CURVES, "cf404_plot_curves")
        series = []
        for i, arm in enumerate(arms):
            src = losses_csv(tmp_path / arm / "losses.csv", scale=1 + i)
            series.append((arm, lc.read_losses(src)))
        out = tmp_path / "curves.png"
        fig, ax = lc.draw(series, out)
        return lc, fig, ax, out

    def test_both_axes_are_logarithmic(self, tmp_path):
        _lc, _fig, ax, _out = self.draw(tmp_path)
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"

    def test_one_curve_per_run_in_one_grey(self, tmp_path):
        """Every run handed over gets a curve, and every run that held takes
        the SAME grey.

        THE FIGURE HAS TWO PANELS. `w3_s08` multiplies the align term by 3,
        so its loss sits on another scale, and on a shared axes it read as
        the worst run of the study. It takes a panel of its own, so the
        curves are counted over the whole figure and not over one axes.

        THIRTEEN COLOURS WERE THIRTEEN TOO MANY. Six of them fell in one
        green-to-brown family inside a band 0.6 wide, under a thirteen-row
        legend no reader could map to a curve. This figure asks one question
        — did the run hold — and `backbone_health.png` already answers that
        question on the same fourteen runs in one grey and one red.
        """
        arms = [a for a, *_ in ARMS]
        lc, fig, _ax, _out = self.draw(tmp_path, arms)
        # One raw trace and one median curve per run, over both panels.
        drawn = [ln for axes in fig.axes for ln in axes.get_lines()]
        assert len(drawn) == 2 * len(arms), len(drawn)
        assert {ln.get_color() for ln in drawn} == {lc.STABLE_COLOUR}

    def test_the_run_that_fell_is_red_and_the_legend_names_its_auc(
            self, tmp_path):
        """"Chance" is 0.50 and no run of this study reached it. The one that
        fell ends at 0.5745, so that is what the legend says."""
        lc = load_module(PLOT_CURVES, "cf404_plot_curves")
        arms = [a for a, *_ in ARMS]
        series = [(arm, lc.read_losses(
            losses_csv(tmp_path / arm / "losses.csv", scale=1 + i)))
            for i, arm in enumerate(arms)]
        fig, _ax = lc.draw(series, tmp_path / "curves.png",
                           fell={arms[0]: 0.5745})
        red = [ln for axes in fig.axes for ln in axes.get_lines()
               if ln.get_color() == lc.COLLAPSED_COLOUR]
        assert len(red) == 2, len(red)
        labels = [x.get_text() for x in fig.legends[0].get_texts()]
        assert len(labels) == 2, labels
        assert any("0.57" in x for x in labels), labels
        assert not any("chance" in x.lower() for x in labels), labels

    def test_the_figure_is_written(self, tmp_path):
        _lc, _fig, _ax, out = self.draw(tmp_path)
        assert out.is_file() and out.stat().st_size > 0

    def test_the_longest_csv_of_an_arm_wins(self, tmp_path):
        """A re-fired leg writes a second CSV under train.py's `_rN` infix.
        Sort order would return whichever name comes first, which can be the
        leg that died at step 200."""
        lc = load_module(PLOT_CURVES, "cf404_plot_curves")
        arm_dir = tmp_path / "a08" / CELL / "leg_40k"
        losses_csv(arm_dir / "run_r2_losses.csv", n=4)
        losses_csv(arm_dir / "run_losses.csv", n=40)
        got = dict(lc.find_curves(tmp_path))
        assert len(got["a08"]) == 40

    def test_a_zero_step_row_does_not_kill_the_log_axis(self, tmp_path):
        """The trainer's first row can be step 0, which has no place on a log
        axis. Drop that row. A crash at redraw time is worse."""
        lc = load_module(PLOT_CURVES, "cf404_plot_curves")
        path = tmp_path / "z.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "loss"])
            w.writerow([0, "3.0"])
            w.writerow([100, "2.0"])
        assert [s for s, _ in lc.read_losses(path)] == [100]


class TestDomainGrid:
    """Deliverable 3 — the per-domain figure, one row per arm.

    IT WAS A RADAR AND THE RADAR DID NOT WORK. Fourteen arms drew fourteen
    near-equal polygons in seven colours that repeated, under a sixteen-row
    legend, and no reader could map a polygon to a row. The grid gives every
    arm a row of its own and prints every value.
    """

    def splits(self, tmp_path, by_arm):
        path = tmp_path / "splits.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stop", "split", "name", "n", "gm_rel_mase"])
            for arm, base in by_arm.items():
                tag = f"{arm}_bb40k_h30k_{ENC}"
                w.writerow([tag, "all", "all", 97, f"{base:.4f}"])
                for i, dom in enumerate(("Econ/Fin", "Energy", "Nature",
                                         "Sales", "Transport", "Web/CloudOps")):
                    w.writerow([tag, "domain", dom, 12,
                                f"{base + 0.01 * i:.4f}"])
        return path

    def row_labels(self, ax):
        return [t.get_text() for t in ax.get_yticklabels()]

    def cells(self, ax):
        """Every number the figure PRINTS in a cell."""
        return [t.get_text() for t in ax.texts]

    def test_one_row_per_arm(self, tmp_path):
        pg = load_module(PLOT_GRID, "cf404_plot_grid")
        src = self.splits(tmp_path, {"a08": 1.19, "s09": 1.11})
        out = tmp_path / "grid.png"
        _fig, ax = pg.draw(pg.read_splits(src), out)
        assert out.is_file()
        assert len(self.row_labels(ax)) == 3, self.row_labels(ax)

    def test_no_row_carries_an_internal_arm_code(self, tmp_path):
        """A reader cannot look up `a08`. Every row names the momentum and
        the schedule, out of `arms.tsv`."""
        pg = load_module(PLOT_GRID, "cf404_plot_grid")
        src = self.splits(tmp_path, {"a08": 1.19, "s09": 1.11})
        _fig, ax = pg.draw(pg.read_splits(src), tmp_path / "grid.png")
        for label in self.row_labels(ax):
            assert "a08" not in label and "s09" not in label, label

    def test_the_best_arm_is_the_first_arm_row(self, tmp_path):
        """The row order IS the ranking, so the grid and the ranking figure
        cannot disagree."""
        pg = load_module(PLOT_GRID, "cf404_plot_grid")
        src = self.splits(tmp_path, {"a08": 1.19, "s09": 1.11})
        _fig, ax = pg.draw(pg.read_splits(src), tmp_path / "grid.png",
                           reference={})
        assert "0.9" in self.row_labels(ax)[0], self.row_labels(ax)

    def test_every_cell_prints_its_own_value(self, tmp_path):
        """Colour alone cannot be read to four figures. The number is there."""
        pg = load_module(PLOT_GRID, "cf404_plot_grid")
        src = self.splits(tmp_path, {"a08": 1.19})
        by_arm = pg.read_splits(src)
        _fig, ax = pg.draw(by_arm, tmp_path / "grid.png", reference={})
        printed = self.cells(ax)
        assert len(printed) == len(by_arm["a08"]), printed
        assert f"{by_arm['a08']['Econ/Fin']:.2f}" in printed, printed

    def test_the_aggregate_is_a_column_of_its_own(self, tmp_path):
        """The 97-config score ties a row to the ranking figure. It is not a
        domain, so it takes the last column."""
        pg = load_module(PLOT_GRID, "cf404_plot_grid")
        src = self.splits(tmp_path, {"a08": 1.19})
        _fig, ax = pg.draw(pg.read_splits(src), tmp_path / "grid.png",
                           reference={})
        columns = [t.get_text() for t in ax.get_xticklabels()]
        assert columns[-1] == pg.ALL, columns

    def test_the_k3_row_is_drawn(self, tmp_path):
        """Without k = 3 on the figure a reader cannot see where the arms sit
        against the score they have to beat."""
        pg = load_module(PLOT_GRID, "cf404_plot_grid")
        src = self.splits(tmp_path, {"a08": 1.19, "s09": 1.11})
        _fig, ax = pg.draw(pg.read_splits(src), tmp_path / "grid.png")
        assert "k = 3" in self.row_labels(ax)[0], self.row_labels(ax)

    def test_the_reference_is_the_run_behind_the_cards_number(self):
        """The row comes out of #373's own committed table. Its 97-config
        aggregate has to BE the card's `k = 3, bb40k` row, or the row is
        another run under the name of that one."""
        pg = load_module(PLOT_GRID, "cf404_plot_grid")
        assert Path(pg.REFERENCE_SPLITS).is_file(), pg.REFERENCE_SPLITS
        _domains, whole = pg._rows(pg.REFERENCE_SPLITS, pg.REFERENCE_KEY)
        assert round(whole, 4) == round(K3_BB40K, 4), whole
        assert len(pg.read_reference()) >= 6

    def test_a_reference_under_another_key_is_dropped(self, tmp_path, capsys):
        """A wrong key would draw another run as k = 3. The figure refuses it
        and says so, rather than drawing a reference nobody can check."""
        pg = load_module(PLOT_GRID, "cf404_plot_grid")
        got = pg.read_reference(pg.REFERENCE_SPLITS, "A4_k3_bb200k_student")
        assert got == {}
        assert f"{K3_BB40K:.4f}" in capsys.readouterr().err

    def test_a_missing_reference_table_still_draws(self, tmp_path, capsys):
        """`make_plots.sh` redraws every 30 minutes. A checkout without #373's
        table gives a figure without that row, not a stack trace."""
        pg = load_module(PLOT_GRID, "cf404_plot_grid")
        assert pg.read_reference(tmp_path / "nope.csv") == {}
        assert "WARN" in capsys.readouterr().err
        src = self.splits(tmp_path, {"a08": 1.19})
        _fig, ax = pg.draw(pg.read_splits(src), tmp_path / "grid.png",
                           reference={})
        assert (tmp_path / "grid.png").is_file()
        assert not any("k = 3" in v for v in self.row_labels(ax))

    def test_the_domains_come_from_the_eval_not_from_a_list(self, tmp_path):
        pg = load_module(PLOT_GRID, "cf404_plot_grid")
        src = self.splits(tmp_path, {"a08": 1.19})
        got = pg.read_splits(src)
        assert set(got["a08"]) == {"Econ/Fin", "Energy", "Nature", "Sales",
                                   "Transport", "Web/CloudOps", pg.ALL}

    def test_a_hole_in_an_arms_table_leaves_the_cell_empty(self, tmp_path):
        """A domain an arm has no row for prints no number. A value there
        would read as a measured score."""
        pg = load_module(PLOT_GRID, "cf404_plot_grid")
        src = self.splits(tmp_path, {"a08": 1.19})
        by_arm = pg.read_splits(src)
        del by_arm["a08"]["Energy"]
        _fig, ax = pg.draw(by_arm, tmp_path / "grid.png", reference={})
        assert len(self.cells(ax)) == len(by_arm["a08"])

    def test_one_wild_cell_does_not_wash_out_the_others(self, tmp_path):
        """A collapsed backbone reaches 3.90 on one domain. A colour span set
        by it drew every other cell near white."""
        pg = load_module(PLOT_GRID, "cf404_plot_grid")
        src = self.splits(tmp_path, {"a08": 1.19, "s09": 1.11})
        by_arm = pg.read_splits(src)
        by_arm["a08"]["Econ/Fin"] = 3.90
        _fig, ax = pg.draw(by_arm, tmp_path / "grid.png", reference={})
        span = ax.get_images()[0].get_clim()[1]
        assert span < math.log2(3.90), span


class TestTheTableAndTheStatement:
    """Deliverable 4 — the table, and the one sentence that reads it."""

    def rows(self, tmp_path, by_arm):
        mt = load_module(MAKE_TABLE, "cf404_make_table")
        return mt, mt.read_scores(scores_csv(tmp_path / "scores.csv", by_arm))

    def test_the_table_holds_every_arm_and_every_reference(self, tmp_path):
        mt, rows = self.rows(tmp_path, {
            arm: 1.19 - 0.001 * i for i, (arm, *_) in enumerate(ARMS)})
        md = mt.table_markdown(rows)
        for arm, *_ in ARMS:
            assert arm in md
        for value in (K3_BB200K, K3_BB40K, K32_BB200K, K32_BB40K,
                      K0_PARENT_BB40K):
            assert f"{value:.4f}" in md, value

    def test_the_statement_names_the_winning_momentum(self, tmp_path):
        mt, rows = self.rows(tmp_path, {"a08": 1.19, "s09": 1.11})
        said = mt.statement(rows)
        assert "s09" in said or "0.9" in said

    def test_the_statement_gives_the_distance_to_the_k3_bb40k_score(self, tmp_path):
        mt, rows = self.rows(tmp_path, {"a08": 1.1962})
        said = mt.statement(rows)
        assert f"{K3_BB40K:.4f}" in said
        assert f"{1.1962 - K3_BB40K:+.4f}" in said

    def test_the_statement_says_when_no_arm_beats_the_reference(self, tmp_path):
        """"Beats", not "goes below". A lower GM-Relative MASE is a BETTER
        score, so "goes below" reads as "is worse than" to a reader who has
        not just checked the direction of the axis."""
        mt, rows = self.rows(tmp_path, {"a08": 1.19, "s09": 1.11})
        said = mt.statement(rows).lower()
        assert "does not beat" in said, said
        assert "below" not in said, said

    def test_the_statement_says_when_an_arm_does_beat_it(self, tmp_path):
        mt, rows = self.rows(tmp_path, {"a08": 1.0700})
        said = mt.statement(rows).lower()
        assert "beats" in said, said
        assert "does not beat" not in said, said

    def test_the_verdict_flips_on_the_k3_bb40k_score(self, tmp_path):
        mt, above = self.rows(tmp_path, {"a08": K3_BB40K + 0.01})
        _mt, below = self.rows(tmp_path, {"a08": K3_BB40K - 0.01})
        assert mt.beats_k3(above) is False
        assert mt.beats_k3(below) is True

    def test_no_score_is_not_a_crash(self, tmp_path):
        mt, rows = self.rows(tmp_path, {})
        with pytest.raises(SystemExit):
            mt.statement(rows)


# --- 8b. One arm, one name ---------------------------------------------------


class TestOneArmOneName:
    """One arm carries ONE name, in the table and on every figure.

    ONE SCHEDULE ONCE CARRIED THREE NAMES. The backbone-health figure said
    "0.9 constant", the ranking figure said "0.9 held", and the table said
    "0.9, fixed". A reader who met the first name and then the third had to
    stop and decide whether the two figures drew one arm. The L_align weight
    carried three names the same way.

    `plot_backbone_health.schedule_label` and `plot_backbone_health.align_label`
    hold the one name now, in the table's own words, and every figure that
    labels an arm calls them.
    """

    # Every figure script that puts an arm's name on the page. The health
    # figure holds the two functions. The other two read them.
    READERS = ("plot_arm_ranking.py", "plot_domain_grid.py")

    # The names the study dropped, each as the whole literal that carried it:
    # `f"{alpha:g} held"` puts " held" in the module on its own. A script that
    # brings one back splits one arm into two for the reader.
    DEAD = (" constant", " held", " rises to 1.0 over ",
            " rising to 1.0 at ", ", L_align x", ", align weight ")

    def health(self):
        return load_module(PLOT_HEALTH, "cf404_health_label")

    def row(self, arm, alpha, end, ramp):
        """One arm of `arms.tsv`, in the columns `scores.csv` gives."""
        return {"arm": arm, "alpha": float(alpha),
                "schedule": "fixed" if end == "-" else "ramp",
                "ramp": 0 if ramp == "-" else int(ramp),
                "align_w": float(ALIGN_W[arm])}

    def literals(self, path):
        """Every string the module can print. Docstrings are not printed."""
        tree = ast.parse(path.read_text())
        docs = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)) or not body:
                continue
            first = body[0]
            if isinstance(first, ast.Expr) and \
                    isinstance(first.value, ast.Constant) and \
                    isinstance(first.value.value, str):
                docs.add(id(first.value))
        return [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and id(n) not in docs]

    def test_a_fixed_arm_carries_the_tables_words(self):
        assert self.health().schedule_label(0.9, 0) == "0.9, fixed"

    def test_a_ramp_arm_carries_the_tables_words(self):
        h = self.health()
        assert h.schedule_label(0.9, 100_000) == "0.9, to 1.0 at 100k"
        assert h.schedule_label(0.8, 200_000) == "0.8, to 1.0 at 200k"

    def test_the_align_weight_carries_the_tables_column(self):
        h = self.health()
        assert h.align_label(3.0) == ", L_align weight 3"

    def test_the_cells_own_weight_names_nothing(self):
        """Thirteen arms hold 1.0. A weight that separates no arm from another
        belongs in no label."""
        assert self.health().align_label(1.0) == ""

    @pytest.mark.parametrize("arm,alpha,end,ramp", ARMS)
    def test_the_table_cell_and_the_figure_label_are_one_string(
            self, arm, alpha, end, ramp):
        """`make_table` writes the momentum column. The figures write the row
        label. Both are this one function's output, so they cannot drift."""
        h = self.health()
        mt = load_module(MAKE_TABLE, "cf404_make_table_label")
        r = self.row(arm, alpha, end, ramp)
        cell = f"{r['alpha']:g}, {mt.schedule_text(r)}"
        assert cell == h.schedule_label(r["alpha"], r["ramp"]), arm

    @pytest.mark.parametrize("arm,alpha,end,ramp", ARMS)
    def test_one_arm_gives_one_label_on_every_figure(
            self, arm, alpha, end, ramp):
        """The ranking figure and the health figure open one arm's row with
        one string, and both name its L_align weight the same way."""
        h = self.health()
        rank = load_module(PLOT_RANKING, "cf404_rank_label")
        r = self.row(arm, alpha, end, ramp)
        name = h.schedule_label(r["alpha"], r["ramp"])
        weight = h.align_label(r["align_w"])
        on_health = dict(h.arms())[arm]
        on_ranking = rank.arm_label(r)
        assert on_health.startswith(name), (arm, on_health)
        assert on_ranking.startswith(name), (arm, on_ranking)
        assert on_health.endswith(weight), (arm, on_health)
        assert weight in on_ranking, (arm, on_ranking)

    @pytest.mark.parametrize("script", READERS)
    def test_every_labelling_figure_reads_the_one_function(self, script):
        code = (EXP / "scripts" / script).read_text()
        assert "plot_backbone_health" in code, script

    @pytest.mark.parametrize(
        "script", ("plot_arm_ranking.py", "plot_domain_grid.py",
                   "plot_backbone_health.py"))
    def test_no_figure_writes_a_name_of_its_own(self, script):
        """A dropped name that comes back is the defect again."""
        for text in self.literals(EXP / "scripts" / script):
            assert text not in self.DEAD, (script, text)


class TestMakePlots:
    """The redraw runs at any point in the study, including before any arm is
    scored, and skips a figure with no input rather than failing."""

    def test_it_survives_an_empty_results_directory(self, tmp_path):
        out = run_sh(MAKE_PLOTS, env={"CF404_RESULTS": str(tmp_path / "results"),
                                      "CF404_PLOTS": str(tmp_path / "plots")})
        assert out.returncode == 0, out.stderr
        assert "SKIP" in out.stdout

    def test_it_draws_every_deliverable(self):
        code = strip_comments(MAKE_PLOTS.read_text())
        for script in ("plot_arm_ranking.py", "plot_reached_two_colours.py",
                       "plot_two_axes.py", "plot_loss_curves.py",
                       "plot_domain_grid.py", "make_table.py",
                       "plot_backbone_health.py"):
            assert script in code, script
