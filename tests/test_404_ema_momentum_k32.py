"""Tests for #404: an EMA-momentum sweep for L_align on the teacher, at k = 32.

The card trains four arms. They share one configuration — #373's cell
`arm6_v2_combab_alignT`, depth k = 32, the mean over the depth copies — and
differ in one hyperparameter, the EMA momentum α. So every guard here asks
one of two questions:

1. Do the four arms differ ONLY in α, and does each arm's α reach the
   trainer? Four arms that share a configuration also share a failure: a
   command line that carries the wrong α trains arm 2 under arm 1's name,
   and no artefact says so. `run_arm.sh` reads α and the reduction back off
   the trainer's own command line for that reason.
2. Does this study write anywhere #373 or #401 wrote? The card compares its
   arms to published numbers, so one overwritten score file is the
   comparison gone.

The card's own deliverables — the momentum figure, the loss curves, the
radar and the table — are covered in section 8.
"""

from __future__ import annotations

import csv
import importlib.util
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
PLOT_CURVES = EXP / "scripts" / "plot_loss_curves.py"
PLOT_RADAR = EXP / "scripts" / "plot_domain_radar.py"
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

# The four arms, exactly as the card's table gives them:
#   arm, α at step 0, α at the end of the ramp, the ramp length.
# A `-` is a flag the arm does not pass. Arms 1 and 2 hold α fixed, so they
# pass no `--ema-tau-end` at all — a repeated flag cannot unset one.
ARMS = (
    ("a08", "0.8", "-", "-"),
    ("a09", "0.9", "-", "-"),
    ("s08", "0.8", "1.0", "200000"),
    ("s09", "0.9", "1.0", "200000"),
)

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
                   PLOT_CURVES, PLOT_RADAR, MAKE_TABLE])
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

    def test_arms_tsv_holds_the_cards_four_rows(self):
        rows = [ln.split("\t") for ln in ARMS_TSV.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
        assert tuple(tuple(r) for r in rows) == ARMS

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

    def test_the_four_arms_write_four_run_names(self, fake_checkout):
        names = set()
        for arm, *_ in ARMS:
            _, argv = trainer_argv(fake_checkout, arm)
            names.add(argv_value(argv, "--run-name"))
        assert len(names) == len(ARMS), names

    def test_the_four_arms_write_four_save_dirs(self, fake_checkout):
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
        #373 and #393 wrote."""
        code = strip_comments(script.read_text())
        default = code.split("${CF_RESULTS:-", 1)[1].split("}", 1)[0]
        assert default.endswith("2026-08-08_rollout_depth/results") or \
            default.endswith("$OUT/results"), default

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

    def test_no_run_name_is_a_prefix_of_another(self):
        """`ckpt_at_step` globs `<name>*_<N>k.pth`, so a name that prefixes
        another resolves to the other arm's checkpoint."""
        names = sorted(study_call(f'cf404_run_name {a}').stdout.strip()
                       for a, *_ in ARMS)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert not b.startswith(a), f"{a!r} prefixes {b!r}"

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
        """Two GPUs, four arms: two run at a time, so the card finishes in
        two passes rather than four."""
        out = dry_run(LAUNCH_BOX, env={"GPUS": "0 1"})
        assert out.returncode == 0, out.stderr
        gpus = [ln.split("gpu=")[1].split()[0]
                for ln in out.stdout.splitlines() if "gpu=" in ln]
        assert sorted(gpus) == ["0", "0", "1", "1"], out.stdout

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
# The repeat spread of #373, 0.6% to 1.3%, which the band around K3_BB40K
# holds.
SPREAD = (0.006, 0.013)


def references():
    return load_module(REFERENCES_PY, "cf404_references")


def scores_csv(path: Path, by_arm: dict[str, float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha = {a: t for a, t, _e, _r in ARMS}
    sched = {a: ("fixed" if e == "-" else "ramp") for a, _t, e, _r in ARMS}
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "alpha", "schedule", "stop", "head_steps",
                    "encoder", "score"])
        for arm, score in by_arm.items():
            w.writerow([arm, alpha[arm], sched[arm], STOP, HEAD_STEPS, ENC,
                        f"{score:.4f}"])
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
    """Deliverable 1 — one point per arm, against the EMA momentum."""

    def draw(self, tmp_path, by_arm):
        mp = load_module(PLOT_MOMENTUM, "cf404_plot_momentum")
        src = scores_csv(tmp_path / "scores.csv", by_arm)
        out = tmp_path / "momentum.png"
        fig, ax = mp.draw(mp.read_scores(src), out)
        return mp, fig, ax, out

    def test_it_draws_one_point_per_arm(self, tmp_path):
        _mp, _fig, ax, out = self.draw(
            tmp_path, {"a08": 1.19, "a09": 1.17, "s08": 1.14, "s09": 1.11})
        assert out.is_file() and out.stat().st_size > 0
        drawn = {t.get_text() for t in ax.texts}
        for arm, *_ in ARMS:
            assert any(arm in d for d in drawn), f"{arm} is not labelled"

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

    def test_the_caption_says_the_dotted_lines_are_not_a_fair_comparison(self):
        """The card asks for this in the caption, not in the report body."""
        text = PLOT_MOMENTUM.read_text()
        assert "CAPTION" in text
        caption = text.split("CAPTION", 1)[1]
        assert "200" in caption and "40" in caption

    def test_a_schedule_arm_and_a_fixed_arm_are_told_apart(self, tmp_path):
        """s08 and a08 share α = 0.8 at step 0, so the marker has to carry
        the schedule or the two points land on one another unexplained."""
        mp, _fig, ax, _out = self.draw(tmp_path, {"a08": 1.19, "s08": 1.14})
        labels = {t.get_text() for t in ax.texts} | {
            ln.get_label() for ln in ax.get_lines()}
        assert any("fixed" in str(v) for v in labels), labels
        assert any("200k" in str(v) or "ramp" in str(v) for v in labels), labels

    def test_it_draws_with_one_arm_scored(self, tmp_path):
        """The figure is redrawn every 30 minutes while the study runs."""
        _mp, _fig, _ax, out = self.draw(tmp_path, {"a08": 1.19})
        assert out.is_file()

    def test_no_score_yet_is_refused_with_a_message_not_a_traceback(self, tmp_path):
        mp = load_module(PLOT_MOMENTUM, "cf404_plot_momentum")
        empty = scores_csv(tmp_path / "scores.csv", {})
        with pytest.raises(SystemExit):
            mp.draw(mp.read_scores(empty), tmp_path / "x.png")


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

    def test_one_curve_per_arm(self, tmp_path):
        _lc, _fig, ax, _out = self.draw(tmp_path, ("a08", "a09", "s08", "s09"))
        labels = {ln.get_label() for ln in ax.get_lines()}
        for arm, *_ in ARMS:
            assert any(arm in str(v) for v in labels), labels

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


class TestDomainRadar:
    """Deliverable 3 — the per-domain figure, as in #373 and #401."""

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

    def test_one_polygon_per_arm(self, tmp_path):
        pr = load_module(PLOT_RADAR, "cf404_plot_radar")
        src = self.splits(tmp_path, {"a08": 1.19, "s09": 1.11})
        out = tmp_path / "radar.png"
        fig, ax = pr.draw(pr.read_splits(src), out)
        assert out.is_file()
        labels = {str(ln.get_label()) for ln in ax.get_lines()}
        assert any("a08" in v for v in labels) and any("s09" in v for v in labels)

    def test_the_radial_axis_holds_the_data_range(self, tmp_path):
        """Every arm sits between about 1.0 and 1.3. An axis from 0 draws the
        four polygons on top of one another and the figure says nothing."""
        pr = load_module(PLOT_RADAR, "cf404_plot_radar")
        src = self.splits(tmp_path, {"a08": 1.19, "s09": 1.11})
        fig, ax = pr.draw(pr.read_splits(src), tmp_path / "radar.png")
        lo, hi = ax.get_ylim()
        assert lo > 0.5, f"the radial axis starts at {lo}"
        assert hi < 1.6

    def test_the_domains_come_from_the_eval_not_from_a_list(self, tmp_path):
        pr = load_module(PLOT_RADAR, "cf404_plot_radar")
        src = self.splits(tmp_path, {"a08": 1.19})
        got = pr.read_splits(src)
        assert set(got["a08"]) == {"Econ/Fin", "Energy", "Nature", "Sales",
                                   "Transport", "Web/CloudOps"}


class TestTheTableAndTheStatement:
    """Deliverable 4 — the table, and the one sentence that reads it."""

    def rows(self, tmp_path, by_arm):
        mt = load_module(MAKE_TABLE, "cf404_make_table")
        return mt, mt.read_scores(scores_csv(tmp_path / "scores.csv", by_arm))

    def test_the_table_holds_every_arm_and_every_reference(self, tmp_path):
        mt, rows = self.rows(tmp_path, {"a08": 1.19, "a09": 1.17,
                                        "s08": 1.14, "s09": 1.11})
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

    def test_the_statement_says_when_no_arm_goes_below(self, tmp_path):
        mt, rows = self.rows(tmp_path, {"a08": 1.19, "s09": 1.11})
        assert "below" in mt.statement(rows).lower()

    def test_the_statement_says_when_an_arm_does_go_below(self, tmp_path):
        mt, rows = self.rows(tmp_path, {"a08": 1.0700})
        said = mt.statement(rows).lower()
        assert "below" in said

    def test_the_verdict_flips_on_the_k3_bb40k_score(self, tmp_path):
        mt, above = self.rows(tmp_path, {"a08": K3_BB40K + 0.01})
        _mt, below = self.rows(tmp_path, {"a08": K3_BB40K - 0.01})
        assert mt.beats_k3(above) is False
        assert mt.beats_k3(below) is True

    def test_no_score_is_not_a_crash(self, tmp_path):
        mt, rows = self.rows(tmp_path, {})
        with pytest.raises(SystemExit):
            mt.statement(rows)


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
        for script in ("plot_momentum.py", "plot_loss_curves.py",
                       "plot_domain_radar.py", "make_table.py"):
            assert script in code, script
