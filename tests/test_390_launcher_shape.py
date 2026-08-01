"""Consistency tests for the #390 arm launcher.

`experiments/2026-08-01_lalign_teacher/scripts/run_arm.sh` retrains the 10
#379 cells that carry an L_align term, with the term pointed at the EMA
teacher. The issue's contract is exact: *each cell keeps its #379 command
line and adds `--align-target teacher`. Nothing else changes.*

So these tests compare the two launchers directly rather than re-listing
flags:

  * per arm, the new case block's flags are the #379 case block's flags
    plus `--align-target teacher`, and nothing else;
  * the shared trainer invocation (batch size, architecture, schedule,
    dataset, dtypes) is character-identical to #379's;
  * only the 10 L_align cells are present — the 20 copied cells are not
    reachable from this launcher;
  * checkpoint names cannot collide with #379's.

The rest of the experiment directory is pinned too. `monitor.sh` and
`sync/sync_loop.sh` have to name every run they guard, and a name table
copied by hand into three files is how a sync loop silently stops pulling an
arm (CLAUDE.md § Remote Machine Monitoring). Both derive their names from
`scripts/arm_names.sh`, and `bb_name()` is checked here against run_arm.sh's
case block for all 10 arms.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "2026-08-01_lalign_teacher"
OLD_LAUNCHER = (REPO_ROOT / "experiments" / "2026-07-21_split_pred_rep_small"
                / "scripts" / "run_arm.sh")
NEW_LAUNCHER = EXP_DIR / "scripts" / "run_arm.sh"
ARM_NAMES_SH = EXP_DIR / "scripts" / "arm_names.sh"
SMOKE_SH = EXP_DIR / "scripts" / "smoke.sh"
ORCHESTRATE_SH = EXP_DIR / "scripts" / "orchestrate.sh"
MONITOR_SH = EXP_DIR / "scripts" / "monitor.sh"
SYNC_LOOP_SH = EXP_DIR / "sync" / "sync_loop.sh"

# The 10 retrained cells: the two L_align arms × the five #379 settings.
ARMS = ("arm5", "arm5_tr1", "arm5_nse", "arm5_ncpc", "arm5_combab",
        "arm6_v2", "arm6_v2_tr1", "arm6_v2_nse", "arm6_v2_ncpc",
        "arm6_v2_combab")

# Cells whose loss has no L_align term — #379's numbers stand, so this
# launcher must not be able to retrain them.
COPIED_ARMS = ("arm1", "arm3", "arm4", "bimoco",
               "arm1_tr1", "arm3_tr1", "arm4_tr1", "bimoco_tr1",
               "arm1_nse", "arm3_nse", "arm4_nse", "bimoco_nse",
               "arm1_ncpc", "arm3_ncpc", "arm4_ncpc", "bimoco_ncpc",
               "arm1_combab", "arm3_combab", "arm4_combab", "bimoco_combab")

ADDED_FLAG = ["--align-target", "teacher"]


def strip_comments(text: str) -> str:
    """Remove full-line bash comments so token-search sees only code."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


@pytest.fixture(scope="module")
def new_source() -> str:
    return NEW_LAUNCHER.read_text()


@pytest.fixture(scope="module")
def new_code(new_source: str) -> str:
    return strip_comments(new_source)


@pytest.fixture(scope="module")
def old_code() -> str:
    return strip_comments(OLD_LAUNCHER.read_text())


def extract_arm_case_body(code: str, arm: str) -> str:
    """Return the body of the `case "$ARM" in ... <arm>) ... ;;` block."""
    m = re.search(rf'(?m)^\s*{re.escape(arm)}\)\s*\n(.*?)\n\s*;;', code,
                  re.DOTALL)
    assert m is not None, f"no case body for arm {arm!r}"
    return m.group(1)


def array_tokens(body: str, name: str) -> list[str]:
    """Flatten a `NAME=(…)` bash array into its whitespace-separated tokens.

    Line continuations are joined first, so a multi-line array reads the
    same as a single-line one.
    """
    m = re.search(rf'{name}=\((.*?)\)', body.replace("\\\n", " "), re.DOTALL)
    return m.group(1).split() if m else []


def flags_of(code: str, arm: str) -> list[str]:
    body = extract_arm_case_body(code, arm)
    return array_tokens(body, "LOSS_ARGS") + array_tokens(body, "EXTRA_ARGS")


def trainer_block(code: str) -> str:
    m = re.search(r'(CUDA_VISIBLE_DEVICES=.*?>>"\$tlog" 2>&1)', code,
                  re.DOTALL)
    assert m is not None, "no trainer invocation found in launcher"
    return m.group(1)


def test_launcher_exists(new_source: str):
    assert new_source, "run_arm.sh is empty"
    assert 'ARM="${1:?' in new_source, (
        "run_arm.sh must take <arm> as its first positional argument.")


@pytest.mark.parametrize("arm", ARMS)
def test_arm_case_block_sets_required_vars(new_code: str, arm: str):
    body = extract_arm_case_body(new_code, arm)
    for var in ("NAME=", "ARM_DESC=", "LOSS_ARGS="):
        assert var in body, (
            f"arm {arm}: case block must set {var} — got:\n{body}")


@pytest.mark.parametrize("arm", ARMS)
def test_flags_are_the_379_command_line_plus_the_new_target(
        new_code: str, old_code: str, arm: str):
    """The issue's contract, checked mechanically against #379's launcher."""
    want = sorted(flags_of(old_code, arm) + ADDED_FLAG)
    got = sorted(flags_of(new_code, arm))
    assert got == want, (
        f"arm {arm}: flags must be #379's plus {' '.join(ADDED_FLAG)}.\n"
        f"  #379: {' '.join(sorted(flags_of(old_code, arm)))}\n"
        f"  #390: {' '.join(got)}")


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_targets_the_teacher(new_code: str, arm: str):
    body = extract_arm_case_body(new_code, arm)
    assert "--align-target teacher" in body.replace("\\\n", " "), (
        f"arm {arm}: the whole point of #390 is `--align-target teacher`.")


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_keeps_the_align_term(new_code: str, arm: str):
    """`--align-target` only means something with L_align switched on —
    train.py rejects the pair, and these are the cells that carry it."""
    assert "--align-loss-weight 1.0" in extract_arm_case_body(new_code, arm)


@pytest.mark.parametrize("arm", ARMS)
def test_checkpoint_names_cannot_collide_with_379(new_code: str,
                                                  old_code: str, arm: str):
    """Same arm, two experiments, one shared `runs/` layout — the NAME must
    differ or a rerun would overwrite #379's artefacts."""
    def name_of(code):
        m = re.search(r'NAME="([^"]+)"',
                      extract_arm_case_body(code, arm))
        assert m is not None, f"arm {arm}: no NAME= in case block"
        return m.group(1)
    assert name_of(new_code) != name_of(old_code)
    assert name_of(new_code).endswith("_alignteacher"), (
        f"arm {arm}: NAME must carry the `_alignteacher` marker.")


@pytest.mark.parametrize("arm", COPIED_ARMS)
def test_cells_without_an_align_term_are_not_reachable(new_code: str,
                                                       arm: str):
    """#379's numbers stand for those 20 cells; retraining them would spend
    compute on an unchanged objective."""
    assert not re.search(rf'(?m)^\s*{re.escape(arm)}\)\s*$', new_code), (
        f"{arm} has no L_align term and must not be a #390 arm.")


def test_trainer_invocation_is_identical_to_379(new_code: str, old_code: str):
    """Everything outside the per-arm case block — architecture, batch size,
    seed, save cadence, dataset, dtypes — must be #379's, verbatim."""
    assert trainer_block(new_code) == trainer_block(old_code)


def test_runs_in_its_own_experiment_directory(new_code: str):
    assert 'OUT="$WT/experiments/2026-08-01_lalign_teacher"' in new_code, (
        "run_arm.sh must write its runs/ and results/ under this "
        "experiment's own directory, never #379's.")


def test_launcher_is_backbone_only(new_code: str):
    forbidden = (
        "QTRAIN=", "QEVAL=", "train_head_cell", "eval_cell", "downstream_hl",
        "--head-nhead", "--quantile-head", "--head-arch", "--forecast-len",
        "gift_eval_full_", "BB_STEPS_K", "QEVAL_EXTRA_ARGS", "GPU_2L",
        "GPU_6L", "HEAD_STEPS", "HEAD_WARMUP",
    )
    for token in forbidden:
        assert token not in new_code, (
            f"run_arm.sh must be backbone-only — found {token!r}")


def test_wave_support_matches_379(new_code: str):
    """Stage 1 → 2 → 3 of the issue is #379's staged-wave protocol: a wave
    trains to TARGET_STEPS, only the final wave writes `_FINAL.pth`, and the
    next wave resumes from the newest `_<N>k.pth`."""
    assert 'TARGET_STEPS="${TARGET_STEPS:-${STEPS:-200000}}"' in new_code
    assert 'FINAL_STEPS="${FINAL_STEPS:-200000}"' in new_code
    assert 'STEPS="$TARGET_STEPS"' in new_code
    assert "WAVE SKIP" in new_code
    assert 'RESUME="--resume $latest"' in new_code


def test_wt_under_tmp_is_rejected(new_code: str):
    assert re.search(r'WT="\$\{WT:-\$HOME/', new_code), (
        "run_arm.sh should default WT under $HOME (a persistent checkout), "
        "never /tmp.")
    assert "/tmp/*|/tmp" in new_code, (
        "run_arm.sh must reject WT under /tmp with a loud ABORT.")


# --- experiment-directory layout -----------------------------------------
#
# Stage 2 of #390 is 10 cells x 3 waves with q-head training and GIFT-Eval
# between waves. The launcher alone does not run that, and CLAUDE.md
# requires a sync loop for the full duration of every run.

SUPPORT_SCRIPTS = (
    pytest.param(SMOKE_SH, id="smoke.sh"),
    pytest.param(ORCHESTRATE_SH, id="orchestrate.sh"),
    pytest.param(MONITOR_SH, id="monitor.sh"),
    pytest.param(SYNC_LOOP_SH, id="sync/sync_loop.sh"),
    pytest.param(ARM_NAMES_SH, id="arm_names.sh"),
)


@pytest.mark.parametrize("path", SUPPORT_SCRIPTS)
def test_support_script_exists_and_parses(path: Path):
    assert path.is_file(), f"{path.name} missing from the experiment directory"
    rc = subprocess.run(["bash", "-n", str(path)], capture_output=True,
                        text=True)
    assert rc.returncode == 0, f"{path.name} is not valid bash:\n{rc.stderr}"


@pytest.mark.parametrize("arm", ARMS)
def test_bb_name_matches_the_launcher(new_code: str, arm: str):
    """The derived name must be the launcher's name, character for
    character — monitor.sh and sync_loop.sh guard whatever bb_name says,
    so a mismatch means an unwatched, unsynced run."""
    m = re.search(r'NAME="([^"]+)"', extract_arm_case_body(new_code, arm))
    assert m is not None, f"arm {arm}: no NAME= in case block"
    got = subprocess.run(
        ["bash", "-c", f'source "{ARM_NAMES_SH}"; bb_name {arm}'],
        capture_output=True, text=True)
    assert got.returncode == 0, f"bb_name {arm} failed:\n{got.stderr}"
    assert got.stdout.strip() == m.group(1)


def test_arm_names_rejects_a_cell_without_an_align_term():
    """arm1 / arm3 / arm4 / bimoco are copied from #379, not rerun."""
    got = subprocess.run(
        ["bash", "-c", f'source "{ARM_NAMES_SH}"; bb_name arm1'],
        capture_output=True, text=True)
    assert got.returncode != 0
    assert "unknown arm" in got.stderr


def test_arm_names_lists_the_ten_cells():
    got = subprocess.run(
        ["bash", "-c", f'source "{ARM_NAMES_SH}"; echo "${{CF390_ARMS[*]}}"'],
        capture_output=True, text=True)
    assert got.returncode == 0
    assert sorted(got.stdout.split()) == sorted(ARMS)


@pytest.mark.parametrize("path", [MONITOR_SH, SYNC_LOOP_SH])
def test_watchers_derive_names_rather_than_retyping_them(path: Path):
    code = strip_comments(path.read_text())
    assert "arm_names.sh" in code, (
        f"{path.name} must source arm_names.sh, not carry its own copy of "
        "the run-name table.")
    assert "bb_name" in code
    # A literal run name in the body means a second, drifting copy.
    assert "bb_small_" not in code, (
        f"{path.name} hard-codes a run name — derive it with bb_name.")


def test_orchestrate_covers_every_cell_and_every_wave():
    code = strip_comments(ORCHESTRATE_SH.read_text())
    for arm in ARMS:
        assert re.search(rf'\b{re.escape(arm)}\b', code), (
            f"orchestrate.sh does not cover {arm}")
    # The issue's schedule: 40k → 100k → 200k, and only the last wave is
    # final (run_arm.sh writes `_FINAL.pth` only when TARGET ≥ FINAL).
    for steps in ("40000", "100000", "200000"):
        assert steps in code, f"orchestrate.sh has no wave at {steps} steps"
    assert "FINAL_STEPS=200000" in code, (
        "waves 1 and 2 must leave FINAL_STEPS at the arm's true end so the "
        "next wave resumes from the newest _<N>k.pth.")


def test_orchestrate_and_monitor_reject_wt_under_tmp():
    for path in (ORCHESTRATE_SH, MONITOR_SH):
        assert "/tmp/*|/tmp" in strip_comments(path.read_text()), (
            f"{path.name} must reject WT under /tmp.")


def test_sync_loop_points_at_a_real_safe_pull():
    """Raw scp writes straight to the destination, so a mid-transfer drop
    corrupts the previous good copy. A wrong path to safe_pull.sh aborts the
    loop, which is the silent no-sync failure CLAUDE.md calls out."""
    line = next((l for l in SYNC_LOOP_SH.read_text().splitlines()
                 if l.strip().startswith("SAFE_PULL=")), None)
    assert line is not None, "sync_loop.sh must resolve safe_pull.sh"
    # Let bash expand it, against the path a real launch would pass.
    got = subprocess.run(
        ["bash", "-c", f'LOCAL_DIR="{EXP_DIR}"; {line}; echo "$SAFE_PULL"'],
        capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    resolved = got.stdout.strip()
    assert Path(resolved).is_file(), (
        f"sync_loop.sh resolves safe_pull.sh to {resolved}, which does not "
        "exist.")


def test_sync_loop_pulls_optimizers_and_uses_per_class_floors():
    code = strip_comments(SYNC_LOOP_SH.read_text())
    assert "_optimizer.pth" in code, (
        "always sync optimizer files — without them a resume loses the step "
        "counter, RNG state and AdamW momentum (CLAUDE.md).")
    # Never one blanket floor for every file class (PR #45 dropped 2.4 MB
    # head checkpoints behind a blanket 70 MB floor).
    for var in ("BACKBONE_MIN", "BACKBONE_OPT_MIN", "TEXT_MIN"):
        assert var in code, f"sync_loop.sh must define {var}"
    assert "/tmp/*|/tmp" in code, (
        "sync_loop.sh must refuse to sync into an ephemeral LOCAL_DIR.")


def test_monitor_writes_into_the_sync_target():
    code = strip_comments(MONITOR_SH.read_text())
    assert 'SYNC="$OUT/sync"' in code, (
        "monitor.sh must copy the irreplaceable CSVs into the experiment's "
        "sync/ target.")
    assert ".tmp" in code and "mv -f" in code, (
        "atomic writes only: download to .tmp, then mv over the old copy.")


@pytest.fixture
def wt_outside_tmp():
    """A scratch WT the /tmp guard accepts (pytest's tmp_path is under /tmp)."""
    root = tempfile.mkdtemp(prefix="cf390-monitor-",
                            dir=os.path.expanduser("~/.cache"))
    yield Path(root)
    shutil.rmtree(root, ignore_errors=True)


def run_monitor(wt: Path, **env_extra):
    env = {**os.environ, "WT": str(wt), "WAVE": "1", **env_extra}
    return subprocess.run(["bash", str(MONITOR_SH), "0"], env=env,
                          capture_output=True, text=True, timeout=120)


def test_monitor_does_not_quit_before_the_first_arm_is_up(wt_outside_tmp):
    """The launch order is monitor FIRST, then orchestrate — so on tick 1
    nothing is alive yet. Treating that as "the wave finished" would exit
    immediately and leave every run unguarded, which is the silent
    no-monitor failure CLAUDE.md is written against."""
    res = run_monitor(wt_outside_tmp, STARTUP_TICKS="3")
    log = (wt_outside_tmp / "experiments" / "2026-08-01_lalign_teacher"
           / "results" / "monitor.log").read_text()
    assert "all arms stopped" not in log, (
        "monitor.sh mistook 'no arm started yet' for 'the wave is over'.")
    assert log.count("no arm up yet") == 3, (
        f"expected 3 startup ticks before giving up, log was:\n{log}")
    assert "THE WAVE IS UNGUARDED" in log, (
        "giving up on a wave that never started must be loud.")
    assert res.returncode == 1, (
        "a wave that never started is a failure, not a clean exit.")


def test_orchestrate_actually_drives_the_launcher(wt_outside_tmp):
    """One wave, end to end, without training: a `_40k.pth` already on disk
    makes run_arm.sh short-circuit, so this exercises the whole
    orchestrate → run_arm → wave-budget path in a second. `bash -n` only
    proves the file parses; this proves the two scripts are wired."""
    exp = wt_outside_tmp / "experiments" / "2026-08-01_lalign_teacher"
    (exp / "runs").mkdir(parents=True)
    (exp / "scripts").symlink_to(EXP_DIR / "scripts")
    (wt_outside_tmp / "experiments" / "2026-04-27_freq-embedding").symlink_to(
        REPO_ROOT / "experiments" / "2026-04-27_freq-embedding")
    (wt_outside_tmp / "experiments" / "hf_token.txt").write_text("hf_dummy\n")
    name = subprocess.run(
        ["bash", "-c", f'source "{ARM_NAMES_SH}"; bb_name arm5'],
        capture_output=True, text=True, check=True).stdout.strip()
    (exp / "runs" / f"{name}_40k.pth").write_bytes(b"x")

    res = subprocess.run(
        ["bash", str(ORCHESTRATE_SH)],
        env={**os.environ, "WT": str(wt_outside_tmp), "WAVE": "1",
             "ARMS": "arm5"},
        capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr

    log = (exp / "results" / "orchestrate_wave1.log").read_text()
    assert "WAVE SKIP" in log, (
        "orchestrate.sh did not reach run_arm.sh's wave-budget check; the "
        f"two scripts are not wired. log:\n{log}")
    assert "arms at/past 40k: 1 / 1" in log, (
        f"wave-1 summary did not count the existing 40k checkpoint:\n{log}")
    state = (exp / "results" / "orchestrate_wave1_state.json").read_text()
    assert '"arms_reached_target": 1' in state


def test_monitor_copies_the_csvs_into_sync(wt_outside_tmp):
    """One real tick against files on disk. The CSVs are the irreplaceable
    artefact — a checkpoint can be retrained, a loss curve cannot. Verified
    by `ls`, not by reading monitor.log (CLAUDE.md)."""
    exp = wt_outside_tmp / "experiments" / "2026-08-01_lalign_teacher"
    runs = exp / "runs"
    runs.mkdir(parents=True)
    name = subprocess.run(
        ["bash", "-c", f'source "{ARM_NAMES_SH}"; bb_name arm5'],
        capture_output=True, text=True, check=True).stdout.strip()
    # Both above their per-class floor — monitor.sh refuses a short file
    # rather than overwriting a good copy with a truncated one.
    (runs / f"{name}_losses.csv").write_text("step,loss\n" + "1,0.5\n" * 400)
    (runs / f"{name}_attn_amplitude.csv").write_text("step,amp\n" + "1,0.1\n" * 100)

    run_monitor(wt_outside_tmp, STARTUP_TICKS="1")

    synced = exp / "sync" / "arm5"
    assert (synced / f"{name}_losses.csv").is_file(), (
        f"monitor.sh did not copy the losses CSV into {synced}")
    assert (synced / f"{name}_attn_amplitude.csv").is_file()
    assert not list(synced.glob("*.tmp")), "a .tmp file was left behind"
