"""Behavioural tests for #390's wave-orchestration scripts.

`tests/test_390_launcher_shape.py` pins what the scripts *say*: the flags,
the names, the wave targets. These pin what they *do* when a wave runs, for
the three failure modes that leave a wave unguarded or a crashed cell
counted as finished:

  * `monitor.sh` must survive a phase boundary. `orchestrate.sh` runs the 10
    cells as 5 sequential pairs, so "nothing is training right now" is true
    4 times per wave and means nothing. Only the orchestrator being gone
    means the wave is over.
  * `orchestrate.sh`'s completion summary decides whether the next wave
    starts. It must count each arm's own checkpoints — `arm5` and
    `arm5_tr1` share a filename prefix, so a loose glob credits a crashed
    cell with its neighbour's progress.
  * `smoke.sh` must fail when the training-dynamics check fails. It is the
    script whose whole job is to fail loudly before a wave commits.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPNAME = "2026-08-01_lalign_teacher"
EXP_DIR = REPO_ROOT / "experiments" / EXPNAME
SCRIPTS = EXP_DIR / "scripts"
ARM_NAMES_SH = SCRIPTS / "arm_names.sh"
MONITOR_SH = SCRIPTS / "monitor.sh"
ORCHESTRATE_SH = SCRIPTS / "orchestrate.sh"
SMOKE_SH = SCRIPTS / "smoke.sh"
SYNC_LOOP_SH = EXP_DIR / "sync" / "sync_loop.sh"


def sh(script: str) -> str:
    """Run a bash snippet with arm_names.sh sourced; return its stdout."""
    return subprocess.run(
        ["bash", "-c", f'source "{ARM_NAMES_SH}"; {script}'],
        capture_output=True, text=True, check=True).stdout.strip()


def bb_name(arm: str) -> str:
    return sh(f"bb_name {arm}")


def poll_until(predicate, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def spawn_marked(marker: str, seconds: int = 300) -> subprocess.Popen:
    """A long-lived process carrying `marker` in its command line.

    `bash -c 'sleep N # marker'` would not do: bash execs the simple command
    and the comment never reaches the command line pgrep reads.
    """
    return subprocess.Popen(
        ["python3", "-c", "import sys, time; time.sleep(float(sys.argv[1]))",
         str(seconds), marker],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def kill_all(*procs: subprocess.Popen) -> None:
    for p in procs:
        if p.poll() is None:
            p.kill()
        p.wait()


@pytest.fixture
def wt():
    """A scratch WT the scripts' /tmp guard accepts (tmp_path is under /tmp)."""
    root = Path(tempfile.mkdtemp(prefix="cf390-wave-",
                                 dir=os.path.expanduser("~/.cache")))
    (root / "experiments" / EXPNAME / "results").mkdir(parents=True)
    (root / "experiments" / EXPNAME / "runs").mkdir()
    yield root
    shutil.rmtree(root, ignore_errors=True)


def wire_launcher(wt: Path) -> None:
    """Everything run_arm.sh checks before it decides to skip an arm."""
    exp = wt / "experiments" / EXPNAME
    (exp / "scripts").symlink_to(SCRIPTS)
    (wt / "experiments" / "2026-04-27_freq-embedding").symlink_to(
        REPO_ROOT / "experiments" / "2026-04-27_freq-embedding")
    (wt / "experiments" / "hf_token.txt").write_text("hf_dummy\n")


def monitor_log(wt: Path) -> str:
    path = wt / "experiments" / EXPNAME / "results" / "monitor.log"
    return path.read_text() if path.is_file() else ""


# --- 1. monitor.sh must not quit in the gap between two phases -------------

def test_monitor_survives_a_phase_gap_and_exits_when_the_orchestrator_is_gone(wt):
    """orchestrate.sh joins each pair before starting the next, so `running
    == 0` happens 4 times per wave with 8 cells still to train. Exiting
    there leaves the rest of the wave unguarded — worse on a re-run, where
    every idempotency skip returns in seconds and no arm is ever seen."""
    results = wt / "experiments" / EXPNAME / "results"
    name = bb_name("arm5")
    arm = spawn_marked(f"--run-name {name}")
    orch = spawn_marked("cf390-fake-orchestrator")
    (results / "orchestrate_wave1.pid").write_text(f"{orch.pid}\n")

    mon = subprocess.Popen(
        ["bash", str(MONITOR_SH), "1"],
        env={**os.environ, "WT": str(wt), "WAVE": "1", "ARMS": "arm5",
             "STARTUP_TICKS": "2"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert poll_until(lambda: "alive=yes" in monitor_log(wt)), (
            f"monitor never saw the arm running:\n{monitor_log(wt)}")

        # Phase boundary: the pair is joined, the next one has not started.
        kill_all(arm)
        with pytest.raises(subprocess.TimeoutExpired):
            mon.wait(timeout=6)
        assert "all arms stopped" not in monitor_log(wt), (
            "monitor.sh mistook a gap between two phases for the end of the "
            f"wave:\n{monitor_log(wt)}")

        # The orchestrator exits — now the wave really is over.
        kill_all(orch)
        assert mon.wait(timeout=30) == 0, (
            "monitor.sh must exit cleanly once nothing is training and the "
            "orchestrator is gone.")
        assert "all arms stopped" in monitor_log(wt)
    finally:
        kill_all(mon, arm, orch)


def test_monitor_does_not_invent_a_death_from_the_previous_waves_log(wt):
    """run_arm.sh appends to results/run_<name>.log across waves, so when
    wave 2 starts the last step in it is wave 1's endpoint. Reading that as
    a step count fires "died at step 40000" for every cell that has not
    reached its wave-2 run yet."""
    results = wt / "experiments" / EXPNAME / "results"
    (results / f"run_{bb_name('arm5')}.log").write_text(
        "[ 40000] loss 1.0000 ema_loss 1.1 gap 0.3 3.5 sps\n")

    subprocess.run(["bash", str(MONITOR_SH), "0"],
                   env={**os.environ, "WT": str(wt), "WAVE": "2",
                        "ARMS": "arm5", "STARTUP_TICKS": "1"},
                   capture_output=True, text=True, timeout=120)

    log = monitor_log(wt)
    assert "died at step" not in log, (
        f"monitor.sh reported a death it never saw start:\n{log}")


def test_monitor_does_not_report_nan_for_an_arm_with_no_log_yet(wt):
    """`grep -c` on a file that does not exist prints nothing, and an empty
    count is not "0". Every arm waiting for its phase has no training log,
    so the loudest line in monitor.log — the one whose stop rule drops a
    cell — fired for all of them, on every tick."""
    subprocess.run(["bash", str(MONITOR_SH), "0"],
                   env={**os.environ, "WT": str(wt), "WAVE": "1",
                        "ARMS": "arm5", "STARTUP_TICKS": "1"},
                   capture_output=True, text=True, timeout=120)

    log = monitor_log(wt)
    assert "NaN in arm5" not in log, (
        f"monitor.sh raised a NaN alert against a missing log:\n{log}")
    assert "nan=0" in log, f"the NaN count should read 0, not empty:\n{log}"


# --- 2. orchestrate.sh must count each arm's own checkpoints ---------------

def run_orchestrate(wt: Path, arms: str, wave: str = "1"):
    return subprocess.run(
        ["bash", str(ORCHESTRATE_SH)],
        env={**os.environ, "WT": str(wt), "WAVE": wave, "ARMS": arms},
        capture_output=True, text=True, timeout=300)


def test_orchestrate_summary_ignores_a_neighbouring_arms_checkpoint(wt):
    """`arm5` and `arm5_tr1` share the `bb_small_arm5` prefix. A glob that
    crosses them lets a crashed cell report as at-target, and that count is
    what decides whether the next wave starts."""
    exp = wt / "experiments" / EXPNAME
    wire_launcher(wt)
    runs = exp / "runs"
    # arm5 crashed early: a 10k snapshot, far short of the 40k wave budget.
    # The FINAL sentinel keeps run_arm.sh from launching a real trainer.
    (runs / f"{bb_name('arm5')}_FINAL.pth").write_bytes(b"x")
    (runs / f"{bb_name('arm5')}_10k.pth").write_bytes(b"x")
    # Its neighbour finished the wave. Different cell, different verdict.
    (runs / f"{bb_name('arm5_tr1')}_40k.pth").write_bytes(b"x")

    res = run_orchestrate(wt, arms="arm5")
    assert res.returncode == 0, res.stdout + res.stderr

    log = (exp / "results" / "orchestrate_wave1.log").read_text()
    assert "arm5: newest checkpoint 10k" in log, (
        f"orchestrate.sh read another arm's checkpoint as arm5's:\n{log}")
    assert "arms at/past 40k: 0 / 1" in log, (
        f"a cell that stopped at 10k was counted as at-target:\n{log}")
    state = (exp / "results" / "orchestrate_wave1_state.json").read_text()
    assert '"arms_reached_target": 0' in state


def test_orchestrate_defaults_to_the_ten_cells(wt):
    """No ARMS override: the sweep is the 10 cells of arm_names.sh, run as
    5 sequential pairs."""
    exp = wt / "experiments" / EXPNAME
    wire_launcher(wt)
    arms = sh('echo "${CF390_ARMS[*]}"').split()
    for arm in arms:
        (exp / "runs" / f"{bb_name(arm)}_FINAL.pth").write_bytes(b"x")

    res = subprocess.run(
        ["bash", str(ORCHESTRATE_SH)],
        env={**os.environ, "WT": str(wt), "WAVE": "1"},
        capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stdout + res.stderr

    log = (exp / "results" / "orchestrate_wave1.log").read_text()
    assert "arms=10" in log and log.count("PHASE ") == 5, (
        f"expected the 10 cells in 5 pairs:\n{log}")
    for arm in arms:
        assert f"arm {arm} done rc=0" in log, f"{arm} was not run:\n{log}"


def test_orchestrate_publishes_a_live_pid_for_the_monitor(wt):
    """monitor.sh watches the orchestrator, not "is anything training right
    now". The PID file is that handle: live for the whole wave, gone after."""
    exp = wt / "experiments" / EXPNAME
    scripts = exp / "scripts"
    scripts.mkdir()
    (scripts / "orchestrate.sh").symlink_to(ORCHESTRATE_SH)
    (scripts / "arm_names.sh").symlink_to(ARM_NAMES_SH)
    # A launcher that takes long enough to observe the wave in flight.
    (scripts / "run_arm.sh").write_text("#!/bin/bash\nsleep 2\n")

    # Two phases, so the PID outlives a phase join. The pair runs in
    # backgrounded subshells; bash resets the EXIT trap there, and a trap
    # that did fire would delete the PID file after the first phase and send
    # every monitor home mid-wave.
    orch = subprocess.Popen(
        ["bash", str(scripts / "orchestrate.sh")],
        env={**os.environ, "WT": str(wt), "WAVE": "1",
             "ARMS": "arm5 arm5_tr1 arm5_nse arm5_ncpc"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pidfile = exp / "results" / "orchestrate_wave1.pid"
    log = exp / "results" / "orchestrate_wave1.log"
    try:
        assert poll_until(lambda: pidfile.is_file()), (
            "orchestrate.sh published no PID, so monitor.sh cannot tell a "
            "phase gap from the end of the wave.")
        assert int(pidfile.read_text().strip()) == orch.pid
        # Phase 1 has been joined and phase 2 has not finished.
        assert poll_until(lambda: "PHASE 2" in log.read_text())
        assert pidfile.is_file(), (
            "the PID file vanished at the first phase join — the monitor "
            "would treat the gap as the end of the wave.")
        assert orch.wait(timeout=60) == 0
        assert not pidfile.exists(), (
            "a PID file left behind after the wave keeps every monitor alive.")
    finally:
        kill_all(orch)


# --- 3. smoke.sh must fail when the dynamics check fails -------------------

GOOD_CSV = ("step,loss,ff,u_batchtime,u_batchtime_e\n"
            "200,1.0,0.5,0.25,0.75\n")
MISSING_COLS_CSV = "step,loss\n200,1.0\n"
HEADER_ONLY_CSV = "step,loss,ff,u_batchtime,u_batchtime_e\n"

STUB_LAUNCHER = """#!/bin/bash
# Stand-in for run_arm.sh: writes the artefacts a 200-step smoke leaves
# behind, so smoke.sh's own checks run without a GPU.
set -euo pipefail
RUNS="$WT/experiments/{expname}/runs"
RES="$WT/experiments/{expname}/results"
mkdir -p "$RUNS" "$RES"
: > "$RUNS/stub_${{1}}_FINAL.pth"
: > "$RUNS/stub_${{1}}_0k.pth"
cat > "$RUNS/stub_${{1}}_losses.csv" <<'CSV'
{csv}CSV
echo "[   200] loss 1.0000 3.50 sps" > "$RES/run_stub_${{1}}.log"
"""


def fake_repo(tmp_path: Path, csv_text: str) -> Path:
    """A repo root holding the real smoke.sh over a stubbed launcher."""
    root = tmp_path / "repo"
    for d in ("src", "scripts", "docs", "tests", "reports"):
        (root / d).mkdir(parents=True)
    exp = root / "experiments" / EXPNAME
    (exp / "scripts").mkdir(parents=True)
    (exp / "sync").mkdir()
    (exp / "README.md").write_text("stub\n")
    (root / "experiments" / "hf_token.txt").write_text("hf_dummy\n")
    (exp / "scripts" / "smoke.sh").symlink_to(SMOKE_SH)
    launcher = exp / "scripts" / "run_arm.sh"
    launcher.write_text(STUB_LAUNCHER.format(expname=EXPNAME, csv=csv_text))
    launcher.chmod(0o755)
    return root


def run_smoke(tmp_path: Path, csv_text: str):
    root = fake_repo(tmp_path, csv_text)
    return subprocess.run(
        ["bash", str(root / "experiments" / EXPNAME / "scripts" / "smoke.sh")],
        env={**os.environ, "ARM": "arm5", "SMOKE_STEPS": "200"},
        capture_output=True, text=True, timeout=300)


def test_smoke_passes_when_the_dynamics_columns_are_there(tmp_path):
    res = run_smoke(tmp_path, GOOD_CSV)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "SMOKE OK" in res.stdout
    assert "ff=0.5" in res.stdout and "u_batchtime=0.25" in res.stdout


def test_smoke_fails_when_a_dynamics_column_is_missing(tmp_path):
    """A missing column means the deliverable plots silently fail. The
    check's exit status has to reach the smoke: `mapfile < <(python3 …)`
    reports mapfile's status, and `set -e` does not see into a process
    substitution, so the failure printed SMOKE OK."""
    res = run_smoke(tmp_path, MISSING_COLS_CSV)
    assert res.returncode != 0, (
        f"smoke.sh passed a CSV with no ff column:\n{res.stdout}")
    assert "SMOKE OK" not in res.stdout
    assert "MISSING_COLS" in res.stdout + res.stderr


def test_smoke_fails_on_a_header_with_no_rows(tmp_path):
    """A run that dies before its first logged step writes the header and
    nothing else."""
    res = run_smoke(tmp_path, HEADER_ONLY_CSV)
    assert res.returncode != 0, (
        f"smoke.sh passed a CSV with no data row:\n{res.stdout}")
    assert "SMOKE OK" not in res.stdout
    assert "NO_ROW" in res.stdout + res.stderr


# --- the wave table is shared, not retyped per script ---------------------

def test_sync_loop_derives_its_checkpoint_steps_from_the_wave_table():
    """A hand-kept step list stops pulling the snapshot the next wave
    resumes from the moment a cadence changes."""
    code = SYNC_LOOP_SH.read_text()
    assert "wave_checkpoint_steps_k" in code, (
        "sync_loop.sh must derive BACKBONE_STEPS_K from arm_names.sh.")
    assert sh("wave_checkpoint_steps_k | tr '\\n' ' '").split() == [
        "2", "10", "20", "30", "40", "50", "75", "100", "125", "150", "175",
        "200"]
    # Derived, not a constant dressed as a function.
    moved = sh("CF390_WAVE_TARGET_STEPS[1]=20000; "
               "wave_checkpoint_steps_k | tr '\\n' ' '").split()
    assert moved[:3] == ["2", "10", "20"] and "30" not in moved
