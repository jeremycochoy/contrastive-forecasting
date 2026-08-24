"""Tests for the two guards #393's leg runner gained after the 08-23 outage.

`reports/2026-08-08_rollout_depth/scripts/run_leg_k.sh` is the one runner
#401, #404 and #409 all train through. Two of its answers cost GPU time on
2026-08-23.

  1. A leg that dies because the Hub is unreachable exited 1, which every
     lane reads as a crashed arm. It now exits `HUB_GATE_RC`, so a lane can
     wait for the network instead of spending its retry ladder in two
     minutes.
  2. A cell whose resume glob matched nothing started FRESH at step 0, even
     with step checkpoints on disk. A fresh start now happens only when the
     cell holds no step checkpoint at all.

Both guards run before the trainer, so these tests need no GPU. `HOLD_ABOVE`
stops each leg one line after the resume decision.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PARENT = REPO_ROOT / "reports" / "2026-08-08_rollout_depth"
RUN_LEG = PARENT / "scripts" / "run_leg_k.sh"
LEG_PATHS = PARENT / "scripts" / "leg_paths.sh"
HUB_GATE = REPO_ROOT / "scripts" / "hub_gate.sh"

CELL = "arm6_v2_combab_alignT"
NAME = f"cf393_{CELL}_cf373k32_cf409_test"
STOP = 40_000

HUB_TAIL = ("ConnectionError: Couldn't reach "
            "'jeremycochoy/gift-pretrain-full-4096' on the Hub "
            "(ConnectionError)")


def hub_gate_rc() -> int:
    out = subprocess.run(
        ["bash", "-c", f'. "{HUB_GATE}" && printf %s "$HUB_GATE_RC"'],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return int(out.stdout)


@pytest.fixture
def runs_root():
    """A durable root. `runs_root()` refuses /tmp and the checkout."""
    root = tempfile.mkdtemp(prefix="cf393-leg-guards-", dir="/var/tmp")
    yield Path(root)
    shutil.rmtree(root, ignore_errors=True)


def stub_checkout(tmp_path: Path, train_body: str) -> Path:
    """A checkout whose trainer is `train_body`, and nothing else."""
    wt = tmp_path / "wt"
    scripts = wt / "experiments" / "2026-04-27_freq-embedding" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "train.py").write_text(train_body)
    (wt / "experiments" / "hf_token.txt").write_text("stub-token\n")
    return wt


def run_leg(runs_root: Path, tmp_path: Path, *, wt: Path | None = None,
            hold: int | None = None, suffix: str = "_cf409_test"):
    res = tmp_path / "results"
    res.mkdir(exist_ok=True)
    if hold is not None:
        (res / "HOLD_ABOVE").write_text(str(hold))
    env = dict(os.environ)
    env.update({
        "WT": str(wt or REPO_ROOT),
        "RUNS": str(runs_root),
        "CF_STUDY_DIR": str(tmp_path),
        "CF_RESULTS": str(res),
        "RUN_SUFFIX": suffix,
        "K": "32",
        "GPU_GATE_LOCKDIR": str(tmp_path),
    })
    out = subprocess.run(["bash", str(RUN_LEG), CELL, str(STOP)],
                         capture_output=True, text=True, env=env,
                         cwd=str(REPO_ROOT), timeout=300)
    log = (res / f"leg_{CELL}.log")
    return out, (log.read_text() if log.exists() else "")


def write_ckpt(runs_root: Path, name: str, step_k: int, leg_k: int = 40):
    d = runs_root / CELL / f"leg_{leg_k}k"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}_{step_k}k.pth"
    path.write_text("stub")
    return path


class TestAFreshStartNeedsACellWithNoCheckpoint:
    """Card rule 3: never fall back to step 0 when a step checkpoint
    exists."""

    def test_an_empty_cell_starts_fresh(self, runs_root, tmp_path):
        out, log = run_leg(runs_root, tmp_path, hold=1)
        assert "FRESH start at step 0" in log
        assert out.returncode == 9, out.stdout + out.stderr

    def test_a_cell_with_its_own_checkpoint_resumes(self, runs_root, tmp_path):
        write_ckpt(runs_root, NAME, 15)
        out, log = run_leg(runs_root, tmp_path, hold=1)
        assert "RESUME from" in log and "15k" in log
        assert "FRESH start" not in log
        assert out.returncode == 9

    def test_a_checkpoint_under_another_run_name_stops_the_leg(
            self, runs_root, tmp_path):
        """The glob is pinned to this leg's run name. A checkpoint written
        under any other name reads as "nothing on disk", and a fresh start
        then throws away every step it holds."""
        stray = write_ckpt(runs_root, "cf393_" + CELL + "_cf373k32_other", 15)
        out, log = run_leg(runs_root, tmp_path, hold=1)
        assert "FRESH start" not in log
        assert out.returncode == 2, out.stdout + out.stderr
        assert stray.name in log

    def test_an_optimizer_file_alone_is_not_a_step_checkpoint(
            self, runs_root, tmp_path):
        d = runs_root / CELL / "leg_40k"
        d.mkdir(parents=True)
        (d / f"{NAME}_best_gap.pth").write_text("stub")
        (d / f"{NAME}_best_gap_optimizer.pth").write_text("stub")
        out, log = run_leg(runs_root, tmp_path, hold=1)
        assert "FRESH start at step 0" in log
        assert out.returncode == 9


class TestALegSaysWhenTheHubWasWhatFailed:
    """Card rule 1: a leg that dies because the Hub is unreachable is not a
    failed arm."""

    def test_a_hub_failure_gives_the_network_code(self, runs_root, tmp_path):
        wt = stub_checkout(tmp_path, f'import sys\nsys.exit(1)\n')
        # The trainer writes its own log, so the runner reads the tail from
        # there. A stub that prints the tail to stdout lands in the same file.
        wt_train = (wt / "experiments" / "2026-04-27_freq-embedding"
                    / "scripts" / "train.py")
        wt_train.write_text(
            "import sys\n"
            f"print({HUB_TAIL!r})\n"
            "sys.exit(1)\n")
        out, log = run_leg(runs_root, tmp_path, wt=wt)
        assert out.returncode == hub_gate_rc(), out.stdout + out.stderr
        assert "Hub" in log

    def test_a_real_crash_still_gives_one(self, runs_root, tmp_path):
        wt = stub_checkout(
            tmp_path,
            "import sys\n"
            "print('torch.OutOfMemoryError: CUDA out of memory')\n"
            "sys.exit(1)\n")
        out, _ = run_leg(runs_root, tmp_path, wt=wt)
        assert out.returncode == 1, out.stdout + out.stderr

    def test_the_network_code_is_not_a_code_the_runner_already_uses(self):
        body = RUN_LEG.read_text()
        rc = hub_gate_rc()
        assert rc not in (0, 1, 2, 4, 9, 10)
        assert "hub_gate.sh" in body


class TestTheStepCheckpointScanIsShared:
    """`step_ckpts` sits beside the resume glob it guards, so both read the
    same layout."""

    def test_leg_paths_holds_the_scan(self):
        assert "step_ckpts()" in LEG_PATHS.read_text()

    def test_it_finds_a_checkpoint_under_any_run_name(self, runs_root):
        write_ckpt(runs_root, "some_other_run", 15)
        out = subprocess.run(
            ["bash", "-c",
             f'. "{LEG_PATHS}" && step_ckpts "{runs_root / CELL}"'],
            capture_output=True, text=True, timeout=60)
        assert out.returncode == 0, out.stderr
        assert "some_other_run_15k.pth" in out.stdout

    def test_it_skips_optimizer_files(self, runs_root):
        d = runs_root / CELL / "leg_40k"
        d.mkdir(parents=True)
        (d / f"{NAME}_15k_optimizer.pth").write_text("stub")
        out = subprocess.run(
            ["bash", "-c",
             f'. "{LEG_PATHS}" && step_ckpts "{runs_root / CELL}"'],
            capture_output=True, text=True, timeout=60)
        assert out.stdout.strip() == ""
