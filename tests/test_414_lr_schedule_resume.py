"""#414: the cosine rate schedule must survive a resume.

Every constant rate of #414's card holds its floor at a different step and then
climbs, so the card added a single cosine anneal (`--lr-final`,
`--lr-cosine-steps`). A leg of that card runs in several parts, and each part
resumes from the last checkpoint through `run_leg_k.sh`, which repeats the
arm's flags. A resume that lost the schedule would silently train the rest of
the pass at a constant rate, and no log line would say so.

Runs `train.py` twice on the smallest CPU arch (100 then 200 steps), the second
time WITHOUT the schedule flags, then verifies:

  (a) the checkpoint carries `lr_schedule`;
  (b) the resumed run says it read the schedule back;
  (c) the resumed optimizer ends at the rate the curve gives for step 200,
      which is the end rate here, and NOT at the `--lr` default;
  (d) a run with no `--lr-final` stores `lr_schedule` as None, so every arm of
      an earlier pass is untouched.

Uses `--mix-ratio 1.0` (pure synth) so no HF token and no network call, and
d_model=8 so both runs finish in seconds.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "experiments" / "2026-04-27_freq-embedding" / "scripts" / "train.py"

LR_START, LR_FINAL, COSINE_STEPS = 1e-2, 1e-4, 200


def _cosine(step, start, final, total):
    t = min(max(step, 0), total)
    return final + 0.5 * (start - final) * (1.0 + math.cos(math.pi * t / total))


def _cmd(save_dir, run_name, total_steps, resume=None, schedule=True):
    cmd = [
        sys.executable, "-u", str(TRAIN_PY),
        "--device", "cpu",
        "--total-steps", str(total_steps),
        "--save-every", "100",
        "--traj-save-every", "100",
        "--batch-size", "2",
        "--lr", str(LR_START),
        "--weight-decay", "0.1",
        "--save-dir", str(save_dir),
        "--run-name", run_name,
        "--mix-ratio", "1.0",
        "--synth-kind", "periodic",
        "--t-raw", "64",
        "--n-channels", "1",
        "--d-model", "8",
        "--n-heads", "2",
        "--num-encoder-layers", "1",
        "--num-layers", "1",
        "--log-every", "50",
        "--seed", "20260728",
        "--loss-shape", "cosine_similarity_batch_full_hh_negs_xshh_allt",
        "--tau", "0.10",
        "--hf-repo", "none",
        "--hf-path", "none",
    ]
    if schedule:
        cmd += ["--lr-final", str(LR_FINAL),
                "--lr-cosine-steps", str(COSINE_STEPS)]
    if resume is not None:
        cmd += ["--resume", resume]
    return cmd


def _run(cmd, cwd, timeout=600):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(cmd, env=env, capture_output=True, text=True,
                          cwd=str(cwd), timeout=timeout)


def _state(path: Path) -> dict:
    companion = path.with_name(path.stem + "_optimizer.pth")
    return torch.load(companion, map_location="cpu", weights_only=False)


def _optim_lr(state: dict) -> float:
    return state["optimizer_state_dict"]["param_groups"][0]["lr"]


def test_schedule_survives_a_resume(tmp_path):
    save_dir = tmp_path / "runs"
    save_dir.mkdir(parents=True)

    first = _run(_cmd(save_dir, "sched", 100), REPO_ROOT)
    assert first.returncode == 0, first.stderr[-3000:]

    ck100 = save_dir / "sched_step100.pth"
    assert ck100.exists(), sorted(p.name for p in save_dir.iterdir())
    held = _state(ck100)["lr_schedule"]
    assert held is not None, "the checkpoint dropped the schedule"
    assert held["lr_start"] == pytest.approx(LR_START)
    assert held["lr_final"] == pytest.approx(LR_FINAL)
    assert held["cosine_steps"] == COSINE_STEPS

    # The resume names NO schedule flag. It must read the curve back.
    second = _run(_cmd(save_dir, "sched", 200, resume=str(ck100),
                       schedule=False), REPO_ROOT)
    assert second.returncode == 0, second.stderr[-3000:]
    assert "resumed schedule from the checkpoint" in second.stdout, \
        second.stdout[-3000:]

    ck200 = next(p for p in save_dir.glob("sched*_step200.pth"))
    got = _optim_lr(_state(ck200))
    want = _cosine(200, LR_START, LR_FINAL, COSINE_STEPS)
    assert got == pytest.approx(want, rel=1e-6), \
        f"resumed run ended at lr={got:g}, the curve gives {want:g}"
    assert got < LR_START / 10, "the rate never annealed"


def test_a_constant_rate_stores_no_schedule(tmp_path):
    save_dir = tmp_path / "runs"
    save_dir.mkdir(parents=True)
    r = _run(_cmd(save_dir, "flat", 100, schedule=False), REPO_ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    ck = save_dir / "flat_step100.pth"
    assert _state(ck)["lr_schedule"] is None
    assert _optim_lr(_state(ck)) == pytest.approx(LR_START)
