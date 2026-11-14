"""#462: the launcher and the score script of MPE at width 64.

The launcher trains the MPE command (results/mpe_command.txt, line 1 of
run_cf421ew_moirai_k3.log) at the size of the width-64 runs. These tests hold
that its trainer command is the MPE command with only the changes of the
card, that a restart resumes from the last checkpoint that loads, and that the
score script scores each saved stop once.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDY = REPO_ROOT / "reports" / "2026-11-14_moirai_width64"
TRAIN_SH = STUDY / "scripts" / "train.sh"
SCORE_SH = STUDY / "scripts" / "score_stops.sh"
NAME = "cf462_mpe_w64"
STOPS = [10, 25, 50, 75, 100, 125, 150, 166]


def flags(argv):
    """``{flag: value, or None for a switch}`` of a trainer command."""
    out, i = {}, 0
    while i < len(argv):
        assert argv[i].startswith("--"), argv[i]
        has_value = i + 1 < len(argv) and not argv[i + 1].startswith("--")
        out[argv[i]] = argv[i + 1] if has_value else None
        i += 2 if has_value else 1
    return out


def mpe_flags():
    words = (STUDY / "results" / "mpe_command.txt").read_text().split()
    assert words[:2] == ["Command", "line:"] and words[2].endswith("train.py")
    return flags(words[3:])


def dry_run(script, root, **env):
    r = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, timeout=300,
        env=dict(os.environ, CF462_ROOT=str(root), DRY_RUN="1", BB_GPU="0",
                 **env))
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def trainer_flags(stdout):
    """The flags after train.py in the command line of a dry run."""
    words = shlex.split(next(line for line in stdout.splitlines()
                             if "train.py" in line))
    script = next(i for i, w in enumerate(words) if w.endswith("train.py"))
    return flags(words[script + 1:])


def checkpoint(path, optimizer=True):
    torch.save({"w": torch.zeros(1)}, path)
    if optimizer:
        torch.save({"step": 1}, str(path)[:-len(".pth")] + "_optimizer.pth")


def test_the_command_is_the_mpe_command_at_width_64(tmp_path):
    run, mpe = trainer_flags(dry_run(TRAIN_SH, tmp_path)), mpe_flags()
    changed = {f for f in mpe if run.get(f, "absent") != mpe[f]}
    assert changed == {"--d-model", "--num-encoder-layers", "--run-name",
                       "--save-dir", "--total-steps", "--extra-save-steps"}
    assert set(run) - set(mpe) == {"--keep-start-window-targets"}
    assert (run["--d-model"], run["--num-encoder-layers"]) == ("64", "1")
    assert (run["--n-heads"], run["--num-layers"]) == ("8", "3")
    assert (run["--batch-size"], run["--grad-clip"]) == ("256", "1.0")
    assert run["--total-steps"] == "166000"
    assert [int(s) // 1000 for s in run["--extra-save-steps"].split(",")] \
        == STOPS
    assert run["--save-dir"] == str(tmp_path / "ckpt")
    assert run["--run-name"] == NAME


def test_a_restart_resumes_the_last_checkpoint_that_loads(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    checkpoint(ckpt / f"{NAME}_10k.pth")
    checkpoint(ckpt / f"{NAME}_r2_25k.pth")
    # A save that a crash cut: no optimizer file, or files that do not load.
    checkpoint(ckpt / f"{NAME}_r2_30k.pth", optimizer=False)
    for name in (f"{NAME}_r2_35k.pth", f"{NAME}_r2_35k_optimizer.pth"):
        (ckpt / name).write_bytes(b"cut")
    run = trainer_flags(dry_run(TRAIN_SH, tmp_path))
    assert run["--resume"] == str(ckpt / f"{NAME}_r2_25k.pth")


def test_a_first_start_resumes_nothing(tmp_path):
    assert "--resume" not in trainer_flags(dry_run(TRAIN_SH, tmp_path))


def test_the_run_trains_nothing_after_its_last_stop(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    checkpoint(ckpt / f"{NAME}_r2_166k.pth")
    out = dry_run(TRAIN_SH, tmp_path)
    assert "train.py" not in out
    assert f"{NAME}_r2_166k.pth" in out


def test_a_cut_optimizer_file_of_the_last_stop_trains_again(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    checkpoint(ckpt / f"{NAME}_165k.pth")
    checkpoint(ckpt / f"{NAME}_166k.pth", optimizer=False)
    (ckpt / f"{NAME}_166k_optimizer.pth").write_bytes(b"cut")
    run = trainer_flags(dry_run(TRAIN_SH, tmp_path))
    assert run["--resume"] == str(ckpt / f"{NAME}_165k.pth")


def test_the_smoke_run_trains_in_its_own_folder(tmp_path):
    run = trainer_flags(dry_run(TRAIN_SH, tmp_path, SMOKE_STEPS="300"))
    assert (run["--total-steps"], run["--extra-save-steps"]) == ("300", "300")
    assert run["--save-dir"] == str(tmp_path / "smoke" / "ckpt")


def test_the_score_script_scores_each_saved_stop_once(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    checkpoint(ckpt / f"{NAME}_10k.pth")
    checkpoint(ckpt / f"{NAME}_r2_25k.pth")
    # The trainer writes the optimizer file after the weights: not ready.
    checkpoint(ckpt / f"{NAME}_50k.pth", optimizer=False)
    results = tmp_path / "results"
    results.mkdir()
    (results / f"score_{NAME}_bb10k_native.txt").write_text("1.0000\n")
    out = dry_run(SCORE_SH, tmp_path, RESULTS=str(results))
    assert [line for line in out.splitlines() if line.startswith("score ")] \
        == [f"score 25k {ckpt / f'{NAME}_r2_25k.pth'} -> "
            f"{results / f'score_{NAME}_bb25k_native.txt'} "
            f"{results / 'per_config' / f'{NAME}_native25k.csv'}"]


def test_the_score_script_skips_a_save_that_does_not_load(tmp_path):
    # A crash cut the optimizer file of 75k. The run trained 75k again from
    # an earlier save, as r2: the score reads that save.
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    checkpoint(ckpt / f"{NAME}_75k.pth", optimizer=False)
    (ckpt / f"{NAME}_75k_optimizer.pth").write_bytes(b"cut")
    checkpoint(ckpt / f"{NAME}_r2_75k.pth")
    out = dry_run(SCORE_SH, tmp_path, RESULTS=str(tmp_path / "results"))
    assert [line.split()[2] for line in out.splitlines()
            if line.startswith("score ")] == [str(ckpt / f"{NAME}_r2_75k.pth")]


def test_the_follower_waits_for_the_trainer_to_start(tmp_path):
    """FOLLOW=1 before train.sh writes its pid file: the follower waits for
    the file, up to FOLLOW_WAIT seconds, and does not stop at once."""
    start = time.monotonic()
    r = subprocess.run(
        ["bash", str(SCORE_SH)], capture_output=True, text=True, timeout=120,
        env=dict(os.environ, CF462_ROOT=str(tmp_path), FOLLOW="1", POLL="1",
                 FOLLOW_WAIT="3", RESULTS=str(tmp_path / "results")))
    assert r.returncode == 0, r.stdout + r.stderr
    # bash SECONDS counts whole seconds of the clock: 3 can come after 2.x s.
    assert time.monotonic() - start >= 2
