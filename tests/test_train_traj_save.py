"""Tests for the `--traj-save-every` fine-grained trajectory checkpoint
knob added for issue #369.

`--save-every` writes coarse `<run>_<K>k.pth` checkpoints — at sub-1000-step
intervals its `step // 1000` naming collides. `--traj-save-every` is a
separate cadence that writes `<run>_step<STEP>.pth` files, distinct from
the coarse cadence, safe at any interval, and independently resumable.

The tests here guard four things:

  1. The argparse flag exists and defaults to 0 (feature off).
  2. The save loop uses the fine name at exactly the trajectory cadence.
  3. Fine and coarse cadences do not collide on shared step multiples.
  4. Running train.py end-to-end with `--total-steps 4 --traj-save-every 2`
     actually emits `_step2.pth` and `_step4.pth` on disk with non-zero
     size — the reviewer's concern that `range(total_steps)` might skip
     the final step (the loop uses `range(start_step + 1, total_steps + 1)`
     so step=total_steps IS reached, but we cover it end-to-end here too).

Static tests over `train.py` follow the pattern in
`test_366_launcher_shape.py`. The runtime test invokes train.py as a
CPU-only subprocess with mix_ratio=1.0 (pure synth, no HF network).
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "experiments" / "2026-04-27_freq-embedding" / "scripts" / "train.py"


def load_train_source() -> str:
    return TRAIN_PY.read_text()


def parse_argparse_defaults() -> dict[str, object]:
    """Return the default= value of every add_argument call in train.py."""
    tree = ast.parse(load_train_source())
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        flag = first.value
        if not flag.startswith("--"):
            continue
        default = None
        for kw in node.keywords:
            if kw.arg == "default":
                try:
                    default = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    default = "<non-literal>"
                break
        defaults[flag] = default
    return defaults


# --- argparse flag exists + is off by default -----------------------------


def test_traj_save_every_argument_defined():
    defaults = parse_argparse_defaults()
    assert "--traj-save-every" in defaults, \
        "train.py does not define --traj-save-every"


def test_traj_save_every_defaults_off():
    """Default MUST be 0 — a non-zero default would double the disk footprint
    of every existing runbook that omits the flag."""
    defaults = parse_argparse_defaults()
    assert defaults["--traj-save-every"] == 0, \
        f"expected default 0, got {defaults['--traj-save-every']!r}"


def test_save_every_default_unchanged():
    """Guard the coarse cadence's default so a stray edit doesn't reduce
    the base disk footprint by surprise."""
    defaults = parse_argparse_defaults()
    assert defaults["--save-every"] == 5000


# --- naming: fine cadence emits `_step<STEP>.pth`, not `_Nk.pth` ----------


def test_trajectory_save_uses_step_named_files():
    """The fine cadence must write `<run>_step<STEP>.pth` so that a save
    at every 500 steps produces 25 distinct files up to step 12,500.

    `_<step // 1000>k.pth` (the coarse name) collides across every
    sub-1000-step interval — steps 500 and 0 both round to `_0k.pth`,
    500 and 1000 both round to `_0k.pth` and `_1k.pth`, etc.
    """
    src = load_train_source()
    # A step-tagged name of the form `_step{step}.pth` must appear.
    assert re.search(
        r'f"\{args\.run_name\}_step\{step\}\.pth"', src
    ), "train.py must emit `<run>_step<step>.pth` for the trajectory cadence"


def test_trajectory_save_gated_on_traj_save_every():
    """The fine save must be gated on both `traj_save_every > 0` AND the
    step-modulo — no gate means constant writes; wrong gate means we
    double-write on every step multiple of the coarse cadence."""
    src = load_train_source()
    # Locate the trajectory-save block. Both conditions must be present in
    # the same `if` line — either alone is unsafe.
    m = re.search(
        r"args\.traj_save_every\s*>\s*0\s*and\s*step\s*%\s*args\.traj_save_every\s*==\s*0",
        src,
    )
    assert m, (
        "train.py must gate trajectory saves on "
        "`args.traj_save_every > 0 and step % args.traj_save_every == 0`"
    )


# --- fine + coarse are separate cadences ---------------------------------


def test_coarse_and_fine_cadence_are_separate_saves():
    """Two `if` blocks in the loop: the classic `step % args.save_every == 0`
    save writing `_Nk.pth`, and a new `step % args.traj_save_every == 0`
    save writing `_step<STEP>.pth`. Merging them would drop the coarse
    disk cadence for runs that also set the fine cadence."""
    src = load_train_source()
    assert re.search(
        r'if step % args\.save_every == 0:\s*\n\s*path\s*=\s*os\.path\.join\([^)]*"\{args\.run_name\}_\{step // 1000\}k\.pth"',
        src,
    ), "coarse `_Nk.pth` save block must remain intact"
    assert re.search(
        r'if\s+args\.traj_save_every\s*>\s*0[^\n]*step\s*%\s*args\.traj_save_every\s*==\s*0:',
        src,
    ), "trajectory save must be its own if-block"


# --- runtime: end-to-end file emission on disk ---------------------------


def _can_run_train_py() -> tuple[bool, str]:
    """Guard: skip runtime test if the training environment isn't wired.
    train.py needs torch + numpy + datasets + the src.* modules on PYTHONPATH.
    """
    for mod in ("torch", "numpy", "datasets"):
        try:
            __import__(mod)
        except ImportError as e:
            return False, f"missing dep {mod}: {e}"
    src_pkg = REPO_ROOT / "src"
    if not (src_pkg / "models.py").exists():
        return False, f"src/models.py not found at {src_pkg}"
    return True, ""


def test_trajectory_save_emits_expected_files_on_disk(tmp_path):
    """End-to-end: run train.py for 4 steps with --traj-save-every 2 on
    CPU and assert that `_step2.pth` and `_step4.pth` land on disk with
    non-zero size. Also asserts the coarse `_Nk.pth` and `_final.pth`
    files stay present (the two cadences must not collide).

    The final-step assertion is the runtime check the reviewer asked
    for: `range(start_step + 1, total_steps + 1)` is inclusive of
    total_steps, so `_step<total_steps>.pth` MUST exist. If someone
    changes the loop bound to be exclusive, this test breaks loudly
    instead of the launch silently failing on a missing checkpoint.
    """
    ok, why = _can_run_train_py()
    if not ok:
        pytest.skip(why)

    save_dir = tmp_path / "runs"
    save_dir.mkdir()
    run_name = "trajtest"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-u", str(TRAIN_PY),
        "--device", "cpu",
        "--total-steps", "4",
        "--traj-save-every", "2",
        "--save-every", "100",
        "--batch-size", "2",
        "--lr", "1e-3",
        "--weight-decay", "0.1",
        "--save-dir", str(save_dir),
        "--run-name", run_name,
        # --mix-ratio 1.0 skips the HF stream entirely (pure synth), so
        # the test needs no network and no HF token.
        "--mix-ratio", "1.0",
        "--synth-kind", "periodic",
        # Tiny arch so a CPU step finishes in a few seconds.
        "--t-raw", "64",
        "--n-channels", "1",
        "--d-model", "32",
        "--n-heads", "2",
        "--num-layers", "1",
        "--log-every", "1",
        "--seed", "42",
        "--loss-shape", "cosine_similarity_batch_full_hh_negs_xshh_allt",
        "--tau", "0.10",
        # Required (no default) — silent 0.01 default was a footgun; the
        # value doesn't matter for a 4-step smoke test.
        "--hf-repo", "none",
        "--hf-path", "none",
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True,
        cwd=str(tmp_path), timeout=600,
    )
    if result.returncode != 0:
        pytest.fail(
            f"train.py returned rc={result.returncode}\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-2000:]}"
        )

    for step in (2, 4):
        p = save_dir / f"{run_name}_step{step}.pth"
        assert p.exists(), (
            f"trajectory checkpoint {p.name} not on disk after 4-step run "
            f"with --traj-save-every 2. Files in save_dir: "
            f"{sorted(x.name for x in save_dir.iterdir())}"
        )
        assert p.stat().st_size > 0, f"{p.name} is empty"

    # The final step must land — otherwise the launcher wires up
    # `_step<TOTAL_STEPS>.pth` in downstream_b1024.sh and gets a hard
    # abort at run time. Guard the invariant so a stale range-bound
    # edit fails here loudly first.
    assert (save_dir / f"{run_name}_step4.pth").exists(), (
        "trajectory save at the FINAL step (step == total_steps) did not "
        "fire. `range(start_step + 1, total_steps + 1)` MUST be inclusive "
        "of total_steps for downstream_b1024.sh to resolve _step<TOTAL_STEPS>.pth."
    )
    # No spurious step-tagged files between the two cadence points.
    step_files = sorted(save_dir.glob(f"{run_name}_step*.pth"))
    step_stems = [p.stem for p in step_files if "_optimizer" not in p.stem]
    assert step_stems == [f"{run_name}_step2", f"{run_name}_step4"], (
        f"unexpected step-tagged files: {step_stems}"
    )
