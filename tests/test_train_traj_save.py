"""Tests for the `--traj-save-every` fine-grained trajectory checkpoint
knob added for issue #369.

`--save-every` writes coarse `<run>_<K>k.pth` checkpoints — at sub-1000-step
intervals its `step // 1000` naming collides. `--traj-save-every` is a
separate cadence that writes `<run>_step<STEP>.pth` files, distinct from
the coarse cadence, safe at any interval, and independently resumable.

The tests here guard three things:

  1. The argparse flag exists and defaults to 0 (feature off).
  2. The save loop uses the fine name at exactly the trajectory cadence.
  3. Fine and coarse cadences do not collide on shared step multiples.

These are static analysis tests over `train.py` (importing it end-to-end
requires the training environment), following the pattern in
`test_366_launcher_shape.py`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


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
