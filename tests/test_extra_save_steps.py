"""Unit tests for the periodic-checkpoint schedule used by train.py (#379).

train.py's inner loop snapshots at `step % save_every == 0`. Some experiments
(notably #379, which evaluates the small-model backbone at 2.5k / 25k / 50k /
100k / 200k) need a few off-cadence checkpoints on top of the regular cadence.
The `snapshot_schedule` helper encodes the union of the two, and this test
pins the union semantics: any duplicate between `save_every` and
`extra_save_steps` is emitted once, not twice, and out-of-range extras are
ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO_ROOT / "experiments" / "2026-04-27_freq-embedding" / "scripts"
sys.path.insert(0, str(TRAIN_DIR))

from train import parse_extra_save_steps, should_snapshot  # noqa: E402


def test_parse_extra_save_steps_none():
    assert parse_extra_save_steps(None) == frozenset()
    assert parse_extra_save_steps("") == frozenset()


def test_parse_extra_save_steps_comma_list():
    assert parse_extra_save_steps("2500,25000") == frozenset({2500, 25000})
    assert parse_extra_save_steps("2500, 25000 , 50000") == frozenset(
        {2500, 25000, 50000})


def test_should_snapshot_regular_cadence():
    extras = frozenset()
    assert should_snapshot(10000, 10000, extras)
    assert not should_snapshot(9999, 10000, extras)
    assert should_snapshot(200000, 10000, extras)


def test_should_snapshot_extra_only():
    extras = frozenset({2500, 25000})
    assert should_snapshot(2500, 10000, extras)
    assert should_snapshot(25000, 10000, extras)
    assert not should_snapshot(2501, 10000, extras)


def test_should_snapshot_extras_union_with_regular():
    extras = frozenset({2500, 25000})
    assert should_snapshot(10000, 10000, extras)
    assert should_snapshot(20000, 10000, extras)
    assert should_snapshot(25000, 10000, extras)
    assert should_snapshot(30000, 10000, extras)


def test_should_snapshot_zero_step_never_saves():
    # Regular cadence check `step % save_every == 0` would fire on step 0;
    # the caller guards against this because save_snapshot at step 0 is a
    # duplicate of the initial state. Extras never include 0 either.
    assert not should_snapshot(0, 10000, frozenset())
    assert not should_snapshot(0, 10000, frozenset({0}))
