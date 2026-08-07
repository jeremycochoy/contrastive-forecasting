#!/usr/bin/env python3
"""#393 — rebuild ladder rows from the eval output on elisa, not from a box.

Every published number in this card is produced on elisa: the cells that
train here evaluate here, and the cells that train on a rented box send
their two checkpoints over and `eval_broker.sh` evaluates them here too.
So elisa's disk already holds every score, its 97-config `all_results.csv`
and its `summary.txt`, whether or not the box that trained the backbone is
still rented.

`results/ladder.csv` does not, and two things keep it behind:

* `ladder.py:stop_scores()` runs a stop's two heads concurrently and
  appends BOTH rows only after both futures return. A stop whose student
  scored hours ago and whose teacher is still evaluating contributes no
  row at all. Box G sat exactly there: `arm6_v2_ncpc_alignT` bb100k
  student measured 1.3904 at 16:12, and the box's ladder.csv still ended
  at the 40k stop.
* The pooled table is built by copying each machine's ladder.csv. Release
  the box and that copy stops being reachable, so a score that cost a
  100k-step backbone and a 30k-step head would leave no trace in the
  table even though the eval that produced it ran here.

Releasing idle boxes is how this card affords anything past bb100k, so
the pooled table must not depend on a box being alive. This writes
`results/per_machine/ladder_evaldirs.csv` in the same schema as every
other per-machine file, and `merge_pooled.sh` folds it in with the rest:
identical rows collapse on the whole-line dedup, and a row only this file
has is kept.

Usage:  python3 scripts/scores_from_evals.py [--runs DIR] [--out FILE]
        python3 scripts/scores_from_evals.py --check   # print, write nothing

A score is only emitted with its evidence beside it: `gift/all_results.csv`
carrying exactly EXPECTED_CONFIGS data rows. A partial eval writes no
score file, but a truncated pull or a half-copied directory could leave
one, and a number with no 97-config table behind it is not a result.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from ladder import (  # noqa: E402
    CELLS,
    LADDER_COLUMNS,
    RUNS_DEFAULT,
    alpha_at,
    head_steps_for,
)

EXPECTED_CONFIGS = 97
CELL_BY_SLUG = {c["slug"]: c for c in CELLS}

# csv.writer's default line terminator is CRLF. Every other file in
# results/per_machine/ is written by ladder.py or copied off a box and ends
# in LF, and merge_pooled.sh compares headers as strings and deduplicates
# rows as whole lines — so a CR would make this file's header disagree with
# its neighbours and would make an identical row from a box look different
# from the same row here. Pin it.
LF = "\n"


def writer(fh):
    return csv.writer(fh, lineterminator=LF)

# <cell>/eval/bb<K>k_<enc>/  — a cell that trains on elisa.
# _broker/<box>/<cell>/bb<K>k_<enc>/  — a cell that trains on a rented box.
STOP_DIR_RE = re.compile(r"^bb(\d+)k_(student|teacher)$")


def config_rows(gift_dir: str) -> int:
    """Data rows in the eval's per-config table, or -1 if there is none."""
    path = os.path.join(gift_dir, "all_results.csv")
    try:
        with open(path, newline="") as fh:
            return max(sum(1 for _ in csv.reader(fh)) - 1, 0)
    except OSError:
        return -1


def read_score(path: str) -> float | None:
    try:
        with open(path) as fh:
            return float(fh.read().strip())
    except (OSError, ValueError):
        return None


def scan(runs: str) -> tuple[list[list], list[str]]:
    """Every (cell, stop, head) with a score and its 97 configs on disk."""
    rows: list[list] = []
    notes: list[str] = []
    seen: set[tuple[str, int, str]] = set()

    for cell_dir, stop_dir, score_path in candidates(runs):
        cell = os.path.basename(cell_dir)
        spec = CELL_BY_SLUG.get(cell)
        if spec is None:
            notes.append(f"skip {cell}: not a cell of this card")
            continue
        m = STOP_DIR_RE.match(os.path.basename(stop_dir))
        if m is None:
            continue
        stop, head = int(m.group(1)) * 1000, m.group(2)

        score = read_score(score_path)
        if score is None:
            continue
        n = config_rows(os.path.join(stop_dir, "gift"))
        if n != EXPECTED_CONFIGS:
            notes.append(f"skip {cell} bb{stop // 1000}k {head}: "
                         f"{n} config rows, expected {EXPECTED_CONFIGS}")
            continue

        key = (cell, stop, head)
        if key in seen:
            continue
        seen.add(key)
        rows.append([cell, spec["arm"], spec["align"] or "", stop, head,
                     head_steps_for(stop), f"{alpha_at(stop):.6f}",
                     f"{score:.6f}"])

    rows.sort(key=lambda r: (r[0], r[3], r[4]))
    return rows, notes


def candidates(runs: str):
    """(cell dir, stop dir, score file) for both eval layouts."""
    broker = os.path.join(runs, "_broker")
    for box in sorted(_subdirs(broker)):
        for cell_dir in sorted(_subdirs(box)):
            for stop_dir in sorted(_subdirs(cell_dir)):
                yield cell_dir, stop_dir, os.path.join(stop_dir, "score.txt")

    for cell_dir in sorted(_subdirs(runs)):
        if os.path.basename(cell_dir) == "_broker":
            continue
        eval_dir = os.path.join(cell_dir, "eval")
        for stop_dir in sorted(_subdirs(eval_dir)):
            # ladder.py's own evaluate() writes the score one level up.
            name = os.path.basename(stop_dir)
            yield cell_dir, stop_dir, os.path.join(eval_dir, f"score_{name}.txt")


def _subdirs(path: str) -> list[str]:
    try:
        return [os.path.join(path, e) for e in os.listdir(path)
                if os.path.isdir(os.path.join(path, e))]
    except OSError:
        return []


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--runs", default=os.environ.get("CF393_RUNS", RUNS_DEFAULT))
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(SCRIPTS_DIR), "results", "per_machine",
        "ladder_evaldirs.csv"))
    p.add_argument("--check", action="store_true",
                   help="print the rows and write nothing")
    a = p.parse_args()

    rows, notes = scan(a.runs)
    for n in notes:
        print(f"[scores] {n}", file=sys.stderr)
    if not rows:
        print("[scores] no eval directory on this machine carries a score",
              file=sys.stderr)
        return 0

    if a.check:
        w = writer(sys.stdout)
        w.writerow(LADDER_COLUMNS)
        w.writerows(rows)
        return 0

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = writer(fh)
        w.writerow(LADDER_COLUMNS)
        w.writerows(rows)
    os.replace(tmp, a.out)
    print(f"[scores] {os.path.basename(a.out)}: {len(rows)} rows "
          f"from eval output under {a.runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
