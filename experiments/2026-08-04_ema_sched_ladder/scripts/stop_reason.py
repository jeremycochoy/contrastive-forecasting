#!/usr/bin/env python3
"""#393 — one row per cell: the stop it reached, the branch of the extend
rule that fired there, and what actually ended it.

The card asks for both facts and they are not the same fact. A cell can be
stopped by the rule (its heads did not improve) or parked by the spend
order (its box went back, or the session ceiling held it). Reading one as
the other is the single easiest way to misreport this study.

`decisions_all.csv` alone does not answer it. One (cell, stop) carries more
than one row:

* `budget_stop` was written for four cells while their bb100k teacher head
  was still evaluating, so the rule had nothing to compare yet. When those
  scores landed the cells were replayed on elisa and `climb()` wrote the
  branch the rule actually gives. `arm6_v2_ncpc_alignT` reads
  `budget_stop` and `none_down` at 100k; the rule stops it on its own
  numbers, 1.2955 -> 1.3904 student and 1.3266 -> 1.3646 teacher, neither
  down. `arm5_combab_alignS` reads `budget_stop` and `one_down`, which is
  the opposite error waiting to be made.
* `session_end` marks a cell held at the ceiling while the rule said
  extend.

And the row order does not disambiguate: `merge_pooled.sh` sorts the
pooled file by cell and stop, so within one stop the rows come out in
whatever order sorting leaves them, not the order they were decided in.

So the branch is not read off a row at all. It is re-derived from the
pooled scores with `ladder.ladder_decision`, the same pure function
`climb()` calls, and the recorded rows are carried alongside as a
cross-check. `rule_matches_record` is `no` when no recorded row at that
stop agrees, which is a bug in this file or in the pooling, not a result.

Usage:  python3 scripts/stop_reason.py [--decisions FILE] [--ladder FILE]
                                       [--out FILE] [--no-probe]

Writes `results/stop_reason.csv`:

    cell,last_stop,rule_branch,extend,heads_next,ended_by,recorded,
    rule_matches_record

`ended_by` is the answer to "why did this cell stop where it stopped":

    rule     the rule said stop
    running  a driver is extending it right now
    budget   the rule said extend; its box went back before it could
    session  the rule said extend; the session ceiling held it
    open     the rule said extend and nothing recorded why it did not

`--no-probe` skips the live-driver check, so the file is reproducible from
the CSVs alone (`running` then reads as `open`).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from ladder import (  # noqa: E402
    CELLS,
    STOP_FIRST,
    STOP_INCREMENT,
    STOP_SECOND,
    experiment_step_cap,
    ladder_decision,
)

COLUMNS = ["cell", "last_stop", "rule_branch", "extend", "heads_next",
           "ended_by", "recorded", "rule_matches_record"]

# Branches climb() writes for a reason other than the extend rule. They say
# where the cell was parked, not what the rule decided.
PARK = {"budget_stop": "budget", "session_end": "session"}

LF = "\n"


def prev_stop(stop: int) -> int | None:
    """The stop before `stop`, or None at the foot of the ladder.

    The inverse of `ladder.next_stop`: 40k, then 100k, then 100k at a time.
    Only used to find the scores the rule compared against.
    """
    if stop <= STOP_FIRST:
        return None
    if stop <= STOP_SECOND:
        return STOP_FIRST
    return stop - STOP_INCREMENT


def read(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return [{k: (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(fh)]


def scores_by_stop(ladder: list[dict], cell: str) -> dict[int, dict]:
    """{stop: {head: GM-Relative MASE}} for one cell, scored heads only."""
    out: dict[int, dict] = {}
    for r in ladder:
        if r["cell"] != cell or r["gm_rel_mase"] == "":
            continue
        out.setdefault(int(r["stop"]), {})[r["head"]] = float(r["gm_rel_mase"])
    return out


def live_cells() -> set[str]:
    """Cells with a ladder driver alive on this machine."""
    try:
        out = subprocess.run(["pgrep", "-af", r"ladder\.py"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    live: set[str] = set()
    for line in out.splitlines():
        m = re.search(r"--cells\s+([A-Za-z0-9_,]+)", line)
        if m:
            live.update(m.group(1).split(","))
    return live


def resolve(decisions: list[dict], ladder: list[dict],
            live: set[str] | None = None,
            cells: list[dict] | None = None) -> list[dict]:
    cells = cells if cells is not None else CELLS
    live = live if live is not None else set()
    cap = experiment_step_cap()
    rows = []
    for cell in cells:
        slug = cell["slug"]
        by_stop = scores_by_stop(ladder, slug)
        if not by_stop:
            continue
        last_stop = max(by_stop)
        previous = by_stop.get(prev_stop(last_stop) or -1) or None
        d = ladder_decision(last_stop, previous, by_stop[last_stop],
                            step_cap=cap)

        recorded = [r["branch"] for r in decisions
                    if r["cell"] == slug and int(r["stop"]) == last_stop]
        parked = [b for b in recorded if b in PARK]

        if not d["extend"]:
            ended = "rule"
        elif slug in live:
            ended = "running"
        elif parked:
            ended = PARK[parked[0]]
        else:
            ended = "open"

        rows.append({
            "cell": slug,
            "last_stop": last_stop,
            "rule_branch": d["branch"],
            "extend": int(d["extend"]),
            "heads_next": " ".join(d["heads"]),
            "ended_by": ended,
            "recorded": " ".join(sorted(set(recorded))),
            "rule_matches_record": "yes" if d["branch"] in recorded else "no",
        })
    return rows


def main() -> int:
    res = os.path.join(os.path.dirname(SCRIPTS_DIR), "results")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--decisions", default=os.path.join(res, "decisions_all.csv"))
    p.add_argument("--ladder", default=os.path.join(res, "ladder_all.csv"))
    p.add_argument("--out", default=os.path.join(res, "stop_reason.csv"))
    p.add_argument("--no-probe", action="store_true",
                   help="do not look for live drivers")
    a = p.parse_args()

    rows = resolve(read(a.decisions), read(a.ladder),
                   live=set() if a.no_probe else live_cells())
    if not rows:
        print(f"stop_reason: no scored cell in {a.ladder}", file=sys.stderr)
        return 1

    tmp = a.out + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, COLUMNS, lineterminator=LF)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, a.out)

    width = max(len(r["cell"]) for r in rows)
    for r in rows:
        flag = "" if r["rule_matches_record"] == "yes" else "   NO RECORDED ROW"
        print(f"{r['cell']:<{width}}  @{r['last_stop']:<7}"
              f" {r['rule_branch']:<14} ended_by={r['ended_by']:<8}"
              f" recorded=[{r['recorded']}]{flag}")
    print(f"  -> {a.out} ({len(rows)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
