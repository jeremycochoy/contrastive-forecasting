#!/usr/bin/env python3
"""#393 — say, in `decisions_all.csv` itself, which of its rows still stand.

The pooled decisions file is an append-only log of every branch any driver
ever recorded, and one (cell, stop) carries several rows. Five of them are
`budget_stop` rows written while a cell's bb100k teacher head was still
evaluating, so the extend rule had nothing to compare and the spend order
parked the cell instead. When those scores landed the cells were replayed
and the rule wrote its real branch — but `merge_pooled.sh` sorts and
deduplicates on the whole line, so row order carries no recency, and the
stale row sits beside the real one with nothing to tell them apart.

A reader taking one row per (cell, stop) gets the wrong branch for half the
study. For `arm5_combab_alignS` they get "the budget stopped it" when the
rule said extend and the cell is extending.

So each row gets three fields it can be read by on its own:

    rule_branch_now  the branch the extend rule gives at that (cell, stop),
                     re-derived here from the pooled scores with
                     ladder.ladder_decision — the same pure function the
                     drivers call
    status           rule    this row IS that branch
                     park    this row records where the spend order or the
                             session ceiling left the cell, not what the
                             rule decided; `rule_branch_now` is the answer
                     stale   neither, and the rule contradicts it
                     open    no scores at that stop, so nothing to check
    written_at       when this annotation ran, UTC

`results/stop_reason.csv` remains the one-row-per-cell answer. This makes
the log underneath it non-misleading rather than replacing it.

Usage:  python3 scripts/annotate_decisions.py [--decisions FILE]
                                              [--ladder FILE] [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from ladder import experiment_step_cap, ladder_decision  # noqa: E402
from stop_reason import PARK, prev_stop, read, scores_by_stop  # noqa: E402

ADDED = ["rule_branch_now", "status", "written_at"]
LF = "\n"


def annotate(decisions: list[dict], ladder: list[dict], stamp: str) -> list[dict]:
    cap = experiment_step_cap()
    # {(cell, stop): branch the rule gives there, from the pooled scores}
    rule_at: dict[tuple[str, int], str] = {}
    cells = {r["cell"] for r in ladder}
    for cell in cells:
        by_stop = scores_by_stop(ladder, cell)
        for stop, heads in by_stop.items():
            previous = by_stop.get(prev_stop(stop) or -1) or None
            rule_at[(cell, stop)] = ladder_decision(
                stop, previous, heads, step_cap=cap)["branch"]

    out = []
    for row in decisions:
        key = (row["cell"], int(row["stop"]))
        now = rule_at.get(key, "")
        if not now:
            status = "open"
        elif row["branch"] == now:
            status = "rule"
        elif row["branch"] in PARK:
            status = "park"
        else:
            status = "stale"
        out.append({**row, "rule_branch_now": now, "status": status,
                    "written_at": stamp})
    return out


def main() -> int:
    res = os.path.join(os.path.dirname(SCRIPTS_DIR), "results")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--decisions", default=os.path.join(res, "decisions_all.csv"))
    p.add_argument("--ladder", default=os.path.join(res, "ladder_all.csv"))
    p.add_argument("--out", default=None, help="default: rewrite --decisions")
    p.add_argument("--stamp", default=None,
                   help="UTC timestamp to write; default now")
    a = p.parse_args()
    out_path = a.out or a.decisions

    decisions, ladder = read(a.decisions), read(a.ladder)
    if not decisions:
        print(f"annotate_decisions: no rows in {a.decisions}", file=sys.stderr)
        return 1
    # Re-annotating an already-annotated file must not stack columns.
    base = [c for c in decisions[0] if c not in ADDED]
    decisions = [{k: r[k] for k in base} for r in decisions]

    stamp = a.stamp or dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    rows = annotate(decisions, ladder, stamp)

    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, base + ADDED, lineterminator=LF)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out_path)

    tally: dict[str, int] = {}
    for r in rows:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    print("  " + os.path.basename(out_path) + ": "
          + ", ".join(f"{n} {s}" for s, n in sorted(tally.items()))
          + f" ({len(rows)} rows, written_at {stamp})")
    for r in rows:
        if r["status"] in ("park", "stale"):
            print(f"    {r['status']:<6} {r['cell']:<24} @{r['stop']:<7}"
                  f" {r['branch']:<14} rule says {r['rule_branch_now']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
