#!/usr/bin/env python3
"""#404 deliverable 4 — the table of scores, and the one statement that reads it.

The table holds the four arms and the five published references. The statement
names the momentum that wins, gives its distance to the k = 3 score at bb40k,
and says whether that arm goes below it.

1.0862 is the comparison, not 1.0660: both this card's arms and that number
stop at 40,000 backbone steps.

Usage:
  make_table.py --scores results/scores.csv --out results/table.md
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _references():
    spec = importlib.util.spec_from_file_location(
        "cf404_refs", HERE / "references.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = _references()

SCHEDULE_TEXT = {"fixed": "fixed", "ramp": "to 1.0 at 200k"}


def read_scores(path) -> list[dict]:
    """The rows of collect.sh's scores.csv, typed and ordered by score."""
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({"arm": r["arm"], "alpha": float(r["alpha"]),
                             "schedule": r["schedule"],
                             "score": float(r["score"])})
            except (KeyError, ValueError, TypeError):
                continue
    return sorted(rows, key=lambda r: r["score"])


def best(rows: list[dict]) -> dict:
    """The arm with the lowest GM-Relative MASE."""
    if not rows:
        raise SystemExit("ABORT: no arm is scored yet")
    return rows[0]


def beats_k3(rows: list[dict]) -> bool:
    """True when the best arm goes below the k = 3 score at bb40k."""
    return best(rows)["score"] < REF.K3_BB40K


def table_markdown(rows: list[dict]) -> str:
    """The card's table: this card's arms first, then the references."""
    out = ["| arm | EMA momentum | GM-Relative MASE | vs k = 3 at bb40k |",
           "|---|---|---|---|"]
    for r in rows:
        alpha = f"{r['alpha']:g}, {SCHEDULE_TEXT.get(r['schedule'], r['schedule'])}"
        out.append(f"| {r['arm']} | {alpha} | {r['score']:.4f} | "
                   f"{r['score'] - REF.K3_BB40K:+.4f} |")
    out += ["", "| reference | GM-Relative MASE |", "|---|---|"]
    for label, value in REF.TABLE:
        out.append(f"| {label} | {value:.4f} |")
    return "\n".join(out)


def statement(rows: list[dict]) -> str:
    """The one sentence the card asks for."""
    win = best(rows)
    delta = win["score"] - REF.K3_BB40K
    schedule = SCHEDULE_TEXT.get(win["schedule"], win["schedule"])
    verdict = ("goes below" if delta < 0 else "does not go below")
    band = (" It lands inside the k = 3 repeat band."
            if REF.enters_band(win["score"]) else "")
    return (f"The EMA momentum {win['alpha']:g} ({schedule}) wins, at "
            f"{win['score']:.4f}. It sits {delta:+.4f} from the k = 3 score "
            f"at bb40k, {REF.K3_BB40K:.4f}, so it {verdict} that score."
            f"{band}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if not Path(args.scores).is_file():
        raise SystemExit(f"ABORT: no scores table at {args.scores}")
    rows = read_scores(args.scores)
    text = table_markdown(rows) + "\n\n" + statement(rows) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
