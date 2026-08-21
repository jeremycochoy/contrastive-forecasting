#!/usr/bin/env python3
"""#401 — the mean arm beside the summed arm, cell by cell.

That pair is the reason this protocol exists. The first run of this card summed
the k + 1 depth copies, so the f-side carried k + 1 times its k = 0 weight
against the terms that hold no f. Every k > 0 backbone of that run collapsed to
one direction and every cell scored far above the k = 0 parent. This protocol
takes the MEAN, so the f-side holds its k = 0 weight at every depth. The
question is what that does to the same cells.

One row per (phase, depth, stop, head budget), from whatever each arm has
scored. The summed arm holds 8 cells and stopped. The mean arm fills its own in
over days, so a row with one side is normal and is kept: a join that dropped
those rows would show an empty table until the last head landed.

`delta` is mean − sum, and GM-Relative MASE is lower-better, so a NEGATIVE
delta is the mean ahead.

Reads the two `scores.csv` tables `collect.sh` writes, one per arm. Writes a
CSV and the same table in Markdown, for the report.

Usage: compare_arms.py [--sum results/scores.csv] \\
           [--mean results/mean/scores.csv] \\
           [--out results/mean/arm_compare.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
import depth_colours as D                                   # noqa: E402

# The arm each table belongs to. `sum` is the stopped comparison arm.
ARMS = ("sum", "mean")

FIELDS = ["phase", "k", "variant", "stop", "head_steps", "encoder",
          "sum", "mean", "delta", "ratio", "better"]

# One cell is a (phase, depth, variant, stop, head budget, encoder). The
# variant is the training schedule: `base` is the card's, and a named one is a
# side run at the same three numbers. Without it in the key, a variant row and
# its base cell join as one, and the second one read overwrites the first.
KEY = ("phase", "k", "variant", "stop", "head_steps", "encoder")


def read_scores(path):
    """`{cell key: score}` from one arm's table. A missing file is an empty arm.

    The mean arm's `scores.csv` does not exist until its first head, and this
    runs from the first minute of the study.

    The summed arm's table was written before the `variant` column existed.
    It holds grid cells only, so a missing column reads as `base` and the two
    arms still join on the same key.
    """
    out = {}
    if not path or not Path(path).is_file():
        return out
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (int(r["phase"]), int(r["k"]), r.get("variant") or "base",
                   int(r["stop"]), int(r["head_steps"]), r["encoder"])
            out[key] = float(r["score"])
    return out


def order(key):
    """The study's own order: phase, then depth as the figures draw it, then
    the stop. A variant sorts after the base cell it sits beside."""
    phase, k, variant, stop, head, _ = key
    rank = D.DEPTHS_DRAWN.index(k) if k in D.DEPTHS_DRAWN else len(
        D.DEPTHS_DRAWN) + k
    return (phase, rank, stop, head, variant != "base", variant)


def join(sum_scores, mean_scores):
    """One row per cell either arm scored, in the study's order."""
    rows = []
    for key in sorted(set(sum_scores) | set(mean_scores), key=order):
        s, m = sum_scores.get(key), mean_scores.get(key)
        row = dict(zip(KEY, key))
        row["sum"] = "" if s is None else f"{s:.4f}"
        row["mean"] = "" if m is None else f"{m:.4f}"
        if s is None or m is None:
            row["delta"] = row["ratio"] = row["better"] = ""
        else:
            row["delta"] = f"{m - s:+.4f}"
            row["ratio"] = f"{m / s:.4f}"
            # Lower is better, so the smaller score wins. An exact tie names
            # neither: one eval is not enough to split two equal numbers.
            row["better"] = "mean" if m < s else ("sum" if s < m else "")
        rows.append(row)
    return rows


def markdown(rows):
    """The same table, for the report. `-` reads better than an empty cell."""
    head = ["phase", "k", "schedule", "backbone stop", "head steps", "sum",
            "mean", "delta", "better"]
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---:"] * len(head)) + "|"]
    for r in rows:
        out.append("| " + " | ".join([
            str(r["phase"]), f"k = {r['k']}", r["variant"], f"{r['stop']:,}",
            f"{r['head_steps']:,}",
            r["sum"] or "-", r["mean"] or "-", r["delta"] or "-",
            r["better"] or "-"]) + " |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sum", default=str(STUDY / "results" / "scores.csv"))
    ap.add_argument("--mean",
                    default=str(STUDY / "results" / "mean" / "scores.csv"))
    ap.add_argument("--out",
                    default=str(STUDY / "results" / "mean" / "arm_compare.csv"))
    a = ap.parse_args(argv)

    rows = join(read_scores(a.sum), read_scores(a.mean))
    if not rows:
        raise SystemExit(
            f"ABORT: no scored cell in either arm ({a.sum}, {a.mean})")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    out.with_suffix(".md").write_text(markdown(rows))

    paired = [r for r in rows if r["better"]]
    won = sum(1 for r in paired if r["better"] == "mean")
    print(f"wrote {out} and {out.with_suffix('.md')}  "
          f"({len(rows)} cell(s), {len(paired)} paired, "
          f"the mean ahead in {won})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
