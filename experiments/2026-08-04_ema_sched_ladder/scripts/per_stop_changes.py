#!/usr/bin/env python3
"""#393 — the raw change each head made between consecutive stops.

The card's caveat asks for this directly: one head seed per cell means a
change smaller than the head-seed spread does not separate from noise, so
the extend rule sometimes fires on noise. The remedy the card prescribes is
to publish the raw per-stop changes and let the reader judge, rather than to
buy more seeds.

Reads `results/ladder_all.csv` (the scores) and `results/decisions.csv` (the
branch that fired at each stop) and writes `results/per_stop_changes.csv`:
one row per cell per head per stop transition, with the change the extend
rule actually read.

Sign: negative means the score went down, which is what the rule calls
"down". `head_budget_moves` marks the 40k-to-100k transitions, where the
head budget goes 15,000 to 30,000 steps and so moves with the backbone.
"""
import csv
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
RES = EXP / "results"


def load_scores():
    rows = {}
    with open(RES / "ladder_all.csv") as fh:
        for r in csv.DictReader(fh):
            rows[(r["cell"], int(r["stop"]), r["head"])] = (
                float(r["gm_rel_mase"]), int(r["head_steps"]))
    return rows


def load_branches():
    """Branch recorded by the rule at each (cell, stop). Park rows are the
    operational record of a session ending, not a rule firing, so they are
    skipped."""
    out = {}
    path = RES / "decisions_all.csv"
    if not path.exists():
        path = RES / "decisions.csv"
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("status", "rule") != "rule":
                continue
            out[(r["cell"], int(r["stop"]))] = r["branch"]
    return out


def main():
    scores = load_scores()
    branches = load_branches()

    cells = sorted({c for c, _, _ in scores})
    out = []
    for cell in cells:
        stops = sorted({s for c, s, _ in scores if c == cell})
        for prev, cur in zip(stops, stops[1:]):
            for head in ("student", "teacher"):
                a = scores.get((cell, prev, head))
                b = scores.get((cell, cur, head))
                if a is None or b is None:
                    continue          # head dropped from the evaluation
                out.append({
                    "cell": cell,
                    "head": head,
                    "from_stop": prev,
                    "to_stop": cur,
                    "from_score": f"{a[0]:.4f}",
                    "to_score": f"{b[0]:.4f}",
                    "change": f"{b[0] - a[0]:+.4f}",
                    "went_down": "yes" if b[0] < a[0] else "no",
                    "branch_at_to_stop": branches.get((cell, cur), ""),
                    "head_budget_moves": "yes" if a[1] != b[1] else "no",
                })

    dst = RES / "per_stop_changes.csv"
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {dst} — {len(out)} transition(s)")

    # Echo the table so the run log carries the numbers too.
    for r in out:
        print(f"{r['cell']:24s} {r['head']:8s} "
              f"{r['from_stop']//1000:>3d}k->{r['to_stop']//1000:<4d}k "
              f"{r['from_score']} -> {r['to_score']}  {r['change']}  "
              f"{r['branch_at_to_stop']}")


if __name__ == "__main__":
    main()
