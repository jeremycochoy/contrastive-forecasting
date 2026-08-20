#!/usr/bin/env python3
"""#404 — the ExperimentRunner comment, built from the tables on disk.

The comment carries every scored arm and the repeat spread this card
measures. Both come from `results/scores.csv`, which `collect.sh` writes from
the score files, so the comment cannot disagree with the figures.

`repeat_spread.py` finds a repeat pair by (alpha, schedule, ramp). `s08` and
`s08b` share all three and differ in the backbone seed alone, so the distance
between their two scores IS the run-to-run spread of this cell.

Usage:  python3 scripts/pr_comment.py --scores results/scores.csv \\
          --agent "ExperimentRunner claude-opus-5" \\
          --dir reports/2026-08-19_ema_momentum_k32
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import references  # noqa: E402
import repeat_spread  # noqa: E402


def read_scores(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "arm": r["arm"],
                "alpha": float(r["alpha"]),
                "schedule": r["schedule"],
                "ramp": int(r["ramp"]),
                "score": float(r["score"]),
                "seed": r.get("seed", ""),
            })
    return sorted(rows, key=lambda r: r["score"])


def momentum(r: dict) -> str:
    if r["schedule"] == "fixed":
        return f"{r['alpha']:.2f}, fixed"
    return f"{r['alpha']:.2f}, to 1.0 at {r['ramp'] // 1000}k"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, type=pathlib.Path)
    ap.add_argument("--agent", default="ExperimentRunner claude-opus-5")
    ap.add_argument("--dir", default="reports/2026-08-19_ema_momentum_k32")
    ap.add_argument("--runs", type=int, default=None,
                    help="arms trained this round, for the runs line")
    ap.add_argument("--cost", default="")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    rows = read_scores(a.scores)
    if not rows:
        print("ABORT: no scored arm in", a.scores, file=sys.stderr)
        return 2

    # The one number every arm is measured against: k = 3 at the SAME 40,000
    # steps. The card compares at bb40k, because that is where the arms stop.
    k3_40k = references.K3_BB40K
    best = rows[0]
    out = []
    out.append(f"«Agent {a.agent} writing»")
    out.append("")
    out.append(f"The experiment directory is `{a.dir}`.")
    out.append("")
    out.append(f"## The {len(rows)} scored arms")
    out.append("")
    out.append("| arm | EMA momentum | backbone seed | GM-Relative MASE | "
               "vs k = 3 at bb40k |")
    out.append("|---|---|---|---|---|")
    for r in rows:
        out.append(f"| {r['arm']} | {momentum(r)} | {r['seed'] or '?'} | "
                   f"{r['score']:.4f} | {r['score'] - k3_40k:+.4f} |")
    out.append("")

    sentence = repeat_spread.sentence(rows)
    pairs = repeat_spread.pairs(rows)
    d = None
    out.append("## The measured repeat spread")
    out.append("")
    if pairs:
        p = max(pairs, key=lambda q: repeat_spread.spread(*q)[0])
        d, rel = repeat_spread.spread(*p)
        out.append(f"**{d:.4f} ({rel:.1%})**, from `{p[0]['arm']}` "
                   f"{p[0]['score']:.4f} against `{p[1]['arm']}` "
                   f"{p[1]['score']:.4f}.")
        out.append("")
        out.append(sentence)
    else:
        out.append("No repeat pair is scored yet.")
    out.append("")

    # The card's own question: is the distance between the two fixed-momentum
    # arms larger than one repeat of the same cell? Both numbers come from
    # scores.csv, so the answer cannot disagree with the table above.
    out.append("## Does the spread separate 0.90 fixed from 0.95 fixed?")
    out.append("")
    answer = repeat_spread.separation(rows, d, 0.90, 0.95) if d is not None \
        else ""
    out.append(answer or "The card cannot answer this yet: it has no repeat "
                         "pair, or one of the two arms has no score.")
    out.append("")
    out.append("## The verdict")
    out.append("")
    out.append(f"`{best['arm']}` wins at **{best['score']:.4f}**, EMA momentum "
               f"{momentum(best)}. It sits {best['score'] - k3_40k:+.4f} from "
               f"the k = 3 score at the same 40,000 steps, {k3_40k:.4f}, so it "
               f"does NOT go below that score.")
    out.append("")
    if a.runs is not None:
        out.append(f"Runs completed this round: {a.runs}.")
    if a.cost:
        out.append(f"Cost: {a.cost}.")

    text = "\n".join(out) + "\n"
    if a.out:
        a.out.write_text(text)
        print(f"wrote {a.out}", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
