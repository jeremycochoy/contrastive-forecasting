#!/usr/bin/env python3
"""#404 — the four seeds of one arm, and what they say about every ranking.

`s08b` was meant to measure this cell's run-to-run spread. It measured a
COLLAPSE instead: its contrastive AUC went 0.91 at 10,000 steps to 0.57 at
40,000, and it scored 1.5459 against s08's 1.1782. Every stable arm of the card
carries backbone seed 20260520, and the one arm at another seed collapsed. Two
readings fit that, and one arm cannot tell them apart:

  - 20260521 was unlucky, and a collapse here is rare, or
  - 20260520 was lucky, and this cell is unstable.

Under the second reading every ranking of this card rests on ONE seed. So the
card trains the same arm at two more seeds, 20260522 and 20260523, and this
module answers the four questions it then asks:

  1. the four scores of the arm, one per seed
  2. how many of the four collapsed, BY THE AUC AT THE STOP
  3. the spread over the seeds that did NOT collapse, in absolute terms
  4. whether that spread separates the 0.9 constant arm from the 0.95 constant
     arm, whose gap is 0.0088

---- What counts as a collapse ------------------------------------------------

The contrastive AUC says whether the backbone still tells a true future from a
false one. 0.5 is chance, and a backbone at chance has learned nothing. A score
table alone cannot show this, because a collapsed backbone still produces a
score.

The line is AUC_THRESHOLD, at the stop. It sits in a wide empty band: the five
stable arms hold 0.93 to 0.98 and the collapsed one holds 0.57. Any line inside
that band classifies the same six arms the same way, so the answer does not
depend on where in the band the line falls.

Usage:
  seed_report.py --scores results/scores.csv --sync-root ~/cf404_sync \
                 --out results/seed_report.md --table results/seed_table.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repeat_spread  # noqa: E402

# A backbone at or below this AUC at the stop has collapsed. The line is 0.80,
# not 0.50: no run of this study reached chance. See above.
AUC_THRESHOLD = 0.80

# The two arms the card asks about, and the schedule they share.
ALPHA_A, ALPHA_B, SCHEDULE = 0.9, 0.95, "fixed"


def read_scores(path: str) -> list[dict]:
    """`scores.csv` as rows with numbers, in the order the file holds them."""
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append({
                "arm": r["arm"],
                "alpha": float(r["alpha"]),
                "schedule": r["schedule"],
                "ramp": int(r["ramp"]),
                "seed": int(r["seed"]),
                "align_w": float(r.get("align_w") or 1.0),
                "stop": int(r["stop"]),
                "score": float(r["score"]),
            })
    return out


def auc_series(sync_root: Path, arm: str, stop: int = 40000):
    """`(steps, auc)` of one arm, from its backbone losses CSV.

    The tree holds one directory per box and one arm per directory. An arm
    trained on a rented box has TWO copies of one file: the box's own sync
    tree, and the canonical tree the driver pulls into. The sync copy is
    whatever the last 15-minute tick landed, so it can stop short of the run.
    The LONGEST copy is the run, and `glob` gives no useful order.
    """
    kk = stop // 1000
    hits = [h for h in sync_root.glob(f"*/sync/{arm}/*/leg_{kk}k/*_losses.csv")
            if not h.name.endswith(".prev")]
    if not hits:
        return [], []
    steps, auc = [], []
    with open(max(hits, key=lambda h: h.stat().st_size), newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("auc"):
                steps.append(int(r["step"]))
                auc.append(float(r["auc"]))
    return steps, auc


def auc_at(sync_root: Path, arm: str, stop: int = 40000) -> float | None:
    """The AUC of one arm at the stop, or `None` when no CSV is here."""
    steps, auc = auc_series(sync_root, arm, stop)
    if not steps:
        return None
    for s, a in zip(reversed(steps), reversed(auc)):
        if s <= stop:
            return a
    return None


def collapsed(auc: float | None) -> bool:
    """Whether an arm at this AUC has collapsed, against `AUC_THRESHOLD`.

    An arm with no CSV is NOT called collapsed. A missing file is a missing
    measurement, and reporting it as a collapse would invent a result.
    """
    return auc is not None and auc <= AUC_THRESHOLD


def _seed(r: dict) -> int:
    """One row's backbone seed as a number.

    Two readers build these rows — this module's own and `pr_comment.py`'s —
    and one of them keeps the seed as text. A comparison between text and a
    number then fails at the sort, hours into a round.
    """
    try:
        return int(r.get("seed", 0) or 0)
    except (TypeError, ValueError):
        return 0


def family(rows: list[dict]) -> list[dict]:
    """The largest set of rows that share a momentum, a schedule, a ramp AND
    an L_align weight.

    Nothing names `s08` here. The repeat family is whichever cell this card
    trained most often, so an arms table that repeats another cell reports that
    one instead.

    The align weight is in the key because a repeat family measures ONE cell
    trained more than once. `w3_s08` shares a momentum, a schedule and a ramp
    with `s08` and moves the weight, so it is a second cell, not a second seed.
    A key without the weight would count it as one and report a deliberate
    change to the objective as run-to-run noise.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[(r["alpha"], r["schedule"], r["ramp"],
                r.get("align_w", 1.0))].append(r)
    if not groups:
        return []
    best = max(groups.values(), key=lambda g: (len(g), -min(_seed(x) for x in g)))
    return sorted(best, key=_seed)


def spread(rows: list[dict]) -> float | None:
    """The absolute range of the scores of `rows`, or `None` below two rows."""
    if len(rows) < 2:
        return None
    return max(r["score"] for r in rows) - min(r["score"] for r in rows)


def report(rows: list[dict], sync_root: Path, stop: int = 40000) -> dict:
    """Every number the card asks for, as one dictionary."""
    fam = family(rows)
    for r in fam:
        r["auc"] = auc_at(sync_root, r["arm"], stop)
        r["collapsed"] = collapsed(r["auc"])
    stable = [r for r in fam if not r["collapsed"]]
    d = spread(stable)
    gap = None
    a = repeat_spread.cell(rows, ALPHA_A, SCHEDULE)
    b = repeat_spread.cell(rows, ALPHA_B, SCHEDULE)
    if a and b:
        gap = abs(a["score"] - b["score"])
    return {
        "family": fam,
        "stable": stable,
        "n_collapsed": sum(1 for r in fam if r["collapsed"]),
        "spread": d,
        "gap": gap,
        "separates": None if (d is None or gap is None) else gap > d,
        "separation": "" if d is None else
        repeat_spread.separation(rows, d, ALPHA_A, ALPHA_B, SCHEDULE),
    }


def spread_sentence(rows: list[dict], rep: dict) -> str:
    """The measured spread in one sentence, for the table.

    The spread is measured over the seeds of ONE arm that did not collapse. The
    list of arms it cannot separate is read over the WHOLE table: a spread of
    this cell applies to every arm of this cell, not only to the repeats.
    """
    stable, d = rep["stable"], rep["spread"]
    if d is None:
        return ""
    names = ", ".join(f"`{r['arm']}`" for r in stable)
    text = (f"{names} are one arm at {len(stable)} backbone seeds that did not "
            f"collapse. They span {d:.4f} "
            f"({d / min(r['score'] for r in stable):.1%}), which is the widest "
            f"repeat this card measures.")

    # THE SPREAD BELONGS TO ONE CELL, NOT TO THE TABLE. This sentence read
    # "all sit within that spread of the best score, so this card does not
    # rank them", over a list that held the WINNER. The winner is the one
    # other cell trained twice, its own two seeds span far less than this,
    # and both of them beat every other arm's best seed. So the claim the
    # spread supports is narrower: it is the arms at ONE seed that this card
    # does not separate.
    win = min(rows, key=lambda r: r["score"])
    win_seeds = [r for r in repeat_spread.seeds_of(rows, win)
                 if not r.get("collapsed")]
    single = [r["arm"] for r in sorted(rows, key=lambda r: r["score"])
              if len(repeat_spread.seeds_of(rows, r)) < 2]
    if len(win_seeds) > 1:
        lo = min(r["score"] for r in win_seeds)
        hi = max(r["score"] for r in win_seeds)
        rest = [r["score"] for r in rows
                if r["arm"] not in {x["arm"] for x in win_seeds}]
        text += (f" The best cell holds {len(win_seeds)} seeds of its own, "
                 f"{lo:.4f} to {hi:.4f}, a span of {hi - lo:.4f}. Its worst "
                 f"seed sits {min(rest) - hi:.4f} from the best seed of every "
                 f"other arm, {min(rest):.4f}.")
    if len(single) > 1:
        text += (" " + ", ".join(f"`{n}`" for n in single) + " carry one seed "
                 "each, and this card does not separate them from each other.")
    return text


def markdown(rep: dict, stop: int = 40000) -> str:
    """The four answers, in the order the card asks them."""
    fam, stable = rep["family"], rep["stable"]
    if not fam:
        return "No repeat family in this table."
    name = fam[0]["arm"]
    lines = [
        f"**The {name} arm at {len(fam)} backbone seeds.** Alpha "
        f"{fam[0]['alpha']} rising to 1.0 at {fam[0]['ramp']}, k = 32, mean "
        f"reduction, align target teacher, {stop} backbone steps, 30,000 head "
        f"steps, head seed 20260722, the 97-config eval.",
        "",
        "| arm | backbone seed | AUC at 40,000 | GM-Relative MASE | verdict |",
        "|---|---|---|---|---|",
    ]
    for r in fam:
        auc = "no CSV" if r["auc"] is None else f"{r['auc']:.3f}"
        verdict = "**collapsed**" if r["collapsed"] else "stable"
        lines.append(f"| `{r['arm']}` | {_seed(r)} | {auc} | "
                     f"{r['score']:.4f} | {verdict} |")
    lines.append("")
    lines.append(
        f"**{rep['n_collapsed']} of {len(fam)} collapsed**, by the AUC at "
        f"{stop} steps against a line at {AUC_THRESHOLD}. The stable arms of "
        f"this card hold 0.93 to 0.98 and the collapsed one holds 0.57, so any "
        f"line inside that band gives the same count.")
    if rep["spread"] is None:
        lines.append("")
        lines.append("Fewer than two seeds survived, so this round measures no "
                     "spread.")
        return "\n".join(lines)
    names = ", ".join(f"`{r['arm']}`" for r in stable)
    lines.append("")
    lines.append(
        f"**The spread over the {len(stable)} seeds that did NOT collapse is "
        f"{rep['spread']:.4f}** in absolute terms: {names} span "
        f"{min(r['score'] for r in stable):.4f} to "
        f"{max(r['score'] for r in stable):.4f}.")
    if rep["separation"]:
        lines.append("")
        lines.append(f"**{rep['separation']}**")
    return "\n".join(lines)


def write_table(rep: dict, path: str) -> None:
    """The family as a CSV, so a reader does not parse the prose."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "alpha", "schedule", "ramp", "seed", "stop",
                    "auc_at_stop", "score", "collapsed"])
        for r in rep["family"]:
            w.writerow([r["arm"], r["alpha"], r["schedule"], r["ramp"],
                        _seed(r), r.get("stop", ""),
                        "" if r["auc"] is None else f"{r['auc']:.6f}",
                        f"{r['score']:.4f}", int(r["collapsed"])])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--sync-root", required=True)
    p.add_argument("--stop", type=int, default=40000)
    p.add_argument("--out")
    p.add_argument("--table")
    args = p.parse_args(argv)

    rows = read_scores(args.scores)
    rep = report(rows, Path(args.sync_root).expanduser(), args.stop)
    text = markdown(rep, args.stop)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    if args.table:
        write_table(rep, args.table)
        print(f"wrote {args.table} — {len(rep['family'])} row(s)")
    if not args.out:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
