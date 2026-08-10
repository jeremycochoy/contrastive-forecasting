#!/usr/bin/env python3
"""#373 figure 2 — GM-Relative MASE by horizon term, depth against k = 0.

This is the figure the hypothesis lives on. The rollout deficit is a horizon
effect: #327 reports short 0.976, medium 1.41, long 1.37, so a per-step gain
that compounds should land on medium and long and leave short alone. A gain
spread evenly over all three is a better model, not the mechanism this card
proposes.

Left panel: the level, one group per horizon term. Right panel: the change
against the same arm's own k = 0, as a percentage, with the card's success
criterion drawn on it (medium+long at least 5% better, short losing less
than 2%).

Reads results/splits.csv and resolves tags through runs.py.

Usage: plot_horizon_split.py --splits results/splits.csv --head student \\
           --out plots/horizon_split_student.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.patches import Patch                   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                              # noqa: E402
import runs as R                                       # noqa: E402

TERMS = ["short", "medium", "long"]
NCONF = {"short": 55, "medium": 21, "long": 21}
plt.rcParams.update(cc.rc())


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["split"] in ("term", "all"):
            out.setdefault(r["stop"], {})[r["name"]] = float(r["gm_rel_mase"])
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--head", default="student")
    args = p.parse_args(argv)

    data = load(args.splits)
    pairs = [(arm, k, base, deep)
             for arm, head, k, base, deep in R.pairs(data)
             if head == args.head
             and base.tag in data and deep.tag in data]
    if not pairs:
        raise SystemExit(f"ABORT: no arm has two depths on the {args.head} "
                         f"head in {args.splits}")
    pairs.sort(key=lambda r: (R.ARM_ORDER.index(r[0]), r[1]))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    w = 0.8 / (len(pairs) + 1)
    all_d = []

    # Left: the level. One bar per arm per depth, k = 0 hollow, depth solid.
    seen0 = set()
    slots = []
    for arm, k, base, deep in pairs:
        if base.tag not in seen0:
            seen0.add(base.tag)
            slots.append((arm, 0, base))
        slots.append((arm, k, deep))
    w = 0.84 / len(slots)
    for i, (arm, k, run) in enumerate(slots):
        col = cc.colour(arm)
        vals = [data[run.tag][t] for t in TERMS]
        xs = [t_i + i * w - 0.42 + w / 2 for t_i in range(len(TERMS))]
        # Hatch is the machine here as it is everywhere else in the report.
        # These bars are levels rather than deltas, so it marks the run that
        # trained on a rented box rather than a pair that crosses one.
        axL.bar(xs, vals, width=w * 0.9,
                color=col if k else "#ffffff", edgecolor=col, linewidth=1.1,
                hatch=None if run.machine == "elisa" else cc.CROSSED_HATCH)

    # Right: the change, one line per (arm, depth).
    for arm, k, base, deep in pairs:
        d = [100.0 * (data[deep.tag][t] / data[base.tag][t] - 1.0)
             for t in TERMS]
        all_d += d
        # The legend says what the pair holds fixed and whether the report
        # stands behind it, because this panel is where the card's criterion
        # is read and two of these five pairs cross a machine.
        note = "" if R.machine_held(base, deep) else "  (two machines)"
        if arm in R.RETRACTED:
            note += "  ✗"
        axR.plot(range(len(TERMS)), d, marker="o", markersize=7,
                 color=cc.colour(arm), linewidth=2.0,
                 linestyle=cc.style(k),
                 markerfacecolor=cc.face(arm),
                 markeredgecolor=cc.colour(arm), markeredgewidth=1.6,
                 label=f"{arm}  k = {k}{note}")
        # Label the long end only. The dashed criterion lines carry the rest;
        # a number on every point is noise.
        axR.annotate(f"{d[-1]:+.1f}%", (len(TERMS) - 1, d[-1]),
                     textcoords="offset points", xytext=(9, -3), ha="left",
                     fontsize=8, color=cc.INK,
                     bbox=dict(fc="#ffffff", ec="none", pad=0.5))

    axL.axhline(1.0, color=cc.PARITY, linewidth=1.2, zorder=0)
    axL.annotate("seasonal naive", (-0.46, 1.0), fontsize=7.5, ha="left",
                 va="bottom", color=cc.INK_SOFT,
                 bbox=dict(fc="#ffffff", ec="none", pad=0.6))
    axL.set_ylim(0, max(v for t in TERMS for run in (s[2] for s in slots)
                        for v in [data[run.tag][t]]) * 1.32)
    axL.set_xticks(range(len(TERMS)))
    axL.set_xticklabels([f"{t}\n({NCONF[t]} configs)" for t in TERMS])
    axL.set_ylabel("GM-Relative MASE  (lower is better)")
    axL.set_title(f"Level, {args.head} head", loc="left")
    axL.legend(handles=[Patch(facecolor="#ffffff", edgecolor=cc.INK_SOFT,
                              label="k = 0"),
                        Patch(facecolor=cc.INK_SOFT, label="k > 0"),
                        Patch(facecolor="#ffffff", edgecolor=cc.INK,
                              hatch=cc.CROSSED_HATCH, linewidth=0.6,
                              label="trained on a rented box")]
               + [Patch(facecolor=cc.COLOUR[c], label=c)
                  for c in ("B9", "B1", "B5", "A3")],
               loc="upper left", fontsize=8, ncol=2)

    axR.axhline(0.0, color=cc.INK, linewidth=1.0)
    lo = min(0.0, min(all_d)) - 6.0
    hi = max(4.0, max(all_d)) + 6.0
    axR.set_ylim(lo, hi)
    axR.axhspan(lo, -5.0, color=cc.COLOUR["B9"], alpha=0.07, zorder=0)
    axR.axhline(-5.0, color=cc.INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
    axR.axhline(2.0, color=cc.INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
    box = dict(fc="#ffffff", ec="none", pad=0.8)
    axR.annotate("criterion: med+long ≤ −5%", (-0.32, -5.0), fontsize=7.5,
                 color=cc.INK_SOFT, ha="left", va="top", bbox=box)
    axR.annotate("short must stay under +2%", (-0.32, 2.0), fontsize=7.5,
                 color=cc.INK_SOFT, ha="left", va="bottom", bbox=box)
    axR.set_xticks(range(len(TERMS)))
    axR.set_xticklabels(TERMS)
    axR.set_xlim(-0.35, 2.75)
    axR.set_ylabel("depth k against its own k = 0  (%, negative is better)")
    axR.set_title("Change, with the card's criterion", loc="left")
    axR.legend(loc="best", fontsize=8, ncol=2)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(pairs)} pair(s), {args.head} head)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
