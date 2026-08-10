#!/usr/bin/env python3
"""#373 figure 4 — does this code snapshot reproduce the published k = 0?

Every group-B delta crosses a code snapshot as well as the depth flag, so
the card asks for a reproduction check before any delta is read. This is it.

One row per backbone. The open marker is the number the parent report
publishes; the filled marker is what this study got when it retrained that
cell's k = 0 on this code.

The rows are grouped by MACHINE, because that is what the check found. Every
retrain on elisa landed on its published value and every retrain on a rented
box missed it, while three of the runs carry the same backbone seed. The
seed does not sort these rows; the machine does.

Two rows change something other than the code:

  B5 published backbone   #379's own checkpoint with this study's head and
                          eval, so a difference there cannot be a training
                          difference at all.
  B5·s3                   the protocol seed retrained on elisa. It is the
                          third corner of the seed / machine square, and it
                          is the row that says which of the two moved B5·s1.

Reads results/score_*.txt, published.py and the registry.

Usage: plot_reproduction.py --results results --out plots/reproduction.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                              # noqa: E402
import runs as R                                       # noqa: E402
from published import PUBLISHED, GATE, verdict         # noqa: E402

plt.rcParams.update(cc.rc())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    res = Path(args.results)

    have = {f.name[len("score_"):-len(".txt")]: f
            for f in res.glob("score_*.txt")}
    rows = []
    for run in R.reproductions(have):
        if run.head != "student":
            continue
        pub = PUBLISHED.get(run.cell, {}).get("student", {}).get(40)
        if pub is None:
            continue
        rows.append((run, pub, float(have[run.tag].read_text().strip())))
    if not rows:
        raise SystemExit(f"ABORT: no retrained k = 0 score in {res}")
    # elisa first, then the rented boxes, then the published backbone; a
    # tighter |Δ| sits higher inside a group.
    def group(run):
        return 0 if run.machine == "elisa" else 1 if run.run else 2
    rows.sort(key=lambda r: (group(r[0]), abs(r[2] - r[1])))

    fig, ax = plt.subplots(figsize=(9.6, 0.62 * len(rows) + 2.1))
    ys = list(range(len(rows)))
    for y, (run, pub, got) in zip(ys, rows):
        col = cc.colour(run.arm) if run.run else cc.INK_SOFT
        ax.plot([pub, got], [y, y], color=col, linewidth=2.0, zorder=2)
        ax.plot(pub, y, marker="o", markersize=11, markerfacecolor="#ffffff",
                markeredgecolor=col, markeredgewidth=2.0, zorder=3)
        ax.plot(got, y, marker="o", markersize=11, markerfacecolor=col,
                markeredgecolor=col, zorder=3)
        d = abs(got - pub)
        ax.annotate(f"|Δ| {d:.4f}   {verdict(d)}",
                    (max(pub, got), y), textcoords="offset points",
                    xytext=(14, -3), fontsize=8.5, color=cc.INK)
    for y in range(1, len(rows)):
        if group(rows[y][0]) != group(rows[y - 1][0]):
            ax.axhline(y - 0.5, color=cc.INK_SOFT, linewidth=0.9,
                       linestyle=(0, (3, 3)), zorder=1)

    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{run.arm}  seed {run.seed}  ·  {run.machine}" for run, _p, _g in rows],
        fontsize=9)
    ax.invert_yaxis()
    lo = min(min(p_, g) for _r, p_, g in rows)
    hi = max(max(p_, g) for _r, p_, g in rows)
    pad = (hi - lo) * 0.12 + 0.01
    ax.set_xlim(lo - pad, hi + pad * 6.5)
    ax.set_xlabel("GM-Relative MASE at bb40k, student head, 97 configs")
    ax.set_title("Published k = 0 against this study's own k = 0   "
                 f"(gate: |Δ| ≤ {GATE})", loc="left", fontsize=12)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=10,
               markerfacecolor="#ffffff", markeredgecolor=cc.INK_SOFT,
               markeredgewidth=2.0, label="published by the parent report"),
        Line2D([], [], marker="o", linestyle="none", markersize=10,
               markerfacecolor=cc.INK_SOFT, markeredgecolor=cc.INK_SOFT,
               label="measured by this study")]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(rows)} row(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
