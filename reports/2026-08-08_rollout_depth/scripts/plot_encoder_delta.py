#!/usr/bin/env python3
"""#373 — teacher encoder minus student encoder, at both depths.

The parent's `encoder_delta.png`, unchanged in form: one point per (cell,
depth), the teacher head's GM-Relative MASE minus the student head's. Below
zero means the teacher encoder gives the better head.

The card asks for this figure because the depth changes how much the
teacher enters the loss: every h index of a duplicated term shifts with the
depth, so MoCo keys and teacher targets are re-read k+1 times per step.

Reads results/splits.csv (`all` rows), written by split_scores.py.

Usage: plot_encoder_delta.py --splits results/splits.csv \\
           --out plots/encoder_delta.png
"""
from __future__ import annotations

import argparse
import csv
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

plt.rcParams.update(cc.rc())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    data = {}
    with open(args.splits) as fh:
        for r in csv.DictReader(fh):
            if r["split"] != "all":
                continue
            run = R.resolve(r["stop"])
            if run is None or run.role != "depth":
                continue
            data[(run.arm, run.k, run.head)] = float(r["gm_rel_mase"])

    pts = []
    for arm in R.ARM_ORDER:
        for k in sorted({kk for a, kk, _h in data if a == arm}):
            s, t = data.get((arm, k, "student")), data.get((arm, k, "teacher"))
            if s is not None and t is not None:
                pts.append((arm, k, t - s))
    if not pts:
        raise SystemExit("ABORT: no (arm, depth) has both heads in "
                         f"{args.splits}")

    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(pts) + 2.0))
    ax.axvline(0.0, color=cc.INK_SOFT, linewidth=1.0)
    for y, (arm, k, d) in enumerate(pts):
        ax.barh(y, d, height=0.55, color=cc.face(arm),
                alpha=1.0 if k else 0.45, edgecolor=cc.colour(arm),
                linewidth=1.4, hatch="///" if cc.hollow(arm) else None)
        ax.text(d + (0.003 if d >= 0 else -0.003), y, f"{d:+.4f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=8)
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels([f"{arm}  k = {k}" for arm, k, _d in pts], fontsize=8.5)
    ax.invert_yaxis()
    lim = max(abs(d) for _a, _k, d in pts) * 2.2 or 0.01
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("teacher head minus student head, GM-Relative MASE "
                  "(negative = teacher encoder wins)")
    ax.legend(handles=[Line2D([], [], color=cc.INK_SOFT, linewidth=8,
                              alpha=0.45, label="k = 0"),
                       Line2D([], [], color=cc.INK_SOFT, linewidth=8,
                              label="k > 0")],
              loc="best", frameon=False, fontsize=8)
    ax.set_title("Which encoder the head is trained on, at bb40k")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(pts)} point(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
