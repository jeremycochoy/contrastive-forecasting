#!/usr/bin/env python3
"""#373 review gap 6 — A3's bb200k student head, drawn twice.

A3 at bb200k reads 1.3998 on the student and 1.2913 on the teacher. Both
heads read ONE backbone file. Everywhere else in this grid the two heads of
one backbone agree to 0.0425 or better, and inside group A to 0.0168, so a
gap of 0.1085 is either a real property of that backbone or one bad head
draw.

The figure puts a second draw of the same head on the same axis. The band is
the ±0.0384 the whole report thresholds on, drawn around the first draw, so
the reader can see at once whether the second draw lands inside it.

Reads the score files. Usage:
  plot_a3_reseed.py --results results --out plots/a3_reseed.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                                # noqa: E402
from published import NOISE_BAND                         # noqa: E402

STOPS = (40, 100, 200)
RESEED_TAG = "A3_k3_bb200k_student_s20260723"
plt.rcParams.update(cc.rc())


def score(res: Path, tag: str):
    p = res / f"score_{tag}.txt"
    try:
        return float(p.read_text().strip())
    except (OSError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = Path(a.results)

    col = cc.COLOUR["A3"]
    st = [score(res, f"A3_k3_bb{s}k_student") for s in STOPS]
    te = [score(res, f"A3_k3_bb{s}k_teacher") for s in STOPS]
    draw2 = score(res, RESEED_TAG)
    if draw2 is None:
        raise SystemExit(f"ABORT: no score for {RESEED_TAG}")

    x = list(range(len(STOPS)))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # The band the report thresholds on, around the first draw's bb200k.
    ax.fill_between([x[-1] - 0.42, x[-1] + 0.42],
                    st[-1] - NOISE_BAND, st[-1] + NOISE_BAND,
                    color=col, alpha=0.12, linewidth=0, zorder=1,
                    label=f"±{NOISE_BAND} head-seed band, around draw 1")

    ax.plot(x, st, "-o", color=col, markersize=7, zorder=3,
            label="student head, seed 20260722")
    ax.plot(x, te, "--s", color=col, markerfacecolor="white", markersize=7,
            zorder=3, label="teacher head, seed 20260722")
    ax.plot([x[-1]], [draw2], "D", color=col, markersize=9,
            markeredgecolor="black", markeredgewidth=1.1, zorder=4,
            label="student head, seed 20260723 (second draw)")
    ax.plot([x[-1], x[-1]], [st[-1], draw2], color="black", linewidth=1.1,
            zorder=2)

    for xi, v in zip(x, st):
        ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    for xi, v in zip(x, te):
        ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=9)
    ax.annotate(f"{draw2:.4f}", (x[-1], draw2), textcoords="offset points",
                xytext=(14, -4), ha="left", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"bb{s}k" for s in STOPS])
    ax.set_xlim(-0.35, len(STOPS) - 0.35 + 0.5)
    ax.set_ylabel("GM-Relative MASE, 97 configs")
    ax.set_title("A3, k = 3: two heads on one backbone, and a second draw "
                 "of one of them")
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    print(f"draw 1 {st[-1]:.4f}  draw 2 {draw2:.4f}  "
          f"spread {abs(draw2 - st[-1]):.4f}  teacher {te[-1]:.4f}")


if __name__ == "__main__":
    main()
