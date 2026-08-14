#!/usr/bin/env python3
"""#373 review gap 6 — A3's bb200k student head, drawn twice.

A3 at bb200k reads 1.3998 on the student and 1.2913 on the teacher. Both
heads read ONE backbone file. Everywhere else in this grid the two heads of
one backbone agree to 0.0425 or better, and inside group A to 0.0168, so a
gap of 0.1084 is either a real property of that backbone or one bad head
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
    fig, ax = plt.subplots(figsize=(8.0, 5.6))

    # The band the report thresholds on, around the first draw's bb200k.
    ax.fill_between([x[-1] - 0.42, x[-1] + 0.42],
                    st[-1] - NOISE_BAND, st[-1] + NOISE_BAND,
                    color=col, alpha=0.12, linewidth=0, zorder=1,
                    label=f"±{NOISE_BAND} head-seed band, around draw 1")

    # The bb200k draws sit on two machines, so the legend names each one.
    # A reader who takes the figure on its own must not read the 0.0100 as a
    # seed effect alone.
    ax.plot(x, st, "-o", color=col, markersize=7, zorder=3,
            label="student head, seed 20260722 (bb200k on the box)")
    ax.plot(x, te, "--s", color=col, markerfacecolor="white", markersize=7,
            zorder=3, label="teacher head, seed 20260722 (bb200k on the box)")
    ax.plot([x[-1]], [draw2], "D", color=col, markersize=9,
            markeredgecolor="black", markeredgewidth=1.1, zorder=4,
            label="student head, seed 20260723, on elisa (second draw)")
    ax.plot([x[-1], x[-1]], [st[-1], draw2], color="black", linewidth=1.1,
            zorder=2)

    # Put each label on the side its own curve leaves free. At bb100k the
    # two curves nearly touch, so a fixed offset prints one label over the
    # other: there the higher point takes the space above and the lower
    # takes the space below.
    for xi, (sv, tv) in enumerate(zip(st, te)):
        # The second draw sits on the student's own x, so at that stop the
        # student's neighbour is the diamond and not the teacher.
        above = sv >= tv and not (xi == x[-1] and draw2 > sv)
        ax.annotate(f"{sv:.4f}", (xi, sv), textcoords="offset points",
                    xytext=(0, 9 if above else -17), ha="center",
                    fontsize=9)
        ax.annotate(f"{tv:.4f}", (xi, tv), textcoords="offset points",
                    xytext=(0, 9 if tv > sv else -17), ha="center",
                    fontsize=9)
    ax.annotate(f"{draw2:.4f}", (x[-1], draw2), textcoords="offset points",
                xytext=(14, -4), ha="left", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"bb{s}k" for s in STOPS])
    ax.set_xlim(-0.35, len(STOPS) - 0.35 + 0.5)
    # Room for the labels themselves, and for the band, which reaches
    # further than any point does.
    lo = min(list(st) + list(te) + [draw2, st[-1] - NOISE_BAND])
    hi = max(list(st) + list(te) + [draw2, st[-1] + NOISE_BAND])
    ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.14 * (hi - lo))
    ax.set_ylabel("GM-Relative MASE, 97 configs")
    ax.set_title("A3, k = 3: two heads on one backbone")
    # The labels name the machine, so they are too wide to sit inside the
    # axes without covering the teacher line. The legend goes under it.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09),
              frameon=False, fontsize=9, ncol=1)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    print(f"draw 1 {st[-1]:.4f}  draw 2 {draw2:.4f}  "
          f"spread {abs(draw2 - st[-1]):.4f}  teacher {te[-1]:.4f}")


if __name__ == "__main__":
    main()
