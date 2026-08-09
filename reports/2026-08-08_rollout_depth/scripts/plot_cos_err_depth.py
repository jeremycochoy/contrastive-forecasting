#!/usr/bin/env python3
"""#373 figure 3 — per-depth forecast error during training.

`1 - cos(f^(j)_t, h_{t+1+j})` per step, one line per depth j = 0..k, against
the k = 0 run's single line. Two questions:

  Does depth 0 pay for the deeper depths?  Compare the k = 3 run's
      `cos_err_d0` with the k = 0 run's `1 - ff`. They measure the same
      thing on the two runs.
  Do the deeper predictions improve at all?  `cos_err_d1..d3` falling is
      the composed operator getting better at its own output.

A k = 0 run writes no `cos_err_d*` column; its depth-0 curve is `1 - ff`.
The presence of `cos_err_d1..d3` on the k = 3 side is also the proof the
depth reached that run's loss.

Usage: plot_cos_err_depth.py --out plots/cos_err_depth.png \\
           --run <cell>:<k>=<losses.csv> [--run ...]
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
from losses_csv import read_by_step, series             # noqa: E402

plt.rcParams.update(cc.rc())
# Depth rides on the alpha within one colour: d0 solid and full, deeper
# depths progressively lighter. The cell keeps its colour.
DEPTH_ALPHA = [1.0, 0.66, 0.46, 0.32]


def smooth(xs, ys, window=25):
    """Running mean, so the per-step noise does not hide the trend."""
    out = []
    for i in range(len(ys)):
        lo = max(0, i - window + 1)
        out.append(sum(ys[lo:i + 1]) / (i + 1 - lo))
    return xs, out


def read(path):
    """The columns this figure needs, one row per step."""
    return read_by_step(path, ["ff"] + [f"cos_err_d{j}" for j in range(8)])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True,
                   metavar="CELL:K=CSV")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    runs = []
    for spec in args.run:
        head, path = spec.split("=", 1)
        cell, k = head.split(":")
        if not Path(path).is_file():
            print(f"  skip {spec}: no {path}", file=sys.stderr)
            continue
        runs.append((cell, int(k), read(path)))
    if not runs:
        raise SystemExit("ABORT: no run had a losses CSV")

    cells = sorted({c for c, _k, _d in runs}, key=lambda c: cc.ORDER.index(c)
                   if c in cc.ORDER else 99)
    fig, axes = plt.subplots(1, len(cells), figsize=(5.4 * len(cells), 4.2),
                             squeeze=False)

    for ax, cell in zip(axes[0], cells):
        col = cc.colour(cell)
        for c, k, d in runs:
            if c != cell:
                continue
            if k == 0:
                xs, ys = series(d, "ff")
                xs, ys = smooth(xs, [1.0 - v for v in ys])
                ax.plot(xs, ys, color=cc.INK, linewidth=1.6,
                        linestyle=cc.style(0), label="k = 0, depth 0 (1 − ff)")
                continue
            for j in range(8):
                key = f"cos_err_d{j}"
                xs, ys = series(d, key)
                if not xs:
                    continue
                xs, ys = smooth(xs, ys)
                ax.plot(xs, ys, color=col, linewidth=1.7,
                        alpha=DEPTH_ALPHA[min(j, len(DEPTH_ALPHA) - 1)],
                        label=f"k = {k}, depth {j}")
        ax.set_title(cc.label(cell), fontsize=9)
        ax.set_xlabel("backbone step")
        ax.set_ylabel("1 − cos(f$^{(j)}_t$, h$_{t+1+j}$)")
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(cells)} cell(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
