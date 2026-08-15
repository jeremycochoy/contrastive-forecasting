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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                              # noqa: E402
import runs as R                                       # noqa: E402
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

    # One panel per TRAINED RUN, not per cell. A3 trained k = 1 and k = 3,
    # and drawing both in one panel put seven lines of one hue at four
    # opacities on one axes, where the two runs cannot be read apart. Each
    # panel now holds one deeper run and the cell's own k = 0 line.
    #
    # Panel order comes from the registry's arm order, not from the cell
    # order: `cc.ORDER` holds cells, so B5's three arms all tied on it and
    # the panels came out in whatever order the set iterated that run.
    k0 = {c: d for c, k, d in runs if k == 0}
    panels = sorted(((c, k) for c, k, _d in runs if k > 0),
                    key=lambda t: (R.ARM_ORDER.index(t[0])
                                   if t[0] in R.ARM_ORDER else 99, t[0], t[1]))
    # Three panels per row. One row of five renders ~200 px per panel at a
    # report column width, which is below what the axis labels need.
    ncol = min(3, len(panels))
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.9 * nrow),
                             squeeze=False)
    flat = [a for row in axes for a in row]
    for ax in flat[len(panels):]:
        ax.set_axis_off()

    for ax, (cell, k) in zip(flat, panels):
        col = cc.colour(cell)
        if cell in k0:
            xs, ys = series(k0[cell], "ff")
            xs, ys = smooth(xs, [1.0 - v for v in ys])
            ax.plot(xs, ys, color=cc.INK, linewidth=1.6,
                    linestyle=cc.style(0), label="k = 0, depth 0")
        d = next(dd for c, kk, dd in runs if c == cell and kk == k)
        for j in range(8):
            xs, ys = series(d, f"cos_err_d{j}")
            if not xs:
                continue
            xs, ys = smooth(xs, ys)
            ax.plot(xs, ys, color=col, linewidth=1.7,
                    alpha=DEPTH_ALPHA[min(j, len(DEPTH_ALPHA) - 1)],
                    label=f"k = {k}, depth {j}")
        ax.set_title(f"{cc.label(cell)}   k = {k}", fontsize=9)
        ax.set_xlabel("backbone step")
        ax.set_ylabel("1 − cos(f$^{(j)}_t$, h$_{t+1+j}$)")
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(panels)} run(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
