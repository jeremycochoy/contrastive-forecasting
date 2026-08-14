#!/usr/bin/env python3
"""#373 figure 4 — rollout fidelity against depth.

`cos(rollout_d, h_{T0+d})` for d = 1..16 on one fixed batch. This measures
the composed operator directly, with no quantile head in the way — the thing
the training objective was changed to improve.

The batch is #379's committed `_latent_movement_batch.pt`, the same one the
parent reports' latent-movement figures use, and the SAME batch carries every
curve here, so all fourteen cells are on one scale. It is a fixed diagnostic
batch, not a held-out one: nothing establishes it is disjoint from
`gift-pretrain-full-4096 / small_v1`, which is what these backbones trained
on.

Two blocks, because the two panels can draw different populations.

  Absolute fidelity needs a `k = 3` checkpoint, and every one of the card's
  fourteen cells has one. Fourteen curves do not separate by hue — no
  categorical palette does — so this block is a SMALL MULTIPLE: one panel per
  cell, the cell's own curve in ink over the other thirteen in furniture
  grey. Identity is the panel title, never a colour a reader has to match.

  The change against `k = 0` needs a `k = 0` checkpoint too, and only the
  four cells this study retrained at `k = 0` have one. Four cells take four
  validated hues, so that block is one axes. Depth rides the line style
  alone: dashed `k = 0`, dotted `k = 1`, solid `k = 3`.

Reads results/rollout_fidelity.csv, written by rollout_fidelity.py.

Usage: plot_rollout_fidelity.py --csv results/rollout_fidelity.csv \\
           --out plots/rollout_fidelity.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
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

NCOL = 7                       # 14 cells, two rows of seven


def read(path):
    """`{(arm, k): {d: cos}}`. A label is `<arm>:<k>` or `<cell>:<k>`."""
    out = defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            run = r["run"]
            arm, _, ktxt = run.rpartition(":")
            if not arm or not ktxt.isdigit():
                raise SystemExit(f"ABORT: {run!r} is not an <arm>:<k> label")
            out[(arm, int(ktxt))][int(r["d"])] = float(r["cos"])
    return out


def cell_curves(curves):
    """`{cell: [(d, cos)]}` at `k = 3`, one entry per card cell.

    A cell the registry holds a depth ladder for is drawn from the arm that
    ladder names, so the curve and the cell's score come off one checkpoint.
    B5 trained three backbones; its grid score is the first of them, and
    `ARM_ORDER` is where that order lives.
    """
    cells = [ln.split("\t")[0] for ln in
             (HERE / "cells.tsv").read_text().splitlines()
             if ln and not ln.startswith("#")]
    out = {}
    for cell in cells:
        arms = [cell] + R.arms_of(cell)
        for arm in arms:
            if (arm, 3) in curves:
                out[cell] = sorted(curves[(arm, 3)].items())
                break
    return out


def place(labels, lo, hi, gap):
    """Vertical positions for end labels: keep the order, force a gap."""
    order = sorted(range(len(labels)), key=lambda i: labels[i][0])
    ys = [labels[i][0] for i in order]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    over = ys[-1] - hi
    if over > 0:
        ys = [y - over for y in ys]
    ys[0] = max(ys[0], lo)
    return {order[i]: ys[i] for i in range(len(order))}


def draw_cells(fig, gs, byc, nrow):
    """One panel per cell: its own k = 3 curve, over the other thirteen."""
    every = list(byc.values())
    vals = [v for pts in every for _d, v in pts]
    lo, hi = min(vals), max(vals)
    pad = 0.08 * (hi - lo)
    axes = []
    for i, (cell, pts) in enumerate(byc.items()):
        ax = fig.add_subplot(gs[i // NCOL, i % NCOL])
        axes.append(ax)
        for other in every:
            ax.plot([d for d, _v in other], [v for _d, v in other],
                    color=cc.CONTEXT, linewidth=0.9, zorder=1)
        ax.plot([d for d, _v in pts], [v for _d, v in pts], color=cc.INK,
                linewidth=2.0, linestyle=cc.style(3), zorder=3)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0.4, 16.6)
        ax.set_xticks([1, 8, 16])
        ax.set_title(cell, fontsize=10.5, pad=3)
        if i % NCOL:
            ax.set_yticklabels([])
        if i // NCOL < nrow - 1:
            ax.set_xticklabels([])
    # A1 and B3 hold one student model, so their curves are one line drawn
    # twice. A reader who sees the two panels agree to six decimals is owed
    # the reason on the panel, not four sections away.
    for i, cell in enumerate(byc):
        if cell in ("A1", "B3"):
            twin = "B3" if cell == "A1" else "A1"
            axes[i].text(0.5, 0.04, f"one student with {twin}",
                         transform=axes[i].transAxes, ha="center",
                         fontsize=7.5, color=cc.INK_SOFT)
    axes[0].set_ylabel("cos(rollout$_d$, h$_{T_0+d}$)", fontsize=9)
    for ax in axes[NCOL * (nrow - 1):]:
        ax.set_xlabel("depth d", fontsize=9)
    return axes


def draw_change(ax, curves, ncell):
    """Every arm's deeper depth against that SAME arm's own k = 0."""
    ends, drawn = [], set()
    for arm in [a for a in R.ARM_ORDER if (a, 0) in curves]:
        base = curves[(arm, 0)]
        for k in sorted(k for (a, k) in curves if a == arm and k > 0):
            deep = curves[(arm, k)]
            ds = sorted(set(base) & set(deep))
            ys = [deep[d] - base[d] for d in ds]
            # The retracted arm takes grey, so its sibling backbone keeps the
            # cell's hue undimmed: the shade channel has no work to do here.
            col = cc.RETRACTED_INK if cc.retracted(arm) else cc.colour(arm)
            ax.plot(ds, ys, color=col, linewidth=cc.width(arm),
                    linestyle=cc.style(k), zorder=3)
            mark = "  ✗ retracted" if cc.retracted(arm) else ""
            ends.append((ys[-1], f"{arm}  k = {k}{mark}", col))
            drawn.add(k)

    ax.axhline(0.0, color=cc.INK_SOFT, linewidth=1.0, zorder=2)
    lo, hi = ax.get_ylim()
    at = place(ends, lo, hi, 0.075 * (hi - lo))
    for i, (_y, text, col) in enumerate(ends):
        ax.text(16.4, at[i], text, color=col, fontsize=9.5,
                va="center", ha="left")
    ax.set_xlim(0.4, 16.4)
    ax.set_xticks(range(2, 17, 2))
    ax.set_xlabel("rollout depth d (tokens)")
    ax.set_ylabel("depth k minus that arm's own k = 0\n(positive is more faithful)",
                  fontsize=9.5)
    ax.legend(handles=[Line2D([], [], color=cc.INK_SOFT, linewidth=1.9,
                              linestyle=cc.style(k), label=f"k = {k}")
                       for k in sorted(drawn)],
              loc="lower right", fontsize=9, ncol=len(drawn))
    cells = {cc.cell_of(a) for a, k in curves if k == 0}
    ax.set_title("Change against the same arm's own k = 0",
                 loc="left", fontsize=13, color=cc.INK, pad=24)
    ax.text(0.0, 1.015, f"{ncell - len(cells)} of the {ncell} cells are "
            "absent here: they read their k = 0 from a published number, and "
            "a published number has no checkpoint",
            transform=ax.transAxes, fontsize=9.5, color=cc.INK_SOFT)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    curves = read(args.csv)
    if not curves:
        raise SystemExit(f"ABORT: no curve in {args.csv}")
    byc = cell_curves(curves)
    if not byc:
        raise SystemExit(f"ABORT: no cell curve in {args.csv}")

    nrow = -(-len(byc) // NCOL)
    fig = plt.figure(figsize=(13.2, 9.9))
    gs = fig.add_gridspec(nrow, NCOL, hspace=0.40, wspace=0.16,
                          top=0.905, bottom=0.585, left=0.055, right=0.985)
    draw_cells(fig, gs, byc, nrow)
    axC = fig.add_subplot(fig.add_gridspec(1, 1, top=0.435, bottom=0.06,
                                           left=0.055, right=0.795)[0])
    draw_change(axC, curves, len(byc))

    fig.text(0.055, 0.965, "Fidelity of the composed forecaster at "
             f"k = 3, {len(byc)} cells, bb40k",
             fontsize=13, color=cc.INK, ha="left")
    fig.text(0.055, 0.943, "each panel is one cell in black over the other "
             f"{len(byc) - 1} in grey; one fixed diagnostic batch for every "
             "curve", fontsize=9.5, color=cc.INK_SOFT, ha="left")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(byc)} cells, {len(curves)} curves)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
