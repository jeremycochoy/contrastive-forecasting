#!/usr/bin/env python3
"""#373 figure 4 — rollout fidelity against depth.

`cos(rollout_d, h_{T0+d})` for d = 1..16 on one fixed batch, k = 3 against
k = 0. This measures the composed operator directly, with no quantile head
in the way — the thing the training objective was changed to improve.

The batch is #379's committed `_latent_movement_batch.pt`, the same one the
parent reports' latent-movement figures use. It is a fixed diagnostic batch,
not a held-out one: nothing establishes it is disjoint from
`gift-pretrain-full-4096 / small_v1`, which is what these backbones trained
on. It holds every curve on one scale, and that is what it is for.

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


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    curves = defaultdict(list)
    with open(args.csv) as fh:
        for r in csv.DictReader(fh):
            curves[r["run"]].append((int(r["d"]), float(r["cos"])))
    if not curves:
        raise SystemExit(f"ABORT: no curve in {args.csv}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    pairs = defaultdict(dict)
    for run, pts in curves.items():
        # `find_artefacts.py --what ckpt` labels a curve `<arm>:<k>`, and
        # the registry turns that into the arm's colour and legend text. No
        # consumer of this study parses a run name itself.
        arm, _, ktxt = run.rpartition(":")
        if not arm or not ktxt.isdigit():
            raise SystemExit(f"ABORT: {run!r} is not an <arm>:<k> label")
        k = int(ktxt)
        pts.sort()
        xs = [d for d, _v in pts]
        ys = [v for _d, v in pts]
        # Fill is the backbone seed, as everywhere: B5 draws three curves in
        # one hue and only the marker tells its second seed from its first.
        axL.plot(xs, ys, color=cc.colour(arm), linestyle=cc.style(k),
                 linewidth=1.9, marker="o", markersize=4,
                 markerfacecolor=cc.face(arm),
                 markeredgecolor=cc.colour(arm))
        pairs[arm][k] = (xs, ys)

    for arm in [a for a in R.ARM_ORDER if a in pairs]:
        byk = pairs[arm]
        if 0 in byk and 3 in byk:
            xs = byk[3][0]
            mark = "  ✗ retracted" if arm in R.RETRACTED else ""
            axR.plot(xs, [b - a for a, b in zip(byk[0][1], byk[3][1])],
                     color=cc.colour(arm), linewidth=1.9, marker="o",
                     markersize=5, markerfacecolor=cc.face(arm),
                     markeredgecolor=cc.colour(arm),
                     label=cc.label(arm) + mark)

    axL.set_xlabel("rollout depth d (tokens)")
    axL.set_ylabel("cos(rollout$_d$, h$_{T_0+d}$)")
    axL.set_title("Fidelity of the composed forecaster")
    axL.legend(handles=[Line2D([], [], color=cc.INK_SOFT, linestyle=cc.style(0),
                               label="k = 0"),
                        Line2D([], [], color=cc.INK_SOFT, linestyle=cc.style(3),
                               label="k = 3")],
               frameon=False, fontsize=9)

    axR.axhline(0.0, color=cc.INK_SOFT, linewidth=1.0)
    axR.set_xlabel("rollout depth d (tokens)")
    axR.set_ylabel("k = 3 minus k = 0  (positive is better)")
    axR.set_title("Change")
    axR.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(curves)} curve(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
