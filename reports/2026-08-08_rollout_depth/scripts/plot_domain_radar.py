#!/usr/bin/env python3
"""#373 figure 1 — per-domain GM-Relative MASE, k = 3 over k = 0.

The parent's `domain_radar.png` with the k = 0 polygon overlaid. The
hypothesis says where the gain should land: Energy, Econ/Fin, Web/CloudOps
and Healthcare, the domains where seasonal naive wins by the largest
margin, and not Nature or Sales, where the k = 0 model already wins.

The radial axis is log2(ratio), so equal multiplicative steps are equal
distances. The pale ring at 1.0 is parity with seasonal naive.

Reads results/splits.csv, written by split_scores.py.

Usage: plot_domain_radar.py --splits results/splits.csv \\
           --out plots/domain_radar.png
"""
from __future__ import annotations

import argparse
import csv
import math
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


def load(path):
    out, counts = {}, {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["split"] != "domain":
                continue
            run = R.resolve(r["stop"])
            if run is None or run.role != "depth":
                continue
            key = (run.arm, run.k, run.head)
            out.setdefault(key, {})[r["name"]] = float(r["gm_rel_mase"])
            counts[r["name"]] = int(r["n"])
    return out, counts


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--head", default="student")
    args = p.parse_args(argv)

    data, counts = load(args.splits)
    # One panel per (arm, deeper depth). The k = 0 polygon is drawn inside
    # each panel against its own arm, never against another arm's.
    panels = [(arm, k) for arm in R.ARM_ORDER
              for k in sorted({kk for a, kk, h in data
                               if a == arm and h == args.head and kk})
              if (arm, 0, args.head) in data]
    if not panels:
        raise SystemExit("ABORT: no arm has two depths on the "
                         f"{args.head} head in {args.splits}")

    domains = sorted(counts, key=lambda d: -counts[d])
    n = len(domains)
    ang = [i / n * 2 * math.pi for i in range(n)] + [0.0]

    allv = [v for arm, k in panels for kk in (0, k)
            for v in data[(arm, kk, args.head)].values()]
    lo, hi = min(allv), max(allv)
    ticks = [t for t in (0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.8, 4.0)
             if lo / 1.1 <= t <= hi * 1.1] or [round(lo, 2), round(hi, 2)]
    if ticks[0] > lo:
        ticks.insert(0, round(lo, 2))
    if ticks[-1] < hi:
        ticks.append(round(hi, 2))

    ncol = min(len(panels), 3)
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 5.5 * nrow),
                             subplot_kw={"projection": "polar"}, squeeze=False)

    for idx, (arm, deep) in enumerate(panels):
        ax = axes[idx // ncol][idx % ncol]
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels([f"{d}\n({counts[d]})" for d in domains], fontsize=7)
        ax.set_ylim(math.log2(ticks[0]) - 0.03, math.log2(ticks[-1]) + 0.03)
        ax.set_yticks([math.log2(t) for t in ticks])
        ax.set_yticklabels([str(t) for t in ticks], fontsize=7)
        ax.plot(ang, [0.0] * len(ang), color=cc.PARITY, linewidth=1.2, zorder=0)

        for k in (0, deep):
            v = [math.log2(data[(arm, k, args.head)].get(d, 1.0))
                 for d in domains]
            v += v[:1]
            ax.plot(ang, v, color=cc.colour(arm), linewidth=1.9,
                    linestyle=cc.style(k), alpha=1.0 if k else 0.7)
        ax.set_title(f"{cc.label(arm)}\nk = 0 against k = {deep}",
                     fontsize=9, pad=16)

    for idx in range(len(panels), nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")

    fig.legend(handles=[Line2D([], [], color=cc.INK_SOFT, linestyle=cc.style(0),
                               label="k = 0"),
                        Line2D([], [], color=cc.INK_SOFT, linestyle=cc.style(3),
                               label="k > 0"),
                        Line2D([], [], color=cc.PARITY,
                               label="seasonal-naive parity")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Per-domain GM-Relative MASE, {args.head} head "
                 "(radial axis log2, lower is better)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=4.0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(panels)} panel(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
