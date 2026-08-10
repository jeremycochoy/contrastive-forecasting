#!/usr/bin/env python3
"""#373 figure 1 — the depth response of every arm this study trained.

One bar per (arm, head, depth): GM-Relative MASE at that depth minus the
SAME arm's own k = 0, over the full 97 configs. Negative is better.

Two reference spans sit behind the bars, and they are not the same kind of
thing:

  head-seed band     ±0.0384, `ema_sched_ladder.md`'s pooled range. It
                     bounds the HEAD seed alone.
  retraining range   the widest disagreement this study measured between two
                     trainings of ONE cell at ONE depth. It is not a seed
                     band: the B5 backbones behind it differ by seed and by
                     machine, and every bar here is the difference of two
                     independent backbone trainings.

A hatched bar is a delta whose two sides trained on different machines. The
study's reproduction table separates on the machine and not on the seed, so
a hatched bar carries a term the study cannot bound. Two bars per head are
not hatched, and they are the ones a reader can lean on.

Reads results/splits.csv (`all` rows) and resolves tags through runs.py.

Usage: plot_depth_response.py --splits results/splits.csv \\
           --out plots/depth_response.png
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
from published import NOISE_BAND                       # noqa: E402

plt.rcParams.update(cc.rc())


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["split"] == "all":
            out[r["stop"]] = float(r["gm_rel_mase"])
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    data = load(args.splits)
    rows = []
    for arm, head, k, base, deep in R.pairs(data):
        a, b = data.get(base.tag), data.get(deep.tag)
        if a is not None and b is not None:
            rows.append((arm, head, k, a, b, R.machine_held(base, deep)))
    if not rows:
        raise SystemExit(f"ABORT: no arm has two depths in {args.splits}")
    rows.sort(key=lambda r: (R.ARM_ORDER.index(r[0]), r[1], r[2]))

    # The widest disagreement between two trainings of one cell at one
    # depth. Read over every pair of B5 backbones, so a third backbone
    # widens or narrows it without an edit here.
    retrain_gap, retrain_why = 0.0, ""
    b5 = [a for a in R.ARM_ORDER if a.startswith("B5·")]
    for k in (0, 3):
        seen = []
        for arm in b5:
            run = R.find_run(arm, k, "depth") or R.find_run(arm, k, "control")
            v = data.get(f"{run.stem}_bb40k_student") if run else None
            if v is not None:
                seen.append((arm, v))
        for i, (a1, v1) in enumerate(seen):
            for a2, v2 in seen[i + 1:]:
                if abs(v2 - v1) > retrain_gap:
                    retrain_gap = abs(v2 - v1)
                    retrain_why = f"{a1} against {a2} at k = {k}"

    fig, ax = plt.subplots(figsize=(9.6, 0.52 * len(rows) + 2.2))
    ys = list(range(len(rows)))
    if retrain_gap:
        ax.axvspan(-retrain_gap, retrain_gap, color=cc.BAND, alpha=0.45,
                   zorder=0)
    ax.axvspan(-NOISE_BAND, NOISE_BAND, color=cc.PARITY, alpha=0.30, zorder=1)
    ax.axvline(0.0, color=cc.INK, linewidth=1.1, zorder=3)

    for y, (arm, head, k, k0, kk, held) in zip(ys, rows):
        d = kk - k0
        col = cc.colour(arm)
        ax.barh(y, d, height=0.62, color=cc.face(arm), edgecolor=col,
                linewidth=1.6, zorder=4)
        # The hatch goes on a second, unfilled bar. Matplotlib draws a
        # hatch in the patch's EDGE colour, and the edge here is the fill
        # colour, so hatching the first bar would draw it invisible.
        if cc.hatch(held):
            ax.barh(y, d, height=0.62, color="none", edgecolor=cc.INK,
                    linewidth=0.0, hatch=cc.CROSSED_HATCH, zorder=5)
        ax.text(d + (0.006 if d >= 0 else -0.006), y,
                f"{k0:.4f} → {kk:.4f}   ({d:+.4f})",
                va="center", ha="left" if d >= 0 else "right",
                fontsize=8, color=cc.INK, zorder=5)

    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{arm}  k = {k}  [{head}]{' ✗' if arm in R.RETRACTED else ''}"
         for arm, head, k, _a, _b, _m in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("GM-Relative MASE, depth k minus the same arm's k = 0   "
                  "(97 configs, negative is better)")
    # Every bar carries its own value label outside the bar end, so the axis
    # has to hold the bar AND the text. 1.75x leaves room for the longest.
    lim = max(abs(b - a) for _r, _h, _k, a, b, _m in rows) * 1.75
    ax.set_xlim(-lim, lim)
    ax.set_title("Rollout depth against the arm's own k = 0, at bb40k",
                 loc="left", fontsize=12)

    handles = [Patch(facecolor=cc.COLOUR[c], label=cc.label(c))
               for c in ("B9", "B1", "B5", "A3") if c in cc.COLOUR]
    handles.append(Patch(facecolor="#ffffff", edgecolor=cc.INK_SOFT,
                         label="second backbone seed (20260521)"))
    handles.append(Patch(facecolor="#ffffff", edgecolor=cc.INK,
                         hatch=cc.CROSSED_HATCH, linewidth=0.6,
                         label="the two sides trained on different machines"))
    if any(a in R.RETRACTED for a, *_ in rows):
        handles.append(Patch(facecolor="#ffffff", edgecolor="#ffffff",
                             label="✗ retracted — see the reproduction table"))
    handles.append(Patch(facecolor=cc.PARITY, alpha=0.30,
                         label=f"head-seed band ±{NOISE_BAND}"))
    if retrain_gap:
        handles.append(Patch(facecolor=cc.BAND, alpha=0.45,
                             label="two trainings of one cell disagreed by "
                                   f"±{retrain_gap:.4f} ({retrain_why})"))
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper left",
              bbox_to_anchor=(0.0, -0.09), borderaxespad=0.0)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(rows)} row(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
