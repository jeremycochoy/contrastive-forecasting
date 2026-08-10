#!/usr/bin/env python3
"""#373 figure 1 — the depth response of every arm this study trained.

One bar per (arm, head, depth): GM-Relative MASE at that depth minus the
SAME arm's own k = 0, over the full 97 configs. Negative is better.

Two reference spans sit behind the bars, and they are not the same kind of
thing:

  head-seed band       ±0.0384, `ema_sched_ladder.md`'s pooled range. It
                       bounds the HEAD seed alone.
  backbone-seed range  the one pair this study measured: B5 trained twice,
                       same recipe, two backbone seeds. That difference is
                       larger than the head-seed band, and every bar here is
                       the difference of two independent backbone trainings.

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
            rows.append((arm, head, k, a, b))
    if not rows:
        raise SystemExit(f"ABORT: no arm has two depths in {args.splits}")
    rows.sort(key=lambda r: (R.ARM_ORDER.index(r[0]), r[1], r[2]))

    # The measured backbone-seed difference, at whichever depth it is widest.
    seed_gap = 0.0
    for k in (0, 3):
        a = data.get(f"B5_k{k}_bb40k_student")
        b = data.get(f"G5_B5_s2_k{k}_bb40k_student")
        if a is not None and b is not None:
            seed_gap = max(seed_gap, abs(b - a))

    fig, ax = plt.subplots(figsize=(9.6, 0.52 * len(rows) + 2.0))
    ys = list(range(len(rows)))
    if seed_gap:
        ax.axvspan(-seed_gap, seed_gap, color=cc.BAND, alpha=0.45, zorder=0)
    ax.axvspan(-NOISE_BAND, NOISE_BAND, color=cc.PARITY, alpha=0.30, zorder=1)
    ax.axvline(0.0, color=cc.INK, linewidth=1.1, zorder=3)

    for y, (arm, head, k, k0, kk) in zip(ys, rows):
        d = kk - k0
        col = cc.colour(arm)
        ax.barh(y, d, height=0.62, color=cc.face(arm), edgecolor=col,
                linewidth=1.6, hatch="///" if cc.hollow(arm) else None,
                zorder=4)
        ax.text(d + (0.006 if d >= 0 else -0.006), y,
                f"{k0:.4f} → {kk:.4f}   ({d:+.4f})",
                va="center", ha="left" if d >= 0 else "right",
                fontsize=8, color=cc.INK, zorder=5)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{arm}  k = {k}  [{head}]" for arm, head, k, _a, _b in rows],
                       fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("GM-Relative MASE, depth k minus the same arm's k = 0   "
                  "(97 configs, negative is better)")
    # Every bar carries its own value label outside the bar end, so the axis
    # has to hold the bar AND the text. 1.75x leaves room for the longest.
    lim = max(abs(b - a) for _r, _h, _k, a, b in rows) * 1.75
    ax.set_xlim(-lim, lim)
    ax.set_title("Rollout depth against the arm's own k = 0, at bb40k",
                 loc="left", fontsize=12)

    handles = [Patch(facecolor=cc.COLOUR[c], label=cc.label(c))
               for c in ("B9", "B1", "B5", "A3") if c in cc.COLOUR]
    handles.append(Patch(facecolor="#ffffff", edgecolor=cc.INK_SOFT,
                         hatch="///", label="second backbone seed (20260521)"))
    handles.append(Patch(facecolor=cc.PARITY, alpha=0.30,
                         label=f"head-seed band ±{NOISE_BAND}"))
    if seed_gap:
        handles.append(Patch(facecolor=cc.BAND, alpha=0.45,
                             label="backbone-seed difference measured on B5, "
                                   f"±{seed_gap:.4f}"))
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper left",
              bbox_to_anchor=(0.0, -0.09), borderaxespad=0.0)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(rows)} row(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
