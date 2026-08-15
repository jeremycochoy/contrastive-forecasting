#!/usr/bin/env python3
"""#373 round 2 — the horizon split, which is the mechanism test.

The card's hypothesis is specific. At eval the forecaster rolls out on its
own output, up to ~45 times on the long configs and not at all on the short
ones, so training the composed operator should pay on medium and long and
not on short. A gain spread evenly over the 97 is a different result: it
says the objective helped, not that the rollout deficit closed.

  panel   the 55 short configs, then the 42 medium+long configs.
  y       GM-Relative MASE.
  dumbbell  hollow dot k = 0, filled dot k = 3, a connector between them.

Dots and not bars, because the reading is "which of two values is lower"
and the interesting range is 0.9 to 1.6. A bar has to start at zero or it
lies about magnitude; a dot does not, so the axis can hold the range the
data lives in.

The `k = 0` side needs PER-CONFIG numbers, and the parents publish only the
97-config aggregate. So a `k = 0` bar appears only where this study holds a
same-code `k = 0` eval of its own: A3, B1, B5 and B9, from round 1. Every
other cell shows its `k = 3` split alone. The report says so rather than
drawing a bar the data does not support.

Usage:
  r2_plot_horizon.py --splits results/splits.csv --head student \\
      --out plots/horizon_split_student.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                           # noqa: E402


sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2_ladder as L                                     # noqa: E402

K3_COLOUR, K0_COLOUR = "#2a78d6", "#9a9995"
INK, INK_SOFT, GRID, PARITY = "#0b0b0b", "#52514e", "#e6e5e1", "#8f8e8a"
SPLITS = ("short", "medium_long")
NICE = {"short": "short (55)", "medium_long": "medium + long (42)"}


def load(path):
    """`{tag: {split name: value}}` for the horizon-term rows."""
    out = defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "term" and r["name"] in SPLITS:
                out[r["stop"]][r["name"]] = float(r["gm_rel_mase"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--head", default="student", choices=list(L.HEADS))
    ap.add_argument("--stop", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    by_tag = load(a.splits)
    if not by_tag:
        print(f"  {a.splits} holds no horizon rows — no horizon figure")
        return 0

    def get(cell, k):
        names = [f"{cell}_k{k}_bb{a.stop}k_{a.head}"]
        if k == 0:
            t = L.k0_tag(cell, a.stop, a.head)
            if t:
                names.insert(0, t)
        else:
            alias = L.TAG_ALIAS.get((cell, a.stop, k))
            if alias:
                names.append(f"{alias}_{a.head}")
        for name in names:
            if name in by_tag:
                return by_tag[name]
        return {}

    cells = [c for c in L.CELLS if get(c, 3)]
    if not cells:
        print(f"  no k = 3 horizon split at bb{a.stop}k, {a.head} head")
        return 0

    fig, axes = plt.subplots(1, 2, figsize=(max(9.0, 0.62 * len(cells) * 2), 4.6),
                             sharey=True)
    for ax, split in zip(axes, SPLITS):
        for i, cell in enumerate(cells):
            v3 = get(cell, 3).get(split)
            v0 = get(cell, 0).get(split)
            if v0 is not None and v3 is not None:
                ax.plot([i, i], [v0, v3], color=INK_SOFT, lw=1.2, zorder=2)
            if v0 is not None:
                ax.plot(i, v0, marker="o", ms=8, mfc="white", mec=K0_COLOUR,
                        mew=1.8, ls="", zorder=3)
            if v3 is not None:
                ax.plot(i, v3, marker="o", ms=8, color=K3_COLOUR, ls="",
                        zorder=4)
                ax.text(i + 0.16, v3, f"{v3:.2f}", va="center", ha="left",
                        fontsize=6.5, color=INK_SOFT, zorder=5)
        ax.axhline(1.0, color=PARITY, lw=0.9, zorder=1)
        ax.set_xlim(-0.6, len(cells) - 0.25)
        ax.set_title(NICE[split], fontsize=10, color=INK)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(cells, fontsize=8, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=8)
    axes[0].set_ylabel("GM-Relative MASE", fontsize=9, color=INK_SOFT)

    handles = [plt.Line2D([], [], ls="", marker="o", ms=8, color=K3_COLOUR,
                          label="k = 3"),
               plt.Line2D([], [], ls="", marker="o", ms=8, mfc="white",
                          mec=K0_COLOUR, mew=1.8,
                          label="k = 0, this study's own (4 cells only)"),
               plt.Line2D([], [], color=PARITY, lw=0.9,
                          label="seasonal-naive parity")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"where the depth lands, by horizon — bb{a.stop}k, "
                 f"{a.head} encoder", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170)
    print(f"  {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
