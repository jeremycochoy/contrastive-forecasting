#!/usr/bin/env python3
"""#373 review item 3 — B1's 2x2, drawn as an interaction plot.

`k = 3` moves two things at once on B1: the total weight on `L_align` goes
1x -> 4x, and the term spreads from t+1 to t+1..t+4. The four corners are
the two-by-two of those factors. An interaction plot puts the horizons on
the x axis and draws one line per weight. Parallel lines mean the two
changes add, and the report's subtraction stands. A gap that opens or
closes means they do not.

Colour is B1's cell colour, per `cell_colours`. The style channel carries
the WEIGHT here and not the depth, which is the one place in this report it
does: the depth already rides the x axis, so reading it twice would waste a
channel and leave the weight unnamed. Marker and legend say the same thing,
so the figure does not rest on style alone.

Reads results/gap4_2x2.json, written by gap4_2x2.py.

Usage: plot_b1_2x2.py --summary results/gap4_2x2.json --out plots/b1_2x2.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                              # noqa: E402

plt.rcParams.update(cc.rc())

COL = cc.colour("B1")
XS = [0, 1]
XLAB = ["t+1\n(k = 0)", "t+1..t+4\n(k = 3)"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    data = json.load(open(args.summary))
    heads = [h for h in ("student", "teacher") if h in data]
    if not heads:
        raise SystemExit("ABORT: the summary holds no head")

    fig, axes = plt.subplots(1, len(heads), figsize=(4.6 * len(heads), 4.3),
                             sharey=True, squeeze=False)

    for ax, head in zip(axes[0], heads):
        d = data[head]
        gm = d["_corners"]
        # 1x weight: k0 -> h.  4x weight: w -> k3.
        for tag, key_lo, key_hi, marker, style, name in [
                ("1x", "k0", "h", "o", (0, (5, 2)), "1x total L_align weight"),
                ("4x", "w", "k3", "s", "solid", "4x total L_align weight")]:
            ys = [gm[key_lo], gm[key_hi]]
            ax.plot(XS, ys, color=COL, linestyle=style, marker=marker,
                    markersize=7, linewidth=cc.width("B1"),
                    markerfacecolor=COL, markeredgecolor=COL, label=name)
            for x, y in zip(XS, ys):
                ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                            xytext=(0, 9 if tag == "1x" else -16),
                            ha="center", fontsize=9, color=COL)

        inter, lo, hi, _ = d["interaction"]
        adds = lo <= 0.0 <= hi
        ax.set_title(f"{head} head\ninteraction {inter:+.4f} "
                     f"[{lo:+.4f}, {hi:+.4f}]"
                     f"\n{'the two changes add' if adds else 'they do not add'}",
                     fontsize=10)
        ax.set_xticks(XS)
        ax.set_xticklabels(XLAB)
        ax.set_xlim(-0.42, 1.42)
        ax.margins(y=0.16)
        ax.set_xlabel("horizons `L_align` covers")

    axes[0][0].set_ylabel("GM-Relative MASE, 97 configs (lower is better)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("B1 at bb40k: the depth, or the weight it carries",
                 fontsize=12)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
