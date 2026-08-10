#!/usr/bin/env python3
"""#373 figure 4 — does this code snapshot reproduce the published k = 0?

Every group-B delta crosses a code snapshot as well as the depth flag, so
the card asks for a reproduction check before any delta is read. This is it.

One row per arm. The open marker is the number the parent report publishes;
the filled marker is what this study got when it retrained that cell's
k = 0. The last row changes the backbone instead of the code: it takes
#379's own published B5 checkpoint and puts this study's head and eval on
it, so a difference there cannot be a training difference at all.

Reads results/score_*.txt and published.py.

Usage: plot_reproduction.py --results results --out plots/reproduction.png
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
from published import PUBLISHED, GATE, verdict         # noqa: E402

plt.rcParams.update(cc.rc())

# arm -> the k = 0 score file this study retrained, on the student head.
RETRAINED = [("B1", "G6_B1_k0_bb40k_student", "code"),
             ("B9", "G2_B9_k0_bb40k_student", "code"),
             ("B5·s2", "G5_B5_s2_k0_bb40k_student", "code"),
             ("B5·s1", "B5_k0_bb40k_student", "code"),
             ("A3", "A3_k0_bb40k_student", "code"),
             ("B5 published backbone", "G1_B5pub_bb40k_student", "backbone")]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    res = Path(args.results)

    rows = []
    for arm, fname, kind in RETRAINED:
        f = res / f"score_{fname}.txt"
        if not f.is_file():
            continue
        got = float(f.read_text().strip())
        pub = PUBLISHED[cc.cell_of(arm.split()[0])]["student"][40]
        rows.append((arm, pub, got, kind))
    if not rows:
        raise SystemExit(f"ABORT: no retrained k = 0 score in {res}")

    fig, ax = plt.subplots(figsize=(9.0, 0.62 * len(rows) + 1.9))
    ys = list(range(len(rows)))
    for y, (arm, pub, got, kind) in zip(ys, rows):
        col = cc.colour(arm.split()[0]) if kind == "code" else cc.INK_SOFT
        ax.plot([pub, got], [y, y], color=col, linewidth=2.0, zorder=2)
        ax.plot(pub, y, marker="o", markersize=11, markerfacecolor="#ffffff",
                markeredgecolor=col, markeredgewidth=2.0, zorder=3)
        ax.plot(got, y, marker="o", markersize=11, markerfacecolor=col,
                markeredgecolor=col, zorder=3)
        d = abs(got - pub)
        ax.annotate(f"|Δ| {d:.4f}   {verdict(d)}",
                    (max(pub, got), y), textcoords="offset points",
                    xytext=(14, -3), fontsize=8.5, color=cc.INK)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{arm}  ({'retrained' if k == 'code' else 're-headed'})"
                        for arm, _p, _g, k in rows], fontsize=9)
    ax.invert_yaxis()
    lo = min(min(p_, g) for _a, p_, g, _k in rows)
    hi = max(max(p_, g) for _a, p_, g, _k in rows)
    pad = (hi - lo) * 0.12 + 0.01
    ax.set_xlim(lo - pad, hi + pad * 6.5)
    ax.set_xlabel("GM-Relative MASE at bb40k, student head, 97 configs")
    ax.set_title("Published k = 0 against this study's own k = 0   "
                 f"(gate: |Δ| ≤ {GATE})", loc="left", fontsize=12)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=10,
               markerfacecolor="#ffffff", markeredgecolor=cc.INK_SOFT,
               markeredgewidth=2.0, label="published by the parent report"),
        Line2D([], [], marker="o", linestyle="none", markersize=10,
               markerfacecolor=cc.INK_SOFT, markeredgecolor=cc.INK_SOFT,
               label="measured by this study")]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(rows)} row(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
