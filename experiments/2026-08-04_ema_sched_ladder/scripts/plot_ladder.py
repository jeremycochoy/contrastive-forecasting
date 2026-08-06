#!/usr/bin/env python3
"""#393 — the ladder as one picture: GM-Relative MASE against backbone step.

Reads `results/ladder_all.csv`, the pooled table, so it draws every machine's
scores and redraws itself as later stops land.

Two panels, because the study asks two questions and one axis cannot answer
both:

  left   every cell, both heads, against the backbone step. Solid is the
         student encoder, dashed the teacher, one colour per cell. Seasonal
         naive sits at 1.0 and nothing in this study is below it.
  right  the change from one stop to the next, per head. Positive is worse.
         The extend rule is exactly the sign of these bars, so a reader can
         see which cells the rule kept and check the branch against the
         picture. The grey band is the largest head-seed range measured in
         this study, pooled over both head budgets (scripts/noise_band.py);
         bars from a cell that carries no head-seed replicate are drawn
         hollow, measured but not tested.

Usage:  python3 scripts/plot_ladder.py [--ladder FILE] [--spread FILE]
                                       [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

import noise_band  # noqa: E402
from cell_label import label as cell_label  # noqa: E402

# Distinct hues, and the head is carried by linestyle as well as colour so
# the panel survives being read in greyscale.
PALETTE = ["#1f4e79", "#c0504d", "#4f8a3d", "#7d5ba6", "#c9891b",
           "#2b8a9e", "#a0522d", "#7f7f7f", "#d63384", "#111111"]
HEAD_STYLE = {"student": ("-", "o"), "teacher": ("--", "s")}


def read_scores(path: str) -> dict:
    """{cell: {head: [(stop, score), ...]}} sorted by stop."""
    out: dict = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("gm_rel_mase") or "").strip():
                continue
            out.setdefault(r["cell"], {}).setdefault(r["head"], []).append(
                (int(r["stop"]), float(r["gm_rel_mase"])))
    for heads in out.values():
        for series in heads.values():
            series.sort()
    return out


def order(scores: dict) -> list[str]:
    """Cells best-first, by the lowest score each one reached."""
    return sorted(scores, key=lambda c: min(
        v for h in scores[c].values() for _, v in h))


def read_band(path: str):
    """(pooled head-seed band, set of cells that carry replicates).

    The band pools both head budgets (scripts/noise_band.py); `path` only
    names the cells that carry a replicate at all.
    """
    band = noise_band.pooled_band()
    if not os.path.exists(path):
        return band, set()
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("range") or "").strip()]
    return band, {r["cell"] for r in rows}


def draw(scores: dict, path: str, band=None, replicated=frozenset()) -> None:
    cells = order(scores)
    colour = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cells)}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 6.2),
                                 gridspec_kw={"width_ratios": [1.15, 1]})

    # --- left: the ladder ---------------------------------------------------
    for cell in cells:
        for head, series in sorted(scores[cell].items()):
            ls, mk = HEAD_STYLE.get(head, ("-", "o"))
            xs = [s / 1000 for s, _ in series]
            ys = [v for _, v in series]
            ax.plot(xs, ys, ls, marker=mk, ms=4, lw=1.6,
                    color=colour[cell], alpha=0.9,
                    label=cell_label(cell, short=True)
                    if head == "student" else None)
    ax.axhline(1.0, color="#333", lw=1.2)
    ax.annotate("seasonal naive", xy=(0.99, 1.0), xycoords=("axes fraction",
                "data"), ha="right", va="bottom", fontsize=8, color="#333")
    ax.set_xlabel("backbone step (thousands)")
    ax.set_ylabel("GM-Relative MASE  (lower is better)")
    ax.set_title("Every run, both heads\nsolid = student encoder, "
                 "dashed = teacher encoder", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, ncol=1, loc="upper left", framealpha=0.9)

    # --- right: the change the rule reads -----------------------------------
    labels, values, colours, hatches, tested = [], [], [], [], []
    for cell in cells:
        for head in ("student", "teacher"):
            series = scores[cell].get(head, [])
            for (s0, v0), (s1, v1) in zip(series, series[1:]):
                mark = "" if cell in replicated else "  *"
                labels.append(
                    f"{cell_label(cell, short=True)}  |  {head} enc  |  "
                    f"{s0//1000}k→{s1//1000}k{mark}")
                values.append(v1 - v0)
                colours.append(colour[cell])
                hatches.append("" if head == "student" else "///")
                tested.append(cell in replicated)
    ypos = range(len(labels))
    if band:
        bx.axvspan(-band, band, color="#9a9a9a", alpha=0.16, zorder=0)
        for edge in (-band, band):
            bx.axvline(edge, color="#9a9a9a", lw=1.0, ls=(0, (4, 3)),
                       zorder=1)
    bars = bx.barh(list(ypos), values, color=colours, alpha=0.9,
                   edgecolor="white", linewidth=0.6, zorder=2)
    for bar, h, ok, col in zip(bars, hatches, tested, colours):
        bar.set_hatch(h)
        if not ok:                       # no replicate: measured, not tested
            bar.set_facecolor("none")
            bar.set_edgecolor(col)
            bar.set_linewidth(1.3)
            bar.set_linestyle((0, (3, 2)))
    bx.axvline(0, color="#333", lw=1.2, zorder=3)
    bx.set_yticks(list(ypos))
    bx.set_yticklabels(labels, fontsize=7)
    bx.invert_yaxis()
    bx.set_xlabel("change in GM-Relative MASE  (left of 0 = improved)")
    sub = "hatched = teacher encoder;  hollow + * = no head-seed replicate"
    if band:
        sub += f"\ngrey band = ±{band:.4f}, {noise_band.LABEL}"
    bx.set_title("Change from one stop to the next\n" + sub, fontsize=9)
    bx.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    res = os.path.join(EXP_DIR, "results")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--ladder", default=os.path.join(res, "ladder_all.csv"))
    p.add_argument("--spread", default=os.path.join(res, "seed_spread.csv"))
    p.add_argument("--out", default=os.path.join(EXP_DIR, "plots",
                                                 "ladder.png"))
    a = p.parse_args()

    if not os.path.exists(a.ladder):
        print(f"plot_ladder: no {a.ladder}", file=sys.stderr)
        return 1
    scores = read_scores(a.ladder)
    if not scores:
        print(f"plot_ladder: no scored row in {a.ladder}", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    band, replicated = read_band(a.spread)
    draw(scores, a.out, band, replicated)
    n = sum(len(s) for h in scores.values() for s in h.values())
    print(f"[out] {a.out}  ({len(scores)} cells, {n} scores)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
