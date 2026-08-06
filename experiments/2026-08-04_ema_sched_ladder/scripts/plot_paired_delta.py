#!/usr/bin/env python3
"""#393 — the bb40k-to-bb100k delta once BOTH of its ends carry a spread.

Reads `results/paired_delta.csv` (scripts/paired_delta.py) and
`results/seed_spread.csv`, and draws the two things the extend rule could
not see.

Every row of the file that carries a delta is drawn. Rows with a single head
seed carry no interval and no test: they are drawn hollow in grey and
labelled `n=1`, measured but not tested. Rows with two or three seeds carry
mean ± t(df, .05)·SE, with df printed per row, because df is 1 for the
two-seed row and 2 for the three-seed rows.

  left   the paired delta per run per head. Each small marker is one head
         seed's own `bb100k(s) - bb40k(s)`; the interval is the mean ±
         t(df, .05)·SE of that row's paired deltas. The rule is a strict
         `<`, so what it reads is exactly the SIGN. An interval crossing
         zero is a branch the head seed alone can flip.
  right  |t| against the two denominators. Grey is the bb100k spread alone,
         which is what a single-ended σ used; colour is the paired standard
         error, which carries both ends. The black caret on each row is that
         row's own critical value t(df, .05).

Colour is the answer, not the identity: a delta whose sign is the same at
every seed is blue, one whose sign flips is orange, and a row with no
replicate is grey. Head is carried by marker shape as well, so neither panel
depends on colour alone.

Usage:  python3 scripts/plot_paired_delta.py [--paired FILE] [--spread FILE]
                                             [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPTS_DIR)

# The same two hues plot_seed_spread.py validated together: ΔE 21.9 under
# protanopia and 27.9 under normal vision against the light surface.
STABLE = "#2c6fb5"
FLIPS = "#c2571b"
INK = "#3f3f3f"
INK_SOFT = "#9a9a9a"
ONE_ENDED = "#c9c9c9"
HEAD_MARKER = {"student": "o", "teacher": "s"}
SEEDS = ["20260722", "20260723", "20260724"]
UNTESTED = "#9a9a9a"


def num(row: dict, key: str):
    v = (row.get(key) or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def one_ended_sigma(spread: list[dict]) -> dict:
    """|delta| / sd(bb100k) — the denominator that covered one end only."""
    out = {}
    for r in spread:
        d, sd = num(r, "delta_mean"), num(r, "sd")
        if d is not None and sd:
            out[(r["cell"], r["head"])] = abs(d) / sd
    return out


def main() -> int:
    res = os.path.join(EXP_DIR, "results")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--paired", default=os.path.join(res, "paired_delta.csv"))
    p.add_argument("--spread", default=os.path.join(res, "seed_spread.csv"))
    p.add_argument("--out", default=os.path.join(EXP_DIR, "plots",
                                                 "paired_delta.png"))
    a = p.parse_args()

    rows = [r for r in read_csv(a.paired) if num(r, "delta_mean") is not None]
    if not rows:
        print("no paired delta rows yet; nothing to plot")
        return 0
    old = one_ended_sigma(read_csv(a.spread)) if os.path.exists(a.spread) else {}

    labels = []
    for r in rows:
        n = int(r["n_seeds"])
        tag = "n=1" if n < 2 else f"df={int(num(r, 'df') or 0)}"
        labels.append(f"{r['cell']}  {r['head']}   [{tag}]")
    y = list(range(len(rows)))[::-1]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13.5, 0.42 * len(rows) + 3.0),
        gridspec_kw={"width_ratios": [1.35, 1.0]})

    for yy, r in zip(y, rows):
        n = int(r["n_seeds"])
        tested = n >= 2 and num(r, "se_paired") is not None
        col = (UNTESTED if not tested
               else STABLE if r["sign_stable"] == "yes" else FLIPS)
        mk = HEAD_MARKER.get(r["head"], "o")
        mean = num(r, "delta_mean")
        if tested:
            t_crit = num(r, "t_crit_05") or 0.0
            axL.errorbar(mean, yy, xerr=t_crit * num(r, "se_paired"),
                         color=col, elinewidth=2.0, capsize=4, capthick=1.4,
                         zorder=2)
        for s_ in SEEDS:
            v = num(r, f"delta_{s_}")
            if v is not None:
                axL.plot(v, yy, mk, ms=4.5, mfc="none", mec=col, mew=1.0,
                         alpha=0.85, zorder=3)
        axL.plot(mean, yy, mk, ms=8, color=col if tested else "none",
                 mec=col, mew=1.4, zorder=4)

    axL.axvline(0.0, color=INK, lw=1.0, zorder=1)
    axL.set_yticks(y)
    axL.set_yticklabels(labels, fontsize=9)
    for tick, r in zip(axL.get_yticklabels(), rows):
        if int(r["n_seeds"]) < 2:
            tick.set_color(UNTESTED)
    axL.set_xlabel("paired  bb100k \u2212 bb40k   (GM-Relative MASE, lower is better)")
    axL.set_title("The delta, both ends at the same head seed\n"
                  "interval = mean \u00b1 t(df, .05)\u00b7SE, df printed per row",
                  fontsize=10, color=INK)
    axL.grid(axis="x", color=INK_SOFT, alpha=0.35, lw=0.6)
    axL.set_axisbelow(True)

    for yy, r in zip(y, rows):
        n = int(r["n_seeds"])
        tested = n >= 2 and num(r, "t_paired") is not None
        if not tested:
            axR.text(0.02, yy, "no replicate \u2014 measured, not tested",
                     fontsize=8.5, color=UNTESTED, va="center")
            continue
        col = STABLE if r["sign_stable"] == "yes" else FLIPS
        t_new = abs(num(r, "t_paired") or 0.0)
        t_old = old.get((r["cell"], r["head"]))
        if t_old is not None:
            axR.barh(yy + 0.19, t_old, height=0.34, color=ONE_ENDED,
                     edgecolor="none", zorder=2)
        axR.barh(yy - 0.19, t_new, height=0.34, color=col, edgecolor="none",
                 zorder=2)
        t_crit = num(r, "t_crit_05")
        if t_crit is not None:
            axR.plot([t_crit], [yy], "|", color=INK, ms=22, mew=1.8, zorder=4)

    axR.set_yticks(y)
    axR.set_yticklabels([])
    axR.set_xlabel("|t|")
    axR.set_title("What the denominator was, and what it is\n"
                  "grey = bb100k spread alone;  colour = both ends, paired",
                  fontsize=10, color=INK)
    axR.grid(axis="x", color=INK_SOFT, alpha=0.35, lw=0.6)
    axR.set_axisbelow(True)
    for ax in (axL, axR):
        ax.set_ylim(-0.8, len(rows) - 0.2)

    handles = [
        Line2D([], [], color=STABLE, lw=3,
               label="same sign at every seed"),
        Line2D([], [], color=FLIPS, lw=3, label="sign flips on the head seed"),
        Line2D([], [], color=UNTESTED, lw=3,
               label="one head seed only: measured, not tested"),
        Line2D([], [], color=ONE_ENDED, lw=3, label="|t| against bb100k spread only"),
        Line2D([], [], color=INK, marker="|", lw=0, ms=12, mew=1.8,
               label="that row's t(df, .05)"),
        Line2D([], [], color=INK, marker="o", lw=0, label="student head"),
        Line2D([], [], color=INK, marker="s", lw=0, label="teacher head"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"  -> {a.out}  ({len(rows)} row(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
