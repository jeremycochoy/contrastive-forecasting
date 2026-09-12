#!/usr/bin/env python3
"""The learning-rate bracket on configuration 1, at 40,000 backbone steps.

WHY THIS FIGURE EXISTS. Every arm of this card trains at 1e-3, the Moirai
recipe. The project does not use muP, and the trainer builds one AdamW group
over all parameters, so a rate that fits `d_model` 64 need not fit 384. A size
verdict taken at a rate that does not fit the width answers a question about
the rate. Three arms move the rate alone and close that hole.

HOW TO READ IT. The shaded band is the two seeds of configuration 1 at 1e-3,
1.2927 and 1.3495. That pair is this card's seed band. A rate arm inside the
shade draws with 1e-3. A rate arm below the shade beats it, and the report
then repeats phase 1 at the winning rate.

Usage:  plot_rates.py [--stop 40000] [--out plots/rates.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import plot_style as S  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
# The cell the bracket moves the rate on: configuration 1 and its repeat.
BASE = ("k3_r100_09", "k3_r100_09b")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default=str(STUDY / "results" / "scores.csv"))
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--stop", type=int, default=S.STOP)
    p.add_argument("--out", default=str(STUDY / "plots" / "rates.png"))
    args = p.parse_args()

    arms = {r["arm"]: r for r in S.read_arms(args.arms)}
    scored = S.read_scores(args.scores, stop=args.stop)
    band, _ = S.effective_band(scored)

    # Every arm of the configuration-1 cell that holds a score: the two 1e-3
    # seeds and the rate arms. Nothing else shares the cell.
    cell = [a for a, r in arms.items()
            if r["k"] == "3" and r["reduce"] == "sum" and r["decay"] == "-"
            and a in scored]
    if not cell:
        raise SystemExit(f"no configuration-1 arm holds a score at {args.stop}")
    pair = sorted(scored[a] for a in BASE if a in scored)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    fig.patch.set_facecolor(S.SURFACE)
    ax.set_facecolor(S.SURFACE)

    if len(pair) == 2:
        ax.axhspan(pair[0], pair[1], color=S.MUTED, alpha=0.14, zorder=1)

    # The two numbers the card measures itself against, so a reader sees how
    # far the best rate arm sits from each.
    ax.axhline(S.PROJECT_BEST, color=S.REFERENCE, linewidth=1.2,
               linestyle="--", zorder=1)
    ax.annotate(f"{S.PROJECT_BEST:.4f}  the project best "
                f"({S.PROJECT_BEST_LABEL})",
                (0.005, S.PROJECT_BEST), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=8,
                color=S.MUTED)
    ax.axhline(S.MOIRAI_SMALL, color=S.REFERENCE, linewidth=1.2,
               linestyle="--", zorder=1)
    ax.annotate(f"{S.MOIRAI_SMALL:.3f}  Moirai-2-Small, the same 97 configs",
                (0.005, S.MOIRAI_SMALL), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=8,
                color=S.MUTED)

    xs, ys, tick = [], [], {}
    for arm in cell:
        rate = float(arms[arm]["lr"])
        value = scored[arm]
        xs.append(rate); ys.append(value)
        # The tick reads the rate as `arms.tsv` writes it, not as `%g` prints
        # a float. `1e-3` and `0.001` are one value and two labels.
        tick[rate] = arms[arm]["lr"]
        # No error bar: every rate arm holds one seed, so the only measured
        # spread is the shaded 1e-3 pair, drawn once.
        ax.plot(rate, value, "o", markersize=8, color=S.SERIES, zorder=3)
        ax.annotate(f"{value:.4f}", (rate, value), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=S.INK, zorder=4)

    # NO CONNECTING LINE. Two seeds sit at 1e-3, so a line sorted by rate runs
    # to one of them and skips the other, which draws a trend between rates
    # this card did not measure. The points and the shaded pair carry the
    # reading on their own.

    ax.set_xscale("log")
    ax.set_xticks(sorted(set(xs)))
    ax.set_xticklabels([tick[x] for x in sorted(set(xs))], fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel("learning rate, one AdamW group over all parameters")
    ax.set_ylabel("GM-Relative MASE — lower is better")
    ax.set_title(f"GM-Relative MASE against learning rate, k = 3, "
                 f"{args.stop:,} steps, 11.4M", color=S.INK, fontsize=12,
                 loc="left", pad=12)
    S.tidy(ax)

    handles = [Line2D([], [], color=S.SERIES, marker="o", linestyle="none",
                      label="11.4M, k = 3, sum, 30,000-step head")]
    if len(pair) == 2:
        handles.append(Patch(color=S.MUTED, alpha=0.14,
                             label=f"{band:.4f} seed band, the 1e-3 pair"))
    fig.legend(handles=handles, frameon=False, fontsize=8, labelcolor=S.INK,
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
    print(f"wrote {args.out} ({len(cell)} arm(s))")


if __name__ == "__main__":
    main()
