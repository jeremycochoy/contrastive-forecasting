#!/usr/bin/env python3
"""The measured grid: the score of every (decay ramp, EMA momentum) cell.

WHY THIS FIGURE EXISTS. The search moved one axis at a time, so
`plot_axes.py` shows each axis on its own. The card then ran a second ramp
family, at momentum 0.940, and the two families do not agree on the best
ramp. That is a fact about the GRID, and only a grid shows it.

WHAT IT SHOWS. Rows are the decay ramp, columns are the EMA momentum the arm
holds at the 40,000-step stop. A cell holds the GM-Relative MASE of that
(ramp, momentum) pair at the stop, the mean when the pair ran at more than
one seed, with the seed count under it. Colour is a one-hue ramp: the darkest
step is the no-decay reference, which no cell reaches. A cell the card never
ran is blank. The cell whose run lost the contrastive task takes the alarm
colour and names the step. The best cell carries a ring.

Usage:
  plot_grid.py --scores results/scores.csv --arms scripts/arms.tsv \
      --verdicts results/auc_verdicts.tsv --out plots/grid.png
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("arm_style", HERE / "arm_style.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

# The sequential blue ramp of the data-viz standard, steps 100 to 700.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scores",
                   default=str(HERE.parent / "results" / "scores.csv"))
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--verdicts",
                   default=str(HERE.parent / "results" / "auc_verdicts.tsv"))
    p.add_argument("--out", default=str(HERE.parent / "plots" / "grid.png"))
    args = p.parse_args(argv)

    arms = S.read_arms(args.arms)
    scored = S.read_scores(args.scores)
    if not scored:
        print(f"no score in {args.scores}", file=sys.stderr)
        return 2
    # `{arm: step}` for each arm whose run lost the contrastive task.
    lost = {}
    with open(args.verdicts, newline="") as fh:
        for rec in csv.DictReader(fh, delimiter="\t"):
            if (rec.get("verdict") or "").strip() == "lost":
                for row in arms:
                    if f"_{row['arm']}_losses" in rec["run"]:
                        lost[row["arm"]] = rec.get("lost_at", "")

    # A cell is (ramp, momentum at the stop). Only cells with a scored run or
    # a lost run are on the grid: an arm that never wrote a step is not a
    # measurement, and the annex names it.
    cells = {}
    for row in arms:
        arm = row["arm"]
        if arm not in scored and arm not in lost:
            continue
        key = (S.decay_ramp(row), round(S.momentum_at(row), 3))
        cells.setdefault(key, []).append(arm)
    ramps = sorted({k[0] for k in cells})
    moms = sorted({k[1] for k in cells})

    values = {k: [scored[a] for a in v if a in scored] for k, v in cells.items()}
    means = {k: statistics.fmean(v) for k, v in values.items() if v}
    best_key = min(means, key=means.get)
    vmin = S.SWEEP_BEST
    vmax = max(means.values())
    cmap = LinearSegmentedColormap.from_list("blue", list(reversed(RAMP)))
    norm = Normalize(vmin, vmax)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for i, ramp in enumerate(ramps):
        for j, mom in enumerate(moms):
            key = (ramp, mom)
            x, y = j, len(ramps) - 1 - i
            if key in means:
                fill = cmap(norm(means[key]))
                ax.add_patch(Rectangle((x + 0.03, y + 0.03), 0.94, 0.94,
                                       facecolor=fill, edgecolor="none"))
                # The ramp runs dark at the reference end, so a low score
                # sits on a dark cell and takes the surface colour as ink.
                dark = norm(means[key]) < 0.55
                ink = S.SURFACE if dark else S.INK
                n = len(values[key])
                text = f"{means[key]:.4f}"
                sub = (f"mean of {n} seeds" if n > 1
                       else cells[key][0])
                ax.text(x + 0.5, y + 0.58, text, ha="center", va="center",
                        fontsize=10, color=ink, fontweight="bold")
                ax.text(x + 0.5, y + 0.32, sub, ha="center", va="center",
                        fontsize=6.5, color=ink)
                if key == best_key:
                    ax.add_patch(Rectangle((x + 0.03, y + 0.03), 0.94, 0.94,
                                           facecolor="none",
                                           edgecolor=S.INK, linewidth=2.0))
            elif key in cells:
                arm = cells[key][0]
                ax.add_patch(Rectangle((x + 0.03, y + 0.03), 0.94, 0.94,
                                       facecolor=S.LOST, alpha=0.85,
                                       edgecolor="none"))
                ax.text(x + 0.5, y + 0.66, "lost the task", ha="center",
                        va="center", fontsize=8, color=S.SURFACE,
                        fontweight="bold")
                ax.text(x + 0.5, y + 0.45, f"at step {int(lost[arm]):,}",
                        ha="center", va="center", fontsize=6.5,
                        color=S.SURFACE)
                ax.text(x + 0.5, y + 0.25, arm, ha="center", va="center",
                        fontsize=6.5, color=S.SURFACE)
            else:
                ax.add_patch(Rectangle((x + 0.03, y + 0.03), 0.94, 0.94,
                                       facecolor="none", edgecolor=S.GRID,
                                       linewidth=0.8))
    ax.set_xlim(0, len(moms))
    ax.set_ylim(0, len(ramps))
    ax.set_xticks([j + 0.5 for j in range(len(moms))])
    ax.set_xticklabels([f"{m:.3f}" for m in moms])
    ax.set_yticks([len(ramps) - 0.5 - i for i in range(len(ramps))])
    ax.set_yticklabels([f"{r:,}" for r in ramps])
    ax.set_xlabel("EMA momentum at the 40,000-step stop")
    ax.set_ylabel("decay ramp, steps")
    ax.set_title("GM-Relative MASE of every (decay ramp, momentum) cell "
                 "at the 40,000-step stop", color=S.INK, fontsize=11,
                 loc="left")
    ax.tick_params(colors=S.MUTED, labelsize=8, length=0)
    ax.xaxis.label.set_color(S.MUTED)
    ax.yaxis.label.set_color(S.MUTED)
    for side in ax.spines.values():
        side.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    bar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("GM-Relative MASE, lower is better", color=S.MUTED,
                  fontsize=8)
    bar.ax.tick_params(colors=S.MUTED, labelsize=8)
    bar.set_ticks([t for t in (1.20, 1.25, 1.30, 1.35) if vmin < t < vmax])
    bar.ax.axhline(S.SWEEP_BEST, color=S.INK, linewidth=1.0)
    bar.ax.text(1.4, S.SWEEP_BEST, f"{S.SWEEP_BEST:.4f}, the reference,\n"
                "no decay", transform=bar.ax.get_yaxis_transform(),
                fontsize=7, color=S.INK, va="bottom", ha="left")
    ax.plot([], [], marker="s", linestyle="none", color=S.LOST, markersize=8,
            label="lost the contrastive task")
    ax.plot([], [], marker="s", linestyle="none", markerfacecolor="none",
            markeredgecolor=S.INK, markersize=8, label="best cell")
    ax.plot([], [], marker="s", linestyle="none", markerfacecolor="none",
            markeredgecolor=S.GRID, markersize=8, label="never run")
    ax.legend(frameon=False, fontsize=8, labelcolor=S.INK, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
    print(f"{args.out}: {len(means)} scored cell(s), {len(lost)} lost, "
          f"{len(ramps) * len(moms) - len(cells)} blank")
    return 0


if __name__ == "__main__":
    sys.exit(main())
