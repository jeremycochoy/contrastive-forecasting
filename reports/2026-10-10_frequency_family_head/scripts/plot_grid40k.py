"""Heat map of the 40k grid: small runs with the flags of violet (muP base 64/3/3,
init output 0 and input 0.3, L_rep 0.1, batch 64, seed 20260520) at width 64,
1 encoder layer and 3 forecaster layers, 40k steps, the EMA ramp to 1.0 over
the 40k steps. The rows are the learning rate, the columns the EMA start
(--ema-tau). Each cell is the forecast GM-Relative MASE of head B4 at 40k
(30k head steps, 97 configs). Lower is better. A grey cell has no score yet.

The heads of the runs trained on cf-recon (vast 56873094) or on elisa. SCORES
holds their score files score_<run>_bb40k_h30k_control.txt (elisa) and a copy
of the results folder of the box. Writes results/grid40k_scores.tsv and
plots/grid40k_heatmap.png.

Usage: python3 plot_grid40k.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

STUDY = Path(__file__).resolve().parent.parent
SCORES = [Path("/home/jupyter/cf_runs/grid40k/ff/results"),
          Path("/home/jupyter/cf_runs/grid40k/ff_box_results")]
TABLE = STUDY / "results" / "grid40k_scores.tsv"
OUT = STUDY / "plots" / "grid40k_heatmap.png"
# learning rate -> (label, run name part); EMA start -> (label, run name part)
LRS = {"5e-5": ("5e-5", "lr5em5"), "1e-4": ("1e-4 (violet)", "lr1em4"),
       "2e-4": ("2e-4", "lr2em4")}
TAUS = {"0.99995": ("0.99995 (violet)", "tau99995"),
        "0.999975": ("0.999975 (lime)", "tau999975")}


def run_score(run):
    """The forecast score of a run, or None. A run with two score files stops the script."""
    paths = [d / f"score_{run}_bb40k_h30k_control.txt" for d in SCORES]
    found = [p for p in paths if p.exists()]
    if len(found) > 1:
        raise SystemExit(f"{run}: a score in two places: {found}")
    return float(found[0].read_text().split()[-1]) if found else None


def grid_scores():
    """Array [learning rate, EMA start] of the scores, NaN for no score."""
    grid = np.full((len(LRS), len(TAUS)), np.nan)
    for i, (_, lr_part) in enumerate(LRS.values()):
        for j, (_, tau_part) in enumerate(TAUS.values()):
            score = run_score(f"g40_{lr_part}_{tau_part}")
            grid[i, j] = np.nan if score is None else score
    return grid


def write_table(grid):
    with open(TABLE, "w") as f:
        f.write("run\tlr\tema_tau\tforecast_gm_mase\n")
        for i, (lr, (_, lr_part)) in enumerate(LRS.items()):
            for j, (tau, (_, tau_part)) in enumerate(TAUS.items()):
                score = "" if np.isnan(grid[i, j]) else f"{grid[i, j]:.4f}"
                f.write(f"g40_{lr_part}_{tau_part}\t{lr}\t{tau}\t{score}\n")


def cell_text(ax, i, j, score, low, high):
    """A score in bold, or a grey "no score yet"."""
    if np.isnan(score):
        ax.text(j, i, "no score yet", ha="center", va="center", fontsize=10, color="#777777")
        return
    dark = (score - low) / (high - low) > 0.5
    ax.text(j, i, f"{score:.4f}", ha="center", va="center", fontsize=13,
            color="white" if dark else "black", fontweight="bold")


def mark_best(ax, grid):
    """A red frame around the lowest score, when there are 2 scores or more."""
    if np.sum(~np.isnan(grid)) < 2:
        return
    i, j = np.unravel_index(np.nanargmin(grid), grid.shape)
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=3))


def color_scale(grid):
    """The lowest and highest score, at least 0.01 apart (one score or two close ones)."""
    low, high = np.nanmin(grid), np.nanmax(grid)
    if high - low < 0.01:
        low, high = (low + high) / 2 - 0.005, (low + high) / 2 + 0.005
    return low, high


def draw_grid(ax, grid, low, high):
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=low, vmax=high, aspect="auto")
    for i in range(len(LRS)):
        for j in range(len(TAUS)):
            cell_text(ax, i, j, grid[i, j], low, high)
    mark_best(ax, grid)
    ax.set_xticks(range(len(TAUS)), [label for label, _ in TAUS.values()])
    ax.set_yticks(range(len(LRS)), [label for label, _ in LRS.values()])
    ax.set(xlabel="EMA start (--ema-tau)", ylabel="learning rate")
    return image


def main():
    grid = grid_scores()
    write_table(grid)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    image = draw_grid(ax, grid, *color_scale(grid))
    fig.colorbar(image, ax=ax, label="forecast GM-Relative MASE (lower is better)")
    ax.set_title(f"40k grid: forecast at 40k ({int(np.sum(~np.isnan(grid)))} of 6 scored)")
    fig.savefig(OUT, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
