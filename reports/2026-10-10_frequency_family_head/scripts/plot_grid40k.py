"""Heat map of the 40k grid and of its second round, on one colour scale.

The grid: small runs with the flags of violet (muP base 64/3/3, init output 0
and input 0.3, L_rep 0.1, batch 64, seed 20260520) at width 64, 1 encoder layer
and 3 forecaster layers, 40k steps, the EMA ramp to 1.0 over the 40k steps. The
rows are the learning rate, the columns the EMA start (--ema-tau).

The second round (the user, 11-02 and 11-03): runs from the cell lr 1e-4,
EMA start 0.99995, each with one change (the losses of blue or an lr
schedule); batch 128 at the lr 1e-4, 2e-4, 4e-4 and 8e-4, and with the warmup
of W and a cosine to 4e-4. Batch 256 runs on the fixed code (the user, 11-03):
the constant lr 4e-4, 8e-4, 1e-3 and 1.6e-3. The grey rows are the planned runs
of the user's plan of 11-04: L_rep 0.05 at the better lr, then 80k steps with the
fixed SIGReg window at both L_rep weights. Its panel has that cell as its first row.

Each cell is the forecast GM-Relative MASE of head B4 at 40k (30k head steps,
97 configs). Lower is better. A grey cell has no score yet. A red frame marks
the lowest score of a set of rows scored the same way: the rows of the raw
rollout feedback (bugged scores) and the batch-256 rows (fixed rollout).

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
BEST = "g40_lr1em4_tau99995"
# run -> (label, lr): the second round, the changes from lr 1e-4, EMA start 0.99995
ROUND2 = {BEST: ("lr 1e-4, EMA start 0.99995\n(no change)", "1e-4"),
          "g40b_lr1em4_tau99995_rep005h": ("losses of blue:\nL_rep 0.05, depths 1/2^j", "1e-4"),
          "g40c_lr1em4_tau99995_cos": ("lr cosine\n1e-4 to 1e-5", "1e-4 to 1e-5"),
          "g40w_lr1em3_wu4k_cos": ("lr warmup to 1e-3,\nthen cosine to 1e-4",
                                   "warmup to 1e-3, cosine to 1e-4"),
          "g40b_lr1em4_tau99995_bs128": ("batch 128, lr 1e-4", "1e-4"),
          "g40b_lr2em4_tau99995_bs128": ("batch 128, lr 2e-4", "2e-4"),
          "g40b_lr4em4_tau99995_bs128": ("batch 128, lr 4e-4", "4e-4"),
          "g40w_lr1em3_wu4k_cos4em4_bs128": ("batch 128, warmup to 1e-3,\nthen cosine to 4e-4",
                                             "warmup to 1e-3, cosine to 4e-4"),
          "g40b_lr8em4_tau99995_bs128": ("batch 128, lr 8e-4", "8e-4"),
          "g40b_lr4em4_tau99995_bs128_sigfix": ("batch 128, lr 4e-4,\nfixed SIGReg window", "4e-4"),
          "g40b_lr4em4_tau99995_bs256_rollfix": ("FIXED ROLLOUT, batch 256, lr 4e-4\n(scored with the fixed rollout)", "4e-4"),
          "g40b_lr8em4_tau99995_bs256_rollfix": ("FIXED ROLLOUT, batch 256, lr 8e-4\n(scored with the fixed rollout)", "8e-4"),
          "g40b_lr1em3_tau99995_bs256_rollfix": ("FIXED ROLLOUT, batch 256, lr 1e-3\n(scored with the fixed rollout)", "1e-3"),
          "g40b_lr1p6em3_tau99995_bs256_rollfix": ("FIXED ROLLOUT, batch 256, lr 1.6e-3\n(scored with the fixed rollout)",
                                                   "1.6e-3")}
# The planned runs of the search, not started: (row label, when). The user (11-03 about
# 21:05 UTC): the heatmap shows them; each one moves into ROUND2 when it starts.
PLANNED = [("PLANNED, FIXED ROLLOUT, batch 256,\nL_rep 0.05, the better lr\n(1e-3 or 8e-4)", "starts at the 1e-3 score"),
           ("PLANNED, FIXED ROLLOUT, batch 256,\n80k steps, fixed SIGReg window,\nthe better lr and L_rep weight",
            "starts at the L_rep 0.05 score"),
           ("PLANNED, FIXED ROLLOUT, batch 256,\n80k steps, fixed SIGReg window,\nthe better lr, the other L_rep weight",
            "starts at the L_rep 0.05 score")]
# run -> its colour in the radar and in the training figure of the second round
ROUND2_COLOURS = {BEST: "#d95f02", "g40b_lr1em4_tau99995_rep005h": "#1f77b4",
                  "g40c_lr1em4_tau99995_cos": "#e7298a", "g40w_lr1em3_wu4k_cos": "#66a61e",
                  "g40b_lr1em4_tau99995_bs128": "#bcbd22", "g40b_lr2em4_tau99995_bs128": "#17becf",
                  "g40b_lr4em4_tau99995_bs128": "#8c564b", "g40w_lr1em3_wu4k_cos4em4_bs128": "#d62728",
                  "g40b_lr8em4_tau99995_bs128": "#6a3d9a", "g40b_lr4em4_tau99995_bs128_sigfix": "#e6ab02",
                  "g40b_lr4em4_tau99995_bs256_rollfix": "#1b9e77", "g40b_lr8em4_tau99995_bs256_rollfix": "#999999",
                  "g40b_lr1em3_tau99995_bs256_rollfix": "#ff00ff", "g40b_lr1p6em3_tau99995_bs256_rollfix": "#000000"}


def run_score(run):
    """The forecast score of a run, or None. A run with two score files stops the script."""
    paths = [d / f"score_{run}_bb40k_h30k_control.txt" for d in SCORES]
    found = [p for p in paths if p.exists()]
    if len(found) > 1:
        raise SystemExit(f"{run}: a score in two places: {found}")
    return float(found[0].read_text().split()[-1]) if found else None


def as_value(score):
    return np.nan if score is None else score


def grid_scores():
    """Array [learning rate, EMA start] of the scores, NaN for no score."""
    grid = np.full((len(LRS), len(TAUS)), np.nan)
    for i, (_, lr_part) in enumerate(LRS.values()):
        for j, (_, tau_part) in enumerate(TAUS.values()):
            grid[i, j] = as_value(run_score(f"g40_{lr_part}_{tau_part}"))
    return grid


def round2_scores():
    """Array [run of ROUND2 and planned run, 1] of the scores, NaN for no score."""
    return np.array([[as_value(run_score(run))] for run in ROUND2] + [[np.nan]] * len(PLANNED))


def score_text(value):
    return "" if np.isnan(value) else f"{value:.4f}"


def write_table(grid, round2):
    with open(TABLE, "w") as f:
        f.write("run\tlr\tema_tau\tchange\tforecast_gm_mase\n")
        for i, (lr, (_, lr_part)) in enumerate(LRS.items()):
            for j, (tau, (_, tau_part)) in enumerate(TAUS.items()):
                f.write(f"g40_{lr_part}_{tau_part}\t{lr}\t{tau}\t\t{score_text(grid[i, j])}\n")
        for k, (run, (label, lr)) in enumerate(ROUND2.items()):
            if run != BEST:
                change = label.replace("\n", " ")
                f.write(f"{run}\t{lr}\t0.99995\t{change}\t{score_text(round2[k, 0])}\n")


def cell_text(ax, i, j, score, low, high, empty="no score yet"):
    """A score in bold, or the grey text ``empty``."""
    if np.isnan(score):
        ax.text(j, i, empty, ha="center", va="center", fontsize=10, color="#777777")
        return
    dark = (score - low) / (high - low) > 0.5
    ax.text(j, i, f"{score:.4f}", ha="center", va="center", fontsize=13,
            color="white" if dark else "black", fontweight="bold")


def mark_best(ax, values, rows):
    """A red frame around the lowest score of ``rows``, when they have 2 scores or more."""
    part = np.full(values.shape, np.nan)
    part[rows] = values[rows]
    if np.sum(~np.isnan(part)) < 2:
        return
    i, j = np.unravel_index(np.nanargmin(part), part.shape)
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=3))


def color_scale(*panels):
    """The lowest and highest score of all panels, at least 0.01 apart."""
    values = np.concatenate([p.ravel() for p in panels])
    low, high = np.nanmin(values), np.nanmax(values)
    if high - low < 0.01:
        low, high = (low + high) / 2 - 0.005, (low + high) / 2 + 0.005
    return low, high


def draw_cells(ax, values, low, high, empty_texts=None, groups=None):
    """The cells on one colour scale; ``empty_texts`` gives the text of an empty row,
    ``groups`` the sets of rows that each get a frame on their lowest score (default: all rows)."""
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=low, vmax=high, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            cell_text(ax, i, j, values[i, j], low, high, (empty_texts or {}).get(i, "no score yet"))
    for rows in groups or [list(range(values.shape[0]))]:
        mark_best(ax, values, rows)
    return image


def draw_grid(ax, grid, low, high):
    image = draw_cells(ax, grid, low, high)
    ax.set_xticks(range(len(TAUS)), [label for label, _ in TAUS.values()])
    ax.set_yticks(range(len(LRS)), [label for label, _ in LRS.values()])
    ax.set(xlabel="EMA start (--ema-tau)", ylabel="learning rate")
    ax.set_title(f"40k grid: forecast at 40k ({int(np.sum(~np.isnan(grid)))} of 6 scored)")
    return image


def draw_round2(ax, round2, low, high):
    planned = {len(ROUND2) + k: when for k, (_, when) in enumerate(PLANNED)}
    fixed = [k for k, run in enumerate(ROUND2) if run.endswith("_rollfix")]
    bugged = [k for k in range(len(ROUND2)) if k not in fixed]
    draw_cells(ax, round2, low, high, planned, groups=[bugged, fixed])
    ax.set_xticks([0], ["lr 1e-4, EMA start 0.99995, batch 64,\nwith the change of the row"])
    labels = [label for label, _ in ROUND2.values()] + [label for label, _ in PLANNED]
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    scored = int(np.sum(~np.isnan(round2[1:len(ROUND2)])))
    ax.set_title(f"Second round: changes from lr 1e-4, EMA start 0.99995\n"
                 f"({scored} of {len(ROUND2) - 1} scored, {len(PLANNED)} planned)")


def main():
    grid, round2 = grid_scores(), round2_scores()
    write_table(grid, round2)
    low, high = color_scale(grid, round2)
    fig, (left, right) = plt.subplots(1, 2, figsize=(14.5, 13), gridspec_kw={"width_ratios": [2, 1.2], "wspace": 0.9})
    image = draw_grid(left, grid, low, high)
    draw_round2(right, round2, low, high)
    fig.colorbar(image, ax=[left, right], label="forecast GM-Relative MASE (lower is better)")
    fig.savefig(OUT, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
