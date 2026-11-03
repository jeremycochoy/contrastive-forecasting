"""Heat maps of the muP grid: runs of 20k steps at width 64 (3 encoder and 3
forecaster layers, muP base width 64, seed 20260520, batch 64, constant lr,
weight decay 0.1, Adam betas 0.9 and 0.98), each scored by the forecast
GM-Relative MASE of head B4 (30k head steps, 97 configs). Lower is better.
The left panel is the learning rate against the output factor (input factor
1). The right panel is the learning rate against the input factor (output
factor 0). Both panels use one colour scale.

The output factor (--mup-init-output) multiplies the init of the output
layers: the attention output projection and the second FFN weight of the last
layer of the encoder and of the forecaster. 0 sets them to 0. The input
factor (--mup-init-input) multiplies the init of the first layer: the
projections of the patch encoders into the width.

The runs trained on cf-recon (vast 56873094): the 4 runs at the factors 1
with experiments 3f6fb790, the others with a33e3bb4 (#435, the init factors).
Their score files score_<head>_bb20k_h30k_control.txt are in SCORES (pulled
from the box). Writes results/mup_grid_scores.tsv and
plots/mup_grid_heatmap.png.

Usage: python3 plot_mup_grid.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STUDY = Path(__file__).resolve().parent.parent
SCORES = Path("/home/jupyter/cf_runs/mup_grid/scores/ff")
TABLE = STUDY / "results" / "mup_grid_scores.tsv"
OUT = STUDY / "plots" / "mup_grid_heatmap.png"
LRS = ["5.6e-5", "1.1e-4", "2.2e-4", "3.2e-4", "4.5e-4", "9e-4"]
OUTPUTS = ["1", "0.1", "0.01", "0"]
INPUTS = ["0.1", "0.3", "1", "3", "10"]
# run -> (learning rate, output factor, input factor, name of its forecast head or None)
RUNS = {
    "mup64_lr5.6e-5": ("5.6e-5", "1", "1", "mup64_lr5p6em5"),
    "mup64_lr1.1e-4": ("1.1e-4", "1", "1", None),
    "mup64_lr2.2e-4": ("2.2e-4", "1", "1", "mup64_lr2p2em4"),
    "mup64_lr4.5e-4": ("4.5e-4", "1", "1", "mup64_lr4p5em4"),
    "mup64_out0p1": ("5.6e-5", "0.1", "1", "mup64_out0p1"),
    "mup64_out0p01": ("5.6e-5", "0.01", "1", "mup64_out0p01"),
    "mup64_out0": ("5.6e-5", "0", "1", "mup64_out0"),
    "mup64_lr2p2em4_out0p01_in1": ("2.2e-4", "0.01", "1", "mup64_lr2p2em4_out0p01_in1"),
    "mup64_lr2p2em4_out0_in1": ("2.2e-4", "0", "1", "mup64_lr2p2em4_out0_in1"),
    "mup64_lr2p2em4_out0_in0p1": ("2.2e-4", "0", "0.1", "mup64_lr2p2em4_out0_in0p1"),
    "mup64_lr2p2em4_out0_in0p3": ("2.2e-4", "0", "0.3", "mup64_lr2p2em4_out0_in0p3"),
    "mup64_lr2p2em4_out0_in3": ("2.2e-4", "0", "3", "mup64_lr2p2em4_out0_in3"),
    "mup64_lr2p2em4_out0_in10": ("2.2e-4", "0", "10", "mup64_lr2p2em4_out0_in10"),
    "mup64_lr4p5em4_out0_in1": ("4.5e-4", "0", "1", "mup64_lr4p5em4_out0_in1"),
    "mup64_lr9em4_out0_in1": ("9e-4", "0", "1", "mup64_lr9em4_out0_in1"),
    "mup64_lr4p5em4_out0_in0p3": ("4.5e-4", "0", "0.3", "mup64_lr4p5em4_out0_in0p3"),
    "mup64_lr3p2em4_out0_in0p3": ("3.2e-4", "0", "0.3", "mup64_lr3p2em4_out0_in0p3"),
    "mup64_lr9em4_out0_in0p3": ("9e-4", "0", "0.3", "mup64_lr9em4_out0_in0p3"),
}
# panel -> (title, x label, column values, index of the column setting in a row,
#           index of the fixed setting, its value)
PANELS = [
    ("Output factor (input factor 1)", "output factor (--mup-init-output)", OUTPUTS, 1, 2, "1"),
    ("Input factor (output factor 0)", "input factor (--mup-init-input)", INPUTS, 2, 1, "0"),
]


def run_score(head):
    """The forecast score of a head, or None (no head or no score file)."""
    path = SCORES / f"score_{head}_bb20k_h30k_control.txt"
    return float(path.read_text().split()[-1]) if head and path.exists() else None


def write_table():
    with open(TABLE, "w") as f:
        f.write("run\tlr\toutput_factor\tinput_factor\tforecast_gm_mase\n")
        for run, (lr, output, inp, head) in RUNS.items():
            score = run_score(head)
            f.write(f"{run}\t{lr}\t{output}\t{inp}\t{'' if score is None else score}\n")


def read_table():
    """[(lr, output factor, input factor, score or None)] from TABLE."""
    with open(TABLE) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [(r["lr"], r["output_factor"], r["input_factor"],
             float(r["forecast_gm_mase"]) if r["forecast_gm_mase"] else None) for r in rows]


def panel_cells(rows, column, fixed, fixed_value):
    """{(lr, column value): score, or "no heads" for a backbone with no head}."""
    return {(row[0], row[column]): "no heads" if row[3] is None else row[3]
            for row in rows if row[fixed] == fixed_value}


def cell_text(ax, i, j, cell, low, high):
    """A score in bold, or a grey status text ("no heads", "not run")."""
    if isinstance(cell, float):
        dark = (cell - low) / (high - low) > 0.5
        ax.text(j, i, f"{cell:.4f}", ha="center", va="center", fontsize=11,
                color="white" if dark else "black", fontweight="bold")
    else:
        ax.text(j, i, cell, ha="center", va="center", fontsize=9, color="#777777")


def draw_panel(ax, rows, panel, low, high, cmap):
    title, xlabel, columns, column, fixed, fixed_value = panel
    cells = panel_cells(rows, column, fixed, fixed_value)
    grid = np.full((len(LRS), len(columns)), np.nan)
    for (lr, value), cell in cells.items():
        if isinstance(cell, float):
            grid[LRS.index(lr), columns.index(value)] = cell
    image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=low, vmax=high, aspect="auto")
    for i, lr in enumerate(LRS):
        for j, value in enumerate(columns):
            cell_text(ax, i, j, cells.get((lr, value), "not run"), low, high)
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(LRS)), LRS)
    ax.set(xlabel=xlabel, ylabel="learning rate")
    ax.set_title(title, fontsize=11)
    return image


def main():
    write_table()
    rows = read_table()
    scores = [row[3] for row in rows if row[3] is not None]
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#eeeeee")
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 5.2))
    for ax, panel in zip(axes, PANELS):
        image = draw_panel(ax, rows, panel, min(scores), max(scores), cmap)
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02,
                 label="forecast GM-Relative MASE (lower is better)")
    fig.suptitle("muP grid at 20k steps")
    fig.savefig(OUT, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
