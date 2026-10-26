"""The parts of the total loss of grn_ema099995 (magenta), grn_ema099995_f6
(grey, 6 forecaster layers), grn_ema099995_f6_rep01 (yellow),
grn_ema09998_rep003_halving (red) and grn_ema099995_rep003_halving (brown),
from their loss files. The colour names are the colours of the runs in the
score figure.

Magenta and grey have the same loss flags: --loss-shape
cosine_similarity_batch_rep_only, L_rep with a weight from 1 to 0 over the
first 10,000 steps, L_align on the teacher latent with rollout depth 3 and the
sum of the 4 depths, SIGReg on e and on h with weight 1.0, CPC off (the
column cpc_aux is blank). Yellow has the flags of grey with the L_rep weight
0.1 at each step. Red has the flags of magenta with the EMA 0.9998, the L_rep
weight 0.03 at each step, and --train-rollout-reduce halving: the L_align of
depth j at the weight 1 / 2^j. Brown has the flags of red with the EMA
0.99995 of magenta. So the column `loss` is the sum of:

  rep_w x l_rep             L_rep with its weight (blank l_rep: 0)
  l_align                   L_align at depth 0, the only depth with a column
  sigreg_e, sigreg_h        SIGReg on e and on h
  the rest                  loss minus the terms above: L_align at depths 1-3
                            (red and brown: depth j at 1 / 2^j)

Each value is the mean over log-spaced bins of steps (BINS bins from 1k to
665k), drawn only when the run logged all the steps of the bin. Writes
results/loss_parts_099995_binned.tsv and plots/loss_composition_099995.png
(the stacked parts of each run: the runs with the L_rep weight 1 to 0 by 10k
in the first row, the runs with an L_rep weight at each step in the second).

Usage: python3 plot_loss_composition_099995.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

STUDY = Path(__file__).resolve().parent.parent
RUNS_DIR = Path("/home/jupyter/cf_runs")
TABLE = STUDY / "results" / "loss_parts_099995_binned.tsv"
OUT_STACK = STUDY / "plots" / "loss_composition_099995.png"
FIRST, TOTAL, BINS = 1000, 665_000, 60
TICKS_K = [1, 10, 40, 100, 200, 400, 665]
STACK_TOP = 2.7  # the total of yellow is 2.6 at 1k
MINI_DIR = RUNS_DIR / "grn_ema099995_f6_rep01/ckpt"
# run -> ([(loss file, last step taken from it)], colour, label)
RUNS = {
    "grn_ema099995": ([(RUNS_DIR / "grn_ema099995/ckpt/grn_ema099995_losses.csv", 180_000),
                       (RUNS_DIR / "grn_ema099995/ckpt_r2/grn_ema099995_r2_losses.csv", None)],
                      "#ff00ff", "Magenta: GRN+EMA 0.99995, 3 forecaster layers (stopped at 391k)"),
    "grn_ema099995_f6": ([(RUNS_DIR / "grn_ema099995_f6/ckpt/grn_ema099995_f6_losses.csv", None)],
                         "#7f7f7f", "Grey: GRN+EMA 0.99995, 6 forecaster layers (stopped at 440k)"),
    "grn_ema099995_f6_rep01": ([(MINI_DIR / "grn_ema099995_f6_rep01_losses.csv", 40_000),
                                (MINI_DIR / "grn_ema099995_f6_rep01_r2_losses.csv", 180_000),
                                (MINI_DIR / "grn_ema099995_f6_rep01_r3_losses.csv", None)],
                               "#ffb000", "Yellow: GRN+EMA 0.99995, 6 layers, L_rep 0.1 (ended at 240k)"),
    "grn_ema09998_rep003_halving": (
        [(RUNS_DIR / "grn_ema09998_rep003_halving/ckpt/grn_ema09998_rep003_halving_losses.csv",
          160_000),
         (RUNS_DIR / "grn_ema09998_rep003_halving/ckpt/grn_ema09998_rep003_halving_r2_losses.csv",
          None)],
        "#e41a1c", "Red: GRN+EMA 0.9998, L_rep 0.03, depths 1/2^j (stopped at 212k)"),
    "grn_ema099995_rep003_halving": (
        [(RUNS_DIR / "grn_ema099995_rep003_halving/ckpt/grn_ema099995_rep003_halving_losses.csv",
          None)],
        "#8c564b", "Brown: GRN+EMA 0.99995, L_rep 0.03, depths 1/2^j (in training)"),
}
# run -> (row, column) of its panel
PANEL = {"grn_ema099995": (0, 0), "grn_ema099995_f6": (0, 1),
         "grn_ema099995_f6_rep01": (1, 0), "grn_ema09998_rep003_halving": (1, 1),
         "grn_ema099995_rep003_halving": (1, 2)}
PARTS = ["align_d0", "align_rest", "sigreg", "rep"]
# part -> (colour, label) of the stacked areas, from the bottom up
PART_STYLE = {
    "align_d0": ("#e6550d", "L_align, depth 0 (column l_align)"),
    "align_rest": ("#fdae6b", "L_align, depths 1 to 3 (the total minus the other parts)"),
    "sigreg": ("#31a354", "SIGReg on e and on h"),
    "rep": ("#c6dbef", "L_rep x its weight"),
}


def row_parts(row):
    """The parts of one loss row: total, rep, align_d0, sigreg_e, sigreg_h
    and rest."""
    rep = float(row["rep_w"]) * float(row["l_rep"]) if row["l_rep"] else 0.0
    total, align = float(row["loss"]), float(row["l_align"])
    sig_e, sig_h = float(row["sigreg_e"]), float(row["sigreg_h"])
    return total, rep, align, sig_e, sig_h, total - rep - align - sig_e - sig_h


def logged_parts(files):
    """(steps, parts array [n, 6]) of the loss ``files``. Each file gives the
    steps after the last step of the file before it, up to its own limit."""
    steps, parts, after = [], [], 0
    for path, last in files:
        with open(path) as f:
            for row in csv.DictReader(f):
                step = int(row["step"])
                if step > after and (last is None or step <= last):
                    steps.append(step)
                    parts.append(row_parts(row))
        after = last if last is not None else steps[-1]
    return np.array(steps), np.array(parts)


def binned(steps, parts):
    """(bin centre, mean parts) of each log-spaced bin logged in full."""
    edges = np.geomspace(FIRST, TOTAL, BINS + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (steps >= low) & (steps < high)
        if steps[-1] >= high and inside.any():
            rows.append((np.sqrt(low * high), *parts[inside].mean(axis=0)))
    return rows


def write_table():
    with open(TABLE, "w") as f:
        f.write("run\tbin_centre_step\ttotal\trep\talign_d0\tsigreg_e\tsigreg_h\talign_rest\n")
        for run, (files, _, _) in RUNS.items():
            for row in binned(*logged_parts(files)):
                f.write(f"{run}\t{row[0]:.0f}\t" + "\t".join(f"{v:.6f}" for v in row[1:]) + "\n")


def read_table():
    """{run: {column: array}} from TABLE, with sigreg = sigreg_e + sigreg_h.
    A run with no complete bin yet has no entry."""
    with open(TABLE) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    table = {}
    for run in RUNS:
        mine = [r for r in rows if r["run"] == run]
        if not mine:
            continue
        cols = {k: np.array([float(r[k]) for r in mine]) for k in mine[0] if k != "run"}
        cols["sigreg"] = cols["sigreg_e"] + cols["sigreg_h"]
        table[run] = cols
    return table


def log_steps(ax):
    ax.set_xscale("log")
    ax.set_xlim(FIRST * 0.9, TOTAL * 1.1)
    ax.xaxis.set_major_locator(FixedLocator([k * 1000 for k in TICKS_K]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{k}k" for k in TICKS_K])
    ax.set_xlabel("Training step (batch 64, log scale)")
    ax.grid(alpha=0.3)


def draw_stack(ax, run, cols):
    """The stacked parts of one run. A note gives the first L_rep value when
    it lies above the panel (the weight 1 to 0 over the first 10k)."""
    ax.stackplot(cols["bin_centre_step"], *[cols[p] for p in PARTS],
                 colors=[PART_STYLE[p][0] for p in PARTS],
                 labels=[PART_STYLE[p][1] for p in PARTS])
    ax.plot(cols["bin_centre_step"], cols["total"], color="black", lw=1.2, label="Total loss")
    ax.set_ylim(0, STACK_TOP)
    ax.set_title(RUNS[run][2], fontsize=10, loc="left")
    if cols["rep"][0] > STACK_TOP:
        ax.text(1100, STACK_TOP * 0.96, f"L_rep x weight: {cols['rep'][0]:.1f} at 1k, 0 from 10k",
                fontsize=8.5, va="top")
    log_steps(ax)


def plot_stack(table):
    """Each run of the table in its PANEL slot. The other slots stay empty.
    The grid has the columns up to the last column with a run."""
    columns = 1 + max(PANEL[run][1] for run in table)
    fig, axes = plt.subplots(2, columns, figsize=(5.6 * columns, 10.0), sharey=True, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for run, cols in table.items():
        axes[PANEL[run]].set_visible(True)
        draw_stack(axes[PANEL[run]], run, cols)
    for row in axes:
        row[0].set_ylabel("Loss, mean over log-spaced bins of steps")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9.5, frameon=False)
    fig.suptitle("Composition of the total loss", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(OUT_STACK, dpi=130)


def main():
    write_table()
    plot_stack(read_table())


if __name__ == "__main__":
    main()
