"""The parts of the total loss of grn_ema099995 (magenta), grn_ema099995_f6
(grey, 6 forecaster layers) and BLK (cf419_cos200k), from their loss files.

The three runs have the same loss flags: --loss-shape
cosine_similarity_batch_rep_only, L_rep with a weight from 1 to 0 over the
first 10,000 steps, L_align on the teacher latent with rollout depth 3 and the
sum of the 4 depths, SIGReg on e and on h with weight 1.0, CPC off (the
column cpc_aux is blank). So the column `loss` is the sum of:

  rep_w x l_rep             L_rep with its weight (blank l_rep: 0)
  l_align                   L_align at depth 0, the only depth with a column
  sigreg_e, sigreg_h        SIGReg on e and on h
  the rest                  loss minus the terms above: L_align at depths 1-3

Each value is the mean over log-spaced bins of steps (BINS bins from 1k to
665k), drawn only when the run logged all the steps of the bin. Writes
results/loss_parts_099995_binned.tsv, plots/loss_composition_099995.png (the
stacked parts of each run) and plots/loss_terms_099995.png. The top row of
loss_terms has the total and SIGReg of the three runs on a log scale. Its
bottom row has, for each run, the forecast error of each rollout depth
j = 0..3 against the student latent, 2 x cos_err_dj = 2 - 2 cos(f^(j)_t,
h_{t+1+j}) (columns cos_err_dj, `rollout_cos_error` in src/metrics.py), and
L_align against the teacher latent: depth 0 (column l_align) and the mean of
depths 1-3 (the rest / 3). The log has no column for L_align at each of the
depths 1-3.

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
BLK_DIR = Path("/home/jupyter/checkpoints_backup/cf-412/vast_lr100x/cf-419c/cos200k/leg_665k")
TABLE = STUDY / "results" / "loss_parts_099995_binned.tsv"
OUT_STACK = STUDY / "plots" / "loss_composition_099995.png"
OUT_TERMS = STUDY / "plots" / "loss_terms_099995.png"
FIRST, TOTAL, BINS = 1000, 665_000, 60
TICKS_K = [1, 10, 40, 100, 200, 400, 665]
STACK_TOP = 2.5
DEPTHS = range(4)  # the rollout depths 0-3 (--train-rollout-depth 3)
DEPTH_COLOURS = ["#7f2704", "#d94801", "#f16913", "#fdae6b"]  # depth 0 darkest
# run -> ([(loss file, last step taken from it)], colour, label)
RUNS = {
    "grn_ema099995": ([(RUNS_DIR / "grn_ema099995/ckpt/grn_ema099995_losses.csv", 180_000),
                       (RUNS_DIR / "grn_ema099995/ckpt_r2/grn_ema099995_r2_losses.csv", None)],
                      "#ff00ff", "GRN+EMA 0.99995 (stopped at 391k)"),
    "grn_ema099995_f6": ([(RUNS_DIR / "grn_ema099995_f6/ckpt/grn_ema099995_f6_losses.csv", None)],
                         "#7f7f7f", "GRN+EMA 0.99995, 6 forecaster layers (in training)"),
    "cf419_cos200k": ([(BLK_DIR / "cf419_cos200k_losses.csv", 140_000),
                       (BLK_DIR / "cf419_cos200k_r2_losses.csv", None)],
                      "#1a1a1a", "BLK"),
}
PARTS = ["align_d0", "align_rest", "sigreg", "rep"]
# part -> (colour, label) of the stacked areas, from the bottom up
PART_STYLE = {
    "align_d0": ("#e6550d", "L_align, depth 0 (column l_align)"),
    "align_rest": ("#fdae6b", "L_align, depths 1 to 3 (the total minus the other parts)"),
    "sigreg": ("#31a354", "SIGReg on e and on h"),
    "rep": ("#c6dbef", "L_rep x its weight (1 to 0 over the first 10k)"),
}


def row_parts(row):
    """The parts of one loss row: total, rep, align_d0, sigreg_e, sigreg_h,
    rest, then 2 x cos_err_dj of the depths 0-3."""
    rep = float(row["rep_w"]) * float(row["l_rep"]) if row["l_rep"] else 0.0
    total, align = float(row["loss"]), float(row["l_align"])
    sig_e, sig_h = float(row["sigreg_e"]), float(row["sigreg_h"])
    errors = [2.0 * float(row[f"cos_err_d{j}"]) for j in DEPTHS]
    return (total, rep, align, sig_e, sig_h, total - rep - align - sig_e - sig_h, *errors)


def logged_parts(files):
    """(steps, parts array [n, 6]) of the loss ``files``; each file gives the
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
        f.write("run\tbin_centre_step\ttotal\trep\talign_d0\tsigreg_e\tsigreg_h\talign_rest\t"
                + "\t".join(f"err_d{j}" for j in DEPTHS) + "\n")
        for run, (files, _, _) in RUNS.items():
            for row in binned(*logged_parts(files)):
                f.write(f"{run}\t{row[0]:.0f}\t" + "\t".join(f"{v:.6f}" for v in row[1:]) + "\n")


def read_table():
    """{run: {column: array}} from TABLE, with sigreg = sigreg_e + sigreg_h."""
    with open(TABLE) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    table = {}
    for run in RUNS:
        mine = [r for r in rows if r["run"] == run]
        cols = {k: np.array([float(r[k]) for r in mine]) for k in mine[0] if k != "run"}
        cols["sigreg"] = cols["sigreg_e"] + cols["sigreg_h"]
        table[run] = cols
    return table


def log_steps(ax, label=True):
    ax.set_xscale("log")
    ax.set_xlim(FIRST * 0.9, TOTAL * 1.1)
    ax.xaxis.set_major_locator(FixedLocator([k * 1000 for k in TICKS_K]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{k}k" for k in TICKS_K])
    if label:
        ax.set_xlabel("Training step (batch 64, log scale)")
    ax.grid(alpha=0.3)


def draw_stack(ax, run, cols):
    """The stacked parts of one run; L_rep goes above the panel before 10k."""
    ax.stackplot(cols["bin_centre_step"], *[cols[p] for p in PARTS],
                 colors=[PART_STYLE[p][0] for p in PARTS],
                 labels=[PART_STYLE[p][1] for p in PARTS])
    ax.plot(cols["bin_centre_step"], cols["total"], color="black", lw=1.2, label="Total loss")
    ax.set_ylim(0, STACK_TOP)
    ax.set_title(RUNS[run][2], fontsize=10, loc="left")
    ax.text(1100, STACK_TOP * 0.96, f"L_rep x weight: {cols['rep'][0]:.1f} at 1k, 0 from 10k",
            fontsize=8.5, va="top")
    log_steps(ax)


def plot_stack(table):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    for ax, run in zip(axes, RUNS):
        draw_stack(ax, run, table[run])
    axes[0].set_ylabel("Loss, mean over log-spaced bins of steps")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9.5, frameon=False)
    fig.suptitle("Composition of the total loss", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(OUT_STACK, dpi=130)


def draw_run_terms(axes, table):
    """Top row: the total and SIGReg of the three runs, one colour per run."""
    titles = {"total": "Total loss", "sigreg_h": "SIGReg on h", "sigreg_e": "SIGReg on e"}
    for ax, (column, title) in zip(axes, titles.items()):
        for run, (_, colour_hex, label) in RUNS.items():
            cols = table[run]
            ax.plot(cols["bin_centre_step"], cols[column], color=colour_hex, lw=2.2, label=label)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11, loc="left")
        log_steps(ax, label=False)
    axes[0].set_ylabel("Mean over log-spaced bins (log scale)")
    axes[0].legend(loc="upper right", fontsize=9)


def draw_depths(ax, run, cols):
    """Bottom row: the error of each rollout depth against the student latent
    (solid), and L_align against the teacher latent (dashed)."""
    steps = cols["bin_centre_step"]
    for j in DEPTHS:
        ax.plot(steps, cols[f"err_d{j}"], color=DEPTH_COLOURS[j], lw=2.0,
                label=f"Depth {j}: 2 - 2 cos(f, student h)")
    ax.plot(steps, cols["align_d0"], color="#08519c", lw=2.0, ls="--",
            label="L_align depth 0: 2 - 2 cos(f, teacher h)")
    ax.plot(steps, cols["align_rest"] / 3, color="#6baed6", lw=2.0, ls="--",
            label="L_align, mean of depths 1-3 (teacher h)")
    ax.set_title(RUNS[run][2], fontsize=10, loc="left")
    log_steps(ax)


def plot_terms(table):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    draw_run_terms(axes[0], table)
    for ax, run in zip(axes[1], RUNS):
        draw_depths(ax, run, table[run])
        ax.sharey(axes[1][0])
    axes[1][0].set_ylim(0.1, 1.15)  # all three runs: BLK goes down to 0.145
    axes[1][0].set_ylabel("Mean over log-spaced bins of steps")
    axes[1][0].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_TERMS, dpi=130)


def main():
    write_table()
    table = read_table()
    plot_stack(table)
    plot_terms(table)


if __name__ == "__main__":
    main()
