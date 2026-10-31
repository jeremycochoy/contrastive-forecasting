"""The parts of the total loss of grn_ema099995 (magenta), grn_ema099995_f6
(grey, 6 forecaster layers), grn_ema099995_f6_rep01 (yellow),
grn_ema099995_f6_rep01_mup (violet), grn_ema099995_f6_rep005_halving_mup
(blue), grn_ema099995_f6_rep01_mup_lr56 (orange), grn_ema0999975_f6_rep01_mup
(lime), grn_ema099995_f6_rep01_mup_hid56 (pink), grn_ema09998_rep003_halving
(red) and grn_ema099995_rep003_halving (brown), from their loss files. The
colour names are the colours of the runs in the score figure.

Magenta and grey have the same loss flags: --loss-shape
cosine_similarity_batch_rep_only, L_rep with a weight from 1 to 0 over the
first 10,000 steps, L_align on the teacher latent with rollout depth 3 and the
sum of the 4 depths, SIGReg on e and on h with weight 1.0, CPC off (the
column cpc_aux is blank). Yellow has the flags of grey with the L_rep weight
0.1 at each step. Violet has the flags of yellow with muP from the muP grid
(base width 64, base layers 3 + 3, output factor 0, input factor 0.3) and the
lr 1e-4. Red has the flags of magenta with the EMA 0.9998, the L_rep
weight 0.03 at each step, and --train-rollout-reduce halving: the L_align of
depth j at the weight 1 / 2^j. Brown has the flags of red with the EMA
0.99995 of magenta. Blue has the flags of violet with the L_rep weight 0.05
and --train-rollout-reduce halving. Orange has the flags of violet with the lr
5.6e-5 of yellow. Lime has the flags of violet with the EMA 0.999975 (1 - tau
halved). Pink has the flags of violet with the lr 3.36e-4 (muP gives the
hidden layers 5.6e-5, the lr of yellow) and the input init factor 1.008. So
the column `loss` is the sum of:

  rep_w x l_rep             L_rep with its weight (blank l_rep: 0)
  l_align                   L_align at depth 0, the only depth with a column
  sigreg_e, sigreg_h        SIGReg on e and on h
  the rest                  loss minus the terms above: L_align at depths 1-3
                            (red, brown and blue: depth j at 1 / 2^j)

Each value is the mean over log-spaced bins of steps (BINS bins from 1k to
665k), drawn only when the run logged all the steps of the bin. Writes
results/loss_parts_099995_binned.tsv and plots/loss_composition_099995.png
(the stacked parts of each run: the runs with 3 forecaster layers in the
first row, the runs with 6 forecaster layers in the second, and orange, lime
and pink, also with 6 forecaster layers, in the third). A dashed line in the
colour of the run marks the stop with its lowest forecast GM-Relative MASE
(results/scores.tsv, control head); a run with no forecast score has no line.

Usage: python3 plot_loss_composition_099995.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

STUDY = Path(__file__).resolve().parent.parent
RUNS_DIR = Path("/home/jupyter/cf_runs")
TABLE = STUDY / "results" / "loss_parts_099995_binned.tsv"
SCORES = STUDY / "results" / "scores.tsv"
OUT_STACK = STUDY / "plots" / "loss_composition_099995.png"
FIRST, TOTAL, BINS = 1000, 665_000, 60
TICKS_K = [1, 10, 40, 100, 200, 400, 665]
STACK_TOP_MIN = 2.7  # the total of yellow is 2.6 at 1k
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
                                (MINI_DIR / "grn_ema099995_f6_rep01_r3_losses.csv", 240_000),
                                (MINI_DIR / "grn_ema099995_f6_rep01_r4_losses.csv", None)],
                               "#ffb000", "Yellow: GRN+EMA 0.99995, 6 layers, L_rep 0.1 (stopped at 432k)"),
    "grn_ema099995_f6_rep01_mup": (
        [(RUNS_DIR / "grn_ema099995_f6_rep01_mup/ckpt/grn_ema099995_f6_rep01_mup_losses.csv", None)],
        "#8a2be2", "Violet: yellow with muP, lr 1e-4 (stopped at 181k)"),
    "grn_ema099995_f6_rep005_halving_mup": (
        [(RUNS_DIR / "grn_ema099995_f6_rep005_halving_mup/ckpt/"
          "grn_ema099995_f6_rep005_halving_mup_losses.csv", None)],
        "#1f77b4", "Blue: violet with L_rep 0.05, depths 1/2^j (stopped at 100k)"),
    "grn_ema099995_f6_rep01_mup_lr56": (
        [(RUNS_DIR / "grn_ema099995_f6_rep01_mup_lr56/ckpt/"
          "grn_ema099995_f6_rep01_mup_lr56_losses.csv", None)],
        "#ff7f0e", "Orange: violet with lr 5.6e-5 (stopped at 268k)"),
    "grn_ema0999975_f6_rep01_mup": (
        [(RUNS_DIR / "grn_ema0999975_f6_rep01_mup/ckpt/grn_ema0999975_f6_rep01_mup_losses.csv",
          None)],
        "#7fff00", "Lime: violet with EMA 0.999975 (still training)"),
    "grn_ema099995_f6_rep01_mup_hid56": (
        [(RUNS_DIR / "grn_ema099995_f6_rep01_mup_hid56/ckpt/"
          "grn_ema099995_f6_rep01_mup_hid56_losses.csv", None)],
        "#ff69b4", "Pink: violet with lr 3.36e-4, input init 1.008 (stopped at 254k)"),
    "grn_ema09998_rep003_halving": (
        [(RUNS_DIR / "grn_ema09998_rep003_halving/ckpt/grn_ema09998_rep003_halving_losses.csv",
          160_000),
         (RUNS_DIR / "grn_ema09998_rep003_halving/ckpt/grn_ema09998_rep003_halving_r2_losses.csv",
          None)],
        "#e41a1c", "Red: GRN+EMA 0.9998, L_rep 0.03, depths 1/2^j (stopped at 212k)"),
    "grn_ema099995_rep003_halving": (
        [(RUNS_DIR / "grn_ema099995_rep003_halving/ckpt/grn_ema099995_rep003_halving_losses.csv",
          300_000),
         (RUNS_DIR / "grn_ema099995_rep003_halving/ckpt/grn_ema099995_rep003_halving_r2_losses.csv",
          None)],
        "#8c564b", "Brown: GRN+EMA 0.99995, L_rep 0.03, depths 1/2^j (stopped at 345k)"),
}
# run -> (row, column) of its panel
PANEL = {"grn_ema099995": (0, 0), "grn_ema09998_rep003_halving": (0, 1),
         "grn_ema099995_rep003_halving": (0, 2), "grn_ema099995_f6": (1, 0),
         "grn_ema099995_f6_rep01": (1, 1), "grn_ema099995_f6_rep01_mup": (1, 2),
         "grn_ema099995_f6_rep005_halving_mup": (1, 3), "grn_ema099995_f6_rep01_mup_lr56": (2, 0),
         "grn_ema0999975_f6_rep01_mup": (2, 1), "grn_ema099995_f6_rep01_mup_hid56": (2, 2)}
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


def stack_top(table):
    """The top of the panels: above the highest total after the first 10k
    (magenta and grey have the L_rep weight 1 to 0 over the first 10k)."""
    highest = max(cols["total"][cols["bin_centre_step"] > 10_000].max(initial=0)
                  for cols in table.values())
    return max(STACK_TOP_MIN, 1.05 * highest)


def best_forecasts():
    """{run: (stop in k, GM-Relative MASE, number of scored stops)} of the
    lowest forecast score of each run (control head)."""
    scores = {}
    with open(SCORES) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["arm"] == "control":
                scores.setdefault(row["run"], []).append((int(row["stop_k"]),
                                                          float(row["gm_rel_mase"])))
    return {run: (*min(rows, key=lambda r: r[1]), len(rows)) for run, rows in scores.items()}


def mark_best(ax, run, best):
    """A dashed line in the colour of the run at the stop of its best forecast."""
    if run not in best:
        return
    stop, score, scored = best[run]
    colour = RUNS[run][1]
    ax.axvline(stop * 1000, color=colour, ls="--", lw=2.0,
               path_effects=[pe.Stroke(linewidth=3.4, foreground="black"), pe.Normal()])
    what = "best forecast" if scored > 1 else "only forecast score so far"
    ax.text(stop * 1000 * 1.08, 0.04, f"{what}:\n{score:.4f} at {stop}k",
            transform=ax.get_xaxis_transform(), color=colour, fontsize=9, fontweight="bold",
            va="bottom", bbox=dict(facecolor="white", alpha=0.9, edgecolor=colour))


def draw_stack(ax, run, cols, top, best):
    """The stacked parts of one run and the stop of its best forecast. A note
    gives the first L_rep value when it lies above the panel (the weight 1 to
    0 over the first 10k)."""
    ax.stackplot(cols["bin_centre_step"], *[cols[p] for p in PARTS],
                 colors=[PART_STYLE[p][0] for p in PARTS],
                 labels=[PART_STYLE[p][1] for p in PARTS])
    ax.plot(cols["bin_centre_step"], cols["total"], color="black", lw=1.2, label="Total loss")
    ax.set_ylim(0, top)
    ax.set_title(RUNS[run][2], fontsize=10, loc="left")
    if cols["rep"][0] > top:
        ax.text(1100, top * 0.96, f"L_rep x weight: {cols['rep'][0]:.1f} at 1k, 0 from 10k",
                fontsize=8.5, va="top")
    log_steps(ax)
    mark_best(ax, run, best)


def plot_stack(table):
    """Each run of the table in its PANEL slot. The other slots stay empty.
    The grid has the rows and columns up to the last row and column with a run."""
    rows = 1 + max(PANEL[run][0] for run in table)
    columns = 1 + max(PANEL[run][1] for run in table)
    fig, axes = plt.subplots(rows, columns, figsize=(5.6 * columns, 5.0 * rows), sharey=True,
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    top = stack_top(table)
    best = best_forecasts()
    for run, cols in table.items():
        axes[PANEL[run]].set_visible(True)
        draw_stack(axes[PANEL[run]], run, cols, top, best)
    for row in axes:
        row[0].set_ylabel("Loss, mean over log-spaced bins of steps")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="black", ls="--", lw=2.0))
    labels.append("Stop with the lowest forecast GM-Relative MASE of the run (in its colour)")
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9.5, frameon=False)
    fig.suptitle("Composition of the total loss", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(OUT_STACK, dpi=130)


def main():
    write_table()
    plot_stack(read_table())


if __name__ == "__main__":
    main()
