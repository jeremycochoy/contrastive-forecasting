"""The loss of green-384 and grey-384 split into its parts, as they enter the training loss (the user, 11-20: 'look
at the different components of the loss and maybe do some stack plot ... see how it is different from the green').

loss = 10 x L_align over the rollout depths + rep_w x L_rep + SIGReg on e + SIGReg on h. The losses CSV logs the
total, L_rep, rep_w, both SIGReg terms and L_align at depth 0 (1 patch ahead, against the teacher). The CSV does not
log L_align of the deeper depths: it is the rest of the total (green: depths 1 to 3, weight 1. grey: depths 1 to 31,
weight 1/d). Each curve is a mean over 1,000 steps.

Row 1: one stack plot per run. Row 2: each part alone, green-384 against grey-384.
Usage: python3 plot_loss_stack_384.py <output png>
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HP = os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64")
RUNS = {"green-384": ("g166b_w384e1f5_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix", "#2ca02c"),
        "grey-384": ("g166b_w384e1f5_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10harm_rollfix", "#7f7f7f")}
PARTS = [("sig_e", "SIGReg on e (weight 1)", "#1f77b4"), ("sig_h", "SIGReg on h (weight 1)", "#9ecae1"),
         ("rep", "L_rep x 0.1", "#ff7f0e"), ("align0", "L_align x 10, 1 patch ahead", "#d62728"),
         ("align_deep", "L_align x 10, the deeper depths", "#f4a582")]
COLUMNS = ("step", "loss", "rep_w", "l_rep", "sigreg_e", "sigreg_h", "l_align")
WINDOW = 1000
STOPS = (25, 50, 75, 100, 125)


def parts(name):
    """{part: values} and the steps, one row per step, from the losses CSV of one run."""
    rows = [r for r in csv.DictReader(open(f"{HP}/{name}/ckpt/{name}_losses.csv")) if r.get("loss")]
    col = {k: np.array([float(r[k]) for r in rows]) for k in COLUMNS}
    out = {"sig_e": col["sigreg_e"], "sig_h": col["sigreg_h"], "rep": col["rep_w"] * col["l_rep"],
           "align0": 10 * col["l_align"]}
    out["align_deep"] = col["loss"] - sum(out.values())
    return col["step"] / 1000, out


def smooth(x):
    """The mean over WINDOW steps, 'valid' only: no edge of half-empty windows."""
    return np.convolve(x, np.ones(WINDOW) / WINDOW, mode="valid")


def stack(ax, label, steps, values, top):
    """The stack plot of the parts of one run, from 0 to ``top``."""
    xs = smooth(steps)
    ax.stackplot(xs, *[smooth(values[k]) for k, _, _ in PARTS], colors=[c for _, _, c in PARTS],
                 labels=[t for _, t, _ in PARTS], alpha=0.9)
    ax.set_title(f"{label}: the training loss split into its parts", loc="left")
    ax.set_ylim(0, top)


def overlay(ax, key, title, data):
    """One part of the loss, one line per run."""
    for label, (steps, values) in data.items():
        ax.plot(smooth(steps), smooth(values[key]), color=RUNS[label][1], lw=1.8, label=label)
    ax.set_title(title, loc="left", fontsize=10)


def stack_row(fig, grid, data):
    """Row 1: one stack plot per run, on one y scale. Returns its axes."""
    top = 1.05 * max(smooth(sum(v.values())).max() for _, v in data.values())
    axes = [fig.add_subplot(grid[0, 0:5]), fig.add_subplot(grid[0, 5:10])]
    for ax, (label, (steps, values)) in zip(axes, data.items()):
        stack(ax, label, steps, values, top)
    axes[0].set_ylabel("Training loss (mean over 1,000 steps)")
    axes[1].legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    return axes


def part_row(fig, grid, data):
    """Row 2: each part alone, the runs on one panel. Returns its axes."""
    small = [fig.add_subplot(grid[1, 2 * i:2 * i + 2]) for i in range(len(PARTS))]
    for ax, (key, title, _) in zip(small, PARTS):
        overlay(ax, key, title, data)
    small[0].legend(fontsize=9.5)
    small[0].set_ylabel("Value in the loss")
    return small


def main(out):
    data = {label: parts(name) for label, (name, _) in RUNS.items()}
    fig = plt.figure(figsize=(22, 11))
    grid = fig.add_gridspec(2, 10)
    for ax in stack_row(fig, grid, data) + part_row(fig, grid, data):
        for s in STOPS:
            ax.axvline(s, color="#bbbbbb", lw=0.8, ls=":")
        ax.set_xlabel("Training step (thousands; dotted: the scored stops)")
        ax.grid(color="#eeeeee")
    fig.suptitle("green-384 and grey-384: the parts of the training loss (green: L_align at depths 0 to 3, weight 1; "
                 "grey: depths 0 to 31, weight 1/d)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
