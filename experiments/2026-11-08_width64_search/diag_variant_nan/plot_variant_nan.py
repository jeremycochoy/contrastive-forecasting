"""The training loss of 'red without SIGReg-F' up to each of its NaN steps, with its GM MASE scores.

Usage: python3 plot_variant_nan.py <output png>
Row 1: pass 1 (from step 0, NaN at 12,325). Row 2: pass 2 (resumed from the 10k save, NaN at 13,756. The third
start gave the same rows). Each: the stacked loss parts (bin means), the GM MASE scores on the right axis.
Row 3: the last steps before each NaN, each loss part divided by its mean over the 500 steps before (1 = usual).
"""
import os
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HP64 = os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64")
NAME = "g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10pw2_rollfix"
CKPT = f"{HP64}/{NAME}/ckpt/{NAME}"
ALIGN_WEIGHT = 10.0
GM = {250: 1.2412, 10000: 1.0884}  # the GM MASE scores of the 0k (250-step) and 10k saves, both of pass 1
PARTS = [  # bottom to top: (key, label, colour)
    ("rep", "L_rep x 0.1", "#1f77b4"),
    ("sig_e", "SIGReg on e", "#2ca02c"),
    ("sig_h", "SIGReg on h", "#98df8a"),
    ("a0", "L_align x 10 at 1 patch ahead", "#d62728"),
    ("arest", "L_align x 10 at 2, 4, 8, 16, 32 patches ahead", "#ff7f0e"),
]


def load_parts(path):
    """Steps and the loss parts of one losses CSV. The deeper L_align terms are the rest of the total."""
    rows = list(csv.DictReader(open(path)))
    col = lambda k: np.array([float(r[k]) for r in rows])
    step, loss = col("step").astype(int), col("loss")
    parts = {"rep": col("rep_w") * col("l_rep"), "sig_e": col("sigreg_e"), "sig_h": col("sigreg_h"),
             "a0": ALIGN_WEIGHT * col("l_align")}
    parts["arest"] = loss - sum(parts.values())
    return step, parts


def bin_means(step, parts, width):
    """The mean of each part over bins of `width` steps. X = the bin centre."""
    edges = np.arange(step[0] - 1, step[-1] + width, width)
    idx = np.digitize(step, edges) - 1
    keep = [i for i in range(len(edges) - 1) if (idx == i).sum() >= width // 2]
    x = np.array([step[idx == i].mean() for i in keep])
    return x, {k: np.array([v[idx == i].mean() for i in keep]) for k, v in parts.items()}


def stack_panel(ax, step, parts, width, nan_step, title):
    x, means = bin_means(step, parts, width)
    ax.stackplot(x, [means[k] for k, _, _ in PARTS], labels=[l for _, l, _ in PARTS],
                 colors=[c for _, _, c in PARTS], alpha=0.85)
    ax.axvline(nan_step, color="black", ls="--", lw=1.5)
    ax.text(nan_step, ax.get_ylim()[1] * 0.97, f" NaN at step {nan_step:,}", va="top", ha="right", fontsize=10)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel(f"training loss ({width}-step means)")
    left = step[0] - 150 if step[0] > 1000 else step[0]  # pass 2: room for the 10k score at its left edge
    ax.set_xlim(left, nan_step + (nan_step - step[0]) * 0.02)
    ax.grid(alpha=0.3)
    gm_axis(ax, left, nan_step)


def gm_axis(ax, first, last):
    right = ax.twinx()
    pts = [(s, v) for s, v in GM.items() if first - 300 <= s <= last]
    right.plot([s for s, _ in pts], [v for _, v in pts], "kP", ms=11, mfc="white", mew=2,
               label="GM MASE of this run's save (right axis)")
    for s, v in pts:
        right.annotate(f"{v:.4f}", (s, v), textcoords="offset points", xytext=(8, 6), fontsize=10)
    right.set_ylim(1.0, 1.32)
    right.set_ylabel("forecast GM-Relative MASE (lower is better)")
    return right


def ratio_panel(ax, step, parts, nan_step, before, title):
    """Each part per step over the last `before` steps, divided by its mean over the 500 steps before them."""
    last = step >= nan_step - before
    base = (step >= nan_step - before - 500) & ~last
    for k, label, colour in PARTS:
        ax.plot(step[last], parts[k][last] / parts[k][base].mean(), color=colour, lw=2, marker="o", ms=3,
                label=label)
    ax.axhline(1.0, color="grey", lw=1)
    ax.axvline(nan_step, color="black", ls="--", lw=1.5)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel("each part / its usual value\n(mean of the 500 steps before)")
    ax.set_xlabel("training step")
    ax.grid(alpha=0.3)


def main(out):
    s1, p1 = load_parts(CKPT + "_losses.csv")
    s2, p2 = load_parts(CKPT + "_r2_losses.csv")
    fig = plt.figure(figsize=(16, 15))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.1, 1.0], hspace=0.38, wspace=0.25, top=0.95)
    top = fig.add_subplot(grid[0, :])
    stack_panel(top, s1, p1, 100, 12325, "Pass 1: from step 0 (its 0k and 10k saves are scored)")
    middle = fig.add_subplot(grid[1, :])
    stack_panel(middle, s2, p2, 50, 13756, "Pass 2: resumed from the 10k save (the third start gave the same rows)")
    middle.set_xlabel("training step")
    handles, labels = top.get_legend_handles_labels()
    right_handles = [plt.Line2D([], [], color="k", marker="P", ls="", ms=11, mfc="white", mew=2)]
    top.legend(handles[::-1] + right_handles, labels[::-1] + ["GM MASE of this run's save (right axis)"],
               loc="upper center", bbox_to_anchor=(0.42, 1.0), fontsize=9, framealpha=0.95)
    ratio_panel(fig.add_subplot(grid[2, 0]), s1, p1, 12325, 30, "Pass 1: the 30 steps before the NaN")
    ratio_panel(fig.add_subplot(grid[2, 1]), s2, p2, 13756, 30, "Pass 2: the 30 steps before the NaN")
    fig.suptitle("Red without SIGReg-F (10-L-ALIGN + L_align at 1, 2, 4, 8, 16, 32 patches ahead): "
                 "the training loss up to each NaN", fontsize=13, y=0.985)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
