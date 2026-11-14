"""Each L_align term of 'red without SIGReg-F' (1, 2, 4, 8, 16 and 32 patches ahead) during pass 2, from the replay.

Usage: python3 plot_align_terms.py <output png>
Row 1: the whole pass 2 (50-step means, log y). Then one panel per collapse episode and one for the NaN (each step),
with SIGReg on e and h on the right axis. Distance 1 is the l_align column of the replay's losses CSV. The other
distances are the copies that diag_replay.py read in the loss (steps.csv). Each term is 2 - 2 cos (unweighted).
"""
import os
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64") + "/diag_variant_nan"
NAME = "g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10pw2_rollfix"
DISTANCES = (1, 2, 4, 8, 16, 32)
COLOURS = ("#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4", "#9467bd")  # one hue per distance
WINDOWS = [(10480, 10560, "collapse near 10.5k"), (11925, 11990, "collapse near 11.95k"),
           (13720, 13756, "before the NaN: step 13,755 gives a non-finite gradient")]


def load():
    losses = {int(r["step"]): r for r in csv.DictReader(open(f"{D}/ckpt/{NAME}_losses.csv"))}
    steps = {int(r["step"]): r for r in csv.DictReader(open(f"{D}/steps.csv")) if r.get("align_d2")}
    common = sorted(set(losses) & set(steps))
    terms = {1: np.array([float(losses[s]["l_align"]) for s in common])}
    terms.update({d: np.array([float(steps[s][f"align_d{d}"]) for s in common]) for d in DISTANCES[1:]})
    sig = {k: np.array([float(losses[s][k]) for s in common]) for k in ("sigreg_e", "sigreg_h")}
    return np.array(common), terms, sig


def bin_means(x, y, width):
    edges = np.arange(x[0], x[-1] + width, width)
    idx = np.digitize(x, edges)
    keep = [i for i in np.unique(idx) if (idx == i).sum() >= width // 2]
    return (np.array([x[idx == i].mean() for i in keep]), np.array([y[idx == i].mean() for i in keep]))


def whole(ax, step, terms):
    for d, colour in zip(DISTANCES, COLOURS):
        x, y = bin_means(step, terms[d], 50)
        ax.plot(x, y, color=colour, lw=2, label=f"{d} patch{'es' if d > 1 else ''} ahead")
    ax.set_yscale("log")
    ax.set_title("Pass 2, resumed from the 10k save: each L_align term (50-step means)", loc="left", fontsize=11)
    ax.set_ylabel("L_align = 2 - 2 cos (log scale)")
    ax.set_xlabel("training step")
    ax.grid(alpha=0.3, which="both")
    ax.legend(ncol=6, fontsize=9, loc="upper left")


def window(ax, step, terms, sig, first, last, title):
    keep = (step >= first) & (step <= last)
    if not keep.any():
        ax.set_title(f"{title}: the replay has not reached it yet", loc="left", fontsize=11)
        return
    for d, colour in zip(DISTANCES, COLOURS):
        ax.plot(step[keep], terms[d][keep], color=colour, lw=2, marker="o", ms=2.5)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel("L_align (each step)")
    ax.set_xlabel("training step")
    ax.grid(alpha=0.3)
    right = ax.twinx()
    right.plot(step[keep], sig["sigreg_e"][keep], color="black", lw=1.5, ls="--", label="SIGReg on e")
    right.plot(step[keep], sig["sigreg_h"][keep], color="grey", lw=1.5, ls=":", label="SIGReg on h")
    right.set_ylabel("SIGReg (dashed, right axis)")
    right.legend(fontsize=8, loc="upper left")


def main(out):
    step, terms, sig = load()
    fig = plt.figure(figsize=(16, 13))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.32, wspace=0.42, top=0.93)
    whole(fig.add_subplot(grid[0, :]), step, terms)
    for i, (first, last, title) in enumerate(WINDOWS):
        window(fig.add_subplot(grid[1, i]), step, terms, sig, first, last, title)
    fig.suptitle(f"Red without SIGReg-F: the 6 L_align terms, from the replay of pass 2 (steps {step[0]:,} to "
                 f"{step[-1]:,})", fontsize=13)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
