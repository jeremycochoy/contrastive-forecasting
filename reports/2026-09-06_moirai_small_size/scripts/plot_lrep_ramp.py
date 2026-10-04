"""L_rep while its weight ramps from 1 to 0: the first 10,000 batch-64 steps
of data of OMB and OCB (loss bug) and of OCF and OEF (loss fixed). The owner
asked for it on 10-02. Reads the per-step loss CSVs on the elisa mirror, as
plot_loss_terms_412.py does. Usage: python3 scripts/plot_lrep_ramp.py
"""
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_loss_terms_412 import LABEL, SOURCES, STUDY, number, read_steps
from run_style import colour, line

RUNS = ["cf412om", "cf412oc", "cf412oc2", "cf412oe2"]
END, BIN = 10000, 100  # batch-64 steps of data
OUT = STUDY / "plots" / "lrep_ramp_bug_vs_fixed.png"


def ramp_curve(run):
    """Median L_rep per BIN batch-64 steps of data, while its weight is above 0."""
    folders, name, scale = SOURCES[run]
    bins = defaultdict(list)
    for step, row in read_steps(folders, name).items():
        value = number(row.get("l_rep"))
        if value is not None and step * scale <= END:
            bins[step * scale // BIN].append(value)
    keys = sorted(bins)
    return [(b + 0.5) * BIN for b in keys], [statistics.median(bins[b]) for b in keys]


def main():
    fig, ax = plt.subplots(figsize=(11, 6.8))
    for run in RUNS:
        xs, ys = ramp_curve(run)
        if xs:
            ax.plot(xs, ys, line(run), color=colour(run), lw=2.4, label=LABEL[run])
    ax.set_xlim(0, END)
    ax.set_xlabel("Data seen, in batch-64 steps (one OMB step counts 4). "
                  "The weight of L_rep falls from 1 to 0 over this window.")
    ax.set_ylabel("L_rep, unweighted (median per 100 batch-64 steps)")
    ax.set_title("L_rep while its weight is above 0\n"
                 "Loss bug: the mean over the patch sizes of the L_rep of each size's own rows. "
                 "Loss fixed: one L_rep over every row of the batch.\n"
                 "L_rep takes its negatives from the rows it compares, so its level depends "
                 "on how many rows it reads.", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")


main()
