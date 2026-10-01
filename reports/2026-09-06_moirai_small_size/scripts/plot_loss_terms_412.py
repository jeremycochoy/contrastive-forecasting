"""#412om against cyan: GM-Relative MASE and the training-loss terms on one
data-seen axis, to see whether the score's rise lines up with a loss term.

`collect` reads the per-step loss CSVs from the elisa mirror of the vast box
and writes results/loss_terms_412.tsv: one median per 4,000 batch-64 steps of
data seen, per run and term. `plot` draws plots/loss_terms_412om_vs_cyan.png
from that file and results/gm_trajectories.tsv. Usage:
    python3 scripts/plot_loss_terms_412.py [--collect]
"""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_style import P, colour, line

STUDY = Path(__file__).resolve().parent.parent
TSV = STUDY / "results" / "loss_terms_412.tsv"
GM = STUDY / "results" / "gm_trajectories.tsv"
OUT = STUDY / "plots" / "loss_terms_412om_vs_cyan.png"
MIRROR = Path("/home/jupyter/checkpoints_backup/cf-412/vast_lr100x")
CYAN = P + "_cos200k"
BIN = 4000  # batch-64 steps of data seen per point
# L_rep exists only while its weight ramps from 1 to 0 (#412om: 2,500 steps,
# 10,000 batch-64 steps of data; the others: 10,000 steps), so its panel
# shows the first 12,000 batch-64 steps with finer bins.
REP_BIN, REP_END = 250, 12000
# run: (CSV files, oldest first so a resumed leg overwrites its failed try;
# batch-64 steps per training step)
SOURCES = {
    "cf412om": (["cf-412om/leg_150k_try1"] + [f"cf-412om/leg_{k}k" for k in
                (10, 25, 50, 75, 100, 125, 150, 166)], "cf412om_k3_losses.csv", 4),
    "cf412oc": ([f"cf-412oc/leg_{k}k" for k in (40, 100, 200)], "cf412oc_k3_losses.csv", 1),
    CYAN: ([f"{CYAN}/arm6_v2_combab_alignT/leg_665k"],
           "cf393_arm6_v2_combab_alignT_cf373k3_cf412_" + CYAN + "_losses.csv", 1),
}
TERMS = ["loss", "l_align", "sigreg_e", "sigreg_h", "top1", "grad_norm"]
LABEL = {"cf412om": "Ours + patch sizes + mean/std + Moirai recipe (batch 256)",
         "cf412oc": "Ours + patch sizes + mean/std, cyan recipe (batch 64)",
         CYAN: "cyan: ours, one patch size, EWMA (batch 64)"}
PANEL = {"gm": "GM-Relative MASE (lower is better)",
         "loss": "Total training loss",
         "l_align": "l_align: predict the teacher latent",
         "sigreg_e": "SIGReg on the embedding",
         "sigreg_h": "SIGReg on the encoding",
         "top1": "Top-1 accuracy of the contrastive match",
         "grad_norm": "Gradient norm before the clip (not logged for cyan)",
         "l_rep": "L_rep while its weight ramps from 1 to 0 (first 12k only)"}


def read_steps(folders, name):
    rows = {}
    for folder in folders:
        path = MIRROR / folder / name
        if path.exists():
            with open(path) as f:
                for r in csv.DictReader(f):
                    rows[int(r["step"])] = r
    return rows


def number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def collect():
    with open(TSV, "w") as out:
        out.write("run\tterm\tdata_seen\tmedian\n")
        for run, (folders, name, scale) in SOURCES.items():
            bins = defaultdict(lambda: defaultdict(list))
            for step, r in read_steps(folders, name).items():
                seen = step * scale
                if seen > 680000:
                    continue
                for term in TERMS:
                    value = number(r.get(term))
                    if value is not None:
                        bins[term][seen // BIN].append(value)
                value = number(r.get("l_rep"))
                if value is not None and seen <= REP_END:
                    bins["l_rep"][seen // REP_BIN].append(value)
            for term, by_bin in bins.items():
                width = REP_BIN if term == "l_rep" else BIN
                for b, values in sorted(by_bin.items()):
                    out.write(f"{run}\t{term}\t{(b + 0.5) * width:.0f}\t{statistics.median(values):.6g}\n")
    print(f"wrote {TSV}")


def load():
    curves = defaultdict(lambda: ([], []))
    with open(TSV) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            xs, ys = curves[(r["run"], r["term"])]
            xs.append(float(r["data_seen"]))
            ys.append(float(r["median"]))
    scale = {"cf412om": 4}
    for arm, stop_k, score in (l.split() for l in open(GM)):
        if arm in SOURCES:
            xs, ys = curves[(arm, "gm")]
            xs.append(int(stop_k) * 1000 * scale.get(arm, 1))
            ys.append(float(score))
    return curves


def plot():
    curves = load()
    fig, axes = plt.subplots(4, 2, figsize=(15, 17))
    for ax, term in zip(axes.flat, ["gm"] + TERMS + ["l_rep"]):
        for run in SOURCES:
            xs, ys = curves.get((run, term), ([], []))
            if xs:
                order = sorted(range(len(xs)), key=xs.__getitem__)
                ax.plot([xs[i] for i in order], [ys[i] for i in order], line(run),
                        color=colour(run), lw=2.4, marker="o" if term == "gm" else None,
                        ms=6, label=LABEL[run])
        ax.set_title(PANEL[term], fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="best")
        if term in ("loss", "grad_norm"):
            ax.set_yscale("log")
    for ax, end, step in [(a, 700000, 100000) for a in axes.flat[:-1]] + [(axes.flat[-1], REP_END, 2000)]:
        ticks = range(0, end + 1, step)
        ax.set_xlim(0, end)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t // 1000}k" for t in ticks])
    for ax in axes[-1]:
        ax.set_xlabel("Data seen, in batch-64 steps. One #412om step counts 4.")
    fig.suptitle("#412om against cyan: the score and the training-loss terms on one axis\n"
                 "Each point is the median over 4,000 batch-64 steps. The contrastive terms "
                 "compare rows within a batch,\nso their level depends on the batch size "
                 "(256 for #412om, 64 for the others): compare the shapes of the curves.",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=110)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    if "--collect" in sys.argv:
        collect()
    plot()
