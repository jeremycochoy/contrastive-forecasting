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

from run_style import CODE, P, colour, line, tagged

STUDY = Path(__file__).resolve().parent.parent
TSV = STUDY / "results" / "loss_terms_412.tsv"
GM = STUDY / "results" / "gm_trajectories.tsv"
# L_rep measured on checkpoints after its ramp, on the whole batch (#412):
# the box lane lrep_measure.sh writes it, one row per checkpoint.
MEASURED = STUDY / "results" / "lrep_after_ramp.tsv"
MEASURED_RUN = {"cyan": P + "_cos200k", "cf412oc": "cf412oc", "cf412om": "cf412om"}
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
    "cf412oc2": ([f"cf-412oc2/leg_{k}k" for k in (40, 100, 200, 300, 400, 500, 665)],
                 "cf412oc2_k3_losses.csv", 1),
    "cf412oe2": ([f"cf-412oe2/leg_{k}k" for k in (40, 100, 200)], "cf412oe2_k3_losses.csv", 1),
    "cf412om2": ([f"cf-412om2/leg_{k}k" for k in (10, 25, 50, 75, 100, 125, 150, 166)],
                 "cf412om2_k3_losses.csv", 4),
    "cf412oa2": ([f"cf-412oa2/leg_{k}k" for k in (40, 100, 200)], "cf412oa2_k3_losses.csv", 1),
    "cf412ow2": (["cf-412ow2/leg_40k"], "cf412ow2_k3_losses.csv", 1),
    "cf412or2": (["cf-412or2/leg_40k", "cf-412or2/leg_100k"], "cf412or2_k3_losses.csv", 1),
    "cf412ol2": (["cf-412ol2/leg_40k"], "cf412ol2_k3_losses.csv", 1),
    "cf412bm": ([f"cf-412bm/leg_{k}k" for k in (10, 25)], "cf412bm_k3_losses.csv", 4),
    CYAN: ([f"{CYAN}/arm6_v2_combab_alignT/leg_665k"],
           "cf393_arm6_v2_combab_alignT_cf373k3_cf412_" + CYAN + "_losses.csv", 1),
}
TERMS = ["loss", "l_align", "sigreg_e", "sigreg_h", "top1", "grad_norm"]
LABEL = {"cf412om": "Ours + patch sizes + mean/std + Moirai recipe, loss bug (batch 256)",
         "cf412oc": "Ours + patch sizes + mean/std, cyan recipe, loss bug (batch 64)",
         "cf412oc2": "Ours + patch sizes + mean/std, cyan recipe, loss fixed (batch 64)",
         "cf412oe2": "Ours + patch sizes + EWMA, cyan recipe, loss fixed (batch 64)",
         "cf412om2": "Ours + patch sizes + mean/std + Moirai recipe, loss fixed (batch 256)",
         "cf412oa2": "Ours + patch sizes + mean/std, lr 5.6e-5, loss fixed (batch 64)",
         "cf412ow2": "Ours + patch sizes + mean/std, warmup to 1e-3 then lr 5.6e-5, loss fixed (batch 64)",
         "cf412or2": "As OWF, L_rep kept at weight 1 (batch 64)",
         "cf412ol2": "As OWR, lr down to 5.6e-5 by 40k (batch 64)",
         "cf412bm": "As OMF, bimoco: L_pred + L_rep with MoCo, tau 1, no L_align (batch 256)",
         CYAN: "cyan: ours, one patch size, EWMA (batch 64)"}
LABEL = {run: tagged(run, text) for run, text in LABEL.items()}
PANEL = {"gm": "GM-Relative MASE (lower is better)",
         "loss": "Total training loss",
         "l_align": "l_align: predict the teacher latent",
         "sigreg_e": "SIGReg on the embedding",
         "sigreg_h": "SIGReg on the encoding",
         "top1": "Top-1 accuracy of the contrastive match",
         "grad_norm": "Gradient norm before the clip (not logged for cyan)",
         "l_rep": "L_rep. Line: as each run computed it during its ramp (OMB and\n"
                  "OCB on one patch size at a time). Dots: on the whole batch,\n"
                  "measured on checkpoints after the ramp (weight 0 in training)"}


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
    if MEASURED.exists():
        with open(MEASURED) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                xs, ys = curves[(MEASURED_RUN[r["run"]], "l_rep_dots")]
                xs.append(float(r["data_seen"]))
                ys.append(float(r["l_rep"]))
    for arm, stop_k, score in (l.split() for l in open(GM)):
        if arm in SOURCES:
            xs, ys = curves[(arm, "gm")]
            xs.append(int(stop_k) * 1000 * SOURCES[arm][2])  # batch-64 steps per step
            ys.append(float(score))
    return curves


def top_dot(curves, x):
    """The highest measured L_rep near data seen x: its label goes above the
    dot, the labels of the lower dots go below theirs."""
    return max(y for run in SOURCES
               for xs, ys in [curves.get((run, "l_rep_dots"), ([], []))]
               for xi, y in zip(xs, ys) if abs(xi - x) <= 0.05 * x)


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
                        ms=6, label=rf"$\mathbf{{{CODE[run]}}}$")
            dots = curves.get((run, "l_rep_dots")) if term == "l_rep" else None
            if dots:
                mark = {CYAN: "o", "cf412oc": "D", "cf412om": "s"}[run]
                ax.plot(*dots, mark, color=colour(run), ms=9, mec="black",
                        zorder=2 if run == CYAN else 3)
                for x, y in zip(*dots):
                    above = y >= top_dot(curves, x)
                    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                                xytext=(8, 6 if above else -14),
                                fontsize=8.5, color=colour(run))
        ax.set_title(PANEL[term], fontsize=12 if term != "l_rep" else 10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="best", ncol=2, framealpha=0.8)
        if term in ("loss", "grad_norm"):
            ax.set_yscale("log")
    for ax in axes.flat:
        end, step = 700000, 100000
        ticks = range(0, end + 1, step)
        ax.set_xlim(0, end)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t // 1000}k" for t in ticks])
    for ax in axes[-1]:
        ax.set_xlabel("Data seen, in batch-64 steps. One OMB, OMF or OBM step counts 4.")
    fig.suptitle("OMB, OCB, OCF, OEF, OMF, OAF, OWF, OWR, OWL and OBM against CYN: the score and the training-loss terms on one axis\n"
                 "Each point is the median over 4,000 batch-64 steps. The contrastive terms "
                 "compare rows within a batch,\nso their level depends on the batch size "
                 "(256 for OMB, OMF and OBM, 64 for the others): compare the shapes of the curves.",
                 fontsize=13)
    # The panels name each run by its code; this legend gives the codes in full.
    proxies = [plt.Line2D([], [], color=colour(run), ls=line(run), lw=2.4) for run in SOURCES]
    fig.legend(proxies, [LABEL[run] for run in SOURCES], loc="lower center", ncol=2,
               fontsize=10.5, framealpha=0.94)
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    fig.savefig(OUT, dpi=110)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    if "--collect" in sys.argv:
        collect()
    plot()
