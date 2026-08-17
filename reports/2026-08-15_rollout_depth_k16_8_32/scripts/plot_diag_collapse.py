#!/usr/bin/env python3
"""#401 diagnosis plots — where the three phase-1 scores come from.

Two figures:

  collapse_onset.png   AUC, u_batchtime (dim usage) and cos_err_d0 against
                       training step, for k = 3 (#373's published A4, the
                       same cell) and #401's k = 8 and k = 16. One panel per
                       metric, log x.

  latent_rank.png      The saved checkpoints, measured through the loader
                       the GIFT-Eval uses: effective rank of the encoder
                       latent, and the mean cosine between the latents of
                       two different series. Values from
                       results/diag/collapse.csv.

Usage:  python3 plot_diag_collapse.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parent.parent
PLOTS = STUDY / "plots"
RES = STUDY / "results"

CF401 = Path("/home/jupyter/checkpoints_backup/cf-401")
CURVES = REPO / "reports/2026-08-08_rollout_depth/curves"

# (label, colour, linestyle, csv). k = 3 is #373's published run of THIS
# cell. The two collapsed arms sit on the same flat line, so they take
# different dash patterns or the upper one hides the lower one.
ARMS = [
    ("k = 3  (#373 A4, published 1.0862)", "#2f6f4e", "-",
     CURVES / "r2/A4_cf393_arm6_v2_combab_alignS_cf373k3_losses.csv"),
    ("k = 8  (#401, scored 2.0357)", "#c9772a", (0, (5, 2)),
     CF401 / "k8/arm6_v2_combab_alignS/leg_40k/"
             "cf393_arm6_v2_combab_alignS_cf373k8_losses.csv"),
    ("k = 16 (#401, scored 4.5297)", "#a63a3a", "-",
     CF401 / "k16/arm6_v2_combab_alignS/leg_40k/"
              "cf393_arm6_v2_combab_alignS_cf373k16_losses.csv"),
]

PANELS = [
    ("auc", "AUC — can the model tell a positive from a negative?",
     0.5, "chance, 0.50"),
    ("u_batchtime", "u_batchtime on h — dimension usage", None, None),
    ("cos_err_d0", "cos_err_d0 = 1 - cos(f_t, h_t+1)", None, None),
]


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for c in rows[0]:
        key = c.strip()
        vals = []
        for r in rows:
            v = (r[c] or "").strip()
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(np.nan)
        out[key] = np.asarray(vals)
    return out


def smooth(step, y, nbins=110):
    """Bin means on a log-spaced step grid.

    The three runs log at different rates — #373's every 200 steps, #401's
    every step — so a running mean of fixed width would smooth them by
    different amounts and the picture would be of the smoother, not of the
    runs. One shared log grid smooths all three the same, and it keeps step 1
    on the plot, which is where the k = 16 arm collapses.
    """
    m = np.isfinite(y) & (step > 0)
    step, y = step[m], y[m]
    if step.size == 0:
        return step, y
    edges = np.unique(np.geomspace(1, max(step.max(), 2), nbins + 1))
    idx = np.clip(np.digitize(step, edges) - 1, 0, len(edges) - 2)
    sx, sy = [], []
    for b in np.unique(idx):
        sel = idx == b
        sx.append(step[sel].mean())
        sy.append(y[sel].mean())
    return np.asarray(sx), np.asarray(sy)


def collapse_onset():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (col, title, ref, reflab) in zip(axes, PANELS):
        for label, colour, ls, path in ARMS:
            if not path.is_file():
                continue
            d = read(path)
            if col not in d:
                continue
            step, y = smooth(d["step"], d[col])
            ax.plot(step, y, color=colour, ls=ls, lw=1.6, label=label)
        if ref is not None:
            ax.axhline(ref, color="0.35", ls=":", lw=1.2)
            ax.text(1.05, ref, reflab, transform=ax.get_yaxis_transform(),
                    va="center", fontsize=8, color="0.35")
        ax.set_xscale("log")
        ax.set_xlabel("backbone step")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
    # A figure legend, below the panels. Every panel is crowded: the two
    # collapsed arms run along the bottom and the healthy one along the top.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, ncol=3, loc="lower center",
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("#401 phase 1 — the k = 8 and k = 16 backbones collapse; "
                 "the same cell at k = 3 does not", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 0.98, 0.95))
    out = PLOTS / "collapse_onset.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def latent_rank():
    rows = list(csv.DictReader(open(RES / "diag/collapse.csv")))
    labels = [r["label"].replace("  ", " ") for r in rows]
    rank = [float(r["eff_rank"]) for r in rows]
    pcos = [float(r["pair_cos"]) for r in rows]
    good = [r["k"] == "0" for r in rows]
    colours = ["#2f6f4e" if g else "#a63a3a" for g in good]
    y = np.arange(len(rows))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    axes[0].barh(y, rank, color=colours, height=0.6)
    axes[0].axvline(1.0, color="0.35", ls=":", lw=1.2)
    axes[0].set_xlabel("effective rank of h  (1.0 = one direction)")
    axes[0].set_title("How many latent directions the encoder uses",
                      fontsize=10)
    axes[1].barh(y, pcos, color=colours, height=0.6)
    axes[1].axvline(1.0, color="0.35", ls=":", lw=1.2)
    axes[1].set_xlim(0, 1.08)
    axes[1].set_xlabel("mean cos(h) between two different series"
                       "  (1.0 = identical)")
    axes[1].set_title("Does the encoder separate two different series?",
                      fontsize=10)
    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
    for ax, vals, fmt in ((axes[0], rank, "{:.2f}"), (axes[1], pcos, "{:.3f}")):
        for yy, v in zip(y, vals):
            ax.text(v, yy, " " + fmt.format(v), va="center", fontsize=8)
    fig.suptitle("The saved checkpoints, through the loader the GIFT-Eval "
                 "uses, on 21 real GIFT-Eval windows", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = PLOTS / "latent_rank.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    PLOTS.mkdir(parents=True, exist_ok=True)
    collapse_onset()
    latent_rank()
