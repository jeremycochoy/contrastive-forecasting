#!/usr/bin/env python3
"""#401 diagnosis plots — where the three phase-1 scores come from.

Two figures:

  collapse_onset.png   AUC, u_batchtime (dim usage) and cos_err_d0 against
                       training step, for k = 3 (#373's published A4, the
                       same cell) and #401's k = 8, 16 and 32 under BOTH
                       reductions. One panel per metric, log x.

  latent_rank.png      The saved checkpoints, measured through the loader
                       the GIFT-Eval uses: effective rank of the encoder
                       latent, and the mean cosine between the latents of
                       two different series. Values from
                       results/diag/collapse.csv.

Usage:  python3 plot_diag_collapse.py
"""
import csv
import sys
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

sys.path.insert(0, str(HERE))
import depth_colours as D                # noqa: E402

CF401 = Path("/home/jupyter/checkpoints_backup/cf-401")
CF401M = Path("/home/jupyter/cf401_sync/box_a/sync")
CURVES = REPO / "reports/2026-08-08_rollout_depth/curves"

# (label, colour, linestyle, csvs). k = 3 is #373's published run of THIS
# cell. The collapsed arms sit on the same flat line, so they take
# different dash patterns or the upper one hides the lower one.
#
# The last two are the MEAN arms of this same card. They are the control the
# figure was missing: they change the reduction and nothing else.
#
# So the two channels carry the two variables, and neither carries both:
#
#   hue          the depth, from `depth_colours.py` — the one map every figure
#                of this report reads. k = 3 is #373's reference and no depth
#                of this study, so it takes the reference ink and no hue.
#   dash         the reduction. Long dash summed, dotted mean, solid the
#                published k = 3 reference.
#
# Every mean arm therefore has its summed twin at the same depth in the same
# hue, and the difference between one pair is the reduction alone.
#
# The mean arms ran a 20k leg first, so their first 40,000 steps span two
# files. They are read in order and concatenated, which is what the summed
# arms' single `leg_40k` file already is.
ARMS = [
    ("k = 3, published reference, 1.0862", D.REF_K3_INK, "-",
     [CURVES / "r2/A4_cf393_arm6_v2_combab_alignS_cf373k3_losses.csv"]),
    ("k = 8 sum, scored 2.0357", D.colour(8), (0, (5, 2)),
     [CF401 / "k8/arm6_v2_combab_alignS/leg_40k/"
              "cf393_arm6_v2_combab_alignS_cf373k8_losses.csv"]),
    ("k = 16 sum, scored 4.5297", D.colour(16), (0, (5, 2)),
     [CF401 / "k16/arm6_v2_combab_alignS/leg_40k/"
              "cf393_arm6_v2_combab_alignS_cf373k16_losses.csv"]),
    ("k = 32 sum, scored 7.9575", D.colour(32), (0, (5, 2)),
     [CF401 / "k32/arm6_v2_combab_alignS/leg_40k/"
              "cf393_arm6_v2_combab_alignS_cf373k32_losses.csv"]),
    ("k = 8 mean, scored 1.2433", D.colour(8), (0, (1, 1.4)),
     [CF401M / "k8/arm6_v2_combab_alignS/leg_20k/"
               "cf393_arm6_v2_combab_alignS_cf373k8_mean_losses.csv",
      CF401M / "k8/arm6_v2_combab_alignS/leg_40k/"
               "cf393_arm6_v2_combab_alignS_cf373k8_mean_losses.csv"]),
    ("k = 32 mean, scored 1.2082", D.colour(32), (0, (1, 1.4)),
     [CF401M / "k32/arm6_v2_combab_alignS/leg_20k/"
                "cf393_arm6_v2_combab_alignS_cf373k32_mean_losses.csv",
      CF401M / "k32/arm6_v2_combab_alignS/leg_40k/"
                "cf393_arm6_v2_combab_alignS_cf373k32_mean_losses.csv"]),
]

# `latent_rank` colours a bar by its DEPTH, out of the same map every other
# figure reads, so no hue means two things across the report. The REDUCTION
# rides on the row label instead of on a second colour channel.
ARM_NAME = {"n/a": "", "sum": "sum", "mean": "mean"}

# The row labels of `latent_rank`, in plain words. `collapse.csv` names a row
# by the issue that produced the checkpoint, which no reader can resolve.
ROW_LABEL = {
    "393 parent  k=0  bb40k": "k = 0 parent",
    "379 B5pub   k=0  bb40k": "k = 0, other study",
}

PANELS = [
    ("auc", "AUC — can the model tell a positive from a negative?",
     0.5, "chance, 0.50"),
    ("u_batchtime", "u_batchtime on h — dimension usage", None, None),
    ("cos_err_d0", "cos_err_d0 = 1 - cos(f_t, h_t+1)", None, None),
]


def read(paths):
    """One arm's curve, over one file or several legs read in order.

    The sync loop rotates a file to `.prev` before the new copy lands, so a
    fetch that dropped mid-transfer leaves a shorter current file beside a
    longer previous one. A losses CSV only appends, so the bigger file is
    strictly more steps.
    """
    rows = []
    for path in paths:
        best = max((p for p in (path, path.with_suffix(path.suffix + ".prev"))
                    if p.is_file()), key=lambda p: p.stat().st_size,
                   default=None)
        if best is None:
            continue
        with open(best, newline="") as f:
            rows += list(csv.DictReader(f))
    if not rows:
        return {}
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
        for label, colour, ls, paths in ARMS:
            d = read(paths)
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
    # A figure legend, below the panels. Every panel is crowded: the summed
    # arms run along the bottom and the mean arms along the top.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, ncol=3, loc="lower center",
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Training-time collapse probes, both reductions, "
                 "against backbone step", fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 0.98, 0.95))
    out = PLOTS / "collapse_onset.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def row_label(r):
    """One bar's name: the reduction, the depth and the backbone stop.

    A checkpoint of another study keeps that fact in its name, because its
    numbers are not this study's and the figure must not read as if they are.
    """
    fixed = ROW_LABEL.get(r["label"])
    if fixed:
        return fixed
    return f"{ARM_NAME.get(r['reduce'], '')}, k = {r['k']}, bb{r['stop_k']}k"


def latent_rank():
    rows = list(csv.DictReader(open(RES / "diag/collapse.csv")))
    labels = [row_label(r) for r in rows]
    rank = [float(r["eff_rank"]) for r in rows]
    pcos = [float(r["pair_cos"]) for r in rows]
    depths = [int(r["k"]) for r in rows]
    colours = [D.colour(k) if k else D.REF_K0_INK for k in depths]
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
    seen = [k for k in (0, 8, 16, 32) if k in depths]
    fig.legend(handles=[plt.Line2D(
                   [], [], lw=6,
                   color=D.colour(k) if k else D.REF_K0_INK,
                   label=f"k = {k}") for k in seen],
               loc="lower center", ncol=len(seen), frameon=False, fontsize=9)
    fig.suptitle("The saved checkpoints, through the loader the GIFT-Eval "
                 "uses, on 21 real GIFT-Eval windows", fontsize=11)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    out = PLOTS / "latent_rank.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    PLOTS.mkdir(parents=True, exist_ok=True)
    collapse_onset()
    latent_rank()
