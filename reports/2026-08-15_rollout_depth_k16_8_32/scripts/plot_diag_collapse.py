#!/usr/bin/env python3
"""#401 diagnosis plots — where the three phase-1 scores come from.

Two figures:

  collapse_onset.png   AUC, u_batchtime (dim usage) and cos_err_d0 against
                       training step, for k = 3 (#373's published A4, the
                       same cell) and #401's k = 8, 16 and 32 under BOTH
                       reductions. One panel per metric, log x.

  latent_rank.png      Every saved checkpoint, measured through the loader
                       the GIFT-Eval uses: effective rank of the encoder
                       latent, and the mean cosine between the latents of
                       two different series. One row of dots per checkpoint
                       set, all 53 points. Values from
                       results/diag/collapse_all.csv.

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

# `latent_rank` colours a dot by its DEPTH, out of the same map every other
# figure reads, so no hue means two things across the report. The REDUCTION
# rides on the row instead of on a second colour channel.
#
# The figure draws every row of `collapse_all.csv`, not a subset: the claim it
# carries is that the two sets do not overlap, and a subset cannot show that.
ARM_NAME = {"n/a": "", "sum": "sum", "mean": "mean"}

# The four rows, top to bottom. A checkpoint of another study keeps that fact
# in its row, because its numbers are not this study's.
ROWS = ["k = 0 parent", "k = 0, other study", "mean", "sum"]


def row_of(r):
    """Which row one checkpoint belongs to."""
    if r["reduce"] == "n/a":
        return "k = 0, other study" if r["label"].startswith("379") \
            else "k = 0 parent"
    return r["reduce"]


# One panel per probe. A title is a LABEL: it names the quantity the panel
# draws, and every name here is one the report's `Terms` list defines. `h` and
# `f_t` are no part of that list, so no title carries them.
PANELS = [
    ("auc", "train AUC", 0.5, "chance, 0.50"),
    ("u_batch", "u_batch — dimension usage of the latent", None, None),
    ("cos_err_d0", "cos_err_d0 — forecast against the next latent",
     None, None),
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


def strip_y(base, n, halfwidth=0.26):
    """The dot heights of one row, spread evenly so no dot hides another."""
    if n == 1:
        return np.asarray([base])
    return base + np.linspace(-halfwidth, halfwidth, n)


def latent_rank():
    rows = list(csv.DictReader(open(RES / "diag/collapse_all.csv")))
    by_row = {name: [] for name in ROWS}
    for r in rows:
        by_row[row_of(r)].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    panels = [
        (axes[0], "eff_rank", "effective rank  (1.0 = one direction)",
         "effective rank of the encoder latent"),
        (axes[1], "pair_cos", "pair cosine  (1.0 = identical)",
         "pair cosine between two different series"),
    ]
    for ax, col, xlabel, title in panels:
        for i, name in enumerate(ROWS):
            group = sorted(by_row[name], key=lambda r: float(r[col]))
            base = len(ROWS) - 1 - i
            ys = strip_y(base, len(group))
            for r, yy in zip(group, ys):
                k = int(r["k"])
                ax.plot(float(r[col]), yy, marker="o", ms=5.5,
                        color=D.colour(k) if k else D.REF_K0_INK,
                        mec="white", mew=0.6, ls="none")
        ax.axvline(1.0, color="0.35", ls=":", lw=1.2)
        ax.set_yticks(range(len(ROWS))[::-1])
        ax.set_yticklabels([f"{n}  ({len(by_row[n])})" for n in ROWS],
                           fontsize=9)
        ax.set_ylim(-0.7, len(ROWS) - 0.3)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
    axes[1].set_xlim(-0.03, 1.08)

    seen = sorted({int(r["k"]) for r in rows})
    fig.legend(handles=[plt.Line2D(
                   [], [], marker="o", ls="none", ms=7,
                   color=D.colour(k) if k else D.REF_K0_INK,
                   label=f"k = {k}") for k in seen],
               loc="lower center", ncol=len(seen), frameon=False, fontsize=9)
    # No suptitle. The markdown caption of the report names this figure, and
    # a second copy inside it is the same label printed twice.
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = PLOTS / "latent_rank.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out} ({len(rows)} checkpoints)")


if __name__ == "__main__":
    PLOTS.mkdir(parents=True, exist_ok=True)
    collapse_onset()
    latent_rank()
