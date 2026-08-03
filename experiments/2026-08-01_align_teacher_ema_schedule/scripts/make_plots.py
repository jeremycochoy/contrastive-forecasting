#!/usr/bin/env python3
"""Figures for #388: teacher-target L_align and the 0.9 -> 1.0 α schedule.

Reads results/drift.csv (post-hoc, 5000-step cadence, from
teacher_latent_drift.py) and results/drift_500.csv (in-training, from
make_results_csvs.py). Writes the report's four figures into plots/:

    drift_headline.png    3x3 arm panel, adjacent 5k-pair drift
    align_fix.png         align_teacher's student and EMA-teacher latent
    cumulative_drift.png  drift vs the first probe, teacher arms
    drift_500.png         adjacent 500-step drift, the four new runs

and one supporting figure into plots/supporting/, which the issue does not
ask for:

    dim_usage.png         how much of h_t stays alive. A flat drift curve
                          is either a stable representation or a dead one,
                          and this separates the two. Needed to read the
                          L_align panels: both align arms sit within ~10%
                          of the collinear floor on the time axis.

Encoding, everywhere: one colour per named series, so a curve is identified
from a single legend row and the legend carries no entry that maps to no
visible curve. The EMA-teacher series are drawn last and dashed, so where a
teacher coincides with its student both stay visible.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ARMS = ["pred", "rep", "align", "align_teacher", "pred_moco",
        "rep_moco", "sigreg_e", "sigreg_h", "cpc"]
TEACHER_ARMS = ["align_teacher", "pred_moco", "rep_moco"]

STUDENT, TEACHER = "#1f77b4", "#d95f02"        # CVD-validated pair
MARK = dict(lw=1.6, marker="o", ms=3.2, markeredgecolor="white",
            markeredgewidth=0.5)

# One named series = one colour = one legend row, Okabe-Ito, equal line
# weight everywhere so no curve is a backdrop for another. The teacher of
# each pair is drawn LAST and dashed, so it renders on top where the two
# coincide and the student shows through the gaps. Shared by the headline,
# the cumulative figure and the 500-step figure.
NAMED = [
    (("const_0.9", "student_h"), "#0072B2",
     "student $h_t$, alpha = 0.9 constant", "-"),
    (("sched_0.9_1.0", "student_h"), "#009E73",
     "student $h_t$, alpha ramps 0.9 to 1.0", "-"),
    (("const_0.9", "teacher_h"), "#E69F00",
     "EMA teacher $h_t$, alpha = 0.9 constant", (0, (4, 3))),
    (("sched_0.9_1.0", "teacher_h"), "#CC79A7",
     "EMA teacher $h_t$, alpha ramps 0.9 to 1.0", (0, (4, 3))),
]
NAMED_ORDER = [0, 2, 1, 3]          # student/teacher pairs side by side

# The drift metric, spelled out inside every figure that plots it.
DRIFT_YLABEL = ("drift of $h_t$: 1 - cosine similarity\n"
                "(0 = no movement, 1 = orthogonal)")

# What each arm trained, in the words of the report's Runs table. Panel
# titles carry this instead of the internal run id.
ARM_TITLE = {
    "align_teacher": "L_align, target = EMA teacher",
    "pred_moco": "L_pred + MoCo negatives",
    "rep_moco": "L_rep + MoCo keys",
}


def named_handles(keys_present):
    """One legend entry per curve actually drawn in the figure."""
    return [Line2D([], [], color=NAMED[i][1], lw=1.5,
                   linestyle=NAMED[i][3], label=NAMED[i][2])
            for i in NAMED_ORDER if NAMED[i][0] in keys_present]


def read_drift(path):
    """{(run, arm, alpha, latent, kind[, step_ref]): [(step, drift_cos), ...]}.

    `vs_initial` rows carry their `step_ref` in the key. A run that resumed
    re-seeds the probe's reference snapshot to the resume step, and two
    references are two curves: joining them would draw the reset as drift.
    `adjacent` rows reference the previous probe by construction, so their
    `step_ref` moves every row and stays out of the key.
    """
    series = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (r["run"], r["arm"], r["alpha"], r["latent"], r["kind"])
            if r["kind"] == "vs_initial":
                key += (int(r["step_ref"]),)
            series[key].append((int(r["step"]), float(r["drift_cos"])))
    for key in series:
        series[key].sort()
    return series


def curves(series, arm, kind):
    """Every (alpha, latent, points) of one arm.

    A resumed run contributes one `vs_initial` entry per reference step, so
    its segments are drawn as separate lines instead of one false curve.
    """
    return sorted((k[2], k[3], v) for k, v in series.items()
                  if k[1] == arm and k[4] == kind)


def log_yticks(ax, ticks, labels=None):
    """Shared log y axis with every tick labelled.

    Matplotlib's default log locator labels decades only, which leaves one
    or two labels over a panel's height and nothing to read a value off.
    """
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    fmt = dict(zip(ticks, labels)) if labels else {}
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: fmt.get(v, f"{v:g}")))


def style_axes(ax, ylabel=None, xlabel="training step", logx=True):
    if logx:
        ax.set_xscale("log")
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


# --- 1. headline -----------------------------------------------------------


def plot_headline(series, out_png):
    # One named series = one colour = one legend row. Okabe-Ito palette,
    # distinct under the three common colour-vision deficiencies and where
    # curves overlap. Equal line weight everywhere: no curve is a backdrop
    # for another.
    # Drawing order = list order; the teacher of each pair goes LAST so it
    # renders on top where the two coincide, and is dashed so the student
    # shows through the gaps. Colour still identifies the curve on its own;
    # the dash is an extra cue, not a replacement.
    named, legend_order = NAMED, NAMED_ORDER
    solo_color, solo_label = "#333333", "student $h_t$"
    kw = dict(lw=1.5, marker="o", ms=3.0,
              markeredgecolor="white", markeredgewidth=0.5)

    fig, axs = plt.subplots(3, 3, figsize=(11.5, 8.5), sharey=True,
                            sharex=True)
    axs = axs.ravel()
    for ax, arm in zip(axs, ARMS):
        pts_of = {(a, l): p for a, l, p in curves(series, arm, "adjacent")}
        if arm in TEACHER_ARMS:
            for z, (key, color, _, ls) in enumerate(named):
                pts = pts_of.get(key, [])
                if not pts:
                    print(f"  (missing {arm} {key})")
                    continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=color, linestyle=ls, zorder=2 + z, **kw)
            ax.set_facecolor("#fdf6f0")
        else:
            for pts in pts_of.values():
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=solo_color, zorder=3, **kw)
        ax.set_title(arm, fontsize=11)
        # The claim this figure carries is a comparison of magnitudes ACROSS
        # panels, so the y scale stays shared. Drift spans 0.0035 to 1.32,
        # nearly three decades, which a linear axis flattens into the zero
        # line for every low-drift arm. Log keeps the panels comparable and
        # separates them. Limits contain every plotted point.
        ax.set_yscale("log")
        ax.set_ylim(2.5e-3, 2.0)
        log_yticks(ax, [0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
        style_axes(ax)
    for ax in axs[:6]:
        ax.set_xlabel("")
    # The y axis is shared by all nine panels, so the label is written once,
    # on the middle-left panel: three copies of a two-line label collide.
    axs[3].set_ylabel(DRIFT_YLABEL)
    handles = [Line2D([], [], color=solo_color, lw=1.5, label=solo_label)]
    handles += [Line2D([], [], color=named[i][1], lw=1.5,
                       linestyle=named[i][3], label=named[i][2])
                for i in legend_order]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Drift over 5000-step windows, one panel per loss term",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


# --- 2. what the fix changed ----------------------------------------------


def plot_align_fix(series, out_png):
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    # The teacher is drawn LAST and dashed, so it renders on top of the
    # student it coincides with and the student shows through the gaps.
    wanted = [
        (("align_teacher_a09", "align_teacher", "const_0.9", "student_h",
          "adjacent"), STUDENT, "student $h_t$, the network being trained",
         "-", dict(MARK, zorder=4)),
        (("align_teacher_a09", "align_teacher", "const_0.9", "teacher_h",
          "adjacent"), TEACHER, "EMA teacher $h_t$, the target of $L_{align}$",
         (0, (4, 3)), dict(lw=1.6, zorder=5)),
    ]
    handles = {}
    for key, color, label, ls, kw in wanted:
        pts = series.get(key, [])
        if not pts:
            print(f"  (missing {key})")
            continue
        handles[label], = ax.plot([p[0] for p in pts], [p[1] for p in pts],
                                  color=color, linestyle=ls, label=label,
                                  **kw)
    ax.axhline(0.0, color="#b0b0b0", linestyle=":", linewidth=0.9)
    style_axes(ax, DRIFT_YLABEL)
    order = [w[2] for w in wanted]
    ax.legend([handles[k] for k in order if k in handles],
              [k for k in order if k in handles], frameon=False, fontsize=9)
    ax.set_title("L_align, teacher as target, alpha = 0.9", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


# --- 3. cumulative drift ---------------------------------------------------


def plot_cumulative(series, out_png):
    fig, axs = plt.subplots(1, 3, figsize=(12, 5.0), sharey=True)
    kw = dict(lw=1.5, marker="o", ms=3.0,
              markeredgecolor="white", markeredgewidth=0.5)
    present = set()
    for ax, arm in zip(axs, TEACHER_ARMS):
        drawn = curves(series, arm, "vs_initial")
        for z, (key, color, _, ls) in enumerate(NAMED):
            for alpha, latent, pts in drawn:
                if (alpha, latent) != key:
                    continue
                present.add(key)
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=color, linestyle=ls, zorder=2 + z, **kw)
        ax.axhline(1.0, color="#b0b0b0", linestyle="--", linewidth=0.9)
        ax.text(0.98, 0.99, "orthogonal (drift = 1)", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#707070")
        ax.set_title(ARM_TITLE[arm], fontsize=11)
        ax.set_ylim(0.0, 1.15)
        style_axes(ax, DRIFT_YLABEL + "\nagainst the 5k checkpoint"
                   if arm == TEACHER_ARMS[0] else None)
        ax.yaxis.label.set_fontsize(9)
    fig.legend(handles=named_handles(present),
               loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Cumulative drift", fontsize=13)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


# --- 4. the 500-step probe -------------------------------------------------


def plot_drift_500(path, out_png):
    series = read_drift(path)
    fig, axs = plt.subplots(1, 3, figsize=(13, 5.0), sharey=True)
    for ax, arm in zip(axs, TEACHER_ARMS):
        drawn = curves(series, arm, "adjacent")
        # The constant-alpha halves of pred_moco and rep_moco come from the
        # loss-term isolation runs, which have no 500-step probe. So each
        # panel legends only the curves that panel actually draws.
        present = set()
        for z, (key, color, _, ls) in enumerate(NAMED):
            for alpha, latent, pts in drawn:
                if (alpha, latent) != key:
                    continue
                present.add(key)
                # 199 points per series: markers would smear, so no markers.
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=color, linestyle=ls, lw=1.5, zorder=2 + z)
        ax.legend(handles=named_handles(present), frameon=False,
                  fontsize=8.5, loc="best")
        ax.set_title(ARM_TITLE[arm], fontsize=11)
        # Linear, the first two probes set the whole range and flatten
        # everything past 1000 steps. Log spans the post-1000 behaviour.
        ax.set_yscale("log")
        # 1593 of the 1600 points sit above 3e-4; a handful of end-of-ramp
        # points reach 1e-5 and would stretch the shared axis over six
        # decades of mostly empty space. Clip the floor to where the data is.
        ax.set_ylim(3e-4, 1.0)
        log_yticks(ax, [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
        style_axes(ax, DRIFT_YLABEL + "\n500-step pairs"
                   if arm == TEACHER_ARMS[0] else None)
        ax.yaxis.label.set_fontsize(9)
    fig.suptitle("Drift over 500-step windows", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


# --- 5. dim usage: is a flat drift curve stability or collapse? -----------


# One named series = one colour, used once in the whole figure, and equal
# line weight, so no curve is a pale backdrop for another. Solid = constant
# α, dashed = the ramp; the dashed curve is drawn last with long gaps, so
# the solid one shows through where the two coincide.
DIM_PANELS = [
    ("Runs trained with L_align",
     [("align", "#7570b3", "-", "align, target = student h_t+1"),
      ("align_teacher_a09", "#0072B2", "-",
       "align_teacher, alpha = 0.9 constant"),
      ("align_teacher_sched", "#0072B2", "--",
       "align_teacher, alpha: 0.9 -> 1.0")]),
    ("Runs trained with MoCo negatives",
     [("pred_moco", "#0072B2", "-", "pred_moco, alpha = 0.9 constant"),
      ("pred_moco_sched", "#0072B2", "--", "pred_moco, alpha: 0.9 -> 1.0"),
      ("rep_moco", "#D55E00", "-", "rep_moco, alpha = 0.9 constant"),
      ("rep_moco_sched", "#D55E00", "--", "rep_moco, alpha: 0.9 -> 1.0")]),
]

# Labelled ticks for the dim-usage axis. The 1/64 floor gets its own label
# so the "both align arms end near the floor" claim is checkable.
DIM_TICKS = [1.0 / 64, 0.02, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6]
DIM_TICK_LABELS = ["1/64 = 0.0156 (collinear)", "0.02", "0.03", "0.05",
                   "0.1", "0.2", "0.4", "0.6"]


def plot_dim_usage(loss_csv, out_png, n_dims=64):
    series = defaultdict(list)
    with open(loss_csv) as fh:
        for r in csv.DictReader(fh):
            series[r["run"]].append((int(r["step"]),
                                     float(r["u_temporal"])))
    for k in series:
        series[k].sort()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.1), sharey=True)
    for ax, (title, entries) in zip(axs, DIM_PANELS):
        for run, color, ls, label in entries:
            pts = series.get(run, [])
            if not pts:
                print(f"  (missing {run} in {os.path.basename(loss_csv)})")
                continue
            # Equal weight for both, and the dashed scheduled curve is
            # long-gapped, so where two curves coincide each stays visible.
            # Same colour for the two runs of one arm, so the ramp is told
            # apart by its dashes alone. Equal line weight: the ramp is drawn
            # last, and its gaps expose the background, so where the two runs
            # coincide the line reads as dashed instead of hiding one of them.
            # The two runs of one arm share a colour and can sit within a
            # line width of each other, so the ramp carries sparse markers:
            # its legend row then maps to something visible even there.
            kw = (dict(lw=2.0) if ls == "-" else
                  dict(lw=1.6, dashes=(4, 4), marker="o", ms=4.5,
                       markevery=110, markeredgecolor="white",
                       markeredgewidth=0.6))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
                    linestyle=ls, label=label, **kw)
        # The floor line is identified by its own y tick, so no text label.
        ax.axhline(1.0 / n_dims, color="#b0b0b0", linestyle=":", linewidth=1.0)
        ax.set_title(title, fontsize=11)
        # Shared log axis, but the data spans 0.0156 to 0.617: the default
        # decade locator labels 10^-1 alone over the whole height and
        # nothing can be read off. Label the ticks, floor included.
        ax.set_yscale("log")
        ax.set_ylim(0.0145, 0.85)
        log_yticks(ax, DIM_TICKS, DIM_TICK_LABELS)
        style_axes(ax, "dimension usage $U$ of $h_t$, across time"
                   if title == DIM_PANELS[0][0] else None)
        ax.legend(frameon=False, fontsize=8.5, loc="best")
    fig.suptitle(
        r"$U=1/(d\cdot\overline{\cos^2})$: "
        "1 = all 64 dimensions used, 1/64 = one direction", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    args = ap.parse_args()
    res = os.path.join(args.exp_dir, "results")
    plots = os.path.join(args.exp_dir, "plots")
    os.makedirs(plots, exist_ok=True)

    series = read_drift(os.path.join(res, "drift.csv"))
    plot_headline(series, os.path.join(plots, "drift_headline.png"))
    plot_align_fix(series, os.path.join(plots, "align_fix.png"))
    plot_cumulative(series, os.path.join(plots, "cumulative_drift.png"))
    plot_drift_500(os.path.join(res, "drift_500.csv"),
                   os.path.join(plots, "drift_500.png"))
    support = os.path.join(plots, "supporting")
    os.makedirs(support, exist_ok=True)
    plot_dim_usage(os.path.join(res, "loss_curve.csv"),
                   os.path.join(support, "dim_usage.png"))


if __name__ == "__main__":
    main()
