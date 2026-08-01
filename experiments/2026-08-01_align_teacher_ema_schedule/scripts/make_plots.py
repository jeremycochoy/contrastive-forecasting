#!/usr/bin/env python3
"""Figures for #388: teacher-target L_align and the 0.9 -> 1.0 α schedule.

Reads results/drift.csv (post-hoc, 5000-step cadence, from
teacher_latent_drift.py), results/drift_500.csv and
results/alpha_schedule.csv (in-training, from make_results_csvs.py).
Writes the report's five figures into plots/:

    drift_headline.png    3x3 arm panel, adjacent 5k-pair drift
    align_fix.png         what swapping L_align's target changed
    alpha_schedule.png    α against training step
    cumulative_drift.png  drift vs the first probe, teacher arms
    drift_500.png         adjacent 500-step drift, the four new runs

and one supporting figure into plots/supporting/, which the issue does not
ask for and the report does not carry:

    dim_usage.png         how much of h_t stays alive. A flat drift curve
                          is either a stable representation or a dead one,
                          and this separates the two. Needed to read the
                          L_align panels: both align arms sit within ~10%
                          of the collinear floor on the time axis.

Encoding used everywhere: colour = which encoder (student / EMA
teacher), line style = the α schedule (solid = constant, dashed =
0.9 -> 1.0).
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

ARMS = ["pred", "rep", "align", "align_teacher", "pred_moco",
        "rep_moco", "sigreg_e", "sigreg_h", "cpc"]
TEACHER_ARMS = ["align_teacher", "pred_moco", "rep_moco"]

STUDENT, TEACHER = "#1f77b4", "#d95f02"        # CVD-validated pair
COLOR = {"student_h": STUDENT, "teacher_h": TEACHER}
LATENT_LABEL = {"student_h": "student $h_t$", "teacher_h": "EMA-teacher $h_t$"}
STYLE = {"none": "-", "const_0.9": "-", "sched_0.9_1.0": "--"}
ALPHA_LABEL = {"const_0.9": r"$\alpha=0.9$ constant",
               "sched_0.9_1.0": r"$\alpha:0.9\rightarrow1.0$"}
MARK = dict(lw=1.6, marker="o", ms=3.2, markeredgecolor="white",
            markeredgewidth=0.5)

# The teacher tracks the student closely enough that on most arms the two
# curves coincide to within a line width. Drawn identically, whichever goes
# on top erases the other and the figure looks like it is missing a curve.
# So the teacher is a wide soft band and the student a thin line on top of
# it: coincidence reads as a blue line inside an orange band, divergence
# reads as two curves.
LATENT_KW = {
    "student_h": dict(lw=1.5, marker="o", ms=3.0, markeredgecolor="white",
                      markeredgewidth=0.5, zorder=3),
    "teacher_h": dict(lw=3.6, alpha=0.45, solid_capstyle="round", zorder=2),
}
RUN_COLOR = {"align_teacher_a09": "#7570b3",
             "align_teacher_sched": "#1b9e77",
             "pred_moco_sched": "#d95f02",
             "rep_moco_sched": "#e7298a"}


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
    """Every (alpha, latent, points) of one arm, constant-α first.

    A resumed run contributes one `vs_initial` entry per reference step, so
    its segments are drawn as separate lines instead of one false curve.
    """
    out = [(k[2], k[3], v) for k, v in series.items()
           if k[1] == arm and k[4] == kind]
    # Teacher first so the student's thin line lands on top of its band.
    return sorted(out, key=lambda t: (t[0], t[1] != "teacher_h"))


def style_axes(ax, ylabel=None, xlabel="training step", logx=True):
    if logx:
        ax.set_xscale("log")
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def legend_handles(alphas):
    h = [Line2D([], [], color=COLOR[l], label=LATENT_LABEL[l],
                lw=LATENT_KW[l]["lw"], alpha=LATENT_KW[l].get("alpha", 1.0))
         for l in ("student_h", "teacher_h")]
    h += [Line2D([], [], color="#555555", lw=1.8, linestyle=STYLE[a],
                 label=ALPHA_LABEL[a]) for a in alphas]
    return h


# --- 1. headline -----------------------------------------------------------


def plot_headline(series, out_png):
    fig, axs = plt.subplots(3, 3, figsize=(11.5, 8.5), sharey=True,
                            sharex=True)
    axs = axs.ravel()
    for ax, arm in zip(axs, ARMS):
        ax.axhline(0.0, color="#b0b0b0", linestyle=":", linewidth=0.9)
        for alpha, latent, pts in curves(series, arm, "adjacent"):
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=COLOR[latent], linestyle=STYLE[alpha],
                    **LATENT_KW[latent])
        if arm in TEACHER_ARMS:
            ax.set_facecolor("#fdf6f0")
        ax.set_title(arm, fontsize=11)
        ax.set_ylim(-0.05, 1.40)
        style_axes(ax, "drift_cos ($h_t$)")
    for ax in axs[:6]:
        ax.set_xlabel("")
    for i, ax in enumerate(axs):
        if i % 3:
            ax.set_ylabel("")
    fig.legend(handles=legend_handles(["const_0.9", "sched_0.9_1.0"]),
               loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Drift of $h_t$ between checkpoints 5000 steps apart "
                 "— shaded panels have an EMA teacher", fontsize=13)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


# --- 2. what the fix changed ----------------------------------------------


def plot_align_fix(series, out_png):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    # Drawn teacher-first so the student's line sits on top of its band;
    # the legend is re-sorted afterwards to read student before teacher.
    wanted = [
        (("align", "align", "none", "student_h", "adjacent"),
         "#7570b3", "#382 `align`: target = student sg($h_{t+1}$)",
         dict(MARK, zorder=3)),
        (("align_teacher_a09", "align_teacher", "const_0.9", "teacher_h",
          "adjacent"), TEACHER, "`align_teacher`: EMA-teacher $h_t$",
         LATENT_KW["teacher_h"]),
        (("align_teacher_a09", "align_teacher", "const_0.9", "student_h",
          "adjacent"), STUDENT, "`align_teacher`: student $h_t$",
         LATENT_KW["student_h"]),
    ]
    handles = {}
    for key, color, label, kw in wanted:
        pts = series.get(key, [])
        if not pts:
            print(f"  (missing {key})")
            continue
        handles[label], = ax.plot([p[0] for p in pts], [p[1] for p in pts],
                                  color=color, linestyle="-", label=label,
                                  **kw)
    ax.axhline(0.0, color="#b0b0b0", linestyle=":", linewidth=0.9)
    style_axes(ax, "drift_cos ($h_t$)")
    order = [w[2] for w in (wanted[0], wanted[2], wanted[1])]
    ax.legend([handles[k] for k in order if k in handles],
              [k for k in order if k in handles], frameon=False, fontsize=9)
    ax.set_title("L_align with the teacher as target, α = 0.9 constant",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


# --- 3. the schedule -------------------------------------------------------


def plot_alpha(alpha_csv, out_png):
    series = defaultdict(list)
    with open(alpha_csv) as fh:
        for r in csv.DictReader(fh):
            series[r["run"]].append((int(r["step"]), float(r["ema_tau"])))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    # The three scheduled runs share one schedule, so their lines coincide
    # exactly. Nested widths keep all four legend entries visible on the
    # figure instead of leaving whichever drew last as the only one.
    items = sorted(series.items())
    for i, (run, pts) in enumerate(items):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=RUN_COLOR.get(run, "#555555"),
                lw=4.0 - 0.9 * i, label=run, solid_capstyle="butt")
    ax.axhline(0.9, color="#b0b0b0", linestyle="--", linewidth=1.0)
    ax.text(0.01, 0.9, "#382 runs: α = 0.9 for the whole run",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=8, color="#707070")
    style_axes(ax, "EMA momentum α", logx=False)
    ax.set_ylim(0.88, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("EMA momentum against training step", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


# --- 4. cumulative drift ---------------------------------------------------


def plot_cumulative(series, out_png):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.1), sharey=True)
    for ax, arm in zip(axs, TEACHER_ARMS):
        for alpha, latent, pts in curves(series, arm, "vs_initial"):
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=COLOR[latent], linestyle=STYLE[alpha],
                    **LATENT_KW[latent])
        ax.axhline(1.0, color="#b0b0b0", linestyle="--", linewidth=0.9)
        ax.text(0.98, 0.99, "orthogonal (drift = 1)", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#707070")
        ax.set_title(arm, fontsize=11)
        ax.set_ylim(0.0, 1.15)
        style_axes(ax, "drift_cos vs the 5k checkpoint"
                   if arm == TEACHER_ARMS[0] else None)
    fig.legend(handles=legend_handles(["const_0.9", "sched_0.9_1.0"]),
               loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Cumulative drift of $h_t$ away from the 5k checkpoint",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


# --- 5. the 500-step probe -------------------------------------------------


def plot_drift_500(path, out_png):
    series = read_drift(path)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.1), sharey=True)
    for ax, arm in zip(axs, TEACHER_ARMS):
        for alpha, latent, pts in curves(series, arm, "adjacent"):
            # 199 points per series: markers would smear, so this panel
            # keeps the band/line pairing but drops them.
            wide = latent == "teacher_h"
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=COLOR[latent], linestyle=STYLE[alpha],
                    lw=2.8 if wide else 1.0, alpha=0.35 if wide else 0.95,
                    zorder=2 if wide else 3)
        ax.axhline(0.0, color="#b0b0b0", linestyle=":", linewidth=0.9)
        ax.set_title(arm, fontsize=11)
        style_axes(ax, "drift_cos ($h_t$), 500-step pairs"
                   if arm == TEACHER_ARMS[0] else None)
    fig.legend(handles=legend_handles(["const_0.9", "sched_0.9_1.0"]),
               loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Drift of $h_t$ between probes 500 steps apart "
                 "(the four #388 runs only)", fontsize=13)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


# --- 6. dim usage: is a flat drift curve stability or collapse? -----------


DIM_PANELS = [
    ("L_align", [("align", "#7570b3", "-", "#382 `align` (student target)"),
                 ("align_teacher_a09", STUDENT, "-",
                  r"`align_teacher`, $\alpha=0.9$"),
                 ("align_teacher_sched", STUDENT, "--",
                  r"`align_teacher`, $\alpha:0.9\rightarrow1.0$")]),
    ("MoCo arms", [("pred_moco", "#1b9e77", "-", r"#382 `pred_moco`"),
                   ("pred_moco_sched", "#1b9e77", "--",
                    r"`pred_moco`, $\alpha:0.9\rightarrow1.0$"),
                   ("rep_moco", "#e7298a", "-", r"#382 `rep_moco`"),
                   ("rep_moco_sched", "#e7298a", "--",
                    r"`rep_moco`, $\alpha:0.9\rightarrow1.0$")]),
]


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
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
                    linestyle=ls, lw=1.5, label=label)
        ax.axhline(1.0 / n_dims, color="#b0b0b0", linestyle=":", linewidth=1.0)
        ax.text(0.01, 1.0 / n_dims, f"collinear $h_t$ (1/{n_dims})",
                transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=8, color="#707070")
        ax.set_title(title, fontsize=11)
        ax.set_yscale("log")
        style_axes(ax, r"dim usage $U$ of $h_t$ across time"
                   if title == DIM_PANELS[0][0] else None)
        ax.legend(frameon=False, fontsize=8.5, loc="best")
    fig.suptitle(r"Dimension usage $U=1/(d\cdot\overline{\cos^2})$ of $h_t$: "
                 "1 is isotropic, 1/64 is collapsed", fontsize=13)
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
    plot_alpha(os.path.join(res, "alpha_schedule.csv"),
               os.path.join(plots, "alpha_schedule.png"))
    plot_cumulative(series, os.path.join(plots, "cumulative_drift.png"))
    plot_drift_500(os.path.join(res, "drift_500.csv"),
                   os.path.join(plots, "drift_500.png"))
    support = os.path.join(plots, "supporting")
    os.makedirs(support, exist_ok=True)
    plot_dim_usage(os.path.join(res, "loss_curve.csv"),
                   os.path.join(support, "dim_usage.png"))


if __name__ == "__main__":
    main()
