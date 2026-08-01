#!/usr/bin/env python3
"""Figures for #388: teacher-target L_align and the 0.9 -> 1.0 α schedule.

Reads results/drift.csv (post-hoc, 5000-step cadence, from
teacher_latent_drift.py), results/drift_500.csv and
results/alpha_schedule.csv (in-training, from make_results_csvs.py).
Writes five PNGs into plots/.

    drift_headline.png    3x3 arm panel, adjacent 5k-pair drift
    align_fix.png         what swapping L_align's target changed
    alpha_schedule.png    α against training step
    cumulative_drift.png  drift vs the first probe, teacher arms
    drift_500.png         adjacent 500-step drift, the four new runs

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
RUN_COLOR = {"align_teacher_a09": "#7570b3",
             "align_teacher_sched": "#1b9e77",
             "pred_moco_sched": "#d95f02",
             "rep_moco_sched": "#e7298a"}


def read_drift(path):
    """{(run, arm, alpha, latent, kind): [(step, drift_cos), ...]}."""
    series = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            series[(r["run"], r["arm"], r["alpha"], r["latent"],
                    r["kind"])].append((int(r["step"]),
                                        float(r["drift_cos"])))
    for key in series:
        series[key].sort()
    return series


def curves(series, arm, kind):
    """Every (alpha, latent, points) of one arm, constant-α first."""
    out = [(k[2], k[3], v) for k, v in series.items()
           if k[1] == arm and k[4] == kind]
    return sorted(out, key=lambda t: (t[0], t[1]))


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
    h = [Line2D([], [], color=COLOR[l], lw=1.8, label=LATENT_LABEL[l])
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
                    color=COLOR[latent], linestyle=STYLE[alpha], **MARK)
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
    wanted = [
        (("align", "align", "none", "student_h", "adjacent"),
         "#7570b3", "-", "#382 `align`: target = student sg($h_{t+1}$)"),
        (("align_teacher_a09", "align_teacher", "const_0.9", "student_h",
          "adjacent"), STUDENT, "-",
         "`align_teacher`: student $h_t$"),
        (("align_teacher_a09", "align_teacher", "const_0.9", "teacher_h",
          "adjacent"), TEACHER, "-",
         "`align_teacher`: EMA-teacher $h_t$"),
    ]
    for key, color, ls, label in wanted:
        pts = series.get(key, [])
        if not pts:
            print(f"  (missing {key})")
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
                linestyle=ls, label=label, **MARK)
    ax.axhline(0.0, color="#b0b0b0", linestyle=":", linewidth=0.9)
    style_axes(ax, "drift_cos ($h_t$)")
    ax.legend(frameon=False, fontsize=9)
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
    for run, pts in sorted(series.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=RUN_COLOR.get(run, "#555555"), lw=1.8, label=run)
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
                    color=COLOR[latent], linestyle=STYLE[alpha], **MARK)
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
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=COLOR[latent], linestyle=STYLE[alpha], lw=1.1,
                    alpha=0.85)
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


if __name__ == "__main__":
    main()
