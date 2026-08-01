#!/usr/bin/env python3
"""Render 5 plots for the #374 latent-drift experiment.

Four drift plots: one per metric (drift_cos, rot_gap, drift_cos_aligned,
cka), 6 arms overlaid on a single axis — the values live on the same
scale so the comparison is direct.

One reference plot: GM-Relative MASE per arm at the eval steps recorded
by #374 (`results/gm_mase_374.csv`, 6L quantile head), on the same
x-axis as the drift plots — so a vertical line at any step aligns
across figures.

Input: ../results/drift_374.csv, ../results/gm_mase_374.csv.
Output: drift_total.png, rot_gap.png, drift_informative.png, cka.png,
        gm_mase_374.png.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DRIFT_CSV = os.path.join(HERE, "..", "results", "drift_374.csv")
GM_CSV = os.path.join(HERE, "..", "results", "gm_mase_374.csv")

ARM_ORDER = ["arm1", "arm3", "arm4", "arm5", "arm6v2", "bimoco"]
ARM_LABEL = {
    "arm1":   "arm 1  L_pred + L_rep",
    "arm3":   "arm 3  L_pred(MoCo) + L_rep",
    "arm4":   "arm 4  pooled + MoCo",
    "arm5":   "arm 5  L_align + L_rep",
    "arm6v2": "arm 6 v2  L_align + L_rep(MoCo)",
    "bimoco": "bimoco  L_pred(MoCo) + L_rep(MoCo)",
}
ARM_COLOR = {
    "arm1":   "#2a78d6",  # blue
    "arm3":   "#eb6834",  # orange
    "arm4":   "#1baf7a",  # aqua
    "arm5":   "#eda100",  # yellow
    "arm6v2": "#e87ba4",  # magenta
    "bimoco": "#008300",  # green
}

METRICS = [
    ("drift_cos",         "drift_total.png",
     "Total per-token drift",
     r"$1 - \langle\cos(h_t^A, h_t^B)\rangle$"),
    ("rot_gap",           "rot_gap.png",
     "Rotational drift",
     r"$\mathrm{drift\_cos} - \mathrm{drift\_cos\_aligned}$  "
     r"(the part removed by the best global feature-axis rotation)"),
    ("drift_cos_aligned", "drift_residual.png",
     "Residual drift (Procrustes-aligned)",
     r"$1 - (1/N)\sum_k \sigma_k(A^\top B)$  "
     r"(the part a linear head cannot absorb)"),
    ("cka",               "cka.png",
     "Linear CKA",
     r"$\mathrm{HSIC}(A,B) / \sqrt{\mathrm{HSIC}(A,A)\,\mathrm{HSIC}(B,B)}$"),
]

XLIM = (0, 51_000)
XTICKS = [0, 12_500, 25_000, 50_000]
XTICK_LABELS = ["0", "12.5k", "25k", "50k"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SPINE_MUTED = "#c3c2b7"


def _load_drift():
    """{arm: [(step_b, {metric: value}), …]} adjacent only, sorted by step."""
    out = defaultdict(list)
    for r in csv.DictReader(open(DRIFT_CSV)):
        if r["kind"] != "adjacent":
            continue
        step_b = int(r["step_b"])
        vals = {m[0]: float(r[m[0]]) for m in METRICS}
        out[r["arm"]].append((step_b, vals))
    for arm in out:
        out[arm].sort(key=lambda kv: kv[0])
    return out


def _load_gm():
    """{head_layers: {arm: [(step, gm_rel_mase), …]}} sorted by step."""
    out = {"2L": defaultdict(list), "6L": defaultdict(list)}
    for r in csv.DictReader(open(GM_CSV)):
        hl = r["head_layers"]
        if hl not in out:
            continue
        out[hl][r["arm"]].append(
            (int(r["step"]), float(r["gm_rel_mase"])))
    for hl in out:
        for arm in out[hl]:
            out[hl][arm].sort(key=lambda kv: kv[0])
    return out


def _style_axis(ax, ylabel, xlabel="training step"):
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.set_xlim(*XLIM)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XTICK_LABELS)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=10, color=TEXT_SECONDARY)
    ax.tick_params(labelsize=9, colors=TEXT_SECONDARY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE_MUTED)


def _plot_drift(metric_key, out_name, title, subtitle, data):
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    for arm in ARM_ORDER:
        pts = data.get(arm, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1][metric_key] for p in pts]
        c = ARM_COLOR[arm]
        ax.plot(xs, ys, color=c, linewidth=2.0, marker="o", markersize=5.5,
                markerfacecolor=c, markeredgecolor="white", markeredgewidth=1.2,
                label=ARM_LABEL[arm])
    _style_axis(ax, ylabel=metric_key,
                xlabel="training step (end of adjacent interval)")
    ax.legend(loc="best", fontsize=8.5, frameon=False,
              labelcolor=TEXT_PRIMARY)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12, x=0.005, y=1.0,
                 ha="left", va="bottom", color=TEXT_PRIMARY)
    out = os.path.join(HERE, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def _plot_gm(gm_data):
    """Side-by-side 2L | 6L quantile-head subplots. Same 6-arm colour
    map as the drift plots, shared y-axis for direct comparison
    across heads."""
    all_ys = [y for hl in ("2L", "6L") for arm in ARM_ORDER
              for _, y in gm_data[hl].get(arm, [])]
    if not all_ys:
        return
    ymin, ymax = min(all_ys), max(all_ys)
    pad = 0.05 * (ymax - ymin)
    ylim = (min(1.0, ymin - pad), ymax + pad)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True,
                             constrained_layout=True)
    for i, hl in enumerate(("2L", "6L")):
        ax = axes[i]
        for arm in ARM_ORDER:
            pts = gm_data[hl].get(arm, [])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            c = ARM_COLOR[arm]
            ax.plot(xs, ys, color=c, linewidth=2.0, marker="o",
                    markersize=6.5, markerfacecolor=c,
                    markeredgecolor="white", markeredgewidth=1.2,
                    linestyle="-" if len(pts) > 1 else "None",
                    label=ARM_LABEL[arm] if i == 0 else None)
        ax.axhline(1.0, color=SPINE_MUTED, linewidth=1.0, linestyle="--")
        _style_axis(ax, ylabel=("GM-Relative MASE (full-97)"
                                if i == 0 else ""))
        ax.set_ylim(*ylim)
        ax.set_title(f"{hl} quantile head", fontsize=10,
                     loc="left", color=TEXT_PRIMARY)
    axes[0].legend(loc="upper right", fontsize=8.5, frameon=False,
                   labelcolor=TEXT_PRIMARY)
    fig.suptitle(
        "GM-Relative MASE (reference, from #374)\n"
        r"eval points on the same x-axis as the drift plots above",
        fontsize=12, x=0.005, y=1.0,
        ha="left", va="bottom", color=TEXT_PRIMARY)
    out = os.path.join(HERE, "gm_mase_374.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": SPINE_MUTED,
        "axes.labelcolor": TEXT_SECONDARY,
    })
    drift_data = _load_drift()
    for metric_key, out_name, title, subtitle in METRICS:
        _plot_drift(metric_key, out_name, title, subtitle, drift_data)
    _plot_gm(_load_gm())


if __name__ == "__main__":
    main()
