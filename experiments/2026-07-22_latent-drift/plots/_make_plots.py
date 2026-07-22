#!/usr/bin/env python3
"""Render 4 latent-drift plots for the #374 arm sweep.

Each PNG is a 2×3 grid, one panel per arm. Each panel shows the
"adjacent-pair" trajectory for one drift metric across training steps.

Input: ../results/drift_374.csv
Output: drift_total.png, rot_gap.png, drift_informative.png, cka.png

Colors: validated dataviz slots 1–6 (blue/orange/aqua/yellow/magenta/green),
one hue per arm, same assignment across all four figures so the reader
carries "arm 1 = blue" between plots.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "..", "results", "drift_374.csv")

# Order matters — colour is assigned by position, so this locks the
# arm ↔ hue mapping across all four figures.
ARM_ORDER = ["arm1", "arm3", "arm4", "arm5", "arm6v2", "bimoco"]
ARM_LABEL = {
    "arm1":   "arm 1  L_pred + L_rep",
    "arm3":   "arm 3  L_pred(MoCo) + L_rep",
    "arm4":   "arm 4  pooled + MoCo",
    "arm5":   "arm 5  L_align + L_rep",
    "arm6v2": "arm 6 v2  L_align + L_rep(MoCo)",
    "bimoco": "bimoco  L_pred(MoCo) + L_rep(MoCo)",
}
# Validated palette slots 1–6 (light mode).
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
     "Uninformative rotation",
     r"$\mathrm{drift\_cos} - \mathrm{drift\_cos\_aligned}$"),
    ("drift_cos_aligned", "drift_informative.png",
     "Informative drift (Procrustes-aligned)",
     r"$1 - (1/N)\sum_k \sigma_k(A^\top B)$"),
    ("cka",               "cka.png",
     "Linear CKA",
     r"$\mathrm{HSIC}(A,B) / \sqrt{\mathrm{HSIC}(A,A)\,\mathrm{HSIC}(B,B)}$"),
]


def _load():
    """{arm: [(step_b, {metric: value}), …]} sorted by step, adjacent only."""
    rows = list(csv.DictReader(open(CSV_PATH)))
    out = defaultdict(list)
    for r in rows:
        if r["kind"] != "adjacent":
            continue
        step_b = int(r["step_b"])
        vals = {m[0]: float(r[m[0]]) for m in METRICS}
        out[r["arm"]].append((step_b, vals))
    for arm in out:
        out[arm].sort(key=lambda kv: kv[0])
    return out


def _make(metric_key, out_name, title, subtitle, data):
    fig, axes = plt.subplots(
        2, 3, figsize=(11, 6.6), sharex=True,
        constrained_layout=True)
    # Global y-limits so all six panels use one axis for direct comparison.
    all_ys = [v[metric_key] for arm in ARM_ORDER for _, v in data.get(arm, [])]
    if not all_ys:
        return
    ymin, ymax = min(all_ys), max(all_ys)
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ylim = (ymin - pad, ymax + pad)
    for ax, arm in zip(axes.ravel(), ARM_ORDER):
        pts = data.get(arm, [])
        colour = ARM_COLOR[arm]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1][metric_key] for p in pts]
            ax.plot(xs, ys, color=colour, linewidth=2.0,
                    marker="o", markersize=5,
                    markerfacecolor=colour, markeredgecolor="white",
                    markeredgewidth=1.2)
        ax.set_title(ARM_LABEL[arm], fontsize=9.5, loc="left",
                     color="#0b0b0b", pad=4)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        ax.set_xlim(0, 51_000)
        ax.set_xticks([0, 12_500, 25_000, 50_000])
        ax.set_xticklabels(["0", "12.5k", "25k", "50k"])
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=8, colors="#52514e")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")
    for ax in axes[-1, :]:
        ax.set_xlabel("training step (end of adjacent interval)",
                      fontsize=9, color="#52514e")
    # Compact 2-line header sitting ABOVE the axes so it can't collide
    # with any panel title.
    fig.suptitle(f"{title}\n{subtitle}",
                 fontsize=12, x=0.005, y=1.0,
                 ha="left", va="bottom", color="#0b0b0b")
    out_path = os.path.join(HERE, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#fcfcfb")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
    })
    data = _load()
    for metric_key, out_name, title, subtitle in METRICS:
        _make(metric_key, out_name, title, subtitle, data)


if __name__ == "__main__":
    main()
