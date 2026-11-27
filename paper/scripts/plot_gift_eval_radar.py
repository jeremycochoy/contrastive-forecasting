"""Radar of the relative MASE per GIFT-Eval dataset, for the Results section.

Reads the per-config tables in paper/data/gift_eval/ (97 configs each) and
writes paper/figures/gift_eval_radar.png. The value of a dataset is the
geometric mean, over its configs, of the MASE of the model divided by the
MASE of seasonal naive. The legend gives the GM-MASE over the 97 configs.

Usage: python3 paper/scripts/plot_gift_eval_radar.py
"""
from collections import defaultdict
from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parent.parent
DATA = PAPER / "data" / "gift_eval"
OUT = PAPER / "figures" / "gift_eval_radar.png"
MASE = "eval_metrics/MASE[0.5]"
TICKS = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

# (table, legend label, colour, line style, line width)
MODELS = [
    ("value_space_w64_166k", "Moirai-style, width 64 (2.4M)", "#E69F00", "-", 1.3),
    ("green_w64_cd135_166k", "Ours, width 64 (0.73M parameters)", "#56B4E9", "-", 1.3),
    ("value_space_w384_166k", "Moirai-style, width 384 (14.4M)", "#D55E00", "-", 1.7),
    ("green_w384_cd125_166k", "Ours, width 384 (15.0M)", "#0072B2", "-", 1.7),
]


def read_mase(name):
    """{config: MASE of the median forecast} of one table."""
    with open(DATA / f"{name}.csv") as handle:
        return {row["dataset"]: float(row[MASE]) for row in csv.DictReader(handle)}


def relative_logs(name, naive):
    """{config: log of MASE / seasonal-naive MASE}."""
    return {cfg: math.log(v / naive[cfg]) for cfg, v in read_mase(name).items()}


def geometric_mean(logs):
    return math.exp(sum(logs) / len(logs))


def per_dataset(logs):
    """{dataset: geometric mean over its configs}. The dataset is the part
    of the config name before the first '/'."""
    groups = defaultdict(list)
    for cfg, value in logs.items():
        groups[cfg.split("/")[0]].append(value)
    return {name: geometric_mean(values) for name, values in groups.items()}


def difficulty_order(per_model):
    """Datasets by their mean log value over the models: the hardest at the
    top, the next ones alternate right and left, the easiest at the bottom."""
    names = sorted(set().union(*per_model))
    mean_log = {d: sum(math.log(m[d]) for m in per_model) / len(per_model) for d in names}
    ranked = sorted(names, key=mean_log.get, reverse=True)
    return ranked[:1] + ranked[1::2] + ranked[2::2][::-1]


def style_axes(ax, labels, angles):
    """Dataset names round the rim, clockwise from the top. Log radius."""
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    for text, angle in zip(ax.get_xticklabels(), angles):
        side = math.sin(angle)
        text.set_horizontalalignment("center" if abs(side) < 0.05 else "left" if side > 0 else "right")
    ax.tick_params(axis="x", pad=10)
    ax.set_rscale("log")
    ax.minorticks_off()
    ax.set_yticks(TICKS)
    ax.set_yticklabels([f"{t:g}" for t in TICKS], fontsize=7.5, color="dimgrey")
    ax.set_ylim(0.55, 5.4)
    ax.set_rlabel_position(180 + 180 / len(labels))  # between the two bottom spokes


def main():
    naive = read_mase("seasonal_naive")
    logs = {name: relative_logs(name, naive) for name, *_ in MODELS}
    per_model = [per_dataset(logs[name]) for name, *_ in MODELS]
    labels = difficulty_order(per_model)
    angles = [n / len(labels) * 2 * math.pi for n in range(len(labels))] + [0.0]
    fig, ax = plt.subplots(figsize=(6.2, 6.7), subplot_kw=dict(polar=True))
    for (name, label, colour, style, width), values in zip(MODELS, per_model):
        radii = [values[d] for d in labels]
        score = geometric_mean(list(logs[name].values()))
        ax.plot(angles, radii + radii[:1], style, color=colour, lw=width, marker="o",
                ms=3.5, label=f"{label}: GM-MASE {score:.4f}",
                zorder=4 if label.startswith("Ours") else 3)  # ours above every Moirai-style line
    ax.plot(angles, [1.0] * len(angles), ":", color="#2ca02c", lw=1.6,
            label="Seasonal naive: 1", zorder=2)
    style_axes(ax, labels, angles)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=9, frameon=False)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
