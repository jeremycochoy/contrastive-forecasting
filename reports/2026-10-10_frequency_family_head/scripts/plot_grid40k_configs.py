"""Radars of the relative MASE per GIFT-Eval dataset for the runs of the 40k grid
(the user, 11-03: the datasets that stay above 1 keep the GM-Relative MASE above
1), in the form of the radars of #419 and #421 (radar_lib.py of
2026-09-06_moirai_small_size): a dataset's value is the geometric mean, over its
configs, of the MASE relative to Seasonal Naive (forecast at 40k, head B4); log
radius; the green ring is 1 (Seasonal Naive), inside is better; the hardest
datasets sit at the top.

Two radars, each with MPE (our copy of Moirai: patch heads, EWMA, new data,
batch 256, at its best checkpoint 166k) as a reference: the 6 cells of the grid,
and the second round (one change each from the best cell).

Writes plots/grid40k_radar.png and results/grid40k_dataset_rel_mase.tsv.

Usage: python3 plot_grid40k_configs.py
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_grid40k import BEST, LRS, ROUND2, ROUND2_COLOURS, SCORES, TAUS

STUDY = Path(__file__).resolve().parent.parent
MOIRAI = STUDY.parent / "2026-09-06_moirai_small_size"
sys.path.insert(0, str(MOIRAI / "scripts"))
from radar_lib import RADAR, SN, bell_order, per_dataset, read_mase, style_axes  # noqa: E402

OUT = STUDY / "plots" / "grid40k_radar.png"
TABLE = STUDY / "results" / "grid40k_dataset_rel_mase.tsv"
MPE = ("moirai421ew_native166k", "MPE: our Moirai, batch 256, 166k", "#000080", "--")
LR_COLOURS = {"5e-5": "#1b9e77", "1e-4": "#d95f02", "2e-4": "#7570b3"}
TAU_STYLES = {"0.99995": "-", "0.999975": ":"}
# run -> (legend label, colour, line style)
GRID_LINES = {f"g40_{lr_part}_{tau_part}": (f"lr {lr}, EMA start {tau}", LR_COLOURS[lr], TAU_STYLES[tau])
              for lr, (_, lr_part) in LRS.items() for tau, (_, tau_part) in TAUS.items()}
SECOND_LINES = {BEST: GRID_LINES[BEST]} | {
    run: (label.replace("\n", " "), ROUND2_COLOURS[run], "-") for run, (label, _) in list(ROUND2.items())[1:]}


def grid_mase():
    """{run: {config: MASE}} of the runs of the grid with a score at 40k."""
    rows = pd.concat([pd.read_csv(d / "config_mase.tsv", sep="\t") for d in SCORES])
    rows = rows[(rows.stop_k == 40) & (rows.arm == "control")]
    return {run: dict(zip(group.config, group.mase)) for run, group in rows.groupby("run")}


def per_dataset_values():
    """{line name: {dataset: GM over its configs of the relative MASE}}: the grid runs and MPE."""
    seasonal_naive = read_mase(SN)
    values = {run: per_dataset(mase, seasonal_naive) for run, mase in grid_mase().items()}
    values[MPE[0]] = per_dataset(read_mase(RADAR / f"{MPE[0]}.csv"), seasonal_naive)
    return values


def gm(mase_by_config):
    """The GM-Relative MASE of a run over all its configs."""
    seasonal_naive = read_mase(SN)
    logs = [math.log(v / seasonal_naive[c]) for c, v in mase_by_config.items()]
    return math.exp(sum(logs) / len(logs))


def draw_radar(ax, values, lines, labels, scores, title):
    angles = [n / len(labels) * 2 * math.pi for n in range(len(labels))] + [0.0]
    for name, (label, colour, style) in lines.items():
        points = [values[name].get(d, float("nan")) for d in labels]
        width = 3.0 if name == BEST else 2.0
        ax.plot(angles, points + points[:1], style, marker="o", color=colour, lw=width, ms=4,
                label=f"{label}   {scores[name]:.4f}", zorder=3)
    ax.plot(angles, [1.0] * len(angles), color="#2ca02c", ls="--", lw=1.8, zorder=4)
    style_axes(ax, labels, angles)
    ax.set_title(title, fontsize=13, pad=28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=10,
              title="run, GM-Relative MASE (forecast at 40k; MPE at 166k)")


def main():
    values = per_dataset_values()
    mase = grid_mase()
    scores = {run: gm(m) for run, m in mase.items()} | {MPE[0]: gm(read_mase(RADAR / f"{MPE[0]}.csv"))}
    grid = {r: v for r, v in GRID_LINES.items() if r in values} | {MPE[0]: MPE[1:]}
    second = {r: v for r, v in SECOND_LINES.items() if r in values} | {MPE[0]: MPE[1:]}
    labels = bell_order({r: values[r] for r in GRID_LINES if r in values})
    pd.DataFrame(values).loc[labels].round(4).to_csv(TABLE, sep="\t", index_label="dataset")
    fig, axes = plt.subplots(1, 2, figsize=(26, 13), subplot_kw=dict(polar=True),
                             gridspec_kw=dict(wspace=0.55))
    draw_radar(axes[0], values, grid, labels, scores, "The 6 cells of the 40k grid")
    draw_radar(axes[1], values, second, labels, scores, "Second round: changes from the best cell")
    fig.suptitle("Relative MASE per GIFT-Eval dataset (geometric mean over its configs). Green ring: "
                 "Seasonal Naive (1.0). Inside is better. The hardest datasets are at the top.",
                 fontsize=14, y=1.03)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
