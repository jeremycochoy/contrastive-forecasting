"""The forecast and the reconstruction of the encoder latent through training,
for the new run (GRN, green) next to ABC, CYN and BLK. Writes
plots/forecast_and_reconstruction.png: three panels on one x axis.

  1. Forecast (B4): the forecaster rolls the latents forward, and the
     standard reconstruction head decodes each rolled latent.
  2. Reconstruction of the encoder latent (R), transformer head.
  3. The same, linear head.

A dashed vertical line joins the scores of one checkpoint in the panels.

Reads the scores of ABC, CYN and BLK of #412 and #425
(reports/2026-09-06_moirai_small_size/results/gm_trajectories.tsv,
reports/2026-10-06_encoder_reconstruction/results/*.tsv), and the scores of
the new run: results/scores.tsv (B4, arm `control`),
results/recon_green_trajectories.tsv and
results/recon_green_lin_trajectories.tsv (R, with the columns arm, stop in thousands of
steps, GM-Relative MASE).

Usage: python3 plot_green_scores.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FixedLocator, NullLocator

STUDY = Path(__file__).resolve().parent.parent
MOIRAI = STUDY.parent / "2026-09-06_moirai_small_size"
RECON = STUDY.parent / "2026-10-06_encoder_reconstruction" / "results"
sys.path.insert(0, str(MOIRAI / "scripts"))
from run_style import P, colour, tagged  # noqa: E402

GREEN = "abc_gift"
GREEN_COLOUR = "#2ca02c"  # the colour of the new run in plot_scores.py
OUT = STUDY / "plots" / "forecast_and_reconstruction.png"
# (arm, legend label, colour, marker): the runs of the figure.
RUNS = [
    (GREEN, r"$\mathbf{GRN}$  the new run: lr 5.6e-5, new data", GREEN_COLOUR, "D"),
    (P + "_lr10x", tagged(P + "_lr10x", "lr 5.6e-5, old data"), colour(P + "_lr10x"), "o"),
    (P + "_cos200k", tagged(P + "_cos200k", "lr cosine to 1e-6 by 200k, old data"),
     colour(P + "_cos200k"), "v"),
    ("cf419_cos200k", tagged("cf419_cos200k", "lr cosine to 1e-6 by 200k, new data"),
     colour("cf419_cos200k"), "*"),
]
# panel title, y range, the tables of ABC, CYN and BLK, the table of the new run
PANELS = [
    ("Forecast (B4)", (1.10, 1.25),
     [MOIRAI / "results" / "gm_trajectories.tsv", RECON / "forecast_425.tsv"], None),
    ("Reconstruction of the encoder latent (R), transformer head", (0.02, 0.08),
     [RECON / "recon_trajectories.tsv"], STUDY / "results" / "recon_green_trajectories.tsv"),
    ("Reconstruction of the encoder latent (R), linear head", (0.12, 0.40),
     [RECON / "recon_lin_trajectories.tsv"], STUDY / "results" / "recon_green_lin_trajectories.tsv"),
]
TICKS_K = [40, 100, 200, 400, 665, 1000, 1330]


def read_table(paths):
    """``{arm: {stop k: score}}`` from tables of arm, stop k and score."""
    points = {}
    for path in paths:
        if Path(path).is_file():
            for line in open(path):
                arm, stop_k, score = line.split()
                points.setdefault(arm, {})[int(stop_k)] = float(score)
    return points


def green_forecast():
    """``{stop k: score}``: the B4 score of the control head of each wave."""
    with open(STUDY / "results" / "scores.tsv") as fh:
        return {int(r["stop_k"]): float(r["gm_rel_mase"])
                for r in csv.DictReader(fh, delimiter="\t")
                if r["run"] == GREEN and r["arm"] == "control"}


def panel_points(base_tables, green_table):
    points = read_table(base_tables)
    points[GREEN] = green_forecast() if green_table is None else read_table([green_table]).get(GREEN, {})
    return points


def draw_panel(ax, title, y_range, points):
    """One line for each run, and an arrow with the value for each point
    above the y range."""
    for arm, label, colour_hex, marker in RUNS:
        stops = sorted(points.get(arm, {}))
        values = [points[arm][s] for s in stops]
        ax.plot([s * 1000 for s in stops], values, color=colour_hex, marker=marker,
                lw=2.2, ms=7, label=label)
        for stop, value in zip(stops, values):
            if value > y_range[1]:
                ax.plot(stop * 1000, y_range[1], "^", ms=11, color=colour_hex, clip_on=False)
                ax.annotate(f"{value:.4f}", (stop * 1000, y_range[1]), xytext=(-6, -14),
                            textcoords="offset points", ha="right", fontsize=9, color=colour_hex)
    ax.set_ylim(*y_range)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel("GM-Relative MASE")
    ax.grid(alpha=0.3)


def style_x(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([k * 1000 for k in TICKS_K]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{k}k" for k in TICKS_K])
    ax.set_xlabel("Data seen, in batch-64 steps (log scale)")


def clipped(ax, value):
    """``value`` inside the y range of ``ax``: an arrow marks a point above it."""
    low, high = ax.get_ylim()
    return min(max(value, low), high)


def draw_links(fig, axes, all_points):
    """For each checkpoint with a score in two panels next to each other, a
    dashed line in the colour of its run from one score to the other."""
    for arm, _, colour_hex, _ in RUNS:
        for upper in range(len(axes) - 1):
            top, bottom = all_points[upper].get(arm, {}), all_points[upper + 1].get(arm, {})
            for stop in sorted(set(top) & set(bottom)):
                x = stop * 1000
                fig.add_artist(ConnectionPatch(
                    xyA=(x, clipped(axes[upper], top[stop])), coordsA=axes[upper].transData,
                    xyB=(x, clipped(axes[upper + 1], bottom[stop])),
                    coordsB=axes[upper + 1].transData,
                    color=colour_hex, ls="--", lw=0.8, alpha=0.6, zorder=0))


def main():
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 12))
    all_points = []
    for ax, (title, y_range, base_tables, green_table) in zip(axes, PANELS):
        all_points.append(panel_points(base_tables, green_table))
        draw_panel(ax, title, y_range, all_points[-1])
    style_x(axes[-1])
    axes[0].legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.42), ncol=2)
    fig.suptitle("Forecast and reconstruction through training (lower is better)", y=0.995)
    fig.tight_layout()
    # The links lie under the panels, so the panels have no background.
    for ax in axes:
        ax.patch.set_visible(False)
    draw_links(fig, axes, all_points)
    fig.savefig(OUT, dpi=130)


if __name__ == "__main__":
    main()
