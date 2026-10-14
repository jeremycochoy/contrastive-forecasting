"""The forecast and the reconstruction of the encoder latent through training,
for the runs of this report (GRN, BLK+EMA, GRN+EMA 0.95, GRN+EMA 0.98, GRN+EMA
0.99, BLK+EMA cosine) next to ABC, CYN and BLK. Writes plots/forecast_and_reconstruction.png: three panels
on one x axis.

  1. Forecast (B4): the forecaster rolls the latents forward, and the
     standard reconstruction head decodes each rolled latent.
  2. Reconstruction of the encoder latent (R), transformer head.
  3. The same, linear head.

A dashed vertical line joins the scores of one checkpoint in the panels.

Reads the scores of ABC, CYN and BLK of #412 and #425
(reports/2026-09-06_moirai_small_size/results/gm_trajectories.tsv,
reports/2026-10-06_encoder_reconstruction/results/*.tsv), and the scores of
the runs of this report (GRN: arm abc_gift, BLK+EMA: arm blk_ema, GRN+EMA 0.95:
arm grn_ema095, GRN+EMA 0.98: arm grn_ema098, GRN+EMA 0.99: arm grn_ema099,
BLK+EMA cosine: arm blk_emacos): results/scores.tsv (B4, arm
`control`), and results/recon_*.tsv (R, with the columns arm, stop in
thousands of steps, GM-Relative MASE; collect_recon.py writes them).
The point at 1k steps of GRN, BLK and ABC is a run of the first 1,000 steps
again, with the flags and the seed of the run: B4 in results/scores.tsv (runs
abc_gift, blk and abc at stop_k 1), R in results/recon_green*.tsv and
results/recon_first_steps*.tsv.

Usage: python3 plot_green_scores.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FixedLocator, NullLocator

STUDY = Path(__file__).resolve().parent.parent
MOIRAI = STUDY.parent / "2026-09-06_moirai_small_size"
RECON = STUDY.parent / "2026-10-06_encoder_reconstruction" / "results"
sys.path.insert(0, str(MOIRAI / "scripts"))
from run_style import P, colour, tagged  # noqa: E402

GREEN = "abc_gift"
GREEN_COLOUR = "#2ca02c"  # the colour of the new run in plot_scores.py
BLK_EMA = "blk_ema"
BLK_EMA_COLOUR = "#ff7f0e"  # no other run of this figure has this colour
GRN_EMA = "grn_ema098"
GRN_EMA_COLOUR = "#6a3d9a"  # its colour in plot_ema_plan.py
GRN_EMA099 = "grn_ema099"
GRN_EMA099_COLOUR = "#e377c2"  # its colour in plot_ema_plan.py
GRN_EMA095 = "grn_ema095"
GRN_EMA095_COLOUR = "#8c564b"  # its colour in plot_ema_plan.py
BLK_EMACOS = "blk_emacos"
BLK_EMACOS_COLOUR = "#d62728"  # its colour in plot_ema_plan.py
# The runs of this report: their forecast comes from results/scores.tsv.
NEW_RUNS = [GREEN, BLK_EMA, GRN_EMA095, GRN_EMA, GRN_EMA099, BLK_EMACOS]
# Older runs with a forecast at 1k steps in results/scores.tsv: arm -> run.
FIRST_STEPS = {"cf419_cos200k": "blk", P + "_lr10x": "abc"}
OUT = STUDY / "plots" / "forecast_and_reconstruction.png"
# (arm, legend label, colour, marker): the runs of the figure.
RUNS = [
    (GREEN, r"$\mathbf{GRN}$  lr 5.6e-5, new data", GREEN_COLOUR, "D"),
    (GRN_EMA095, r"$\mathbf{GRN+EMA\ 0.95}$  as GRN, EMA of the teacher 0.95 to 1.0 over 665k",
     GRN_EMA095_COLOUR, "p"),
    (GRN_EMA, r"$\mathbf{GRN+EMA\ 0.98}$  as GRN, EMA of the teacher 0.98 to 1.0 over 665k",
     GRN_EMA_COLOUR, "X"),
    (GRN_EMA099, r"$\mathbf{GRN+EMA\ 0.99}$  as GRN, EMA of the teacher 0.99 to 1.0 over 665k",
     GRN_EMA099_COLOUR, "h"),
    (BLK_EMA, r"$\mathbf{BLK+EMA}$  as BLK, EMA of the teacher 0.9 to 1.0 over 665k",
     BLK_EMA_COLOUR, "P"),
    (BLK_EMACOS, r"$\mathbf{BLK+EMA\ cosine}$  as BLK, EMA of the teacher 0.9 to 1.0 on a cosine "
     "by 200k, then 1.0", BLK_EMACOS_COLOUR, "s"),
    (P + "_lr10x", tagged(P + "_lr10x", "lr 5.6e-5, old data"), colour(P + "_lr10x"), "o"),
    (P + "_cos200k", tagged(P + "_cos200k", "lr cosine to 1e-6 by 200k, old data"),
     colour(P + "_cos200k"), "v"),
    ("cf419_cos200k", tagged("cf419_cos200k", "lr cosine to 1e-6 by 200k, new data"),
     colour("cf419_cos200k"), "*"),
]
OURS = STUDY / "results"
# The R tables of collect_recon.py (first_steps: the 1k point of BLK and ABC).
OUR_R_TABLES = ["green", "blk_ema", "grn_ema095", "grn_ema098", "grn_ema099", "blk_emacos",
                "first_steps"]
# panel title, y range, the score tables, and whether the forecast of the
# runs of this report comes from results/scores.tsv
PANELS = [
    ("Forecast (B4)", (1.10, 1.27),
     [MOIRAI / "results" / "gm_trajectories.tsv", RECON / "forecast_425.tsv"], True),
    ("Reconstruction of the encoder latent (R), transformer head", (0.02, 0.08),
     [RECON / "recon_trajectories.tsv"]
     + [OURS / f"recon_{table}_trajectories.tsv" for table in OUR_R_TABLES], False),
    ("Reconstruction of the encoder latent (R), linear head", (0.12, 0.40),
     [RECON / "recon_lin_trajectories.tsv"]
     + [OURS / f"recon_{table}_lin_trajectories.tsv" for table in OUR_R_TABLES], False),
]
TICKS_K = [1, 10, 40, 100, 200, 400, 665, 1330]


def read_table(paths):
    """``{arm: {stop k: score}}`` from tables of arm, stop k and score."""
    points = {}
    for path in paths:
        if Path(path).is_file():
            for line in open(path):
                arm, stop_k, score = line.split()
                points.setdefault(arm, {})[int(stop_k)] = float(score)
    return points


def new_run_forecast(run, stops=None):
    """``{stop k: score}``: the B4 score of the control head of each wave of
    ``run`` (only the stops in ``stops`` when given)."""
    with open(STUDY / "results" / "scores.tsv") as fh:
        return {int(r["stop_k"]): float(r["gm_rel_mase"])
                for r in csv.DictReader(fh, delimiter="\t")
                if r["run"] == run and r["arm"] == "control"
                and (stops is None or int(r["stop_k"]) in stops)}


def panel_points(tables, forecast):
    points = read_table(tables)
    if forecast:
        for run in NEW_RUNS:
            points[run] = new_run_forecast(run)
        for arm, run in FIRST_STEPS.items():
            points.setdefault(arm, {}).update(new_run_forecast(run, stops={1}))
    return points


def draw_panel(ax, title, y_range, points):
    """One line for each run, and an arrow with the value for each point
    above the y range."""
    for arm, label, colour_hex, marker in RUNS:
        stops = sorted(points.get(arm, {}))
        if not stops:
            continue
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
    for ax, (title, y_range, tables, forecast) in zip(axes, PANELS):
        all_points.append(panel_points(tables, forecast))
        draw_panel(ax, title, y_range, all_points[-1])
    style_x(axes[-1])
    # Blank rows after the older runs: the runs of this report fill the left column.
    handles, labels = axes[0].get_legend_handles_labels()
    new_labels = {label for arm, label, _, _ in RUNS if arm in NEW_RUNS}
    blanks = max(0, 2 * sum(label in new_labels for label in labels) - len(handles))
    axes[0].legend(handles + [Line2D([], [], alpha=0)] * blanks, labels + [""] * blanks,
                   fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=2)
    fig.suptitle("Forecast and reconstruction through training (lower is better)", y=0.995)
    fig.tight_layout()
    # The links lie under the panels, so the panels have no background.
    for ax in axes:
        ax.patch.set_visible(False)
    draw_links(fig, axes, all_points)
    fig.savefig(OUT, dpi=130)


if __name__ == "__main__":
    main()
