"""The forecast and the reconstruction of the encoder latent through training,
for the runs of this report (GRN+EMA 0.99995, GRN+EMA 0.99995 with a
forecaster of 6 layers, the same with the L_rep weight 0.1 at each
step, the same with muP, the same with the L_rep weight 0.05 and the depth
weights 1 / 2^j, the muP run with lr 5.6e-5, with the EMA 0.999975, or with
lr 3.36e-4 and the input init 1.008, GRN+EMA 0.9998 with the L_rep weight 0.03 and the depth
weights 1 / 2^j, the same with the EMA 0.99995), next to the forecast of MPE
(Moirai, patch heads, EWMA, new data, batch 256: x 4 for the data seen).
Writes plots/forecast_and_reconstruction.png: three panels on one x axis.

  1. Forecast (B4): the forecaster rolls the latents forward, and the
     standard reconstruction head decodes each rolled latent.
  2. Reconstruction of the encoder latent (R), transformer head.
  3. The same, linear head.

A dashed vertical line joins the scores of one checkpoint in the panels. The
legend gives each run a three-letter code (from its colour: MAG, GRY, YEL, VIO,
BLU, ORA, LIM, PNK, RED, BRN), and for orange, lime and pink the result that
the user expected before their start.

Reads the forecast of MPE of #412 and #425
(reports/2026-09-06_moirai_small_size/results/gm_trajectories.tsv,
reports/2026-10-06_encoder_reconstruction/results/*.tsv), and the scores of
the runs of this report (GRN+EMA 0.99995: arm grn_ema099995, with 6
forecaster layers: arm grn_ema099995_f6, and with the L_rep weight 0.1: arm
grn_ema099995_f6_rep01, and with muP (the settings of the muP grid, lr 1e-4):
arm grn_ema099995_f6_rep01_mup, and its variants: arms
grn_ema099995_f6_rep01_mup_lr56, grn_ema0999975_f6_rep01_mup and
grn_ema099995_f6_rep01_mup_hid56, GRN+EMA 0.9998 with the L_rep weight 0.03 and
the depth weights 1 / 2^j: arm grn_ema09998_rep003_halving, and the same with
the EMA 0.99995: arm grn_ema099995_rep003_halving): results/scores.tsv (B4,
arm `control`), and results/recon_*.tsv (R, with the columns arm, stop in
thousands of steps, GM-Relative MASE, from collect_recon.py).

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
from run_style import STYLE, colour, line, tagged  # noqa: E402
from plot_gm_rates import XSCALE  # noqa: E402  (batch-256 runs: 4)

GRN_EMA099995 = "grn_ema099995"
GRN_EMA099995_COLOUR = "#ff00ff"
GRN_EMA099995_F6 = "grn_ema099995_f6"
GRN_EMA099995_F6_COLOUR = "#7f7f7f"  # no other run of this figure has this colour
GRN_EMA099995_F6_REP01 = "grn_ema099995_f6_rep01"
GRN_EMA099995_F6_REP01_COLOUR = "#ffb000"  # no other run of this figure has this colour
GRN_EMA099995_F6_REP01_MUP = "grn_ema099995_f6_rep01_mup"
GRN_EMA099995_F6_REP01_MUP_COLOUR = "#8a2be2"  # no other run of this figure has this colour
GRN_EMA099995_F6_REP005_HALVING_MUP = "grn_ema099995_f6_rep005_halving_mup"
# no other run of this figure has this colour
GRN_EMA099995_F6_REP005_HALVING_MUP_COLOUR = "#1f77b4"
GRN_EMA099995_F6_REP01_MUP_LR56 = "grn_ema099995_f6_rep01_mup_lr56"
GRN_EMA099995_F6_REP01_MUP_LR56_COLOUR = "#ff7f0e"  # orange
GRN_EMA0999975_F6_REP01_MUP = "grn_ema0999975_f6_rep01_mup"
GRN_EMA0999975_F6_REP01_MUP_COLOUR = "#7fff00"  # lime
GRN_EMA099995_F6_REP01_MUP_HID56 = "grn_ema099995_f6_rep01_mup_hid56"
GRN_EMA099995_F6_REP01_MUP_HID56_COLOUR = "#ff69b4"  # pink
GRN_EMA09998_REP003_HALVING ="grn_ema09998_rep003_halving"
GRN_EMA09998_REP003_HALVING_COLOUR = "#e41a1c"  # no other run of this figure has this colour
GRN_EMA099995_REP003_HALVING = "grn_ema099995_rep003_halving"
GRN_EMA099995_REP003_HALVING_COLOUR = "#8c564b"  # no other run of this figure has this colour
# The runs of this report: their forecast comes from results/scores.tsv.
NEW_RUNS = [GRN_EMA099995, GRN_EMA099995_F6,
            GRN_EMA099995_F6_REP01, GRN_EMA099995_F6_REP01_MUP,
            GRN_EMA099995_F6_REP005_HALVING_MUP, GRN_EMA099995_F6_REP01_MUP_LR56,
            GRN_EMA0999975_F6_REP01_MUP, GRN_EMA099995_F6_REP01_MUP_HID56,
            GRN_EMA09998_REP003_HALVING, GRN_EMA099995_REP003_HALVING]
# The newest runs: drawn above the others, so that their first points show.
ON_TOP = {GRN_EMA099995, GRN_EMA099995_F6, GRN_EMA099995_F6_REP01,
          GRN_EMA099995_F6_REP01_MUP, GRN_EMA099995_F6_REP005_HALVING_MUP,
          GRN_EMA099995_F6_REP01_MUP_LR56, GRN_EMA0999975_F6_REP01_MUP,
          GRN_EMA099995_F6_REP01_MUP_HID56,
          GRN_EMA09998_REP003_HALVING, GRN_EMA099995_REP003_HALVING}
MPE = "cf421ew_moirai_native"
# Its R (transformer head 0.12 to 0.21) lies far above the R panels.
FORECAST_ONLY = {MPE}
OUT = STUDY / "plots" / "forecast_and_reconstruction.png"
# (arm, legend label, colour, marker): the runs of the figure. Each label opens
# with the three-letter code of the run (from its colour); a run that changes one
# setting of another run names that run by its code. "expected" gives what the
# user expected of the run before its start (the user, 10-29 and 10-31).
RUNS = [
    (GRN_EMA099995, r"$\mathbf{MAG}$  GRN+EMA 0.99995, 3 forecaster layers, lr 5.6e-5, "
     "stopped at 391k", GRN_EMA099995_COLOUR, "8"),
    (GRN_EMA099995_F6, r"$\mathbf{GRY}$  MAG with 6 forecaster layers (the size of MPE), "
     "stopped at 440k", GRN_EMA099995_F6_COLOUR, "H"),
    (GRN_EMA099995_F6_REP01, r"$\mathbf{YEL}$  GRY with L_rep 0.1, stopped at 432k",
     GRN_EMA099995_F6_REP01_COLOUR, "^"),
    (GRN_EMA099995_F6_REP01_MUP, r"$\mathbf{VIO}$  YEL with muP, lr 1e-4, stopped at 181k",
     GRN_EMA099995_F6_REP01_MUP_COLOUR, "p"),
    (GRN_EMA099995_F6_REP005_HALVING_MUP,
     r"$\mathbf{BLU}$  VIO with L_rep 0.05, depths $1/2^j$, stopped at 100k",
     GRN_EMA099995_F6_REP005_HALVING_MUP_COLOUR, "P"),
    (GRN_EMA099995_F6_REP01_MUP_LR56,
     r"$\mathbf{ORA}$  VIO with lr 5.6e-5 (the lr of YEL), stopped at 268k" "\n"
     "(expected: better than VIO)", GRN_EMA099995_F6_REP01_MUP_LR56_COLOUR, "o"),
    (GRN_EMA0999975_F6_REP01_MUP,
     r"$\mathbf{LIM}$  VIO with the EMA 0.999975, to 665k" "\n"
     "(expected: better than VIO)", GRN_EMA0999975_F6_REP01_MUP_COLOUR, "v"),
    (GRN_EMA099995_F6_REP01_MUP_HID56,
     r"$\mathbf{PNK}$  VIO with lr 3.36e-4 (hidden lr 5.6e-5 of YEL), input init 1.008," "\n"
     "stopped at 254k (expected: better than VIO, maybe as good as YEL)",
     GRN_EMA099995_F6_REP01_MUP_HID56_COLOUR, "*"),
    (GRN_EMA09998_REP003_HALVING,
     r"$\mathbf{RED}$  GRN+EMA 0.9998, L_rep 0.03, depths $1/2^j$, stopped at 212k",
     GRN_EMA09998_REP003_HALVING_COLOUR, "X"),
    (GRN_EMA099995_REP003_HALVING, r"$\mathbf{BRN}$  RED with the EMA 0.99995, stopped at 345k",
     GRN_EMA099995_REP003_HALVING_COLOUR, "s"),
    (MPE, tagged(MPE, "Moirai, patch heads, EWMA, new data"), colour(MPE), "d"),
]
OURS = STUDY / "results"
# The R tables of collect_recon.py.
OUR_R_TABLES = ["grn_ema099995", "grn_ema099995_f6",
                "grn_ema099995_f6_rep01", "grn_ema099995_f6_rep01_mup",
                "grn_ema099995_f6_rep005_halving_mup", "grn_ema099995_f6_rep01_mup_lr56",
                "grn_ema0999975_f6_rep01_mup", "grn_ema099995_f6_rep01_mup_hid56",
                "grn_ema09998_rep003_halving", "grn_ema099995_rep003_halving"]
# panel title, y range, the score tables, and whether the forecast of the
# runs of this report comes from results/scores.tsv
PANELS = [
    ("Forecast (B4)", (1.10, 1.31),
     [MOIRAI / "results" / "gm_trajectories.tsv", RECON / "forecast_425.tsv"], True),
    ("Reconstruction of the encoder latent (R), transformer head", (0.02, 0.11),
     [RECON / "recon_trajectories.tsv"]
     + [OURS / f"recon_{table}_trajectories.tsv" for table in OUR_R_TABLES], False),
    ("Reconstruction of the encoder latent (R), linear head", (0.12, 0.50),
     [RECON / "recon_lin_trajectories.tsv"]
     + [OURS / f"recon_{table}_lin_trajectories.tsv" for table in OUR_R_TABLES], False),
]
TICKS_K = [1, 10, 40, 100, 200, 400, 665, 1330]


def read_table(paths):
    """``{arm: {data seen in thousands of batch-64 steps: score}}`` from
    tables of arm, stop k and score."""
    points = {}
    for path in paths:
        if Path(path).is_file():
            for row in open(path):
                arm, stop_k, score = row.split()
                points.setdefault(arm, {})[int(stop_k) * XSCALE.get(arm, 1)] = float(score)
    return points


def new_run_forecast(run):
    """``{stop k: score}``: the B4 score of the control head of each wave of
    ``run``."""
    with open(STUDY / "results" / "scores.tsv") as fh:
        return {int(r["stop_k"]): float(r["gm_rel_mase"])
                for r in csv.DictReader(fh, delimiter="\t")
                if r["run"] == run and r["arm"] == "control"}


def panel_points(tables, forecast):
    points = read_table(tables)
    if forecast:
        for run in NEW_RUNS:
            points[run] = new_run_forecast(run)
    else:
        for arm in FORECAST_ONLY:
            points.pop(arm, None)
    return points


def draw_panel(ax, title, y_range, points):
    """One line for each run, and an arrow with the value for each point
    above the y range. The y range goes down to the lowest point."""
    low = y_range[0]
    for arm, label, colour_hex, marker in RUNS:
        stops = sorted(points.get(arm, {}))
        if not stops:
            continue
        values = [points[arm][s] for s in stops]
        low = min(low, min(values) - 0.02 * (y_range[1] - y_range[0]))
        ax.plot([s * 1000 for s in stops], values, color=colour_hex, marker=marker,
                ls=line(arm) if arm in STYLE else "-", lw=2.2, ms=7, label=label,
                zorder=3 if arm in ON_TOP else 2)
        for stop, value in zip(stops, values):
            if value > y_range[1]:
                ax.plot(stop * 1000, y_range[1], "^", ms=11, color=colour_hex, clip_on=False)
                ax.annotate(f"{value:.4f}", (stop * 1000, y_range[1]), xytext=(-6, -14),
                            textcoords="offset points", ha="right", fontsize=9, color=colour_hex)
    ax.set_ylim(low, y_range[1])
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel("GM-Relative MASE")
    ax.grid(alpha=0.3)


def style_x(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([k * 1000 for k in TICKS_K]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{k}k" for k in TICKS_K])
    ax.set_xlabel("Data seen, in batch-64 steps (log scale). One step at batch 256 (MPE) "
                  "is equal to 4 steps at batch 64.")


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


def legend_entries(axes):
    """(handles, labels) of the runs drawn in any panel, in the order of RUNS:
    a run with R scores and no forecast score yet also gets its entry."""
    found = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            found.setdefault(label, handle)
    labels = [label for _, label, _, _ in RUNS if label in found]
    return [found[label] for label in labels], labels


def main():
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 12))
    all_points = []
    for ax, (title, y_range, tables, forecast) in zip(axes, PANELS):
        all_points.append(panel_points(tables, forecast))
        draw_panel(ax, title, y_range, all_points[-1])
    style_x(axes[-1])
    # Blank rows after the older runs: the runs of this report fill the left column.
    handles, labels = legend_entries(axes)
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
