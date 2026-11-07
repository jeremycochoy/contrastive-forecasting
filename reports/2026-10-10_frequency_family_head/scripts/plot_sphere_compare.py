"""The forecast GM-Relative MASE through training of the runs with the latent sphere (--latent-sphere,
contrastive-forecasting #453) and of the same runs without it, at batch 256 (the user, 11-06: "the sphere and
non-sphere and how they compare to each other"): width 64 to 80k steps and width 384 to 166k steps. The scores
of the sphere runs use the head path of #454. The points come from results/grid40k_gm_points.tsv and
results/ss256_gm_points.tsv (arm, stop k, score; stop 0.25: the save at 250 steps).

Usage: python3 plot_sphere_compare.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

STUDY = Path(__file__).resolve().parent.parent
TABLES = [STUDY / "results" / "grid40k_gm_points.tsv", STUDY / "results" / "ss256_gm_points.tsv"]
OUT = STUDY / "plots" / "sphere_compare.png"
# (run, label, colour, marker size): the runs without the sphere in cool hues, with the sphere in warm hues,
# in the colours of the other figures. The sphere runs have smaller markers: two equal scores both show.
RUNS = [
    ("g80b_lr1em3_tau99995_bs256_rep01_sigfix_ss_rollfix", "S256: width 64, no sphere", "#17becf", 11),
    ("g80b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_rollfix", "Width 64 with the latent sphere", "#800000", 6),
    ("grn_ema099995_f6_rep01_ss256", "S36: width 384, no sphere", "#006400", 11),
    ("grn_ema099995_f6_rep01_ss256_sphere", "Width 384 with the latent sphere", "#dc143c", 6),
]
# A point of a second run of the first 250 steps with other random numbers than the run (torch 2.8.0 on
# elisa, 2.9.1 on the box where the run started): an open marker.
OTHER_RANDOM = {("grn_ema099995_f6_rep01_ss256_sphere", 0.25)}
TICKS_K = [0.25, 1, 10, 20, 40, 80, 166]


def read_points():
    """``{run: {stop k: score}}`` of TABLES."""
    points = {}
    for path in TABLES:
        for row in open(path):
            run, stop_k, score = row.split()
            points.setdefault(run, {})[float(stop_k)] = float(score)
    return points


def draw_run(ax, run, label, colour, size, scores):
    stops = sorted(scores)
    ax.plot([k * 1000 for k in stops], [scores[k] for k in stops], color=colour, lw=2, label=label)
    for k in stops:
        open_marker = (run, k) in OTHER_RANDOM
        ax.plot(k * 1000, scores[k], marker="D", ms=size, color=colour, ls="none",
                markerfacecolor="white" if open_marker else colour, markeredgewidth=2 if open_marker else 1)


def main():
    points = read_points()
    fig, ax = plt.subplots(figsize=(10, 6))
    for run, label, colour, size in RUNS:
        if points.get(run):
            draw_run(ax, run, label, colour, size, points[run])
    ax.set_xscale("log")
    ax.set_xlim(180, 200_000)  # to the end of the 166k runs
    ax.xaxis.set_major_locator(FixedLocator([k * 1000 for k in TICKS_K]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{k:g}k" for k in TICKS_K])
    ax.set_xlabel("Training step at batch 256 (log scale)")
    ax.set_ylabel("Forecast GM-Relative MASE (lower is better)")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="D", ms=6, color="#555555", markerfacecolor="white", markeredgewidth=2,
                          ls="none"))
    labels.append("Second run of the first 250 steps,\nwith other random numbers than the run")
    ax.legend(handles, labels, fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)


if __name__ == "__main__":
    main()
