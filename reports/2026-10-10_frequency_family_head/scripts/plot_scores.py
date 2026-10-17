"""The figures of the frequency family.

forecast_through_training.png: the B4 GM-Relative MASE of `abc_gift` (the ABC
recipe on the GiftEvalPretrain stream) at each ABC stop, with the standard
head and with the family head `shared_strict`, next to ABC and BLK.
blk_200k_family_heads.png: the five heads of the BLK 200k wave.

Reads results/scores.tsv (collect_scores.py) and the ABC and BLK scores of
reports/2026-09-06_moirai_small_size/results/gm_trajectories.tsv. The colour
of ABC and BLK comes from run_style.py of that report.

Usage: python3 plot_scores.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

STUDY = Path(__file__).resolve().parent.parent
MOIRAI = STUDY.parent / "2026-09-06_moirai_small_size"
sys.path.insert(0, str(MOIRAI / "scripts"))
from run_style import P, colour, tagged  # noqa: E402

SCORES = STUDY / "results" / "scores.tsv"
TRAJECTORIES = MOIRAI / "results" / "gm_trajectories.tsv"
OUT_CURVES = STUDY / "plots" / "forecast_through_training.png"
OUT_BLK = STUDY / "plots" / "blk_200k_family_heads.png"
NEW_RUN_COLOUR = "#2ca02c"  # no run of run_style.py has this colour
# The y range of the curves. A point above it gets an arrow at the top edge
# and its value as text, so the range shows the differences near the best.
Y_RANGE = (1.11, 1.235)
ABC, BLK = P + "_lr10x", "cf419_cos200k"
STOPS_K = [40, 100, 140, 180, 200, 240, 300, 360, 400, 460]
BLK_ARMS = [
    ("control", "Standard head (the control)"),
    ("shared_strict", "Family, shared body, strict"),
    ("shared_draw", "Family, shared body, draw"),
    ("heads_strict", "Family, 4 complete heads, strict"),
    ("heads_draw", "Family, 4 complete heads, draw"),
]


def read_scores():
    """{(run, arm): {stop_k: GM-Relative MASE}} of results/scores.tsv."""
    scores = {}
    with open(SCORES) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            key = (row["run"], row["arm"])
            scores.setdefault(key, {})[int(row["stop_k"])] = float(row["gm_rel_mase"])
    return scores


def read_trajectory(run):
    """{stop_k: GM-Relative MASE} of one run of gm_trajectories.tsv."""
    points = {}
    for line in open(TRAJECTORIES):
        name, stop_k, score = line.split()
        if name == run:
            points[int(stop_k)] = float(score)
    return points


def best_label(label, points):
    """The label, with the lowest score of the curve and its stop."""
    stop, score = min(points.items(), key=lambda item: item[1])
    return f"{label}. Best {score:.4f} at {stop}k"


def draw_curve(ax, points, label, colour_hex, **style):
    stops = sorted(points)
    ax.plot([s * 1000 for s in stops], [points[s] for s in stops],
            color=colour_hex, label=best_label(label, points), lw=2.4, ms=7, **style)


def off_scale(points):
    """The stops of a curve with a score above the y range."""
    return [stop for stop in sorted(points) if points[stop] > Y_RANGE[1]]


def note_off_scale(ax, curves):
    """An arrow at the top edge for each point above the y range, and a box
    with their values."""
    notes = []
    for name, points, colour_hex, face in curves:
        for stop in off_scale(points):
            ax.plot(stop * 1000, Y_RANGE[1], "^", ms=13, color=colour_hex, mfc=face,
                    mew=2, clip_on=False, zorder=5)
            notes.append(f"{name} at {stop}k: {points[stop]:.4f}")
    if notes:
        ax.text(0.45, 0.97, "Above the scale (arrows):\n" + "\n".join(notes),
                transform=ax.transAxes, ha="center", va="top", fontsize=10,
                bbox=dict(fc="white", ec="#999999"))


def style_curves(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([s * 1000 for s in STOPS_K]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{s}k" for s in STOPS_K])
    ax.set_xlabel("Data seen, in batch-64 steps (log scale)")
    ax.set_ylabel("GM-Relative MASE, 97-config GIFT-Eval (lower is better)")
    ax.set_title("Forecast score through training")
    ax.set_ylim(*Y_RANGE)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.11))


def plot_curves(scores):
    fig, ax = plt.subplots(figsize=(11, 8))
    standard, family = scores[("abc_gift", "control")], scores[("abc_gift", "shared_strict")]
    draw_curve(ax, standard, "New run: the ABC recipe on the new data, standard head",
               NEW_RUN_COLOUR, marker="o")
    draw_curve(ax, family, "New run, family head (shared body, strict)", NEW_RUN_COLOUR,
               marker="o", ls="--", mfc="white")
    note_off_scale(ax, [("New run, standard head", standard, NEW_RUN_COLOUR, NEW_RUN_COLOUR),
                        ("New run, family head", family, NEW_RUN_COLOUR, "white")])
    draw_curve(ax, read_trajectory(ABC), tagged(ABC, "lr 5.6e-5, old data, standard head"),
               colour(ABC), marker="s")
    draw_curve(ax, read_trajectory(BLK), tagged(BLK, "lr cosine by 200k, new data, standard head"),
               colour(BLK), marker="*")
    style_curves(ax)
    fig.tight_layout()
    fig.savefig(OUT_CURVES, dpi=130)


def draw_blk_heads(ax, scores):
    control = scores[("blk", "control")][200]
    for row, (arm, label) in enumerate(BLK_ARMS):
        value = scores[("blk", arm)][200]
        change = "" if arm == "control" else f"  ({(value / control - 1) * 100:+.2f}%)"
        ax.plot(value, row, "o", ms=11, color=colour(BLK),
                mfc=colour(BLK) if arm == "control" else "white", mew=2)
        ax.text(value + 0.0006, row, f"{value:.4f}{change}", va="center", fontsize=11)
    ax.axvline(control, color=colour(BLK), ls=":", lw=1.2)
    ax.set_yticks(range(len(BLK_ARMS)))
    ax.set_yticklabels([label for _, label in BLK_ARMS], fontsize=11)
    ax.invert_yaxis()


def plot_blk_heads(scores):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    draw_blk_heads(ax, scores)
    ax.set_xlim(1.120, 1.162)
    ax.set_xlabel("GM-Relative MASE, 97-config GIFT-Eval (lower is better)")
    ax.set_title("BLK at 200k: five forecast heads")
    ax.grid(axis="x", alpha=0.3)
    fig.text(0.01, 0.02, "strict: a series trains only the decoder of its frequency.   "
             "draw: a series draws a decoder from the range of its frequency, as Moirai does.",
             fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT_BLK, dpi=130)


def main():
    scores = read_scores()
    OUT_CURVES.parent.mkdir(exist_ok=True)
    plot_curves(scores)
    plot_blk_heads(scores)


if __name__ == "__main__":
    main()
