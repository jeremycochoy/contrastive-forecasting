"""GM-Relative MASE against data seen, one line per run, in three groups: ours
with one patch size, ours with patch sizes 8 to 128, and our copy of Moirai.
Draws two figures: gm_mase_rates.png with the runs the owner follows, and
gm_mase_rates_all.png with every run. Reads results/gm_trajectories.tsv
(scripts/gm_trajectories.py writes it)."""
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt

from run_style import P, colour, line, tagged

STUDY = Path(__file__).resolve().parent.parent
TSV = STUDY / "results" / "gm_trajectories.tsv"
OUT = STUDY / "plots" / "gm_mase_rates.png"
OUT_ALL = STUDY / "plots" / "gm_mase_rates_all.png"

# One legend column per group of runs: the group's name, then per run its
# arm, label, line width and marker. The colour and the line style come from
# run_style.py, so a run looks the same in every figure. Off both figures:
# the lr 5.6e-4 run (arm P), b2cos (1.1M, the wrong width),
# cf421z_moirai_native (fp16), cf421fb_moirai_native (GRU bound) and
# cf415_moirai (a separate head, a mistake). Their scores stay in
# results/gm_trajectories.tsv.
GROUPS = [
    ("Ours, one patch size, EWMA, batch 64", [
        (P + "_lr10x",    "lr 5.6e-5, old data",                      3.4, "o"),
        (P + "_lr10xb",   "lr 5.6e-5, old data, second seed",         3.4, "s"),
        (P + "_lr30x",    "lr 1.8e-5, old data",                      4.0, "o"),
        (P + "_lr100x",   "lr 5.6e-6, old data",                      4.0, "^"),
        (P + "_cos665k",  "lr cosine 6e-5→1e-6 over 665k, old data", 4.0, "D"),
        (P + "_cos200k",  "lr cosine 5e-5→1e-6 by 200k, old data",   4.0, "v"),
        ("cf419_cos200k", "lr cosine by 200k, new data",              4.0, "*"),
    ]),
    ("Ours, patch sizes 8 to 128, new data", [
        ("cf412om",  "mean/std, lr 1e-3 cosine by 166k, batch 256, loss bug",   4.4, "h"),
        ("cf412oc",  "mean/std, lr cosine 5.6e-5→1e-6 by 200k, loss bug",       4.4, "p"),
        ("cf412oc2", "mean/std, lr cosine 5.6e-5→1e-6 by 200k, loss fixed",     4.4, ">"),
        ("cf412oe2", "EWMA, lr cosine 5.6e-5→1e-6 by 200k, loss fixed",         4.4, "8"),
        ("cf412om2", "mean/std, lr 1e-3 cosine by 166k, batch 256, loss fixed", 4.4, "H"),
        ("cf412oa2", "mean/std, lr 5.6e-5, loss fixed",                         4.4, "o"),
        ("cf412ow2", "mean/std, warmup to 1e-3, then lr 5.6e-5 from 20k, loss fixed", 4.4, "s"),
        ("cf412or2", "as OWF, L_rep kept at weight 1", 4.4, "p"),
        ("cf412ol2", "as OWR, ramp down to 5.6e-5 by 40k", 4.4, (6, 1, 0)),
    ]),
    ("Moirai: our copy, batch 256", [
        ("cf419_moirai_native",   "own head, new data",                        2.4, "<"),
        ("cf415_moirai_native",   "own head, old data",                        2.4, "o"),
        ("cf421f_moirai_native",  "patch heads, mean/std, new data",           4.4, "X"),
        ("cf421ew_moirai_native", "patch heads, EWMA, new data",               4.4, "d"),
        ("cf421n_moirai_native",  "patch heads, mean/std, RMS term, new data", 3.6, "P"),
    ]),
]
# The owner took TWN, OCB, MON and MOO off the main figure on 10-03.
OFF_MAIN = {P + "_lr10xb", "cf412oc", "cf419_moirai_native", "cf415_moirai_native"}
# Moirai and cf412om train at batch 256: one of their steps holds the data
# of four batch-64 steps.
XSCALE = {"cf415_moirai": 4, "cf415_moirai_native": 4, "cf419_moirai_native": 4,
          "cf421z_moirai_native": 4, "cf421f_moirai_native": 4, "cf421fb_moirai_native": 4,
          "cf421n_moirai_native": 4, "cf421ew_moirai_native": 4, "cf412om": 4,
          "cf412om2": 4}
# Where the last score of a line prints, so two lines that end together part.
END_LABEL_OFFSET = {P + "_lr100x": (9, 4), P + "_cos200k": (9, -12), "cf412om": (-62, 6),
                    "cf412oc2": (9, 6), "cf412oa2": (9, 7), "cf412oe2": (9, 5), "cf412om2": (-22, -18),
                    "cf412ow2": (9, -9), "cf412ol2": (9, 2),
                    "cf421ew_moirai_native": (-45, -18), "cf421n_moirai_native": (-48, -16)}
# A white outline keeps a score label readable where a line crosses it.
HALO = [patheffects.withStroke(linewidth=3, foreground="white")]
# How far left of its point an off-the-chart label starts, so two such labels part.
OFF_LABEL_DIV = {"cf412om": 2.4}
BEST, BAND, PROJECT_BEST = 1.1369, 0.008, 1.0651
YMIN, YMAX = 0.90, 1.70


def load_points():
    points = defaultdict(list)
    for line in open(TSV):
        arm, stop_k, score = line.split()
        points[arm].append((int(stop_k) * 1000, float(score)))
    return points


def draw_series(ax, arm, width, marker, points):
    """One run's line and its score labels. Returns the line."""
    x, y = zip(*sorted(points))
    x = [v * XSCALE.get(arm, 1) for v in x]
    shown = [min(v, YMAX - 0.004) for v in y]
    hue = colour(arm)
    handle, = ax.plot(x, shown, line(arm), color=hue, lw=width, marker=marker, ms=8,
                      zorder=3)
    above = [(xi, yi) for xi, yi in zip(x, y) if yi > YMAX]
    if above:
        # One label lists the points off the chart and points at the last of
        # them. It sits inside the axes, clear of the title: right of a point
        # near the left edge, left of any other. The arrow leaves the label
        # from its side nearest the point.
        xi = above[-1][0]
        right_of_point = xi < 100000
        label_x = xi * 1.12 if right_of_point else xi / OFF_LABEL_DIV.get(arm, 1.9)
        values = ", ".join(f"{yi:.4f}" for _, yi in above)
        ax.annotate(f"{values}, off the chart", (xi, YMAX - 0.004),
                    xytext=(label_x, YMAX - 0.012), va="center",
                    fontsize=9, color=hue, weight="bold", path_effects=HALO,
                    arrowprops=dict(arrowstyle="->", color=hue, lw=1,
                                    relpos=(0, 0.5) if right_of_point else (1, 0.5)))
    if y[-1] <= YMAX:
        ax.annotate(f"{y[-1]:.4f}", (x[-1], shown[-1]), textcoords="offset points",
                    xytext=END_LABEL_OFFSET.get(arm, (9, -3)), fontsize=9,
                    color=hue, weight="bold", path_effects=HALO)
    return handle


def draw_legend(ax, groups, handles):
    """One column per group of runs, under the group's name in bold."""
    rows = 1 + max(len(runs) for _, runs in groups)
    blank = plt.Line2D([], [], linestyle="none")
    entries = []
    for name, runs in groups:
        column = [(blank, name)] + [(handles[arm], tagged(arm, label))
                                    for arm, label, *_ in runs if arm in handles]
        entries += column + [(blank, "")] * (rows - len(column))
    legend = ax.legend(*zip(*entries), ncol=len(GROUPS), fontsize=10, loc="upper center",
                       bbox_to_anchor=(0.5, -0.115), framealpha=0.94)
    for text in legend.get_texts()[::rows]:
        text.set_fontweight("bold")


def draw_references(ax):
    ax.axhline(BEST, color="#2ca02c", ls=":", lw=1.6, zorder=1)
    ax.axhspan(BEST - BAND, BEST + BAND, color="#2ca02c", alpha=0.10, zorder=0)
    ax.text(42000, BEST - 0.016, f"{BEST}: best of ours, old data, 665k.  "
            "Shaded: the seed band, 0.008", fontsize=9.5, color="#2ca02c")
    ax.axhline(PROJECT_BEST, color="#555555", ls="-.", lw=1.4, zorder=1)
    ax.text(110000, PROJECT_BEST + 0.005, f"{PROJECT_BEST}: best of ours at 1.1M parameters, "
            "200k steps", fontsize=9.5, color="#555555")
    ax.axvline(665000, color="#555555", ls=":", lw=1.2, zorder=1)
    ax.text(675000, YMAX - 0.012, "one pass\nover the data", fontsize=9,
            color="#555555", va="top")


def style_axes(ax):
    ax.set_xscale("log")
    ticks = [40000, 100000, 200000, 400000, 665000, 1000000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["40k", "100k", "200k", "400k", "665k", "1,000k"])
    ax.set_xlabel("Data seen, in batch-64 steps (log scale). The Moirai runs, OMB and OMF "
                  "train at batch 256, so one of their steps counts 4.")
    ax.set_ylabel("GM-Relative MASE, 97-config GIFT-Eval (lower is better)")
    ax.set_title("GM-Relative MASE against data seen. 11.4M parameters unless marked.\n"
                 "Solid lines, Ours: our contrastive model. Dashed lines, Moirai: "
                 "our copy of Moirai, trained on the values with its schedule.\n"
                 "Old data: GiftEvalPretrain series of 4,096 points or more. "
                 "New data: all of GiftEvalPretrain.\n"
                 "Loss bug: the contrastive terms of a patch-size run read the rows of one patch size at a time.")
    ax.grid(alpha=0.3)
    ax.set_ylim(YMIN, YMAX)


def draw_figure(groups, out):
    points = load_points()
    fig, ax = plt.subplots(figsize=(12.5, 9.0))
    handles = {arm: draw_series(ax, arm, width, marker, points[arm])
               for _, runs in groups for arm, _, width, marker in runs if arm in points}
    draw_references(ax)
    style_axes(ax)
    draw_legend(ax, groups, handles)
    fig.tight_layout()
    fig.savefig(out, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    followed = [(name, [run for run in runs if run[0] not in OFF_MAIN]) for name, runs in GROUPS]
    draw_figure(followed, OUT)
    draw_figure(GROUPS, OUT_ALL)


main()
