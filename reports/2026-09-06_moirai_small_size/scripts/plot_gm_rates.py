"""GM-Relative MASE against data seen: each backbone rate, the two cosine
anneals, the value-space reference of #415, and the #419 runs on all of
GiftEvalPretrain. Reads
results/gm_trajectories.tsv (scripts/gm_trajectories.py writes it)."""
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = Path(__file__).resolve().parent.parent
TSV = STUDY / "results" / "gm_trajectories.tsv"
OUT = STUDY / "plots" / "gm_mase_rates.png"

P = "k3_r100_09_lr56_fix09_dec10k"
# arm, label, colour, line width, marker, line style
SERIES = [
    # Ours: our contrastive model. The lr 5.6e-4 run (arm P) keeps its scores in
    # results/gm_trajectories.tsv; the owner took it off this figure.
    (P + "_lr10x",    "Ours, lr 5.6e-5, seed a",                 "#1f77b4", 3.4, "o", "-"),
    (P + "_lr10xb",   "Ours, lr 5.6e-5, seed b",                 "#7fb8e0", 3.4, "s", "-"),
    (P + "_lr30x",    "Ours, lr 1.8e-5",                         "#d62728", 4.0, "o", "-"),
    (P + "_lr100x",   "Ours, lr 5.6e-6",                         "#ff9f40", 4.0, "^", "-"),
    (P + "_cos665k",  "Ours, lr cosine 6e-5→1e-6 over 665k",     "#9467bd", 4.0, "D", "-"),
    (P + "_cos200k",  "Ours, lr cosine 5.6e-5→1e-6 by 200k",     "#17becf", 4.0, "v", "-"),
    ("b2cos",         "Ours 1.1M, lr cosine 5.6e-5→1e-6 by 200k", "#e377c2", 3.0, "P", "-."),
    ("cf419_cos200k", "Ours, lr cosine by 200k, new data",       "#1a1a1a", 4.0, "*", "-"),
    # Moirai: our copy of Moirai, trained on the values with its schedule.
    ("cf415_moirai",  "Moirai, separate head (B4)",              "#8c564b", 3.2, "X", "-"),
    ("cf415_moirai_native", "Moirai, own head",                  "#8c564b", 2.4, "o", ":"),
    ("cf419_moirai_native", "Moirai, own head, new data",        "#9a9a00", 3.0, "o", "--"),
    ("cf421z_moirai_native", "Moirai + patch heads + mean/std, fp16, new data", "#c51b7d", 3.4, "D", "-"),
    ("cf421f_moirai_native", "Moirai + patch heads + mean/std, new data", "#1b9e77", 3.8, "X", "-"),
    ("cf421fb_moirai_native", "Moirai + patch heads + mean/std + GRU bound, new data", "#d4a017", 3.8, "h", "-"),
]
# Moirai trains at batch 256: one of its steps holds the data of four
# batch-64 steps.
XSCALE = {"cf415_moirai": 4, "cf415_moirai_native": 4, "cf419_moirai_native": 4,
          "cf421z_moirai_native": 4, "cf421f_moirai_native": 4, "cf421fb_moirai_native": 4}
# Where the last score of a line prints, so two lines that end together part.
END_LABEL_OFFSET = {P + "_lr100x": (9, 4), P + "_cos200k": (9, -12)}
BEST, BAND, PROJECT_BEST = 1.1369, 0.008, 1.0651
YMIN, YMAX = 1.05, 1.66


def load_points():
    points = defaultdict(list)
    for line in open(TSV):
        arm, stop_k, score = line.split()
        points[arm].append((int(stop_k) * 1000, float(score)))
    return points


def draw_series(ax, arm, label, colour, width, marker, style, points):
    x, y = zip(*sorted(points))
    x = [v * XSCALE.get(arm, 1) for v in x]
    shown = [min(v, YMAX - 0.004) for v in y]
    ax.plot(x, shown, style, color=colour, lw=width, marker=marker, ms=8,
            label=label, zorder=3)
    for xi, yi in zip(x, y):
        if yi > YMAX:
            # The label sits beside the point, inside the axes, clear of the
            # title: right of a point near the left edge, left of any other.
            label_x = xi * 1.12 if xi < 100000 else xi / 1.9
            ax.annotate(f"{yi:.4f}, off the chart", (xi, YMAX - 0.004),
                        xytext=(label_x, YMAX - 0.012), va="center",
                        fontsize=9, color=colour, weight="bold",
                        arrowprops=dict(arrowstyle="->", color=colour, lw=1))
    if y[-1] <= YMAX:
        ax.annotate(f"{y[-1]:.4f}", (x[-1], shown[-1]), textcoords="offset points",
                    xytext=END_LABEL_OFFSET.get(arm, (9, -3)), fontsize=9,
                    color=colour, weight="bold")


def draw_references(ax):
    ax.axhline(BEST, color="#2ca02c", ls=":", lw=1.6, zorder=1)
    ax.axhspan(BEST - BAND, BEST + BAND, color="#2ca02c", alpha=0.10, zorder=0)
    ax.text(42000, BEST - 0.016, f"{BEST}: best of ours, old data, 665k.  "
            "Shaded: the seed band, 0.008", fontsize=9.5, color="#2ca02c")
    ax.axhline(PROJECT_BEST, color="#e8173c", ls="--", lw=1.4, zorder=1)
    ax.text(42000, PROJECT_BEST + 0.005, f"{PROJECT_BEST}: project best "
            "(ours, 1.1M parameters, 200k steps)", fontsize=9.5, color="#e8173c")
    ax.axvline(665000, color="#555555", ls=":", lw=1.2, zorder=1)
    ax.text(675000, YMAX - 0.012, "one pass\nover the data", fontsize=9,
            color="#555555", va="top")


def main():
    points = load_points()
    fig, ax = plt.subplots(figsize=(12.5, 9.0))
    for arm, *style in SERIES:
        if arm in points:
            draw_series(ax, arm, *style, points[arm])
    draw_references(ax)
    ax.set_xscale("log")
    ticks = [40000, 100000, 200000, 400000, 665000, 1000000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["40k", "100k", "200k", "400k", "665k", "1,000k"])
    ax.set_xlabel("Data seen, in batch-64 steps (log scale). Moirai trains at "
                  "batch 256, so one of its steps counts 4.")
    ax.set_ylabel("GM-Relative MASE, 97-config GIFT-Eval (lower is better)")
    ax.set_title("GM-Relative MASE against data seen. 11.4M parameters unless marked.\n"
                 "Ours: our contrastive model. Moirai: our copy of Moirai, trained "
                 "on the values with its schedule.\n"
                 "Old data: GiftEvalPretrain series of 4,096 points or more. "
                 "New data: all of GiftEvalPretrain.")
    ax.grid(alpha=0.3)
    ax.set_ylim(YMIN, YMAX)
    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.115),
              ncol=3, framealpha=0.94, title="one line per run")
    fig.tight_layout()
    fig.savefig(OUT, dpi=135)
    print(f"wrote {OUT}")


main()
