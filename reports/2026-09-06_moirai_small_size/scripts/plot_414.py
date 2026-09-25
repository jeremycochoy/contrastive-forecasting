"""GM-Relative MASE against steps for the EMA and L_rep-decay arms of #414.

Thin lines: the arms at 5.6e-4, and the plain cell at 1e-3. Thick lines: the
same settings at 5.6e-5. Reads results/gm_trajectories.tsv.
"""
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = Path(__file__).resolve().parent.parent
TSV = STUDY / "results" / "gm_trajectories.tsv"
OUT = STUDY / "plots" / "gm_mase_414.png"
P = "k3_r100_09_lr56"
OTHER_RATES = ("lr10x", "lr30x", "lr100x", "cos", "aw03")
BAND_AT_56E4, PROJECT_BEST = 0.0649, 1.0651


def load_points():
    points = defaultdict(list)
    for line in open(TSV):
        arm, stop_k, score = line.split()
        points[arm].append((int(stop_k) * 1000, float(score)))
    return {arm: sorted(v) for arm, v in points.items() if len(v) >= 2}


def split_arms(points):
    at_56e4 = [a for a in points if a.startswith(P + "_")
               and not any(tag in a for tag in OTHER_RATES)]
    at_56e5 = [a for a in points if a.startswith(P + "_") and a.endswith("_lr10x")]
    thin = ["k3_r100_09", P] + sorted(at_56e4)
    return [a for a in thin if a in points], sorted(at_56e5)


def label(arm):
    names = {"k3_r100_09": "plain, 1e-3", P: "plain"}
    return names.get(arm, arm.replace(P + "_", ""))


def draw(ax, points, thin, thick):
    for i, arm in enumerate(thin):
        x, y = zip(*points[arm])
        ax.plot(x, y, "-o", color=plt.cm.tab20(i), lw=1.2, ms=3.5, alpha=0.6,
                label=label(arm))
    for i, arm in enumerate(thick):
        x, y = zip(*points[arm])
        ax.plot(x, y, "-o", color=plt.cm.Dark2(i), lw=3.0, ms=7,
                label=label(arm).replace("_lr10x", "") + ", at 5.6e-5")
    best_thin = min(s for arm in thin for _, s in points[arm])
    ax.axhspan(best_thin, best_thin + BAND_AT_56E4, color="grey", alpha=0.13, zorder=0)
    ax.text(480000, best_thin + 0.052, "seed band at 5.6e-4, 0.0649",
            fontsize=9, color="dimgrey")
    ax.axhline(PROJECT_BEST, color="crimson", ls="--", lw=1.4)
    ax.text(42000, PROJECT_BEST + 0.0045, "1.0651, the project best (1.1M parameters)",
            fontsize=9, color="crimson")


def main():
    points = load_points()
    thin, thick = split_arms(points)
    fig, ax = plt.subplots(figsize=(11, 6.8))
    draw(ax, points, thin, thick)
    ax.set_xscale("log")
    ax.set_xticks([40000, 100000, 200000, 400000, 665000])
    ax.set_xticklabels(["40k", "100k", "200k", "400k", "665k"])
    ax.set_xlabel("backbone training steps (log scale). 665,000 steps is one pass over the data")
    ax.set_ylabel("GM-Relative MASE, 97-config GIFT-Eval (lower is better)")
    ax.set_title("The EMA and L_rep-decay settings, at 11.4M parameters\n"
                 "Thin lines train at 5.6e-4 or 1e-3, and each one ends above its "
                 "40k score. Thick lines train at 5.6e-5.")
    ax.set_ylim(1.05, 1.48)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, ncol=3, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT, dpi=135)
    print(f"wrote {OUT}: {len(thin)} thin lines, {len(thick)} thick lines")


main()
