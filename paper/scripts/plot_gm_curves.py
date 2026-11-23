"""GM-MASE against the training step, for the Results section.

Two panels, width 64 and width 384. Each has our model, the Moirai-style
value-space model of the same width, our cooldown points after a dotted
line at the save the cooldown starts from, and seasonal naive at 1.

Reads paper/data/gm_curves/gm_mase_by_step.csv (sources in its README) and
writes paper/figures/gm_curves.pdf.
Usage: python3 paper/scripts/plot_gm_curves.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parent.parent
DATA = PAPER / "data" / "gm_curves" / "gm_mase_by_step.csv"
OUT = PAPER / "figures" / "gm_curves.pdf"
INK, MUTED = "#222222", "#888888"
# Okabe-Ito: blues for ours, oranges for the Moirai-style models. The lighter
# pair is width 64, as in the radar of Figure fig:gift-radar.
OURS = {64: "#56B4E9", 384: "#0072B2"}
MOIRAI = {64: "#E69F00", 384: "#D55E00"}
COOLDOWN_START = {64: 135, 384: 125}  # the save each cooldown starts from (k)
TITLE = {64: "width 64", 384: "width 384"}


def load():
    data = defaultdict(list)
    with open(DATA) as f:
        for r in csv.DictReader(f):
            data[(int(r["width"]), r["series"])].append(
                (int(r["step_k"]), float(r["gm_mase"])))
    return {k: sorted(v) for k, v in data.items()}


def main():
    data = load()
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for ax, width in zip(axes, (64, 384)):
        mx, my = zip(*data[(width, "moirai_style")])
        start = COOLDOWN_START[width]
        # Ours: the saves before the cooldown starts, then the cooled-down
        # saves, as one line.
        ours = [p for p in data[(width, "ours")] if p[0] <= start]
        ours += data[(width, "ours_cooldown")]
        ax.plot(mx, my, color=MOIRAI[width], lw=1.4, marker="s", ms=3.5,
                label="Moirai-style, value space")
        ax.plot(*zip(*ours), color=OURS[width], lw=1.4, marker="o", ms=3.5,
                label="ours")
        ax.axvline(start, color=INK, lw=0.8, ls=":")
        ax.text(start, 0.98, " cooldown", transform=ax.get_xaxis_transform(),
                fontsize=7, va="top", color=INK)
        ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
        ax.set_title(TITLE[width], fontsize=8)
        ax.set_xlabel("training step (thousands)", fontsize=7)
        ax.set_xlim(0, 172)
        ax.tick_params(labelsize=6, colors=MUTED)
        ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    axes[0].set_ylabel("GM-MASE (lower is better)", fontsize=7)
    for ax in axes:
        ax.legend(frameon=False, fontsize=6.5, loc="upper right", bbox_to_anchor=(0.97, 0.9))
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
