#!/usr/bin/env python3
"""#341 — design schematic as a COMPARISON CHAIN (not a 2x2 grid: the enc3 x
bottleneck cell is never run and irrelevant, so a grid wastes a cell). From the
#339 stop-grad winner (arm 2), a depth step gives arm 3, then a width step gives
arm 4. Arm 1 is the same enc6 x bottleneck architecture with the stop-grad removed
— the control that isolates stop-grad's role in arm 4's collapse. Env: OUT.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.environ.get(
    "OUT",
    "/tmp/cf-341/experiments/2026-06-11_stopgrad_capacity/plots/design_grid.png",
)

NEW, REUSE, NOSG = "#2ca02c", "#7f7f7f", "#c0504d"
CW, CH = 1.34, 0.76

fig, ax = plt.subplots(figsize=(10.6, 4.7))


def card(cx, cy, title, sub, fc, ec, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (cx - CW / 2, cy - CH / 2), CW, CH,
        boxstyle="round,pad=0.02,rounding_size=0.05", linewidth=2.0,
        edgecolor=ec, facecolor=fc, linestyle="--" if dashed else "-", zorder=3))
    ax.text(cx, cy + CH * 0.20, title, ha="center", va="center",
            fontsize=13, fontweight="bold", color=ec, zorder=4)
    ax.text(cx, cy - CH * 0.17, sub, ha="center", va="center",
            fontsize=9.5, color="#222", zorder=4)
    return cx, cy


# main chain: arm 2 -> arm 3 -> arm 4 (left to right)
y0 = 1.30
x2, x3, x4 = 0.0, 1.98, 3.96
card(x2, y0, "arm 2", "enc3 · full · sg\nreused — #339 winner", "#eef6ee", REUSE)
card(x3, y0, "arm 3", "enc6 · full · sg\nNEW", "#eaf5ea", NEW)
card(x4, y0, "arm 4", "enc6 · bn · sg\nNEW", "#eaf5ea", NEW)
# control hanging below arm 4 (same architecture, stop-grad removed)
y1 = 0.02
card(x4, y1, "arm 1", "enc6 · bn · NO sg\nreused — #336 control", "#fbeaea", NOSG)


def harrow(xa, xb, y, color):
    ax.add_patch(FancyArrowPatch((xa, y), (xb, y), arrowstyle="-|>",
                 mutation_scale=20, lw=2.2, color=color, zorder=5))


# depth step: arm 2 -> arm 3
harrow(x2 + CW / 2, x3 - CW / 2, y0, "#1a6e1a")
ax.text((x2 + x3) / 2, y0 + CH / 2 + 0.10, "depth step\nenc3→enc6", ha="center",
        va="bottom", fontsize=10, color="#1a6e1a", fontweight="bold")
# width step: arm 3 -> arm 4
harrow(x3 + CW / 2, x4 - CW / 2, y0, "#b8860b")
ax.text((x3 + x4) / 2, y0 + CH / 2 + 0.10, "width step\nfull→bottleneck", ha="center",
        va="bottom", fontsize=10, color="#b8860b", fontweight="bold")
# control: arm 4 -> arm 1 (remove stop-grad), vertical dashed
ax.add_patch(FancyArrowPatch((x4, y0 - CH / 2), (x4, y1 + CH / 2), arrowstyle="-|>",
             mutation_scale=20, lw=2.2, color=NOSG, zorder=5, linestyle="--"))
ax.text(x4 + CW / 2 + 0.14, (y0 + y1) / 2, "remove\nstop-grad\n(control)", ha="left",
        va="center", fontsize=9.5, color=NOSG, fontweight="bold")

ax.set_xlim(-CW / 2 - 0.35, x4 + CW / 2 + 1.55)
ax.set_ylim(y1 - CH / 2 - 0.30, y0 + CH / 2 + 0.62)
ax.axis("off")
ax.set_title("The four arms as a comparison chain: two capacity steps under stop-grad, "
             "plus the no-stop-grad control", fontsize=12.5, pad=8)

handles = [
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="w", markeredgecolor=NEW,
               markeredgewidth=2, markersize=13, label="new this card (stop-grad)"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="w", markeredgecolor=REUSE,
               markeredgewidth=2, markersize=13, label="reused reference (stop-grad)"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="w", markeredgecolor=NOSG,
               markeredgewidth=2, markersize=13, label="reused control (NO stop-grad)"),
]
ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9,
          bbox_to_anchor=(0.0, 0.0))

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
