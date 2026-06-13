#!/usr/bin/env python3
"""#341 — design schematic: the four arms placed on the encoder-depth x
forecaster-width plane (all stop-grad), so the experiment's shape is shown, not
tabulated in prose. Arms 1-2 are reused references; 3-4 are new. Arm 1 shares the
enc6 x bottleneck cell with arm 4 but is the no-stop-grad reference (marked). The
two comparisons the card makes are drawn as arrows: the depth step (a2->a3) and
the bottleneck step (a3->a4). Env: OUT.
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

# grid coords: x = forecaster width, y = encoder depth
XS = {"full": 0, "bn": 1}          # full-width left, 128-bottleneck right
YS = {"enc3": 1, "enc6": 0}        # 3-layer top, 6-layer bottom
XLAB = ["6-layer full-width\nforecaster", "128-wide bottleneck\nforecaster"]
YLAB = ["6-layer encoder", "3-layer encoder"]  # row 0 = enc6, row 1 = enc3

NEW = "#2ca02c"
REUSE = "#7f7f7f"
NOSG = "#c0504d"

# (x, y, title, sub, facecolor, edgecolor)
CARDS = [
    ("full", "enc3", "arm 2", "enc3 · full · sg\nreused — #339 winner", "#eef6ee", REUSE),
    ("full", "enc6", "arm 3", "enc6 · full · sg\nNEW", "#eaf5ea", NEW),
    ("bn",   "enc6", "arm 4", "enc6 · bn · sg\nNEW", "#eaf5ea", NEW),
]

fig, ax = plt.subplots(figsize=(9.2, 6.2))
cw, ch = 0.78, 0.62


def cell_center(xk, yk):
    return XS[xk] + 0.5, (1 - YS[yk]) + 0.5  # invert y so enc6 sits at bottom visually


def draw_card(xk, yk, title, sub, fc, ec, lw=2.0, dashed=False):
    cx, cy = cell_center(xk, yk)
    box = FancyBboxPatch(
        (cx - cw / 2, cy - ch / 2), cw, ch,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw, edgecolor=ec, facecolor=fc,
        linestyle="--" if dashed else "-", zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy + 0.14, title, ha="center", va="center",
            fontsize=13, fontweight="bold", color=ec, zorder=4)
    ax.text(cx, cy - 0.08, sub, ha="center", va="center",
            fontsize=9.5, color="#222", zorder=4)
    return cx, cy


# the three stop-grad arms
centers = {}
for xk, yk, title, sub, fc, ec in CARDS:
    centers[(xk, yk)] = draw_card(xk, yk, title, sub, fc, ec)

# arm 1: the no-stop-grad reference shares the enc6 x bottleneck cell with arm 4.
# Draw it as a small offset/dashed tag so the reader sees it is the SAME architecture
# minus the stop-grad, not a fifth cell.
cx4, cy4 = centers[("bn", "enc6")]
ax.text(cx4, cy4 - ch / 2 - 0.135,
        "arm 1 — same cell, NO stop-grad (reused #336 reference)",
        ha="center", va="center", fontsize=9, color=NOSG, fontweight="bold", zorder=4)

# empty stop-grad cell (enc3 x bottleneck) — never run
ecx, ecy = cell_center("bn", "enc3")
ax.add_patch(FancyBboxPatch(
    (ecx - cw / 2, ecy - ch / 2), cw, ch,
    boxstyle="round,pad=0.02,rounding_size=0.06",
    linewidth=1.4, edgecolor="#bbbbbb", facecolor="#f4f4f4",
    linestyle=":", zorder=2))
ax.text(ecx, ecy, "(not run)", ha="center", va="center",
        fontsize=10, color="#999", style="italic", zorder=3)

# comparison arrows
# depth step: a2 (enc3 full) -> a3 (enc6 full), vertical
c2 = centers[("full", "enc3")]; c3 = centers[("full", "enc6")]
ax.add_patch(FancyArrowPatch(
    (c2[0] - 0.12, c2[1] - ch / 2), (c3[0] - 0.12, c3[1] + ch / 2),
    arrowstyle="-|>", mutation_scale=18, lw=2.0, color="#1a6e1a", zorder=5))
ax.text(c2[0] - 0.30, (c2[1] + c3[1]) / 2, "depth step\nenc3→enc6", ha="right",
        va="center", fontsize=9, color="#1a6e1a", fontweight="bold")

# bottleneck step: a3 (enc6 full) -> a4 (enc6 bn), horizontal
c4 = centers[("bn", "enc6")]
ax.add_patch(FancyArrowPatch(
    (c3[0] + cw / 2, c3[1] - 0.04), (c4[0] - cw / 2, c4[1] - 0.04),
    arrowstyle="-|>", mutation_scale=18, lw=2.0, color="#b8860b", zorder=5))
ax.text((c3[0] + c4[0]) / 2, c3[1] + 0.12, "width step\nfull→bottleneck", ha="center",
        va="bottom", fontsize=9, color="#b8860b", fontweight="bold")

# axes cosmetics
ax.set_xlim(-0.55, 2.05)
ax.set_ylim(-0.35, 2.15)
ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(XLAB, fontsize=10.5)
ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(YLAB, fontsize=10.5)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Four arms on the encoder-depth × forecaster-width plane "
             "(all stop-grad unless marked)", fontsize=12.5, pad=12)

# legend chips
handles = [
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="w",
               markeredgecolor=NEW, markeredgewidth=2, markersize=13, label="new this card (stop-grad)"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="w",
               markeredgecolor=REUSE, markeredgewidth=2, markersize=13, label="reused reference (stop-grad)"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="w",
               markeredgecolor=NOSG, markeredgewidth=2, markersize=13, label="reused reference (NO stop-grad)"),
]
ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9,
          bbox_to_anchor=(-0.02, -0.02))

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
