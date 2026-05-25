#!/usr/bin/env python3
"""#316 schematic — what the experiment changes: 1-step (β) vs 12-step forecast.
Writes plots/experiment.png. Pure matplotlib, no data."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "plots", "experiment.png")
OUT = os.path.abspath(OUT)

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

def box(x, y, w, h, text, fc, ec="#333", fs=10, bold=False, zorder=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4, zorder=zorder))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=zorder + 0.1)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, lw=1.6, color="#444"))

ENC = "#cfe3f7"; HEAD = "#ffe0b3"; PRED = "#d6f0d6"

# ---- Row 1: k = 1  (β) ----
ax.text(0.1, 4.55, "k = 1   (this is β — predict the next latent)", fontsize=12, fontweight="bold", color="#1f5fa8")
box(0.3, 3.5, 1.9, 0.85, "causal\nencoder", ENC)
arrow(2.2, 3.92, 2.9, 3.92)
box(2.9, 3.55, 0.8, 0.75, "$h_t$", "#eee", fs=12)
arrow(3.7, 3.92, 4.4, 3.92)
box(4.4, 3.5, 1.7, 0.85, "1 forecaster\nhead", HEAD)
arrow(6.1, 3.92, 6.8, 3.92)
box(6.8, 3.55, 2.6, 0.75, "predict $\\hat h_{t+1}$", PRED, fs=11, bold=True)

# ---- Row 2: k = 12 ----
ax.text(0.1, 2.75, "k = 12   (predict the next 12 latents)", fontsize=12, fontweight="bold", color="#b01818")
box(0.3, 1.55, 1.9, 0.85, "same causal\nencoder", ENC)
arrow(2.2, 1.97, 2.9, 1.97)
box(2.9, 1.6, 0.8, 0.75, "$h_t$", "#eee", fs=12)
arrow(3.7, 1.97, 4.4, 1.97)
# stack of heads: two faint copies behind (up-left), labeled box in front
for i in (2, 1):
    box(4.4 - i*0.13, 1.5 + i*0.13, 1.7, 0.85, "", HEAD, zorder=1)
box(4.4, 1.5, 1.7, 0.85, "12 heads\n(head k → $\\hat h_{t+k}$)", HEAD, fs=9, zorder=3)
arrow(6.1, 1.97, 6.8, 1.97)
box(6.8, 1.6, 2.6, 0.75, "predict $\\hat h_{t+1}\\,...\\,\\hat h_{t+12}$", PRED, fs=10, bold=True)

# ---- shared note ----
ax.text(0.1, 0.55,
        "Same in both rows: the causal encoder and all training settings. Only the forecast\n"
        "horizon changes — at k=1 the loss is byte-identical to β.",
        fontsize=9.5, color="#333")
ax.text(0.1, 0.02,
        "Shown here is the headline test (β's forecaster head + β's negatives). Two controls also swap\n"
        "the head (transformer→linear) and, in one, the negative set — see the report's table.",
        fontsize=9.0, style="italic", color="#555")

plt.tight_layout()
plt.savefig(OUT, dpi=120, bbox_inches="tight")
print("wrote", OUT)
