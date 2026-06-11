#!/usr/bin/env python3
"""Backbone schematic + where the stop-grad cuts the positive's gradient."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "/tmp/cf-sgpos/experiments/2026-06-10_stopgrad_positive/plots/arch_stopgrad.png"
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")

def box(x, y, w, h, text, fc="#eef3f8", ec="#2f6da8", fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")

def arrow(p0, p1, color="#333333", ls="-", lw=1.6, label=None, lab_dxy=(0, 0.16),
          fs=9, lc=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 color=color, ls=ls, lw=lw))
    if label:
        mx, my = (p0[0] + p1[0]) / 2 + lab_dxy[0], (p0[1] + p1[1]) / 2 + lab_dxy[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs,
                color=lc or color)

Y = 3.2
box(0.15, Y, 1.75, 0.85, "input window\n4096 steps × 1 ch")
box(2.45, Y, 1.15, 0.85, "EWMA\nnorm")
box(4.15, Y, 1.45, 0.85, "GRU patch\nencoder")
box(6.15, Y, 2.05, 0.85, "causal transformer\nencoder — 3 layers")
box(8.75, Y, 0.62, 0.85, "h", fc="#dcead8", ec="#3a7d3a", fs=13, bold=True)
box(9.95, Y, 2.35, 0.85, "causal transformer\nforecaster — 6 layers\nfull width, no bottleneck", fs=9)
box(12.4, Y, 0.5, 0.85, "f", fc="#f6e3d6", ec="#b65d23", fs=13, bold=True)
for x0, x1 in [(1.9, 2.45), (3.6, 4.15), (5.6, 6.15), (8.2, 8.75), (9.37, 9.95), (12.3, 12.4)]:
    arrow((x0, Y + 0.42), (x1, Y + 0.42))
ax.text(8.75 + 0.31, Y + 1.05, "encoder output\n= contrastive target", ha="center",
        fontsize=8.5, color="#3a7d3a")
ax.text(0.15, 4.42, "d_model 384 · 6 attention heads · 16.7M parameters — identical in both arms",
        fontsize=9, color="#444444", style="italic", va="top")

# Loss node
box(4.6, 0.25, 4.9, 1.0,
    "InfoNCE loss\npositive: sim(h$_{t+1}$, f$_{t+1}$)/τ  (in numerator and denominator)\n"
    "negatives: other times · other series · rest of batch", fc="#f4f0e2", ec="#8a7a2a", fs=9.5)

# forward arrows into the loss
arrow((9.06, Y - 0.06), (7.6, 1.32), color="#3a7d3a", label="target  h$_{t+1}$", lab_dxy=(0.25, -0.62))
arrow((12.6, Y - 0.06), (9.35, 1.1), color="#b65d23", label="forecast  f$_{t+1}$", lab_dxy=(0.75, 0.12))

# gradient arrows out of the loss
arrow((9.6, 1.35), (12.45, Y - 0.12), color="#b65d23", ls="--", lw=1.8,
      label="∂L/∂f — both arms", lab_dxy=(0.9, -0.25))
arrow((6.9, 1.4), (8.85, Y - 0.12), color="#888888", ls="--", lw=1.8)
ax.text(6.25, 2.82, "reference arm:\n∂L/∂h flows back to h", fontsize=9, color="#555555",
        ha="center")
# the stop-grad cut
ax.plot([7.78, 8.12], [2.18, 2.52], color="#d62728", lw=3.5)
ax.plot([7.78, 8.12], [2.52, 2.18], color="#d62728", lw=3.5)
ax.plot([4.35, 7.6], [2.3, 2.32], color="#d62728", lw=1.0, ls=":")
ax.text(2.55, 2.3, "stop-grad arm: detach(h$_{t+1}$)\nin the positive — encoder gets\ngradient from negatives only",
        fontsize=10, color="#d62728", ha="center", va="center", fontweight="bold")

fig.tight_layout()
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
