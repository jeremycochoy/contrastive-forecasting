#!/usr/bin/env python3
"""Generate V2 scaling curve: GM-Relative MASE vs backbone training steps."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# V2 data points (backbone steps, GM-Relative MASE)
steps = [30_000, 60_000, 112_000]
mase = [1.2564, 1.2742, 1.2751]
labels = ['30k', '60k', '112k\n(1 epoch)']

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# Plot data
ax.plot(steps, mase, 'o-', color='#2196F3', linewidth=2, markersize=10,
        label='V2 (shuffled data)', zorder=3)

# Annotate points
for s, m, l in zip(steps, mase, labels):
    ax.annotate(f'{m:.4f}', (s, m), textcoords='offset points',
                xytext=(0, 14), ha='center', fontsize=10, fontweight='bold')

# Naive baseline
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
           label='Naive baseline (1.000)')

# Reference models
refs = {
    'Sundial': 0.673, 'TimesFM': 0.680, 'PatchTST': 0.762,
    'Chronos': 0.786, 'Moirai': 0.809,
}
for name, val in refs.items():
    ax.axhline(y=val, color='#FF9800', linestyle=':', linewidth=1, alpha=0.4)
    ax.text(steps[-1] * 1.02, val, name, fontsize=7, color='#FF9800',
            va='center', alpha=0.7)

ax.set_xlabel('Backbone Training Steps', fontsize=12)
ax.set_ylabel('GM-Relative MASE (lower is better)', fontsize=12)
ax.set_title('V2 Scaling Curve (Shuffled Data)\nGIFT-Eval Performance vs Training Steps', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(0, steps[-1] * 1.15)
ax.set_ylim(0.5, 1.45)
ax.grid(True, alpha=0.3)
ax.set_xticks(steps)
ax.set_xticklabels(labels)

plt.tight_layout()
plt.savefig('experiments/gift-eval/fig_v2_scaling_curve.png', dpi=150)
print('Saved experiments/gift-eval/fig_v2_scaling_curve.png')
