#!/usr/bin/env python3
"""Schematic of the (A_norm, B_norm, C) crossfade triplet (#328), one example.

All three windows are z-normalised and added to the batch together: the two
parents A_norm, B_norm, and their blend C, which copies A's past, ramps across a
transition band, and copies B's future. The contrastive loss then sees C with
both of its exact parents as batch-mates (A shares C's past, B shares C's
future). Bottom: the per-sample blend weight s(t) rising 0 -> 1 across the band.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get(
    "OUT",
    "/tmp/cf-328/experiments/2026-06-03_crossfade_triplet/plots/triplet_schematic.png")
rng = np.random.default_rng(11)
T = 512
t = np.arange(T)


def series(rng):
    s = np.zeros(T)
    for _ in range(3):
        f = rng.uniform(1.0, 4.0) / T * 2 * np.pi
        s += rng.uniform(0.5, 1.5) * np.sin(f * t + rng.uniform(0, 2 * np.pi))
    s += rng.uniform(-1.2, 1.2) * (t / T)
    return (s - s.mean()) / s.std()


A, B = series(rng), series(rng)
l, lp = 0.42 * T, 0.62 * T                       # explicit band for a clean illustration
s = np.clip((t - l) / (lp - l), 0.0, 1.0)
C = (1 - s) * A + s * B

fig, (ax, axs) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
ax.axvspan(l, lp, color="0.9", lw=0)
ax.plot(t, A, color="#9bb8d3", lw=1.6, label="A_norm (parent, in batch)")
ax.plot(t, B, color="#e0a96d", lw=1.6, label="B_norm (parent, in batch)")
ax.plot(t, C, color="#1a1a1a", lw=2.2, label="C = blend (in batch)")
ax.text(0.20 * T, ax.get_ylim()[1] * 0.86, "C copies A's past", ha="center", fontsize=10, color="#3a6ea5")
ax.text(0.82 * T, ax.get_ylim()[1] * 0.86, "C copies B's future", ha="center", fontsize=10, color="#c8821f")
ax.text(0.52 * T, ax.get_ylim()[0] * 0.92, "transition", ha="center", fontsize=9, color="0.4")
ax.set_ylabel("value (z-normalised)")
ax.set_title("Crossfade triplet: both z-normalised parents and their blend, all added to the batch")
ax.legend(loc="lower left", fontsize=9, ncol=3)

axs.axvspan(l, lp, color="0.9", lw=0)
axs.plot(t, s, color="#2f6da8", lw=2)
axs.set_ylim(-0.08, 1.08)
axs.set_ylabel("blend s(t)")
axs.set_xlabel("time within the window")
axs.text(0.04 * T, 0.12, "0 -> use A", fontsize=9, color="#3a6ea5")
axs.text(0.83 * T, 0.80, "1 -> use B", fontsize=9, color="#c8821f")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
