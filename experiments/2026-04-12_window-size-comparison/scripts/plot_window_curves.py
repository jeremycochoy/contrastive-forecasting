#!/usr/bin/env python3
"""Plot window-size comparison curves.

Two figures, written to ../plots/:
  1) gap_vs_step.png      — contrastive gap vs training step (3 arms)
  2) gap_vs_walltime.png  — contrastive gap vs wall-clock minutes (3 arms)

Data sources (relative to this script, run from the scripts/ dir):
  - gap-vs-step: parsed directly from the committed training logs in ../logs/
    (authoritative: the per-1000-step `gap=` field each run printed).
  - gap-vs-walltime: there are NO timestamps in the logs, so the wall-time
    axis is taken from the "Results by Wall Time" table in the report
    (../window-size-comparison.md). Hard-coded below and tagged as such.
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "..", "logs")
PLOT_DIR = os.path.join(HERE, "..", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# --- arm styling -----------------------------------------------------------
ARMS = {
    "w32": dict(log="window_test_w32.log",
                label="W=32, bs=32 (23.4 GB, 5.7 sps)",
                color="#1f77b4", marker="o"),
    "w16": dict(log="window_test_w16.log",
                label="W=16, bs=24 (14.7 GB, 4.7 sps)",
                color="#d62728", marker="s"),
    "w16_bs28": dict(log="window_test_w16_bs28.log",
                     label="W=16, bs=28 (17.6 GB, 3.9 sps)",
                     color="#2ca02c", marker="^"),
}

LINE_RE = re.compile(r"\[\s*(\d+)\]\s.*\bgap=([0-9.]+)")


def parse_log(path):
    """Return (steps, gaps) parsed from a training log's `gap=` field."""
    steps, gaps = [], []
    with open(path) as fh:
        for line in fh:
            m = LINE_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                gaps.append(float(m.group(2)))
    return steps, gaps


# --- Figure 1: gap vs step (from logs) -------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
best = {}
for key, cfg in ARMS.items():
    steps, gaps = parse_log(os.path.join(LOG_DIR, cfg["log"]))
    ax.plot(steps, gaps, marker=cfg["marker"], color=cfg["color"],
            label=cfg["label"], linewidth=2, markersize=6)
    bi = max(range(len(gaps)), key=lambda i: gaps[i])
    best[key] = (steps[bi], gaps[bi])

# annotate the two 10k-step headline arms' best points
for key in ("w32", "w16"):
    s, g = best[key]
    ax.annotate(f"{g:.3f}", (s, g), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=9,
                color=ARMS[key]["color"], fontweight="bold")

ax.set_xlabel("training step")
ax.set_ylabel("contrastive gap  (FF − FP)")
ax.set_title("Contrastive gap vs step — single run per arm (ARMA val, seed 0)")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
out1 = os.path.join(PLOT_DIR, "gap_vs_step.png")
fig.savefig(out1, dpi=130)
plt.close(fig)
print("wrote", out1, "best:", {k: v for k, v in best.items()})

# --- Figure 2: gap vs wall time (from report's wall-time table) ------------
# Source: ../window-size-comparison.md "Results by Wall Time" (+ bs=28 follow-up).
# (minutes, gap) pairs. No timestamps exist in the logs, hence this table.
WALL = {
    "w32": [(2.9, 0.056), (5.8, 0.068), (8.8, 0.067), (11.7, 0.073),
            (14.6, 0.074), (17.5, 0.075), (20.5, 0.079), (23.4, 0.073),
            (26.3, 0.082), (29.2, 0.082)],
    "w16": [(3.5, 0.064), (7.1, 0.064), (10.6, 0.072), (14.2, 0.077),
            (17.7, 0.078), (21.3, 0.080), (24.8, 0.088), (28.4, 0.091),
            (31.9, 0.093), (35.5, 0.092)],
    # bs=28 was wall-time-limited to 29.2 min, stopping at step 6895 (gap .082).
    # per-step sps=3.9 -> 1000 steps ~= 4.3 min; place markers on that cadence.
    "w16_bs28": [(4.3, 0.064), (8.5, 0.069), (12.8, 0.073), (17.1, 0.078),
                 (21.4, 0.080), (25.6, 0.082)],
}

fig, ax = plt.subplots(figsize=(7.5, 4.8))
for key, cfg in ARMS.items():
    xs = [p[0] for p in WALL[key]]
    ys = [p[1] for p in WALL[key]]
    ax.plot(xs, ys, marker=cfg["marker"], color=cfg["color"],
            label=cfg["label"], linewidth=2, markersize=6)

# mark the 29.2-min wall-time budget of the W=32 run
ax.axvline(29.2, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.text(29.2, ax.get_ylim()[1], " W=32 budget (29 min)",
        rotation=90, va="top", ha="left", fontsize=8, color="gray")

ax.set_xlabel("wall-clock time (minutes)")
ax.set_ylabel("contrastive gap  (FF − FP)")
ax.set_title("Contrastive gap vs wall time — single run per arm")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
out2 = os.path.join(PLOT_DIR, "gap_vs_walltime.png")
fig.savefig(out2, dpi=130)
plt.close(fig)
print("wrote", out2)
