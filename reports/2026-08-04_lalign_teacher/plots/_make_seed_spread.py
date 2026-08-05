"""How far one cell moves under nothing but a head seed.

Each frozen backbone gets its quantile head retrained under two extra seeds
and the full 97-config eval re-run each time. Each row is one cell; the dots
are its per-seed GM-Relative MASE and the bar is the range they span. A
difference smaller than that cell's own range is not separable from
head-seed noise.

**The range is a property of the cell, not of the report.** The four
backbone-40k rows, drawn in their own colour, are where the controlled
comparison lives and are measured on both sides of it. They span 0.0018 to
0.0747 — a factor of forty — so a delta that clears the smallest of them
sits inside the largest. Nothing here is a global bar, and the six
un-replicated controlled cells get no bar at all.

Reads `results/seed_spread.csv` (written by
`experiments/2026-08-01_lalign_teacher/scripts/seed_spread.py`).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SRC = HERE.parent / "results" / "seed_spread.csv"

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

CONTROLLED_BB = 40000
C_40K, C_LONG = "#0f6f6f", "#8b1e8b"

with open(SRC, newline="") as fh:
    rows = list(csv.DictReader(fh))
# Backbone step first, so the 40k block the controlled deltas are judged
# against reads as one group instead of interleaving with the long runs.
rows.sort(key=lambda r: (int(r["bb_steps"]), -float(r["range"])))

fig, ax = plt.subplots(figsize=(13.5, 0.42 * len(rows) + 1.5))
for y, r in enumerate(rows[::-1]):
    vals = [float(v) for v in r["values"].split()]
    col = C_40K if int(r["bb_steps"]) == CONTROLLED_BB else C_LONG
    ax.plot([min(vals), max(vals)], [y, y], color=col, lw=6, alpha=0.3,
            solid_capstyle="butt")
    ax.plot(vals, [y] * len(vals), "o", color=col, ms=7)
    ax.text(max(vals) + 0.012, y, f"range {float(r['range']):.4f}"
            f"  ({float(r['range_rel']) * 100:.1f}% of the cell's lowest seed)",
            va="center", fontsize=9)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{r['arm_slug'].replace('_', ' ')}  "
                    f"bb {int(r['bb_steps']) // 1000}k  {r['align_target']}"
                    for r in rows[::-1]])
for lab, r in zip(ax.get_yticklabels(), rows[::-1]):
    lab.set_color(C_40K if int(r["bb_steps"]) == CONTROLLED_BB else INK)
ax.set_xlabel("Aggregate GM-Relative MASE, 97 GIFT-Eval B4 configs, "
              "horizon 16  (lower is better)")
at40 = [float(r["range"]) for r in rows
        if int(r["bb_steps"]) == CONTROLLED_BB]
ax.set_title("Same frozen backbone, same head budget, different head seed"
             "   —   teal = backbone 40k, where every controlled delta lives"
             f"\nthe four 40k ranges span {min(at40):.4f} to {max(at40):.4f}, "
             f"a factor of {max(at40) / min(at40):.0f}: no one number is the "
             "bar", fontsize=11)
ax.grid(True, axis="x", color=GRID, alpha=0.6)
lo = min(float(v) for r in rows for v in r["values"].split())
hi = max(float(v) for r in rows for v in r["values"].split())
ax.set_xlim(lo - 0.04, hi + 0.62)
fig.tight_layout()
out = HERE / "head_seed_spread.png"
fig.savefig(out)
print(f"wrote {out}  ({len(rows)} cells)")
