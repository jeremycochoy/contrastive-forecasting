"""How far one cell moves under nothing but a head seed.

Four frozen backbones, the quantile head retrained under two extra seeds,
the full 97-config eval re-run each time. Each row is one cell; the dots are
its per-seed GM-Relative MASE and the bar is the range they span. A
teacher-vs-student difference smaller than this range is not separable from
head-seed noise.

Reads `results/seed_spread.csv` (written by
`experiments/2026-08-01_lalign_teacher/scripts/seed_spread.py`).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SRC = HERE.parent / "results" / "seed_spread.csv"
PENDING = [("arm6_v2", 100000, 3)]  # (arm, bb step, seeds planned)

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

with open(SRC, newline="") as fh:
    rows = list(csv.DictReader(fh))
rows.sort(key=lambda r: -float(r["range"]))

fig, ax = plt.subplots(figsize=(13.5, 3.6))
for y, r in enumerate(rows[::-1]):
    vals = [float(v) for v in r["values"].split()]
    ax.plot([min(vals), max(vals)], [y, y], color="#8b1e8b", lw=6, alpha=0.3,
            solid_capstyle="butt")
    ax.plot(vals, [y] * len(vals), "o", color="#8b1e8b", ms=7)
    n_want = next((p[2] for p in PENDING
                   if p[0] == r["arm_slug"]
                   and str(p[1]) == r["bb_steps"]), len(vals))
    tag = "" if n_want == len(vals) else f"  ({len(vals)}/{n_want} seeds)"
    ax.text(max(vals) + 0.012, y, f"range {float(r['range']):.4f}"
            f"  ({float(r['range_rel']) * 100:.1f}% of the cell's lowest seed){tag}",
            va="center", fontsize=9)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{r['arm_slug'].replace('_', ' ')}  "
                    f"bb {int(r['bb_steps']) // 1000}k"
                    for r in rows[::-1]])
ax.set_xlabel("Aggregate GM-Relative MASE, 97 GIFT-Eval B4 configs, "
              "horizon 16  (lower is better)")
ax.set_title("Same frozen backbone, same head budget, different head seed",
             fontsize=11)
ax.grid(True, axis="x", color=GRID, alpha=0.6)
ax.set_xlim(1.13, 2.75)
fig.tight_layout()
out = HERE / "head_seed_spread.png"
fig.savefig(out)
print(f"wrote {out}  ({len(rows)} cells)")
