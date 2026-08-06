"""Teacher encoder minus student encoder, per cell per stop.

Both heads of a stop are trained and evaluated on their own encoder, so the
difference is the only paired teacher-vs-student comparison the ladder gives.
The grey band is the largest head-seed range measured in this study
(results/seed_spread.csv).
"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
EXP = HERE.parent.parent.parent / "experiments" / "2026-08-04_ema_sched_ladder"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
                     "axes.edgecolor": MUTED, "xtick.color": INK, "ytick.color": INK})

vals = {}
with open(EXP / "results" / "ladder_all.csv") as f:
    for r in csv.DictReader(f):
        vals[(r["cell"], int(r["stop"]), r["head"])] = float(r["gm_rel_mase"])

BAND = max(float(r["range"]) for r in csv.DictReader(open(EXP / "results" / "seed_spread.csv")))

cells = sorted({c for c, _s, _h in vals}, reverse=True)
STOPS = [(40000, "#2a78d6", "o", "bb40k"), (100000, "#eb6834", "s", "bb100k"),
         (200000, "#008300", "D", "bb200k")]

fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.axvspan(-BAND, BAND, color=MUTED, alpha=0.18, zorder=0,
           label=f"±{BAND:.4f}, largest head-seed range measured here")
ax.axvline(0, color=INK, lw=1.2, zorder=2)
for stop, colour, marker, lab in STOPS:
    xs, ys = [], []
    for i, c in enumerate(cells):
        s, t = vals.get((c, stop, "student")), vals.get((c, stop, "teacher"))
        if s is None or t is None:
            continue
        xs.append(t - s)
        ys.append(i)
    ax.plot(xs, ys, marker, color=colour, ms=7, ls="none", zorder=3, label=lab)
ax.set_yticks(range(len(cells)))
ax.set_yticklabels(cells, fontsize=9)
ax.set_xlabel("teacher head − student head, GM-Relative MASE  (left of 0 = teacher lower)")
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
ax.set_title("Teacher encoder against student encoder, same stop, same backbone", fontsize=11)
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
fig.tight_layout()
out = HERE / "teacher_vs_student.png"
fig.savefig(out)
print("wrote", out, "band", BAND)
