"""Teacher encoder minus student encoder, per cell per stop.

Both heads of a stop are trained and evaluated on their own encoder, so the
difference is the only paired teacher-vs-student comparison the ladder gives.
The grey band is the largest head-seed range measured in this study, pooled
over both head budgets (experiments/.../scripts/noise_band.py). Pooling
matters: `results/seed_spread.csv` holds 30,000-step heads only, and these
differences include stops whose heads ran 15,000 steps.
"""
from pathlib import Path
import csv
import sys
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

sys.path.insert(0, str(EXP / "scripts"))
import noise_band  # noqa: E402

BAND = noise_band.pooled_band(str(EXP / "results" / "paired_delta.csv"))

cells = sorted({c for c, _s, _h in vals}, reverse=True)
STOPS = [(40000, "#2a78d6", "o", "bb40k"), (100000, "#eb6834", "s", "bb100k"),
         (200000, "#008300", "D", "bb200k")]

fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.axvspan(-BAND, BAND, color=MUTED, alpha=0.18, zorder=0,
           label=f"±{BAND:.4f}, {noise_band.LABEL}")
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
ax.legend(fontsize=8.5, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, -0.14), ncol=4)
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = HERE / "teacher_vs_student.png"
fig.savefig(out)
print("wrote", out, "band", BAND)
