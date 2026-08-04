"""Bar plot of GM-Relative MASE at backbone step 200k, head 30k steps.

Only the cells that improved from backbone 40k to 100k were extended to 200k,
so this ranking is not the full 30-cell field. Each bar also carries the cell's
100k value, so the extension's effect is readable without switching figures.

Error bars are the measured head-seed range from `results/seed_spread.csv`
(min to max over the replicate seeds of that cell). Cells without a replicate
carry no bar.
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _cells as C

HERE = Path(__file__).parent
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

SPREAD = {}
with open(Path(__file__).parent.parent / "results" / "seed_spread.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        vals = [float(v) for v in r["values"].split()]
        SPREAD[(r["arm_slug"], int(r["bb_steps"]))] = (min(vals), max(vals))

rows = []
for arm, var, vals in C.all_cells():
    if vals[2] is None: continue
    rows.append((vals[2], vals[1], arm, var))
rows.sort(key=lambda r: r[0])

fig, ax = plt.subplots(figsize=(12, 6))
xs = list(range(len(rows)))
ys = [r[0] for r in rows]
prev = [r[1] for r in rows]
colours = [C.ARM_COLOR[r[2]] for r in rows]
labels = [C.label(r[2], r[3]) for r in rows]

slugs = [f"{r[2]}{r[3]}" for r in rows]
err_lo, err_hi, n_rep = [], [], 0
for x, v, slug in zip(xs, ys, slugs):
    lo_hi = SPREAD.get((slug, 200000))
    if lo_hi is None:
        err_lo.append(0.0); err_hi.append(0.0)
    else:
        n_rep += 1
        err_lo.append(v - lo_hi[0]); err_hi.append(lo_hi[1] - v)
# Dot and range, not bars: the values span 1.18-1.89 on a metric whose
# reference is 1.0, so a bar drawn from zero would compress every difference.
ax.vlines(xs, [min(v, p) for v, p in zip(ys, prev)],
          [max(v, p) for v, p in zip(ys, prev)], color=GRID, lw=2.4, zorder=1)
for x, v, lo, hi in zip(xs, ys, err_lo, err_hi):
    if lo or hi:
        ax.plot([x, x], [v - lo, v + hi], color=INK, lw=1.3, zorder=3,
                solid_capstyle="butt")
        for e in (v - lo, v + hi):
            ax.plot([x - 0.12, x + 0.12], [e, e], color=INK, lw=1.3, zorder=3)
ax.scatter(xs, prev, marker="o", s=52, facecolor="white", edgecolor=MUTED,
           linewidths=1.4, zorder=4)
ax.scatter(xs, ys, marker="o", s=95, color=colours, zorder=5)

for x, v, p_, e_lo, e_hi in zip(xs, ys, prev, err_lo, err_hi):
    ax.text(x + 0.16, v, f"{v:.4f}", ha="left", va="center", fontsize=8.5,
            color=INK)
    d = v - p_
    ax.text(x, max(v + e_hi, p_) + 0.022, f"{d:+.3f}", ha="center",
            va="bottom", fontsize=8.5,
            color="#2e8b57" if d < 0 else "#c04040", weight="bold")

ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("Aggregate GM-Relative MASE  (lower is better)")
ax.grid(True, axis="y", color=GRID, alpha=0.6)
ax.axhline(1.0, color=MUTED, ls="--", lw=1.0, zorder=0)
ax.set_ylim(0.96, max(max(y + e for y, e in zip(ys, err_hi)), max(prev)) + 0.09)
# Legend keys encode fill only, never colour: the filled dots are coloured per
# recipe, so a coloured key would read as "only that recipe is 200k".
handles = [
    Line2D([], [], marker="o", ls="none", markersize=7, markerfacecolor="white",
           markeredgecolor=INK, markeredgewidth=1.4, label="hollow = backbone 100k"),
    Line2D([], [], marker="o", ls="none", markersize=9, markerfacecolor=INK,
           markeredgecolor=INK, label="filled = backbone 200k"),
    Line2D([], [], ls="none", marker="", label="colour = recipe"),
]
ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=False,
          handletextpad=0.8)
n_down = sum(1 for v, p, _a, _v in rows if v < p)
ax.set_title(
    f"GM-Relative MASE at backbone step 200k, head 30k steps, GIFT-Eval B4\n"
    f"filled = backbone 200k, hollow = 100k; whisker = head-seed range",
    fontsize=10.5)
fig.tight_layout()
out = HERE / "eval_2L_gm_mase_bars_200k.png"
fig.savefig(out)
print(f"wrote {out}  ({len(rows)} cells, {n_down} improved again)")
