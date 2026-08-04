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
ax.bar(xs, ys, color=colours, yerr=[err_lo, err_hi], capsize=6, width=0.62,
       error_kw={"ecolor": INK, "elinewidth": 1.2})
# Where the cell sat at 100k, so each bar shows what the extension did.
ax.scatter(xs, prev, marker="_", s=420, color=INK, linewidths=1.8, zorder=4,
           label="value at backbone 100k")

for x, v, p, e_hi in zip(xs, ys, prev, err_hi):
    # Value inside the bar, change above whichever of the two marks is higher,
    # so neither collides with the bar fill or the 100k tick.
    # When the cell worsened the 100k tick sits inside the bar, so drop the
    # value label below it rather than printing the two on top of each other.
    v_y = (min(v, p) - 0.012) if p < v else (v - 0.012)
    ax.text(x, v_y, f"{v:.4f}", ha="center", va="top", fontsize=8.5,
            color="white", weight="bold")
    d = v - p
    # Clear the error-bar cap as well as the bar top, else the cap strikes
    # through the change label on cells that carry replicate seeds.
    ax.text(x, max(v + e_hi, p) + 0.018, f"{d:+.3f}", ha="center", va="bottom",
            fontsize=8.5, color="#2e8b57" if d < 0 else "#c04040", weight="bold")

ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("Aggregate GM-Relative MASE  (lower is better)")
ax.grid(True, axis="y", color=GRID, alpha=0.6)
ax.set_ylim(min(min(ys), min(prev)) - 0.06,
            max(max(y + e for y, e in zip(ys, err_hi)), max(prev)) + 0.075)
ax.legend(loc="upper left", fontsize=9, frameon=False)
n_down = sum(1 for v, p, _a, _v in rows if v < p)
ax.set_title(
    f"GM-Relative MASE at backbone step 200k, head 30k steps, GIFT-Eval B4\n"
    f"{len(rows)} cells; error bars = measured head-seed range "
    f"({n_rep} of {len(rows)} cells have replicate seeds)",
    fontsize=10.5)
fig.tight_layout()
out = HERE / "eval_2L_gm_mase_bars_200k.png"
fig.savefig(out)
print(f"wrote {out}  ({len(rows)} cells, {n_down} improved again)")
