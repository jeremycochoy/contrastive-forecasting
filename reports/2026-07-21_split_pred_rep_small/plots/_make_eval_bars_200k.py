"""Bar plot of GM-Relative MASE at backbone step 200k, head 30k steps.

Only the 10 cells that improved from backbone 40k to 100k were extended to
200k, so this ranking covers those 10 — it is not the full 30-cell field the
40k and 100k rankings show. Each bar also carries the cell's 100k value, so
the extension's effect is readable without switching figures.
"""
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
SEED_NOISE = 0.01

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

ax.bar(xs, ys, color=colours, yerr=SEED_NOISE, capsize=6, width=0.62,
       error_kw={"ecolor": INK, "elinewidth": 1.2})
# Where the cell sat at 100k, so each bar shows what the extension did.
ax.scatter(xs, prev, marker="_", s=420, color=INK, linewidths=1.8, zorder=4,
           label="value at backbone 100k")

for x, v, p in zip(xs, ys, prev):
    # Value inside the bar, change above whichever of the two marks is higher,
    # so neither collides with the bar fill or the 100k tick.
    # When the cell worsened the 100k tick sits inside the bar, so drop the
    # value label below it rather than printing the two on top of each other.
    v_y = (min(v, p) - 0.012) if p < v else (v - 0.012)
    ax.text(x, v_y, f"{v:.4f}", ha="center", va="top", fontsize=8.5,
            color="white", weight="bold")
    d = v - p
    ax.text(x, max(v, p) + 0.012, f"{d:+.3f}", ha="center", va="bottom",
            fontsize=8.5, color="#2e8b57" if d < 0 else "#c04040", weight="bold")

ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("Aggregate GM-Relative MASE  (lower is better)")
ax.grid(True, axis="y", color=GRID, alpha=0.6)
ax.set_ylim(min(min(ys), min(prev)) - 0.06, max(max(ys), max(prev)) + 0.075)
ax.legend(loc="upper left", fontsize=9, frameon=False)
n_down = sum(1 for v, p, _a, _v in rows if v < p)
ax.set_title(
    f"GM-Relative MASE at backbone step 200k, head 30k steps, GIFT-Eval B4\n"
    f"only the {len(rows)} cells extended past 100k — {n_down} improved again, "
    f"{len(rows) - n_down} worsened\n"
    f"seasonal-naive parity (1.0) is below the axis; no cell reaches it",
    fontsize=10.5)
fig.tight_layout()
out = HERE / "eval_2L_gm_mase_bars_200k.png"
fig.savefig(out)
print(f"wrote {out}  ({len(rows)} cells, {n_down} improved again)")
