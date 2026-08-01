"""GM-MASE for every cell across backbone horizons, one combined chart.

All 30 cells have backbone 40k and 100k; the 10 that improved over that stretch
were extended to 200k.

The right-hand labels carry only the cell name, its last value and its change.
The loss recipe and the knobs each setting turns are decoded once in the two
legends instead of being repeated in all 30 labels — spelled out per label they
took more width than the curves themselves.
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

X = [0, 1, 2]
YMIN, YMAX = 1.13, 1.82
SEED_NOISE = 0.01
LABEL_X = 2.04          # just outside the axes; the figure margin holds the text

def spread(anchors, gap, lo, hi):
    order = sorted(range(len(anchors)), key=lambda i: anchors[i])
    pos = list(anchors); prev = lo - gap
    for i in order:
        pos[i] = max(pos[i], prev + gap); prev = pos[i]
    prev = hi + gap
    for i in reversed(order):
        pos[i] = min(pos[i], prev - gap); prev = pos[i]
    return pos

cells = C.all_cells()
best_val = min(v for _a, _v, vals in cells for v in vals if v is not None)

fig, ax = plt.subplots(figsize=(15.5, 10))
# Axes take the full left portion; labels live in the right margin.
fig.subplots_adjust(left=0.055, right=0.775, top=0.90, bottom=0.155)

anchors, drawn = [], []
for arm, var, vals in cells:
    colour, st = C.ARM_COLOR[arm], C.VAR_STYLE[var]
    pts = [(x, min(v, YMAX)) for x, v in zip(X, vals) if v is not None]
    if len(pts) > 1:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour,
                ls=st["ls"], lw=st["lw"], marker=st["marker"],
                markersize=st["ms"], markeredgewidth=0, alpha=0.92, zorder=3)
    for x, v in zip(X, vals):
        if v is not None and v > YMAX:
            ax.annotate(f"{v:.2f}↑", xy=(x, YMAX - 0.006), fontsize=8,
                        color=colour, ha="center", va="top", zorder=5)
    anchors.append(pts[-1][1]); drawn.append((arm, var, vals, pts[-1]))

ys = spread(anchors, gap=(YMAX - YMIN) / (len(anchors) + 1),
            lo=YMIN + 0.005, hi=YMAX - 0.005)
for (arm, var, vals, (last_x, last_v)), y_lab in zip(drawn, ys):
    colour = C.ARM_COLOR[arm]
    have = [v for v in vals if v is not None]
    delta = f"  {have[-1] - have[-2]:+.3f}" if len(have) > 1 else ""
    tag = "  ←200k" if last_x == 2 else ""
    ax.plot([last_x + 0.02, LABEL_X - 0.02], [last_v, y_lab], color=colour,
            lw=0.7, alpha=0.5, zorder=1, clip_on=False)
    ax.text(LABEL_X, y_lab, f"{C.label(arm, var)}  {have[-1]:.3f}{delta}{tag}",
            color=colour, fontsize=8.5, va="center", ha="left", clip_on=False)

ax.axhspan(best_val - SEED_NOISE, best_val + SEED_NOISE,
           color=MUTED, alpha=0.13, zorder=0)
ax.set_xticks(X)
ax.set_xticklabels(["backbone 40k\nhead 15k", "backbone 100k\nhead 30k",
                    "backbone 200k\nhead 30k"], fontsize=10)
ax.set_xlim(-0.04, 2.02)
ax.set_ylim(YMIN, YMAX)
ax.set_ylabel("Aggregate GM-Relative MASE  (lower is better)")
ax.grid(True, axis="y", color=GRID, alpha=0.7)

from matplotlib.lines import Line2D
recipe_handles = [Line2D([], [], color=C.ARM_COLOR[a], lw=2.4,
                         label=f"{a}  —  {C.ARM_LOSS[a]}") for a in C.ARMS]
setting_handles = [Line2D([], [], color=INK, ls=C.VAR_STYLE[v]["ls"],
                          marker=C.VAR_STYLE[v]["marker"],
                          markersize=C.VAR_STYLE[v]["ms"], markeredgewidth=0,
                          lw=C.VAR_STYLE[v]["lw"],
                          label=f"{C.VAR_SHORT[v]}  ({C.variant_knobs('arm1', v)})")
                   for v in C.VARIANTS]
setting_handles.append(Line2D([], [], color=MUTED, lw=7, alpha=0.35,
                              label=f"±{SEED_NOISE} band around the best cell ({best_val:.3f})"))
leg1 = fig.legend(handles=recipe_handles, loc="lower left", ncol=3, fontsize=8.5,
                  frameon=False, title="loss recipe (colour)",
                  bbox_to_anchor=(0.04, 0.004))
leg1._legend_box.align = "left"
leg2 = fig.legend(handles=setting_handles, loc="lower left", ncol=2, fontsize=8.5,
                  frameon=False, title="setting (line style)",
                  bbox_to_anchor=(0.60, 0.004))
leg2._legend_box.align = "left"

n200 = sum(1 for _a, _v, vals in cells if vals[2] is not None)
down = sum(1 for _a, _v, vals in cells
           if vals[2] is not None and vals[1] is not None and vals[2] < vals[1])
ax.set_title("GM-Relative MASE across backbone horizons — all 30 cells\n"
             f"the {n200} that improved over 40k→100k were extended to 200k, of which "
             f"{down} improved again;  seasonal-naive parity (1.0) is below the axis — "
             "no cell reaches it", fontsize=11.5, pad=12)
out = HERE / "eval_2L_gm_mase_progression.png"
fig.savefig(out)
print(f"wrote {out}  ({len(cells)} cells, {n200} at 200k, {down} improved again)")
