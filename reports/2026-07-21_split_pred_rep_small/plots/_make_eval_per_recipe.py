"""GM-MASE across backbone horizons, one panel per loss recipe.

Companion to the combined chart in _make_eval_progression.py: same data, split
by recipe so each recipe's five settings can be read without tracing a line
across 30 others.

All 30 cells have backbone 40k and 100k; the 10 that improved over that stretch
were extended to 200k. Splitting by recipe keeps every label beside its own line
inside its own panel — a single combined panel needs 30 leader lines to the
right margin, which is what made the earlier version hard to read.
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

X = [0, 1, 2]           # 40k, 100k, 200k
YMIN, YMAX = 1.13, 1.80  # arm1 combab's 3.13 at 40k is annotated, not drawn
SEED_NOISE = 0.01

cells = C.all_cells()
by_arm = {a: [(v, vals) for (aa, v, vals) in cells if aa == a] for a in C.ARMS}

def spread(anchors, gap, lo, hi):
    order = sorted(range(len(anchors)), key=lambda i: anchors[i])
    pos = list(anchors); prev = lo - gap
    for i in order:
        pos[i] = max(pos[i], prev + gap); prev = pos[i]
    prev = hi + gap
    for i in reversed(order):
        pos[i] = min(pos[i], prev - gap); prev = pos[i]
    return pos

fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6), sharex=True, sharey=True)
best_val = min(v for _a, _v, vals in cells for v in vals if v is not None)

for ax, arm in zip(axes.flat, C.ARMS):
    colour = C.ARM_COLOR[arm]
    # Every other recipe in grey, so each panel keeps the full context.
    for other, rows in by_arm.items():
        if other == arm: continue
        for var, vals in rows:
            pts = [(x, min(v, YMAX)) for x, v in zip(X, vals) if v is not None]
            if len(pts) > 1:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color="#d8d6cf", lw=0.9, zorder=1)

    anchors, drawn = [], []
    for var, vals in by_arm[arm]:
        pts = [(x, min(v, YMAX)) for x, v in zip(X, vals) if v is not None]
        st = C.VAR_STYLE[var]
        if len(pts) > 1:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour,
                    ls=st["ls"], lw=st["lw"], marker=st["marker"],
                    markersize=st["ms"], markeredgewidth=0, zorder=4)
        for x, v in zip(X, vals):
            if v is not None and v > YMAX:
                ax.annotate(f"{v:.2f}↑", xy=(x, YMAX - 0.006), fontsize=7.5,
                            color=colour, ha="center", va="top", zorder=5)
        last_x, last_v = pts[-1]
        anchors.append(last_v); drawn.append((var, vals, last_x, last_v))

    ys = spread(anchors, gap=(YMAX - YMIN) / 8.0, lo=YMIN + 0.02, hi=YMAX - 0.02)
    for (var, vals, last_x, last_v), y_lab in zip(drawn, ys):
        have = [v for v in vals if v is not None]
        delta = f"  {have[-1] - have[-2]:+.3f}" if len(have) > 1 else ""
        tag = "←200k" if last_x == 2 else ""
        ax.plot([last_x + 0.06, 2.28], [last_v, y_lab], color=colour,
                lw=0.6, alpha=0.5, zorder=2)
        ax.text(2.33, y_lab, f"{C.VAR_SHORT[var]} {have[-1]:.3f}{delta} {tag}",
                color=colour, fontsize=8, va="center", ha="left")

    ax.axhspan(best_val - SEED_NOISE, best_val + SEED_NOISE,
               color=MUTED, alpha=0.13, zorder=0)
    ax.set_title(f"{arm}   {C.ARM_LOSS[arm]}", fontsize=10.5, color=colour, pad=6)
    ax.set_xlim(-0.12, 3.55)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(X)
    ax.set_xticklabels(["bb 40k\nhead 15k", "bb 100k\nhead 30k", "bb 200k\nhead 30k"],
                       fontsize=8.5)
    ax.grid(True, axis="y", color=GRID, alpha=0.7)

for ax in axes[:, 0]:
    ax.set_ylabel("GM-Relative MASE  (lower is better)", fontsize=9.5)

from matplotlib.lines import Line2D
handles = [Line2D([], [], color=INK, ls=C.VAR_STYLE[v]["ls"],
                  marker=C.VAR_STYLE[v]["marker"], markersize=C.VAR_STYLE[v]["ms"],
                  markeredgewidth=0, lw=C.VAR_STYLE[v]["lw"],
                  label=f"{C.VAR_SHORT[v]}  ({C.variant_knobs('arm1', v)})")
           for v in C.VARIANTS]
handles += [Line2D([], [], color="#d8d6cf", lw=1.4, label="the other five recipes"),
            Line2D([], [], color=MUTED, lw=6, alpha=0.35,
                   label=f"±{SEED_NOISE} band around the best cell ({best_val:.3f})")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.005))

n200 = sum(1 for _a, _v, vals in cells if vals[2] is not None)
down = sum(1 for _a, _v, vals in cells
           if vals[2] is not None and vals[1] is not None and vals[2] < vals[1])
fig.suptitle("GM-Relative MASE across backbone horizons, one panel per loss recipe\n"
             f"all 30 cells at 40k and 100k; the {n200} that improved were extended to 200k, "
             f"of which {down} improved again;  seasonal-naive parity (1.0) is below the axis — "
             f"no cell reaches it  (sigreg_e=0 also applies to arm1/3/4 combab)",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0.075, 1, 0.93])
out = HERE / "eval_2L_gm_mase_per_recipe.png"
fig.savefig(out)
print(f"wrote {out}  ({len(cells)} cells, {n200} at 200k, {down} improved again)")
