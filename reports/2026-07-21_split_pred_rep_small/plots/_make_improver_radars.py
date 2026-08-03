"""Per-domain radar for each cell that kept improving to backbone 200k.

One panel per cell, each with its own radial scale so a cell's own shape across
horizons is readable — a shared scale would flatten the cells whose values move
by only a few hundredths. Every panel overlays the three horizons, so what the
extra training changed per domain is visible rather than only the all-config
total.

Panels: the 5 cells that improved from backbone 100k to 200k, plus
`arm6_v2 combab`, the lowest cell overall, which is flat over that stretch and
is here as the reference the others are chasing.
"""
from pathlib import Path
import math
import matplotlib.pyplot as plt

import _cells as C

HERE = Path(__file__).parent
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

# Horizon → (line style, alpha, marker) — later horizons drawn stronger.
H_STYLE = [
    {"ls": ":",  "lw": 1.5, "alpha": 0.75, "marker": "o", "ms": 4, "name": "bb 40k"},
    {"ls": "--", "lw": 1.8, "alpha": 0.85, "marker": "s", "ms": 4, "name": "bb 100k"},
    {"ls": "-",  "lw": 2.6, "alpha": 1.00, "marker": "P", "ms": 7, "name": "bb 200k"},
]

cells = C.all_cells()
improving = [(vals[2], arm, var, vals) for arm, var, vals in cells
             if vals[1] is not None and vals[2] is not None and vals[2] < vals[1]]
improving.sort(key=lambda r: r[0])
best = min(((vals[2] if vals[2] is not None else 9e9), arm, var, vals)
           for arm, var, vals in cells)
panels = list(improving)
if not any(a == best[1] and v == best[2] for _t, a, v, _vals in panels):
    panels.insert(0, best)          # reference first

_, ref_counts = C.per_domain_relative_mase("arm6_v2_combab", 200, 30000)
DOMAINS = sorted(ref_counts, key=lambda d: -ref_counts[d])
N = len(DOMAINS)
ANG = [n / N * 2 * math.pi for n in range(N)] + [0.0]
r = math.log2

ncol = 3
nrow = math.ceil(len(panels) / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(16, 5.9 * nrow),
                         subplot_kw={"projection": "polar"})
axes = axes.flat if hasattr(axes, "flat") else [axes]

for ax, (_tot, arm, var, vals) in zip(axes, panels):
    series = []
    for i, (bb, hd) in enumerate(C.HORIZONS):
        if vals[i] is None: continue
        d, _c = C.per_domain_relative_mase(f"{arm}{var}", bb, hd)
        if d: series.append((i, d, vals[i]))
    allv = [d[k] for _i, d, _t in series for k in DOMAINS]
    lo, hi = min(allv + [1.0]), max(allv)          # always include parity
    pad = (r(hi) - r(lo)) * 0.08 or 0.05
    ax.set_ylim(r(lo) - pad, r(hi) + pad)

    ticks, t = [], 0.5
    while t <= hi * 1.35:                          # round ratio ticks in range
        if lo / 1.12 <= t <= hi * 1.12: ticks.append(t)
        t *= 1.25 if t < 2 else 1.4
    if 1.0 not in ticks: ticks.append(1.0)
    ticks = sorted(set(round(x, 2) for x in ticks))
    ax.set_yticks([r(t) for t in ticks])
    ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=7, color=MUTED)

    ax.set_theta_offset(math.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(ANG[:-1])
    ax.set_xticklabels([f"{d}\n({ref_counts[d]})" for d in DOMAINS], fontsize=8)
    ax.grid(color=GRID, alpha=0.9)
    ax.set_rlabel_position(180 / N)
    ax.plot(ANG, [r(1.0)] * (N + 1), color="#c04040", lw=1.6, ls="--", zorder=3)

    colour = C.ARM_COLOR[arm]
    for i, d, tot in series:
        st = H_STYLE[i]
        vv = [r(d[k]) for k in DOMAINS]; vv += vv[:1]
        ax.plot(ANG, vv, color=colour, ls=st["ls"], lw=st["lw"], alpha=st["alpha"],
                marker=st["marker"], markersize=st["ms"], markeredgewidth=0,
                zorder=4, label=f"{st['name']}  ({tot:.3f})")
    ax.fill(ANG, [r(series[-1][1][k]) for k in DOMAINS] +
                 [r(series[-1][1][DOMAINS[0]])], color=colour, alpha=0.07, zorder=1)

    delta = vals[2] - vals[1]
    tag = "reference — flat over 100k→200k" if abs(delta) < 0.01 else f"{delta:+.3f} over 100k→200k"
    ax.set_title(f"{C.label(arm, var)}   {tag}", fontsize=10.5, color=colour, pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.14), fontsize=7.5, frameon=False)

for ax in list(axes)[len(panels):]:
    ax.set_visible(False)

fig.suptitle("Per-domain GM-Relative MASE across backbone horizons — the cells that kept improving to 200k, "
             "plus the best cell as reference\n"
             "each panel has its own radial scale (log2 ratio); red ring = seasonal-naive parity; "
             "domain config counts in brackets", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = HERE / "eval_domain_radar_improvers.png"
fig.savefig(out)
print(f"wrote {out}  ({len(panels)} panels: " +
      ", ".join(C.label(a, v) for _t, a, v, _vals in panels) + ")")
