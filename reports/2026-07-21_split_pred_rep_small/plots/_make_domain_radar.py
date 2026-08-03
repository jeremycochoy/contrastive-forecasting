"""Per-domain GM-Relative MASE radars.

The headline number is one geometric mean over all 97 GIFT-Eval configs. These
radars split that same geometric mean by dataset domain, so a cell's strong and
weak domains are visible instead of averaged away.

Two panels:
  left  — the 5 lowest cells by their last evaluated backbone step
  right — every cell that improved from backbone 100k to 200k

The radial axis is log2(ratio): the plotted quantity is a ratio whose headline
value is a geometric mean, so equal multiplicative steps are equal distances.
On a linear axis one 4.0 outlier squashes every other cell into the centre.
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

def _lighten(hexcolour, amount):
    """Blend a hex colour toward white by `amount` in [0, 1]."""
    h = hexcolour.lstrip("#")
    rgb = [int(h[i:i+2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{int(c + (255 - c) * amount):02x}" for c in rgb)

cells = C.all_cells()

scored = []
for arm, var, vals in cells:
    i, v = C.last_horizon(vals)
    if v is not None:
        scored.append((v, arm, var, C.HORIZONS[i][0], C.HORIZONS[i][1]))
scored.sort(key=lambda r: r[0])
topn = scored[:5]

improved = []
for arm, var, vals in cells:
    if vals[1] is not None and vals[2] is not None and vals[2] < vals[1]:
        improved.append((vals[2], arm, var, 200, 30000))
improved.sort(key=lambda r: r[0])

def series(arm, var, bb, hd):
    return C.per_domain_relative_mase(f"{arm}{var}", bb, hd)

_, ref_counts = series("arm6_v2", "_combab", 200, 30000)
DOMAINS = sorted(ref_counts, key=lambda d: -ref_counts[d])
N = len(DOMAINS)
ANG = [n / N * 2 * math.pi for n in range(N)] + [0.0]

allv = []
for _v, arm, var, bb, hd in topn + improved:
    d, _c = series(arm, var, bb, hd)
    if d: allv += [d[k] for k in DOMAINS if k in d]
LO, HI = min(allv), max(allv)
# Ratio ticks, chosen to bracket the data with round multiplicative steps.
TICKS = [t for t in (0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.8, 4.0)
         if LO / 1.08 <= t <= HI * 1.08]
if TICKS[0] > LO: TICKS.insert(0, round(LO, 2))
if TICKS[-1] < HI: TICKS.append(round(HI, 2))
r = math.log2
RMIN, RMAX = r(TICKS[0]) - 0.03, r(TICKS[-1]) + 0.03

def draw(ax, rows, title):
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(RMIN, RMAX)
    ax.set_xticks(ANG[:-1])
    ax.set_xticklabels([f"{d}\n({ref_counts[d]} cfg)" for d in DOMAINS], fontsize=9)
    ax.set_yticks([r(t) for t in TICKS])
    ax.set_yticklabels([f"{t:g}" for t in TICKS], fontsize=7.5, color=MUTED)
    ax.grid(color=GRID, alpha=0.9)
    ax.set_rlabel_position(180 / N)

    # Seasonal-naive parity. Inside this ring the model beats seasonal-naive.
    ax.plot(ANG, [r(1.0)] * (N + 1), color="#c04040", lw=1.8, ls="--", zorder=3)
    ax.fill(ANG, [r(1.0)] * (N + 1), color="#c04040", alpha=0.06, zorder=0)

    handles = []
    # Two cells of the same arm share a colour; shade the later one lighter so
    # the pair is separable by more than line style alone.
    seen = {}
    for val, arm, var, bb, hd in rows:
        d, _c = series(arm, var, bb, hd)
        if not d: continue
        vals = [r(d[k]) for k in DOMAINS]
        vals += vals[:1]
        style = C.VAR_STYLE[var]
        nth = seen.get(arm, 0); seen[arm] = nth + 1
        colour = C.ARM_COLOR[arm] if nth == 0 else _lighten(C.ARM_COLOR[arm], min(0.22 + 0.26 * nth, 0.62))
        ln, = ax.plot(ANG, vals, color=colour, lw=2.0, ls=style["ls"],
                      marker=style["marker"], markersize=style["ms"],
                      markeredgewidth=0, zorder=4,
                      label=f"{C.label(arm, var)} @ bb{bb}k — all-config {val:.3f}")
        handles.append(ln)
        ax.fill(ANG, vals, color=colour, alpha=0.05, zorder=1)
        # Call out any domain that lands outside the tick range's comfort zone.
        for ang, k in zip(ANG[:-1], DOMAINS):
            if d[k] >= 2.0:
                ax.annotate(f"{d[k]:.2f}", xy=(ang, r(d[k])), fontsize=7.5,
                            color=colour, ha="left", va="bottom", zorder=5)
    ax.set_title(title, fontsize=10.5, pad=18)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              fontsize=8.5, frameon=False, ncol=1)

fig, axes = plt.subplots(1, 2, figsize=(14.5, 9.4),
                         subplot_kw={"projection": "polar"})
draw(axes[0], topn, "5 lowest cells, each at its last evaluated backbone step")
draw(axes[1], improved, "Cells that improved from backbone 100k to 200k (shown at 200k)")
fig.suptitle("GM-Relative MASE per dataset domain — the headline geometric mean split by domain\n"
             "inside the red ring = better than seasonal-naive;  radial axis is log2(ratio)",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0.10, 1, 0.94])
out = HERE / "eval_domain_radar.png"
fig.savefig(out)
print(f"wrote {out}")
print("  panel 1:", ", ".join(f"{C.label(a,v)}@bb{bb}k" for _, a, v, bb, _ in topn))
print("  panel 2:", ", ".join(C.label(a, v) for _, a, v, _, _ in improved))
print(f"  ratio range {LO:.3f}–{HI:.3f}, ticks {TICKS}")
