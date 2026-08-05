"""Does latent movement track the GM-MASE improvement of the extended cells?

Three panels, left to right:

  1. `h_t` drift between adjacent checkpoints for the cells whose
     GM-Relative MASE fell from backbone 100k to 200k, plus `arm6_v2 combab`,
     the lowest cell overall.
  2. The GM-Relative MASE of those same cells on the same log-step axis, so the
     two curves can be read against each other at a glance.
  3. The test the first two panels only suggest: mean `h_t` drift over the
     100k→200k stretch against the GM-MASE change over that same stretch, for
     every extended cell. Panels 1 and 2 show only cells selected for having
     improved, which cannot answer whether drift separates improvers from the
     rest; panel 3 includes the cells that worsened, so it can.

Drift values come from `results/latent_movement_pairs.csv`.
"""
from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt

import _cells as C

HERE = Path(__file__).parent
PAIRS = HERE.parent / "results" / "latent_movement_pairs.csv"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BETTER, WORSE = "#2e8b57", "#c04040"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

# arm1 contributes three of the five focus cells; the shared recipe hue makes
# those three lines one blur, so panels 1 and 2 key colour by cell, on a
# lightness ladder wide enough to tell them apart.
CELL_COLOR = {("arm1", "_nse"): "#9ecae1", ("arm1", "_ncpc"): "#2a78d6",
              ("arm1", "_combab"): "#08306b"}


def cell_color(arm, var):
    return CELL_COLOR.get((arm, var), C.ARM_COLOR[arm])


drift = {}
with open(PAIRS) as fh:
    for row in csv.DictReader(fh):
        drift.setdefault(row["arm_slug"], []).append(
            (int(row["step_later"]), float(row["drift_h"]), float(row["drift_e"])))
for k in drift: drift[k].sort()

cells = C.all_cells()
extended = [(arm, var, vals) for arm, var, vals in cells if vals[2] is not None]
improving = [(vals[2], arm, var, vals) for arm, var, vals in extended if vals[2] < vals[1]]
improving.sort(key=lambda r: r[0])
best_arm, best_var = min(((vals[2], arm, var) for arm, var, vals in extended))[1:]
focus = list(improving)
if not any(a == best_arm and v == best_var for _t, a, v, _vals in focus):
    bvals = next(vals for a, v, vals in extended if a == best_arm and v == best_var)
    focus.insert(0, (bvals[2], best_arm, best_var, bvals))

def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]: j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1): r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")

fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.4))

# --- panel 1: drift trajectories ---------------------------------------------
ax = axes[0]
for _t, arm, var, _vals in focus:
    pts = drift.get(f"{arm}{var}", [])
    if not pts: continue
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=cell_color(arm, var),
            ls=C.VAR_STYLE[var]["ls"], lw=2.0, marker=C.VAR_STYLE[var]["marker"],
            markersize=5, markeredgewidth=0, label=C.label(arm, var))
ax.axvspan(100_000, 200_000, color=MUTED, alpha=0.12, zorder=0)
ax.set_xscale("log")
ax.set_xlabel("training step of the later checkpoint (log)")
ax.set_ylabel("1 − cos(h_prev, h_next)   (higher = more movement)")
ax.set_title(f"1.  h_t drift between adjacent checkpoints\n"
             f"the {len(focus)} of 8 extended cells that improved over 100k→200k;"
             f"  shaded: the extension", fontsize=10.5)
ax.grid(True, color=GRID, alpha=0.7)
ax.legend(fontsize=8, frameon=False, loc="lower left")

# --- panel 2: GM-MASE on the same axis ---------------------------------------
ax = axes[1]
STEPS = [40_000, 100_000, 200_000]
for _t, arm, var, vals in focus:
    xs = [s for s, v in zip(STEPS, vals) if v is not None]
    ys = [v for v in vals if v is not None]
    ax.plot(xs, ys, color=cell_color(arm, var), ls=C.VAR_STYLE[var]["ls"], lw=2.0,
            marker=C.VAR_STYLE[var]["marker"], markersize=7, markeredgewidth=0,
            label=C.label(arm, var))
ax.axvspan(100_000, 200_000, color=MUTED, alpha=0.12, zorder=0)
ax.set_xscale("log")
ax.set_xlabel("backbone step (log)")
ax.set_ylabel("Aggregate GM-Relative MASE  (lower is better)")
ax.set_title(f"2.  GM-Relative MASE, the same {len(focus)} cells, same axis",
             fontsize=10.5)
ax.set_ylim(1.13, 1.85)
ax.grid(True, color=GRID, alpha=0.7)

# --- panel 3: the actual test, all 10 extended cells -------------------------
ax = axes[2]
xs, ys, cols, labs = [], [], [], []
for arm, var, vals in extended:
    pts = [p for p in drift.get(f"{arm}{var}", []) if p[0] > 100_000]
    if not pts: continue
    mean_late = sum(p[1] for p in pts) / len(pts)
    d = vals[2] - vals[1]
    xs.append(mean_late); ys.append(d)
    cols.append(BETTER if d < 0 else WORSE); labs.append(C.label(arm, var))
ax.axhline(0, color=INK, lw=1.0)
ax.scatter(xs, ys, c=cols, s=90, zorder=3)
# arm6_v2 combab and arm6_v2 ncpc land on nearly the same point; alternate the
# annotation side so neither is written over the other.
order = sorted(range(len(xs)), key=lambda i: (xs[i], ys[i]))
side = {}
for rank_i, i in enumerate(order):
    close = any(abs(xs[i] - xs[j]) < 0.02 and abs(ys[i] - ys[j]) < 0.02
                for j in order[:rank_i])
    side[i] = -1 if close else 1
for i, (x, y, lab) in enumerate(zip(xs, ys, labs)):
    dx, ha = (6, "left") if side[i] > 0 else (-6, "right")
    ax.annotate(lab, xy=(x, y), xytext=(dx, 4), textcoords="offset points",
                fontsize=8, ha=ha)
rho = spearman(xs, ys)
from scipy import stats as _stats
_p = _stats.spearmanr(xs, ys).pvalue
ax.text(0.03, 0.05, f"Spearman rho = {rho:+.2f},  n = {len(xs)},  p = {_p:.2f}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, color=INK)
ax.set_xlabel("mean h_t drift over the 100k→200k checkpoints")
ax.set_ylabel("GM-Relative MASE change, 100k → 200k")
ax.set_title(f"3.  Late h_t drift against the 100k→200k GM-MASE change\n"
             f"all {len(xs)} extended cells;  green = improved, red = worsened",
             fontsize=10.5)
ax.grid(True, color=GRID, alpha=0.7)

fig.suptitle("Latent movement against the GM-MASE improvement of the extended cells", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = HERE / "drift_vs_improvement.png"
fig.savefig(out)
print(f"wrote {out}")
print(f"  focus cells: " + ", ".join(C.label(a, v) for _t, a, v, _vals in focus))
from scipy import stats as _st
_res = _st.spearmanr(xs, ys)
print(f"  panel 3: n={len(xs)}  Spearman rho={rho:+.3f}  p={_res.pvalue:.3f}")
for x, y, lab in sorted(zip(xs, ys, labs), key=lambda r: r[1]):
    print(f"    {lab:<17} late_drift={x:.3f}  dMASE={y:+.4f}")
