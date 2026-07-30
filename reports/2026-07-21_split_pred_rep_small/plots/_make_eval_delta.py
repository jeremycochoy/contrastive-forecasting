"""Change in GM-Relative MASE between backbone 40k and backbone 100k, one
horizontal bar per cell, sorted. Negative (green, left) = better at 100k."""
from pathlib import Path
import re
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
REP = HERE.parent / "results" / "eval_gm_mase"
EXP = HERE.parent.parent.parent / "experiments" / "2026-07-21_split_pred_rep_small" / "eval_gm_mase"

ARMS = ["arm1", "arm3", "arm4", "arm5", "arm6_v2", "bimoco"]
VARIANTS = [("", "base"), ("_tr1", "tr1"), ("_nse", "nse"),
            ("_ncpc", "ncpc"), ("_combab", "combab")]
BETTER, WORSE = "#2e8b57", "#c04040"
SEED_NOISE = 0.01

def read(slug, bb, hd):
    for p in (REP / f"{slug}_bb{bb}k_hd{hd}s_summary.txt",
              EXP / f"{slug}_bb{bb}k_hd{hd}s" / "summary.txt"):
        if not p.exists(): continue
        m = re.search(r"Aggregate.*\((\d+) configs\):\s+([0-9.]+)", p.read_text())
        if m: return float(m.group(2))
    return None

rows = []
for arm in ARMS:
    for var, short in VARIANTS:
        v40, v100 = read(f"{arm}{var}", 40, 15000), read(f"{arm}{var}", 100, 30000)
        if v40 is None or v100 is None: continue
        rows.append((f"{arm} {short}", v40, v100, v100 - v40))
rows.sort(key=lambda r: r[3])

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})
fig, ax = plt.subplots(figsize=(11, 9))
ys = list(range(len(rows)))
deltas = [r[3] for r in rows]
# arm1 combab starts from a 3.125 outlier at 40k; its -1.37 change would
# squash every other bar, so bars are clipped and the value is written out.
XLO, XHI = -0.42, 0.40
drawn = [max(min(d, XHI), XLO) for d in deltas]
colours = [BETTER if d < 0 else WORSE for d in deltas]
ax.barh(ys, drawn, color=colours, height=0.7)
ax.axvline(0, color=INK, lw=1.0)
ax.axvspan(-SEED_NOISE, SEED_NOISE, color=MUTED, alpha=0.18, zorder=0,
           label=f"+/-{SEED_NOISE} seed-noise band")

for y, (label, v40, v100, d) in zip(ys, rows):
    clipped = d < XLO or d > XHI
    x = max(min(d, XHI), XLO)
    if clipped:                      # write inside the bar, it runs off-axis
        ax.text(x + 0.008, y, f"{d:+.3f}  (bar clipped)", va="center", ha="left",
                fontsize=8, color="white", weight="bold")
    else:
        off = 0.006 if d >= 0 else -0.006
        ax.text(x + off, y, f"{d:+.3f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=8)

ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.set_ylim(-0.8, len(rows) - 0.2)
ax.set_xlabel("change in GM-Relative MASE, backbone 100k minus backbone 40k")
ax.set_xlim(XLO, XHI)
ax.grid(True, axis="x", color=GRID, alpha=0.6)
ax.legend(loc="lower right", fontsize=9, frameon=False)
n_better = sum(1 for d in deltas if d < 0)
ax.set_title(f"Change in GM-Relative MASE from 40k to 100k backbone\n"
             f"{n_better} of {len(rows)} cells improve (green, left); "
             f"{len(rows) - n_better} worsen (red, right)", fontsize=11)
fig.tight_layout()
out = HERE / "eval_2L_gm_mase_delta.png"
fig.savefig(out)
print(f"wrote {out}  ({len(rows)} cells, {n_better} improving)")
