"""Bar plot of 2L GM-Relative MASE across the 8 eval candidates, #379.

Error bars: ±0.01 GM-Rel MASE = seed-noise band estimated from the
2026-05-08 tau-sweep paired re-runs (referenced in the LeJEPA sigreg-tau
report, annex F). Applied uniformly here.
"""
from pathlib import Path
import re
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
EVAL_ROOT = HERE.parent.parent.parent / "experiments" / "2026-07-21_split_pred_rep_small" / "eval_gm_mase"

# Order = ranking order used at pick time (best-1ff → best-trajectory → least-shaky).
ARMS = [
    ("arm6_v2 tr1",     "arm6_v2_tr1",     "#dcbb60"),
    ("bimoco tr1",      "bimoco_tr1",      "#66c4c4"),
    ("arm3 tr1",        "arm3_tr1",        "#f4a680"),
    ("bimoco combab",   "bimoco_combab",   "#00a3a3"),
    ("arm5 tr1",        "arm5_tr1",        "#c98cc9"),
    ("arm5 nse",        "arm5_nse",        "#c58fc5"),
    ("arm5 ncpc",       "arm5_ncpc",       "#8b1e8b"),
    ("arm6_v2 combab",  "arm6_v2_combab",  "#b8860b"),
    ("arm3 combab",     "arm3_combab",     "#eb6834"),
    ("arm4 tr1",        "arm4_tr1",        "#7fc17f"),
    ("arm4 nse",        "arm4_nse",        "#008300"),
]
SEED_NOISE = 0.01

def read_agg(arm_slug):
    p = EVAL_ROOT / f"{arm_slug}_bb40k_hd15000s" / "summary.txt"
    if not p.exists(): return None
    m = re.search(r"([0-9]+\.[0-9]+)", p.read_text())
    return float(m.group(1)) if m else None

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})
fig, ax = plt.subplots(figsize=(11, 5.5))
xs, ys, cs, labs = [], [], [], []
for i, (label, slug, colour) in enumerate(ARMS):
    v = read_agg(slug)
    xs.append(i); labs.append(label); cs.append(colour)
    ys.append(v if v is not None else float('nan'))

bars = ax.bar(xs, ys, color=cs, yerr=SEED_NOISE, capsize=6,
              error_kw={"ecolor": INK, "elinewidth": 1.2})
for x, v in zip(xs, ys):
    if v == v:
        ax.text(x, v + 0.015, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(x, 0.02, "queued", ha="center", va="bottom", fontsize=8,
                color=MUTED, style="italic")
ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=30, ha="right")
ax.set_ylabel("Aggregate GM-Relative MASE  (lower is better)")
ax.set_title(
    "2L GM-Relative MASE at backbone step 40k, head 15k steps, GIFT-Eval B4 full-97\n"
    f"error bars: ±{SEED_NOISE} seed-noise band (from 2026-05-08 τ-sweep paired reruns)",
    fontsize=10)
ax.grid(True, axis="y", color=GRID, alpha=0.6)
finite = [v for v in ys if v == v]
if finite:
    ymin = min(finite) - 3 * SEED_NOISE   # ~3 seed-noise bands below min
    ymax = max(finite) + 3 * SEED_NOISE
    ax.set_ylim(ymin, ymax)
fig.tight_layout()
out = HERE / "eval_2L_gm_mase_bars.png"
fig.savefig(out)
print(f"wrote {out}")
