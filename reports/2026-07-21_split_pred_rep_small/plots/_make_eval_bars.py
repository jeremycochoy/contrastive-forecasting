"""Bar plot of 2L GM-Relative MASE at backbone step 40k, all evaluated arms, #379.

Includes the original 11 cells + the 12 new cells from the 2026-07-27 vast batch
(plus arm3_ncpc/arm4_combab which finished only partial gift-eval on vast — those
land under a red hatched bar to flag config coverage <97).

Error bars: ±0.01 GM-Rel MASE = seed-noise band from the 2026-05-08 τ-sweep
paired re-runs (LeJEPA sigreg-tau report annex F).

Aggregate is read preferentially from the report-flat summary file
`results/eval_gm_mase/<slug>_bb40k_hd15000s_summary.txt` (populated by the
vast sync loop), falling back to the experiments-side
`<slug>_bb40k_hd15000s/summary.txt` for cells run locally on elisa."""
from pathlib import Path
import re
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
EVAL_ROOT_EXP = HERE.parent.parent.parent / "experiments" / "2026-07-21_split_pred_rep_small" / "eval_gm_mase"
EVAL_ROOT_REP = HERE.parent / "results" / "eval_gm_mase"

ARMS = [
    ("arm1 base",       "arm1",            "#7fb0e8"),
    ("arm1 tr1",        "arm1_tr1",        "#7fb0e8"),
    ("arm1 nse",        "arm1_nse",        "#a6c8ee"),
    ("arm1 ncpc",       "arm1_ncpc",       "#2a78d6"),
    ("arm1 combab",     "arm1_combab",     "#2a78d6"),
    ("arm3 base",       "arm3",            "#f4a680"),
    ("arm3 tr1",        "arm3_tr1",        "#f4a680"),
    ("arm3 nse",        "arm3_nse",        "#f5b39a"),
    ("arm3 ncpc",       "arm3_ncpc",       "#eb6834"),
    ("arm3 combab",     "arm3_combab",     "#eb6834"),
    ("arm4 base",       "arm4",            "#7fc17f"),
    ("arm4 tr1",        "arm4_tr1",        "#7fc17f"),
    ("arm4 nse",        "arm4_nse",        "#7fc17f"),
    ("arm4 ncpc",       "arm4_ncpc",       "#008300"),
    ("arm4 combab",     "arm4_combab",     "#008300"),
    ("arm5 base",       "arm5",            "#c98cc9"),
    ("arm5 tr1",        "arm5_tr1",        "#c98cc9"),
    ("arm5 nse",        "arm5_nse",        "#c58fc5"),
    ("arm5 ncpc",       "arm5_ncpc",       "#8b1e8b"),
    ("arm5 combab",     "arm5_combab",     "#8b1e8b"),
    ("arm6_v2 base",    "arm6_v2",         "#dcbb60"),
    ("arm6_v2 tr1",     "arm6_v2_tr1",     "#dcbb60"),
    ("arm6_v2 nse",     "arm6_v2_nse",     "#dcc385"),
    ("arm6_v2 ncpc",    "arm6_v2_ncpc",    "#b8860b"),
    ("arm6_v2 combab",  "arm6_v2_combab",  "#b8860b"),
    ("bimoco base",     "bimoco",          "#66c4c4"),
    ("bimoco tr1",      "bimoco_tr1",      "#66c4c4"),
    ("bimoco nse",      "bimoco_nse",      "#7fd1d1"),
    ("bimoco ncpc",     "bimoco_ncpc",     "#00a3a3"),
    ("bimoco combab",   "bimoco_combab",   "#00a3a3"),
]
SEED_NOISE = 0.01

def read_agg(arm_slug):
    """Return (gm_rel_mase, n_configs) or (None, None) if missing.

    Prefers the report-flat file (holds full-97 aggregates from vast + local),
    falls back to the experiments-side summary."""
    for p in (EVAL_ROOT_REP / f"{arm_slug}_bb40k_hd15000s_summary.txt",
              EVAL_ROOT_EXP / f"{arm_slug}_bb40k_hd15000s" / "summary.txt"):
        if not p.exists(): continue
        text = p.read_text()
        m_agg = re.search(r"Aggregate\s+GM-Relative\s+MASE\s+\((\d+)\s+configs\):\s+([0-9]+\.[0-9]+)", text)
        if m_agg:
            return float(m_agg.group(2)), int(m_agg.group(1))
    return None, None

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})
fig, ax = plt.subplots(figsize=(14, 5.5))
rows = []
for label, slug, colour in ARMS:
    v, n = read_agg(slug)
    if v is None: continue
    rows.append((label, colour, v, n))
rows.sort(key=lambda r: r[2])
xs = list(range(len(rows)))
labs = [r[0] for r in rows]
cs = [r[1] for r in rows]
ys = [r[2] for r in rows]
ns = [r[3] for r in rows]
hatches = ["///" if n < 97 else None for n in ns]

YMAX = 1.75  # truncate — outliers >YMAX are marked with an arrow + value
bars = ax.bar(xs, ys, color=cs, yerr=SEED_NOISE, capsize=6,
              edgecolor=[INK if n < 97 else c for n, c in zip(ns, cs)],
              linewidth=[1.6 if n < 97 else 0.8 for n in ns],
              error_kw={"ecolor": INK, "elinewidth": 1.2})
for bar, hatch in zip(bars, hatches):
    if hatch: bar.set_hatch(hatch)
ax.axhline(1.0, color="#c04040", lw=1.2, linestyle="--",
           label="seasonal-naive reference (MASE=1)")
for x, v, n in zip(xs, ys, ns):
    tag = f"{v:.4f}" + (f"\n({n} cfg)" if n < 97 else "")
    if v > YMAX:
        ax.annotate(f"↑ {v:.2f}", xy=(x, YMAX - 0.02), ha="center", va="top",
                    fontsize=9, color="#c04040", weight="bold")
    else:
        ax.text(x, v + 0.015, tag, ha="center", va="bottom", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=45, ha="right")
ax.set_ylabel("Aggregate GM-Relative MASE  (lower is better)")
n_full = sum(1 for n in ns if n == 97)
ax.set_title(
    f"2L GM-Relative MASE at backbone step 40k, head 15k steps, GIFT-Eval B4  "
    f"({n_full}/{len(rows)} cells full-97, hatched = partial)\n"
    f"error bars: ±{SEED_NOISE} seed-noise band (from 2026-05-08 τ-sweep paired reruns)",
    fontsize=10)
ax.grid(True, axis="y", color=GRID, alpha=0.6)
ax.legend(loc="lower right", fontsize=9, frameon=False)
if ys:
    ymin = min(v for v in ys if v <= YMAX) - 3 * SEED_NOISE
    ymin = min(ymin, 1.0 - 3 * SEED_NOISE)
    ax.set_ylim(ymin, YMAX)
fig.tight_layout()
out = HERE / "eval_2L_gm_mase_bars.png"
fig.savefig(out)
print(f"wrote {out}  ({len(rows)} cells, {n_full} full-97, {len(rows)-n_full} partial)")
