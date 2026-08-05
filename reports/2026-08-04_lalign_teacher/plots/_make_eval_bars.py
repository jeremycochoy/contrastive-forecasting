"""Bar plot of 2L GM-Relative MASE at backbone step 40k, all evaluated arms, #379.

Includes the original 11 cells + the 12 new cells from the 2026-07-27 vast batch
(plus arm3_ncpc/arm4_combab which finished only partial gift-eval on vast — those
land under a red hatched bar to flag config coverage <97).

Error bars are the measured head-seed range from `results/seed_spread.csv`,
teacher rows at this backbone step, min to max over that cell's replicate
seeds. Two of the thirty cells carry one; the rest ran a single head seed and
get no bar, because a spread borrowed from another cell would be wrong by up
to a factor of forty.

Aggregate is read preferentially from the report-flat summary file
`results/eval_gm_mase/<slug>_bb40k_hd15000s_summary.txt` (populated by the
vast sync loop), falling back to the experiments-side
`<slug>_bb40k_hd15000s/summary.txt` for cells run locally on elisa."""
from pathlib import Path
import re
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
EVAL_ROOT_EXP = HERE.parent.parent.parent / "experiments" / "2026-08-01_lalign_teacher" / "eval_gm_mase"
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

def rose_over_0_40k():
    """Slugs whose backbone loss ended the 0-40k window above where it started.

    Read from `results/anomaly_windows.csv`, the same file section 7 of the
    report uses. A 40k bar drawn on such a backbone is not a peer of the rest,
    so the figure marks it.
    """
    import csv
    path = HERE.parent / "results" / "anomaly_windows.csv"
    slugs = {slug for _l, slug, _c in ARMS}
    out = set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["window"] != "0k-40k" or float(r["delta"]) <= 0:
                continue
            rest = r["run"].removeprefix("bb_small_")
            hit = [g for g in slugs if rest.startswith(g + "_")]
            if hit:
                out.add(max(hit, key=len))
    return out


def head_seed_spread():
    """{slug: (min, max)} over the replicate head seeds, teacher side, 40k.

    Keyed on the arm alone, so the teacher row is the one to take: at 40k the
    student side of an arm is a different backbone under the same key.
    """
    import csv
    path = HERE.parent / "results" / "seed_spread.csv"
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["align_target"] != "teacher" or int(r["bb_steps"]) != 40000:
                continue
            vals = [float(v) for v in r["values"].split()]
            out[r["arm_slug"]] = (min(vals), max(vals))
    return out


INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})
fig, ax = plt.subplots(figsize=(14, 5.5))
# Recipe colours sit in shared families with no room for a 6-way key on 30 bars,
# so the bars carry one distinction only: retrained with --align-target teacher
# or not. Two colours, two legend entries.
RETRAINED, OTHER = "#8b1e8b", "#c9c7bf"
SEED_INK = "#0f6f6f"      # the head-seed figure's colour for 40k cells
ROSE = rose_over_0_40k()
SPREAD = head_seed_spread()
rows = []
for label, slug, colour in ARMS:
    v, n = read_agg(slug)
    if v is None: continue
    teacher = label.startswith("arm5 ") or label.startswith("arm6_v2 ")
    rows.append((label + (" ⟲" if teacher else "") + (" ‡" if slug in ROSE else "")
                 + (" †" if slug in SPREAD else ""),
                 RETRAINED if teacher else OTHER, v, n, slug in ROSE,
                 SPREAD.get(slug)))
rows.sort(key=lambda r: r[2])
xs = list(range(len(rows)))
labs = [r[0] for r in rows]
cs = [r[1] for r in rows]
ys = [r[2] for r in rows]
ns = [r[3] for r in rows]
rose = [r[4] for r in rows]
spreads = [r[5] for r in rows]
hatches = ["///" if n < 97 else ("xxx" if r else None)
           for n, r in zip(ns, rose)]

bars = ax.bar(xs, ys, color=cs,
              edgecolor=[INK if (n < 97 or r) else c
                         for n, c, r in zip(ns, cs, rose)],
              linewidth=[1.6 if (n < 97 or r) else 0.8
                         for n, r in zip(ns, rose)])
for bar, hatch in zip(bars, hatches):
    if hatch: bar.set_hatch(hatch)
ax.axhline(1.0, color="#c04040", lw=1.2, linestyle="--",
           label="seasonal-naive reference (MASE=1)")
# Where a cell was re-headed under extra seeds, the whisker spans what those
# seeds measured. The other 28 ran one head seed and stay bare.
for x, sp in zip(xs, spreads):
    if sp is None:
        continue
    lo, hi = sp
    ax.plot([x, x], [lo, hi], color=SEED_INK, lw=1.8, solid_capstyle="butt",
            zorder=5)
    for y in (lo, hi):
        ax.plot([x - 0.24, x + 0.24], [y, y], color=SEED_INK, lw=1.8,
                zorder=5)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [Patch(facecolor=RETRAINED, label="⟲ retrained, --align-target teacher"),
           Patch(facecolor=OTHER, label="earlier sweep, no L_align"),
           Patch(facecolor="white", edgecolor=INK, hatch="xxx",
                 label="‡ backbone loss rose over 0–40k"),
           Line2D([0], [0], color=SEED_INK, lw=1.8,
                  label="† measured range over 3 head seeds "
                        f"({len(SPREAD)} of {len(rows)} cells)")]
for x, v, r in zip(xs, ys, rose):
    if r:
        ax.annotate("loss rose over 0–40k", xy=(x - 0.62, v * 0.80),
                    rotation=90, ha="right", va="center", fontsize=8,
                    color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=INK, lw=0.6))
# Bars start at zero, so every bar is drawn to scale and the tallest one fits.
for x, v, n in zip(xs, ys, ns):
    tag = f"{v:.4f}" + (f"\n({n} cfg)" if n < 97 else "")
    ax.text(x, v + 0.03, tag, ha="center", va="bottom", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=45, ha="right")
ax.set_ylabel("Aggregate GM-Relative MASE  (lower is better)")
n_full = sum(1 for n in ns if n == 97)
ax.set_title(
    f"GM-Relative MASE at backbone step 40k, head 15k steps, GIFT-Eval B4  "
    f"({n_full}/{len(rows)} cells full-97)",
    fontsize=10)
ax.grid(True, axis="y", color=GRID, alpha=0.6)
ax.legend(handles=handles + [ax.lines[0]], loc="upper left", fontsize=9, frameon=False)
if ys:
    ax.set_ylim(0.0, max(ys) * 1.10)
fig.tight_layout()
out = HERE / "eval_2L_gm_mase_bars.png"
fig.savefig(out)
print(f"wrote {out}  ({len(rows)} cells, {n_full} full-97, {len(rows)-n_full} partial)")
