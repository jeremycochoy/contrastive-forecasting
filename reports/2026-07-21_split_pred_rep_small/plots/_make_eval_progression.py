"""GM-MASE progression across backbone horizons. All 30 cells have bb=40k and
bb=100k; the 10 that improved from 40k to 100k were extended to bb=200k.
Lines connect the horizons a cell has; labels are de-overlapped with leader
lines and carry the cell's last value."""
from pathlib import Path
import re
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
REP = HERE.parent / "results" / "eval_gm_mase"
EXP = HERE.parent.parent.parent / "experiments" / "2026-07-21_split_pred_rep_small" / "eval_gm_mase"

ARM_COLOR = {"arm1": "#2a78d6", "arm3": "#eb6834", "arm4": "#008300",
             "arm5": "#8b1e8b", "arm6_v2": "#b8860b", "bimoco": "#00a3a3"}
ARM_LOSS = {
    "arm1":    "L_pred + L_rep",
    "arm3":    "L_pred_moco + L_rep",
    "arm4":    "pooled + MoCo",
    "arm5":    "L_align + L_rep",
    "arm6_v2": "L_align + L_rep_moco",
    "bimoco":  "L_pred_moco + L_rep_moco",
}
def variant_annotation(arm, var):
    if var == "":        return "tau=0.10, cpc=1, sigreg_e=1"
    if var == "_tr1":    return "tau=1.0"
    if var == "_nse":    return "sigreg_e=0"
    if var == "_ncpc":   return "cpc=0"
    if var == "_combab":
        parts = ["tau=1.0", "cpc=0"]
        if arm in ("arm1", "arm3", "arm4"): parts.append("sigreg_e=0")
        return " + ".join(parts)
VAR_STYLE = {
    "":        {"ls": "-",  "marker": "o", "lw": 2.0, "ms": 8, "short": "base"},
    "_tr1":    {"ls": "--", "marker": "s", "lw": 1.5, "ms": 7, "short": "tr1"},
    "_nse":    {"ls": ":",  "marker": "^", "lw": 1.5, "ms": 7, "short": "nse"},
    "_ncpc":   {"ls": "-.", "marker": "D", "lw": 1.5, "ms": 7, "short": "ncpc"},
    "_combab": {"ls": "-",  "marker": "P", "lw": 2.0, "ms": 9, "short": "combab"},
}
ARMS = ["arm1", "arm3", "arm4", "arm5", "arm6_v2", "bimoco"]
VARIANTS = ["", "_tr1", "_nse", "_ncpc", "_combab"]
# x position per horizon, and the head-training steps used at each
HORIZONS = [(40, 15000, 40), (100, 30000, 100), (200, 30000, 160)]

def read(slug, bb, hd):
    for p in (REP / f"{slug}_bb{bb}k_hd{hd}s_summary.txt",
              EXP / f"{slug}_bb{bb}k_hd{hd}s" / "summary.txt"):
        if not p.exists(): continue
        m = re.search(r"Aggregate.*\((\d+) configs\):\s+([0-9.]+)", p.read_text())
        # Only full-97 values are comparable.
        if m and m.group(1) == "97": return float(m.group(2))
    return None

def spread(anchors, gap, lo, hi):
    """Push label positions apart to `gap` while keeping their vertical order."""
    order = sorted(range(len(anchors)), key=lambda i: anchors[i])
    pos = list(anchors)
    prev = lo - gap
    for i in order:
        pos[i] = max(pos[i], prev + gap); prev = pos[i]
    prev = hi + gap
    for i in reversed(order):
        pos[i] = min(pos[i], prev - gap); prev = pos[i]
    return pos

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})
fig, ax = plt.subplots(figsize=(16, 10))

YMAX, YMIN = 1.85, 1.13
cells = []
for arm in ARMS:
    for var in VARIANTS:
        slug = f"{arm}{var}"
        vals = [(x, read(slug, bb, hd)) for bb, hd, x in HORIZONS]
        if all(v is None for _, v in vals): continue
        cells.append((arm, var, vals))

for arm, var, vals in cells:
    style, colour = VAR_STYLE[var], ARM_COLOR[arm]
    pts = [(x, min(v, YMAX)) for x, v in vals if v is not None]
    if len(pts) > 1:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, alpha=0.9,
                linestyle=style["ls"], linewidth=style["lw"],
                marker=style["marker"], markersize=style["ms"], markeredgewidth=0)
    elif pts:
        ax.plot([pts[0][0]], [pts[0][1]], color=colour, alpha=0.5, linestyle="",
                marker=style["marker"], markersize=style["ms"])
    for x, v in vals:
        if v is not None and v > YMAX:
            ax.annotate(f"^{v:.2f}", xy=(x, YMAX - 0.01), fontsize=7,
                        color=colour, ha="center", va="top")

# Label at each cell's LAST available horizon, de-overlapped at the right edge.
anchors, labelled = [], []
for arm, var, vals in cells:
    last = [(x, v) for x, v in vals if v is not None][-1]
    anchors.append(min(last[1], YMAX)); labelled.append((arm, var, vals, last))
ys = spread(anchors, gap=(YMAX - YMIN) / (len(anchors) + 1), lo=YMIN, hi=YMAX)
LABEL_X = 178
for (arm, var, vals, last), y_anchor, y_lab in zip(labelled, anchors, ys):
    colour = ARM_COLOR[arm]
    have = [v for _, v in vals if v is not None]
    delta = f"  ({have[-1] - have[-2]:+.3f})" if len(have) > 1 else ""
    mark = " ←200k" if last[0] == 160 else ""
    text = (f"{arm} {VAR_STYLE[var]['short']}  {last[1]:.3f}{delta}{mark}"
            f"   [{ARM_LOSS[arm]} | {variant_annotation(arm, var)}]")
    ax.plot([last[0] + 1, LABEL_X - 1], [y_anchor, y_lab], color=colour,
            lw=0.6, alpha=0.45, zorder=0)
    ax.text(LABEL_X, y_lab, text, color=colour, fontsize=7.5, va="center", ha="left")

ax.axhline(1.0, color="#c04040", lw=1.2, linestyle="--")
ax.set_xticks([x for _, _, x in HORIZONS])
ax.set_xticklabels(["bb 40k\n(head 15k)", "bb 100k\n(head 30k)", "bb 200k\n(head 30k)"])
ax.set_xlim(25, 372)
ax.set_ylim(YMIN, YMAX)
ax.set_ylabel("Aggregate GM-Relative MASE  (lower is better)")
ax.grid(True, axis="y", color=GRID, alpha=0.6)

from matplotlib.lines import Line2D
var_handles = [Line2D([], [], color=INK, linestyle=VAR_STYLE[v]["ls"],
                      marker=VAR_STYLE[v]["marker"], markersize=VAR_STYLE[v]["ms"],
                      markeredgewidth=0, linewidth=VAR_STYLE[v]["lw"],
                      label=VAR_STYLE[v]["short"]) for v in VARIANTS]
ax.legend(handles=var_handles, loc="lower left", fontsize=8, frameon=False,
          ncol=5, title="variant")

n200 = sum(1 for _, _, vals in cells if vals[2][1] is not None)
down = sum(1 for _, _, vals in cells
           if vals[2][1] is not None and vals[2][1] < vals[1][1])
ax.set_title(f"GM-Relative MASE across backbone horizons  "
             f"(30 cells at 40k and 100k; the {n200} that improved 40k→100k were "
             f"extended to 200k, of which {down} improved again)", fontsize=11)
fig.tight_layout()
out = HERE / "eval_2L_gm_mase_progression.png"
fig.savefig(out)
print(f"wrote {out}  ({len(cells)} cells, {n200} at 200k, {down} improved again)")
