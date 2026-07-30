"""GM-MASE progression: for every cell where both bb=40k (head 15k) and
bb=100k (head 30k) are landed, draw a line 40k -> 100k. Cells with only one
endpoint show as isolated markers. Colour by arm; variant styles are given
in-line labels at the 100k endpoint, de-overlapped and joined by leader lines."""
from pathlib import Path
import re
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
REP = HERE.parent / "results" / "eval_gm_mase"
EXP = HERE.parent.parent.parent / "experiments" / "2026-07-21_split_pred_rep_small" / "eval_gm_mase"

# Arm palette (same base colour per arm across variants)
ARM_COLOR = {"arm1": "#2a78d6", "arm3": "#eb6834", "arm4": "#008300",
             "arm5": "#8b1e8b", "arm6_v2": "#b8860b", "bimoco": "#00a3a3"}
# Loss-recipe short label per arm.
ARM_LOSS = {
    "arm1":    "L_pred + L_rep",
    "arm3":    "L_pred_moco + L_rep",
    "arm4":    "pooled + MoCo",
    "arm5":    "L_align + L_rep",
    "arm6_v2": "L_align + L_rep_moco",
    "bimoco":  "L_pred_moco + L_rep_moco",
}
# Variant -> knobs turned. arm 1/3/4 combab also sets sigreg_e=0.
def variant_annotation(arm, var):
    if var == "":        return "tau=0.10, cpc=1, sigreg_e=1"
    if var == "_tr1":    return "tau=1.0"
    if var == "_nse":    return "sigreg_e=0"
    if var == "_ncpc":   return "cpc=0"
    if var == "_combab":
        parts = ["tau=1.0", "cpc=0"]
        if arm in ("arm1", "arm3", "arm4"): parts.append("sigreg_e=0")
        return " + ".join(parts)
# Variant -> linestyle + marker
VAR_STYLE = {
    "":        {"ls": "-",  "marker": "o", "lw": 2.0, "ms": 8, "short": "base"},
    "_tr1":    {"ls": "--", "marker": "s", "lw": 1.5, "ms": 7, "short": "tr1"},
    "_nse":    {"ls": ":",  "marker": "^", "lw": 1.5, "ms": 7, "short": "nse"},
    "_ncpc":   {"ls": "-.", "marker": "D", "lw": 1.5, "ms": 7, "short": "ncpc"},
    "_combab": {"ls": "-",  "marker": "P", "lw": 2.0, "ms": 9, "short": "combab"},
}
ARMS = ["arm1", "arm3", "arm4", "arm5", "arm6_v2", "bimoco"]
VARIANTS = ["", "_tr1", "_nse", "_ncpc", "_combab"]

def read(slug, bb, hd):
    for p in (REP / f"{slug}_bb{bb}k_hd{hd}s_summary.txt",
              EXP / f"{slug}_bb{bb}k_hd{hd}s" / "summary.txt"):
        if not p.exists(): continue
        m = re.search(r"Aggregate.*\((\d+) configs\):\s+([0-9.]+)", p.read_text())
        if m: return float(m.group(2))
    return None

def spread(anchors, gap, lo, hi):
    """Push label positions apart to `gap` while keeping their vertical order."""
    order = sorted(range(len(anchors)), key=lambda i: anchors[i])
    pos = list(anchors)
    prev = lo - gap
    for i in order:                      # upward pass
        pos[i] = max(pos[i], prev + gap)
        prev = pos[i]
    prev = hi + gap
    for i in reversed(order):            # downward pass, keeps everything inside
        pos[i] = min(pos[i], prev - gap)
        prev = pos[i]
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
        v40, v100 = read(slug, 40, 15000), read(slug, 100, 30000)
        if v40 is None and v100 is None: continue
        cells.append((arm, var, v40, v100))

for arm, var, v40, v100 in cells:
    style, colour = VAR_STYLE[var], ARM_COLOR[arm]
    v40p = min(v40, YMAX) if v40 is not None else None
    v100p = min(v100, YMAX) if v100 is not None else None
    if v40p is not None and v100p is not None:
        ax.plot([40, 100], [v40p, v100p], color=colour, alpha=0.9,
                linestyle=style["ls"], linewidth=style["lw"],
                marker=style["marker"], markersize=style["ms"], markeredgewidth=0)
    else:
        x, y = (40, v40p) if v40p is not None else (100, v100p)
        ax.plot([x], [y], color=colour, alpha=0.5, linestyle="",
                marker=style["marker"], markersize=style["ms"])
    for v, x in ((v40, 40), (v100, 100)):
        if v is not None and v > YMAX:
            ax.annotate(f"^{v:.2f}", xy=(x, YMAX - 0.01), fontsize=7,
                        color=colour, ha="center", va="top")

# Labels at the right edge: anchor on the 100k value, de-overlap, leader lines.
labelled = [c for c in cells if c[3] is not None]
anchors = [min(c[3], YMAX) for c in labelled]
ys = spread(anchors, gap=(YMAX - YMIN) / (len(anchors) + 1), lo=YMIN, hi=YMAX)
LABEL_X = 118
for (arm, var, v40, v100), y_anchor, y_lab in zip(labelled, anchors, ys):
    colour = ARM_COLOR[arm]
    delta = "" if v40 is None else f"  ({v100 - v40:+.3f})"
    text = (f"{arm} {VAR_STYLE[var]['short']}  {v100:.3f}{delta}"
            f"   [{ARM_LOSS[arm]} | {variant_annotation(arm, var)}]")
    ax.plot([101, LABEL_X - 1], [y_anchor, y_lab], color=colour,
            lw=0.6, alpha=0.45, zorder=0)
    ax.text(LABEL_X, y_lab, text, color=colour, fontsize=7.5, va="center", ha="left")

ax.axhline(1.0, color="#c04040", lw=1.2, linestyle="--")
ax.set_xticks([40, 100])
ax.set_xticklabels(["bb 40k\n(head 15k)", "bb 100k\n(head 30k)"])
ax.set_xlim(25, 305)
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

n_paired = sum(1 for c in cells if c[2] is not None and c[3] is not None)
down = sum(1 for c in cells if c[2] is not None and c[3] is not None and c[3] < c[2])
ax.set_title(f"GM-Relative MASE, backbone 40k -> 100k  "
             f"({n_paired} cells with both endpoints; {down} improve, {n_paired - down} worsen; "
             f"labels carry the 100k value and the change)", fontsize=11)
fig.tight_layout()
out = HERE / "eval_2L_gm_mase_progression.png"
fig.savefig(out)
print(f"wrote {out}  ({n_paired} paired lines, {down} improving)")
