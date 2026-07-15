"""GM-Relative MASE trajectory per arm (2L / 6L), (best_step, best_GM) and
(12500, last_GM). Two points per (arm, HL) at present — the 25k extension
will add a third when it lands.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
EXP = ROOT / "experiments" / "2026-07-10_split_pred_rep"

C_ARM1, C_ARM3, C_ARM4 = "#2a78d6", "#eb6834", "#008300"
C_ARM5, C_ARM6, C_BIMOCO = "#8b1e8b", "#b8860b", "#00a3a3"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

# (label, results-dir, base name, best_step, colour). Labels use a short slug
# that names what each arm actually is, followed by the report's arm-N tag for
# tie-back:
#   split         — arm 1: baseline split L_pred + L_rep, no MoCo.
#   split+moco    — arm 3: split with MoCo on L_pred (teacher-keys on the
#                   cross-batch f↔h family).
#   pooled+moco   — arm 4: arm C's pooled xshh_allt shape + MoCo negatives.
#   byol+rep      — arm 5: L_align (BYOL, positive only) + L_rep, no InfoNCE.
#   moco-align+rep — arm 6: L_align_moco (MoCo-style same-time encoder align)
#                   + L_rep.
#   bimoco        — split with MoCo on BOTH L_pred AND L_rep (teacher-keys on
#                   every h-anchored family).
ARMS = [
    ("L_pred + L_rep",                    "results",           "gift_eval_full_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",           12500, C_ARM1),
    ("L_pred_moco + L_rep",               "results",           "gift_eval_full_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",      11800, C_ARM3),
    ("L_pooled_moco",                     "results_arm4",      "gift_eval_full_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090",           600, C_ARM4),
    ("L_align + L_rep",                   "results_arm5",      "gift_eval_full_lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090",             11800, C_ARM5),
    ("L_align + L_rep_moco (arm 6)",      "results_arm6_v2",   "gift_eval_full_lalign_lrepmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6v2_tau090",   8700, C_ARM6),
    ("L_pred_moco + L_rep_moco (bimoco)", "results_bimoco_v2", "gift_eval_full_split_pred_rep_bimoco_v2_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090", 12400, C_BIMOCO),
]


def gm(path: Path) -> float | None:
    if not path.exists():
        return None
    m = re.search(r"Aggregate GM-Relative MASE.*?([0-9]+\.[0-9]+)", path.read_text())
    return float(m.group(1)) if m else None


fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for i, HL in enumerate(("2L", "6L")):
    ax = axes[i]
    for label, rd, base, best_step, colour in ARMS:
        best = gm(EXP / rd / (base + f"_{HL}") / "summary.txt")
        last = gm(EXP / rd / (base + f"_last_{HL}") / "summary.txt")
        if best is None or last is None:
            continue
        ax.plot([best_step, 12500], [best, last], color=colour, lw=1.5,
                marker="o", markersize=6, label=label)
        # step-25k `last` will land here as a third point once the extension eval finishes.
    ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")
    ax.set_xlabel("backbone step")
    ax.set_title(f"{HL} quantile head", fontsize=10)
    ax.grid(True, color=GRID, alpha=0.6)
    ax.set_xlim(0, 15000)
axes[0].set_ylabel("Aggregate GM-Relative MASE (full-97)")
axes[0].legend(loc="upper right", fontsize=9, frameon=False)
fig.suptitle(
    "GM-Relative MASE per arm at (best-loss step, 12,500 last). "
    "L_pred + L_rep collapses to a single point at step 12,500 (its FINAL.pth md5 = final.pth). "
    "25k extension in flight adds a third point per (arm, HL) when the eval cell lands.",
    fontsize=9)
fig.tight_layout()
out = HERE / "gm_curve_per_arm.png"
fig.savefig(out)
print(f"wrote {out}")
