"""GM-Relative MASE trajectory per arm across backbone steps.

Points per (arm, HL): whichever of these are available in results/,
each read as `gift_eval_full_<base>[_suffix]_<HL>L/summary.txt`:
  * step ~2,000  (suffix `_2k` — 25k-prolongation Round A)
  * arm's best-loss step   (suffix ""    — `best` cell)
  * step 12,500  (suffix `_last`)
  * step 25,000  (suffix `_25k` — 25k-prolongation Round B)
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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

# (label, results-dir, base name, best_step, colour).
ARMS = [
    ("arm 1 (split)",                  "results",           "gift_eval_full_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",           12500, C_ARM1),
    ("arm 3 (split + MoCo)",           "results",           "gift_eval_full_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",      11800, C_ARM3),
    ("arm 4 (pooled + MoCo)",          "results_arm4",      "gift_eval_full_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090",           600, C_ARM4),
    ("arm 5 (L_align + L_rep)",        "results_arm5",      "gift_eval_full_lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090",             11800, C_ARM5),
    ("arm 6 (L_align + L_rep_moco)",   "results_arm6_v2",   "gift_eval_full_lalign_lrepmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6v2_tau090",   8700, C_ARM6),
    ("arm bimoco (L_pred_moco + L_rep_moco)", "results_bimoco_v2", "gift_eval_full_split_pred_rep_bimoco_v2_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090", 12400, C_BIMOCO),
]


def gm(path: Path) -> float | None:
    if not path.exists():
        return None
    m = re.search(r"Aggregate GM-Relative MASE.*?([0-9]+\.[0-9]+)", path.read_text())
    return float(m.group(1)) if m else None


# arm C (champion cross_C) references from the sigreg cross experiment.
SIGREG = ROOT / "experiments" / "2026-06-28_sigreg_lambda_tau_cross"
_gm = pd.read_csv(SIGREG / "results" / "gm_table.csv")
_cc = _gm[_gm["arm"] == "cross_C"]
ARM_C_REF = {(r["head"], r["ckpt"]): float(r["gm"]) for _, r in _cc.iterrows()}

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for i, HL in enumerate(("2L", "6L")):
    ax = axes[i]
    for j, ck in enumerate(("best", "last")):
        ref = ARM_C_REF[(HL, ck)]
        ax.axhline(ref, color=MUTED, lw=1.2, ls="--",
                   label="arm C ref (champion)" if (i == 0 and j == 0) else None)
    for label, rd, base, best_step, colour in ARMS:
        # (step, suffix) — each read as f"{base}{suffix}_{HL}"
        candidates = [
            (2000, "_2k"),
            (best_step, ""),
            (12500, "_last"),
            (25000, "_25k"),
            (50000, "_50k"),
        ]
        pts = []
        for step, suffix in candidates:
            val = gm(EXP / rd / (base + suffix + f"_{HL}") / "summary.txt")
            if val is not None:
                pts.append((step, val))
        # dedupe on step (arm 1's best == last since FINAL.pth md5 = final.pth)
        pts = sorted({step: val for step, val in pts}.items())
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=colour, lw=1.5, marker="o", markersize=6, label=label)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")
    ax.set_xlabel("backbone step")
    ax.set_title(f"{HL} quantile head", fontsize=10)
    ax.grid(True, color=GRID, alpha=0.6)
    ax.set_xlim(0, 51000)
axes[0].set_ylabel("Aggregate GM-Relative MASE (full-97)")
axes[0].legend(loc="upper right", fontsize=9, frameon=False)
fig.suptitle(
    "GM-Relative MASE per arm across backbone step\n"
    "best / last cells: 30k-step best-loss q-head (+10k-step resume for last);  "
    "2k / 25k / 50k cells: fresh 40k-step q-head on that backbone snapshot",
    fontsize=9)
fig.tight_layout()
out = HERE / "gm_curve_per_arm.png"
fig.savefig(out)
print(f"wrote {out}")
