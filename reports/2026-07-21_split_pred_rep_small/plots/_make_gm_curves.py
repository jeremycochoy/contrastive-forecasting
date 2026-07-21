"""GM-Relative MASE trajectory per arm × backbone-step cell, #379.

Reads `experiments/2026-07-21_split_pred_rep_small/results/
gift_eval_full_{TAG}_{sk}k_{HL}L/summary.txt` for each of the 5 cells
(sk ∈ {2, 25, 50, 100, 200}) × 6 arms × 2 head-layer sizes = 60 cells.
`TAG` is the arm's `NAME[3:]` (strip the `bb_` prefix); the name stubs
below must match `run_arm.sh`'s per-arm case block verbatim.

Emits two files (one per HL) matching the report scaffold's placeholders:
  plots/gm_curve_per_arm_2L.png
  plots/gm_curve_per_arm_6L.png
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
EXP = ROOT / "experiments" / "2026-07-21_split_pred_rep_small"

# One color per arm, matching #374's palette.
C_ARM1, C_ARM3, C_ARM4 = "#2a78d6", "#eb6834", "#008300"
C_ARM5, C_ARM6, C_BIMOCO = "#8b1e8b", "#b8860b", "#00a3a3"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

# (label, TAG stub, colour). TAG stub = NAME without the `bb_` prefix,
# matching run_arm.sh + summary directory naming.
ARMS = [
    ("arm 1  (L_pred + L_rep)",
     "small_arm1_split_pred_rep_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090",
     C_ARM1),
    ("arm 3  (L_pred_moco + L_rep)",
     "small_arm3_split_pred_rep_moco_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090",
     C_ARM3),
    ("arm 4  (pooled + MoCo)",
     "small_arm4_xshh_allt_moco_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090",
     C_ARM4),
    ("arm 5  (L_align + L_rep)",
     "small_arm5_lalign_lrep_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090",
     C_ARM5),
    ("arm 6 v2  (L_align + L_rep_moco)",
     "small_arm6_v2_lalign_lrepmoco_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090",
     C_ARM6),
    ("bimoco  (L_pred_moco + L_rep_moco)",
     "small_bimoco_split_pred_rep_moco_bothsides_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090",
     C_BIMOCO),
]

# The 5 backbone-step cells the sweep evaluates (matches run_arm.sh
# BB_STEPS_K default). x-axis in steps, plotted linearly out to 210k.
STEPS_K = (2, 25, 50, 100, 200)


def gm(path: Path) -> float | None:
    if not path.exists():
        return None
    m = re.search(r"Aggregate GM-Relative MASE.*?([0-9]+\.[0-9]+)", path.read_text())
    return float(m.group(1)) if m else None


def render(hl: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, tag, colour in ARMS:
        pts = []
        for sk in STEPS_K:
            val = gm(EXP / "results" / f"gift_eval_full_{tag}_{sk}k_{hl}" / "summary.txt")
            if val is not None:
                pts.append((sk * 1000, val))
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=colour, lw=1.5, marker="o", markersize=6, label=label)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")
    ax.set_xlabel("backbone step")
    ax.set_ylabel("Aggregate GM-Relative MASE (full-97)")
    ax.set_xlim(0, 210_000)
    ax.grid(True, color=GRID, alpha=0.6)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_title(
        f"{hl} quantile head — GM-Relative MASE per arm across backbone "
        f"step (fresh 40k head per cell)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    render("2L", HERE / "gm_curve_per_arm_2L.png")
    render("6L", HERE / "gm_curve_per_arm_6L.png")
