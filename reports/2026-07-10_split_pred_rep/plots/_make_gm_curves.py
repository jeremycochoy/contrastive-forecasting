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
C_ARMC = "#52514e"
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


# arm C (SIGReg-cross champion recipe) — seed-2 retrain trajectory.
# The original seed-20260520 arm C per-task file was never committed and
# is now gone; a same-recipe seed-2 retrain lives here with per-task files
# at backbone steps 12,500 / 25,000 / 50,000.
ARM_C_STEPS = (12500, 25000, 50000)
ARM_C_DIR = EXP / "results_armC_seed2"


def arm_c(head: str) -> list[tuple[int, float]]:
    pts = []
    for step in ARM_C_STEPS:
        v = gm(ARM_C_DIR / f"gift_eval_full_armC_seed2_step{step}_{head}" / "summary.txt")
        if v is not None:
            pts.append((step, v))
    return pts


fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for i, HL in enumerate(("2L", "6L")):
    ax = axes[i]
    # arm C (seed-2) — solid line, distinct grey.
    c_pts = arm_c(HL)
    if c_pts:
        xs, ys = zip(*c_pts)
        ax.plot(xs, ys, color=C_ARMC, lw=1.8, marker="D", markersize=6,
                label="arm C (SIGReg champion, seed 2)" if i == 0 else None)
    for label, rd, base, best_step, colour in ARMS:
        # (step, suffix) — each read as f"{base}{suffix}_{HL}"
        candidates = [
            (2000, "_2k"),
            (best_step, ""),
            (12500, "_last"),
            (25000, "_25k"),
            (50000, "_50k"),
        ]
        # Head protocol differs: 2k/25k/50k = fresh 40k head; best/last =
        # 30k best-loss head (+10k resume). Marker shape (solid disk vs hollow
        # circle) carries that distinction; a single per-arm trajectory line
        # joins every evaluated cell so the reader can follow the arm's
        # downstream quality across backbone step.
        fresh, resumed = {}, {}
        for step, suffix in candidates:
            val = gm(EXP / rd / (base + suffix + f"_{HL}") / "summary.txt")
            if val is None:
                continue
            (fresh if suffix in ("_2k", "_25k", "_50k") else resumed)[step] = val
        # Combined trajectory (dedup: fresh wins if both protocols land at
        # the same backbone step — e.g. arm 1 `last`==step-12,500).
        combined = dict(resumed); combined.update(fresh)
        combined = sorted(combined.items())
        if not combined:
            continue
        xs, ys = zip(*combined)
        ax.plot(xs, ys, color=colour, lw=1.5, label=label)
        # Solid disks for fresh-head cells, hollow circles for resumed-head cells.
        if fresh:
            fx, fy = zip(*sorted(fresh.items()))
            ax.plot(fx, fy, color=colour, lw=0, marker="o", markersize=6)
        if resumed:
            rx, ry = zip(*sorted(resumed.items()))
            ax.plot(rx, ry, color=colour, lw=0, marker="o", markersize=7,
                    markerfacecolor="none", markeredgewidth=1.6)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")
    ax.set_xlabel("backbone step")
    ax.set_title(f"{HL} quantile head", fontsize=10)
    ax.grid(True, color=GRID, alpha=0.6)
    ax.set_xlim(0, 51000)
axes[0].set_ylabel("Aggregate GM-Relative MASE (full-97)")
from matplotlib.lines import Line2D  # noqa: E402

handles, labels = axes[0].get_legend_handles_labels()
handles += [
    Line2D([], [], color=INK, lw=1.5, marker="o", markersize=6,
           label="fresh 40k head (2k / 25k / 50k)"),
    Line2D([], [], color=INK, lw=1.5, marker="o", markersize=7,
           markerfacecolor="none", markeredgewidth=1.6,
           label="30k best-loss head, +10k resume (best / last)"),
]
axes[0].legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
fig.suptitle(
    "GM-Relative MASE per arm across backbone step  —  line joins every evaluated cell of that arm\n"
    "solid disks = fresh 40k-step q-head on that snapshot;  "
    "hollow circles = 30k-step best-loss q-head (+10k-step resume for last)",
    fontsize=9)
fig.tight_layout()
out = HERE / "gm_curve_per_arm.png"
fig.savefig(out)
print(f"wrote {out}")
