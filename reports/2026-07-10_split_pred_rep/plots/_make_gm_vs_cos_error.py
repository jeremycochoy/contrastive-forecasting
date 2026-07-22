"""Per-arm stacked GM-Relative MASE (2L head) above 1-ff, shared x-axis.

4 rows × 3 columns. For each arm the top panel is the 2L GIFT-Eval score at
each backbone-step cell (2k / best / last / 25k / 50k where available); the
panel directly below is 1 − ⟨cos(f̂, f_true)⟩ over training with vertical
dotted lines at the same backbone steps, so a vertical drop between the two
panels reads (eval, training-time alignment) per arm.

Rows 1–2: arm 1, arm 3, arm 4.
Rows 3–4: arm 5, arm 6 v2, bimoco.

Same x-range [0, 51 000] on every panel.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
EXP = ROOT / "experiments" / "2026-07-10_split_pred_rep"

RUNS = [
    ("arm 1  (L_pred + L_rep)", "runs", "results",
     "bb_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
     "split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
     "#2a78d6",
     [(2000, "_2k"), (12500, "_last"), (25000, "_25k")]),
    ("arm 3  (L_pred_moco + L_rep)", "runs", "results",
     "bb_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
     "split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
     "#eb6834",
     [(2000, "_2k"), (11800, ""), (12500, "_last"), (25000, "_25k")]),
    ("arm 4  (pooled + MoCo)", "runs_arm4", "results_arm4",
     "bb_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090",
     "allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090",
     "#008300",
     [(600, ""), (2000, "_2k"), (12500, "_last"), (25000, "_25k"), (50000, "_50k")]),
    ("arm 5  (L_align + L_rep)", "runs_arm5", "results_arm5",
     "bb_lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090",
     "lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090",
     "#8b1e8b",
     [(2000, "_2k"), (11800, ""), (12500, "_last"), (25000, "_25k"), (50000, "_50k")]),
    ("arm 6  (L_align + L_rep_moco)", "runs_arm6_v2", "results_arm6_v2",
     "bb_lalign_lrepmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6v2_tau090",
     "lalign_lrepmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6v2_tau090",
     "#b8860b",
     [(2000, "_2k"), (8700, ""), (12500, "_last"), (25000, "_25k"), (50000, "_50k")]),
    ("bimoco  (L_pred_moco + L_rep_moco)", "runs_bimoco_v2", "results_bimoco_v2",
     "bb_split_pred_rep_bimoco_v2_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
     "split_pred_rep_bimoco_v2_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
     "#00a3a3",
     [(2000, "_2k"), (12400, ""), (12500, "_last"), (25000, "_25k")]),
]

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})


def load_ff(runs_dir: str, name: str) -> pd.DataFrame:
    full = EXP / runs_dir / f"{name}_losses_full.csv"
    base = full if full.exists() else EXP / runs_dir / f"{name}_losses.csv"
    df = pd.read_csv(base, usecols=["step", "ff"])
    r2 = EXP / runs_dir / f"{name}_r2_losses.csv"
    if r2.exists() and pd.read_csv(r2, usecols=["step"])["step"].max() > 12500:
        df = pd.concat([df, pd.read_csv(r2, usecols=["step", "ff"])], ignore_index=True)
    ext = EXP / runs_dir / f"{name}_ext25k_losses.csv"
    if ext.exists():
        d = pd.read_csv(ext, usecols=["step", "ff"])
        d["step"] += 12500
        df = pd.concat([df, d], ignore_index=True)
    r3 = EXP / runs_dir / f"{name}_r3_losses.csv"
    if r3.exists():
        df = pd.concat([df, pd.read_csv(r3, usecols=["step", "ff"])], ignore_index=True)
    return df.sort_values("step").reset_index(drop=True)


def gm(path: Path) -> float | None:
    if not path.exists():
        return None
    m = re.search(r"Aggregate GM-Relative MASE.*?([0-9]+\.[0-9]+)", path.read_text())
    return float(m.group(1)) if m else None


def plot_arm(ax_gm, ax_ff, run):
    label, runs_dir, res_dir, bb_name, base_qh, colour, candidates = run

    pts = []
    for step, suffix in candidates:
        val = gm(EXP / res_dir / f"gift_eval_full_{base_qh}{suffix}_2L" / "summary.txt")
        if val is not None:
            pts.append((step, val))
    pts = sorted({s: v for s, v in pts}.items())
    if pts:
        xs, ys = zip(*pts)
        ax_gm.plot(xs, ys, color=colour, marker="o", lw=1.5, markersize=6)
    ax_gm.axhline(1.0, color=MUTED, lw=0.8, ls="--")
    ax_gm.set_xlim(0, 51000)
    ax_gm.set_title(f"{label} — 2L head", fontsize=9)
    ax_gm.grid(True, color=GRID, alpha=0.6)
    ax_gm.set_ylabel("GM-Relative MASE")
    for step, _ in candidates:
        ax_gm.axvline(step, color=INK, ls=":", lw=0.8, alpha=0.6)

    df = load_ff(runs_dir, bb_name)
    df = df[df["step"] >= 100]
    ax_ff.plot(df["step"], 1.0 - df["ff"], color=colour, lw=1.2)
    for step, _ in candidates:
        ax_ff.axvline(step, color=INK, ls=":", lw=0.8, alpha=0.6)
    ax_ff.set_xlim(0, 51000)
    ax_ff.set_ylabel("1 − ff")
    ax_ff.set_xlabel("backbone step")
    ax_ff.grid(True, color=GRID, alpha=0.6)


fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True)

for col in range(3):
    plot_arm(axes[0, col], axes[1, col], RUNS[col])
for col in range(3):
    plot_arm(axes[2, col], axes[3, col], RUNS[col + 3])

fig.suptitle(
    "Per arm: 2L GIFT-Eval GM-Relative MASE (top) vs 1 − ⟨cos(f̂, f_true)⟩ over training (bottom). "
    "Dotted verticals mark the backbone steps evaluated.",
    fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = HERE / "gm_2L_vs_cos_error_per_arm.png"
fig.savefig(out)
print(f"wrote {out}")
