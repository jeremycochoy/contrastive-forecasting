"""The highest GM-Relative MASE cell at each backbone step, against its
recipe-mates.

Top row: backbone training loss, log step axis. The flagged run is drawn in
its recipe colour, every other setting of the same recipe in grey, so the
flagged curve is read against the runs it should look like.

Bottom row: mean attention logit magnitude over the encoder layers, same
runs, same axis, from `results/attn_amplitude/`.

Curves come from `results/training_curves/`; a resumed run writes `_r2` /
`_r3` files, so a full trajectory is the concatenation ordered by step.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
CURVES = HERE.parent / "results" / "training_curves"
ATTN = HERE.parent / "results" / "attn_amplitude"

INK, MUTED, GRID, OTHER = "#0b0b0b", "#898781", "#e1e0d9", "#c9c7c0"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

# (recipe stem template, colour, flagged setting, panel label)
GROUPS = [
    ("bb_small_arm1{v}_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#2a78d6", "_combab", "arm1 combab  (highest at bb 40k)"),
    ("bb_small_arm5{v}_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#8b1e8b", "_nse", "arm5 nse  (highest at bb 200k)"),
    ("bb_small_arm6_v2{v}_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#b8860b", "", "arm6_v2 base  (highest at bb 100k)"),
]
SETTINGS = ["", "_tr1", "_nse", "_ncpc", "_combab"]


def _stem(template: str, variant: str) -> str:
    """Run stem for one cell, preferring this branch's teacher-target run."""
    base = template.format(v=variant)
    for cand in (f"{base}_alignteacher", base):
        if (CURVES / f"{cand}_losses.csv").exists():
            return cand
    return base


def loss_curve(stem: str):
    parts = []
    for suffix in ("", "_r2", "_r3"):
        p = CURVES / f"{stem}{suffix}_losses.csv"
        if p.exists():
            parts.append(pd.read_csv(p, usecols=["step", "loss"]))
    if not parts:
        return None
    # No step floor: the report's window table reads the mean of the first 34
    # logged steps, so the figure must start at the first logged step too.
    return pd.concat(parts, ignore_index=True).sort_values("step").reset_index(drop=True)


def attn_curve(stem: str):
    parts = []
    for suffix in ("", "_r2", "_r3"):
        p = ATTN / f"{stem}{suffix}_attn_amplitude.csv"
        if p.exists():
            parts.append(pd.read_csv(p, usecols=["step", "qk_logit_maxabs"]))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    return df.groupby("step", as_index=False)["qk_logit_maxabs"].mean()


fig, axes = plt.subplots(2, 3, figsize=(15, 7.4))

for col, (template, colour, flagged, label) in enumerate(GROUPS):
    ax_loss, ax_attn = axes[0][col], axes[1][col]
    for variant in SETTINGS:
        stem = _stem(template, variant)
        hot = variant == flagged
        style = dict(color=colour, lw=2.0, zorder=3) if hot else \
            dict(color=OTHER, lw=1.0, zorder=1)
        d = loss_curve(stem)
        if d is not None and not d.empty:
            ax_loss.plot(d["step"], d["loss"], **style)
        a = attn_curve(stem)
        if a is not None and not a.empty:
            ax_attn.plot(a["step"], a["qk_logit_maxabs"], **style)
    for ax in (ax_loss, ax_attn):
        ax.set_xscale("log")
        ax.grid(True, color=GRID, alpha=0.6, which="both")
    ax_attn.set_yscale("log")
    ax_loss.set_title(label, fontsize=10, color=colour)
    ax_attn.set_xlabel("training step (log)")

axes[0][0].set_ylabel("backbone training loss")
axes[1][0].set_ylabel("mean qk logit magnitude (log)")

fig.suptitle("Backbone loss and attention logit magnitude of the highest "
             "GM-Relative MASE cell at each backbone step (coloured), against "
             "the other settings of the same recipe (grey)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = HERE / "per_run_loss.png"
fig.savefig(out)
print(f"wrote {out}")
