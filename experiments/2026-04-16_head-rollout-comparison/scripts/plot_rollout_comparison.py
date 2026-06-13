#!/usr/bin/env python3
"""Recompute B1-B4 GM-Relative MASE and plot the 6-variant headline figure.

GM-Relative MASE = geometric mean over the 97 GIFT-Eval configs of
    (model MASE[0.5]) / (seasonal-naive MASE)

The B1-B4 eval CSVs (results/B{1,2,3,4}/all_results.csv) hold per-config
MASE[0.5] but NOT the seasonal-naive baseline. Seasonal-naive (SN_MASE) is
dataset-intrinsic and identical across experiments, so we join it in by config
string from the v2 GIFT-Eval summary
(../2026-04-13_gift-eval/results/v2_pair_30k_summary.txt).

A1=1.275 and A2=1.262 are cited prior/baseline value-space evals (A1 is the v2
baseline documented in notes/EXECUTION_PLAN.md and notes/DESIGN.md); their
per-config CSVs are not committed here, so they are plotted as cited values,
not recomputed.

Run from anywhere; all paths are relative to this script's location.
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
SN_SUMMARY = os.path.join(
    EXP, "..", "2026-04-13_gift-eval", "results", "v2_pair_30k_summary.txt"
)


def load_sn_mase(path):
    """Parse Config -> SN_MASE from the GIFT-Eval summary table."""
    rows = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            # Data rows: 'config  MASE  SN_MASE  Relative' with '/' in config.
            if len(parts) == 4 and "/" in parts[0]:
                try:
                    rows[parts[0]] = float(parts[2])
                except ValueError:
                    continue
    return pd.Series(rows, name="SN_MASE")


def gm_rel_mase(variant):
    """Recompute GM-Relative MASE for one B-variant from committed data."""
    csv = os.path.join(EXP, "results", variant, "all_results.csv")
    df = pd.read_csv(csv)
    df = df.set_index("dataset")
    model_mase = df["eval_metrics/MASE[0.5]"]
    sn = load_sn_mase(SN_SUMMARY)
    joined = pd.concat([model_mase, sn], axis=1, join="inner")
    n_model = len(model_mase)
    n_join = len(joined)
    rel = joined["eval_metrics/MASE[0.5]"] / joined["SN_MASE"]
    gm = float(np.exp(np.mean(np.log(rel))))
    return gm, n_model, n_join


# --- Recompute B1-B4 -------------------------------------------------------
recomputed = {}
print(f"{'variant':<8}{'GM-Rel':>10}{'n_model':>10}{'n_joined':>10}")
for v in ["B1", "B2", "B3", "B4"]:
    gm, nm, nj = gm_rel_mase(v)
    recomputed[v] = gm
    print(f"{v:<8}{gm:>10.4f}{nm:>10d}{nj:>10d}")

# A1/A2 are cited prior values (no per-config CSV committed here).
A1, A2 = 1.275, 1.262

# --- Headline bar chart ----------------------------------------------------
order = ["A1", "A2", "B1", "B2", "B3", "B4"]
vals = {
    "A1": A1,
    "A2": A2,
    "B1": recomputed["B1"],
    "B2": recomputed["B2"],
    "B3": recomputed["B3"],
    "B4": recomputed["B4"],
}
# Value-space (A) vs latent-space (B) coloured distinctly.
colors = {
    "A1": "#c44e52",
    "A2": "#c44e52",  # value-space: red
    "B1": "#4c72b0",
    "B2": "#4c72b0",
    "B3": "#4c72b0",
    "B4": "#4c72b0",  # latent-space: blue
}
labels = {
    "A1": "A1\nvalue, slide-128",
    "A2": "A2\nvalue, slide-16",
    "B1": "B1\nlatent, decode-end",
    "B2": "B2\nlatent, crop-16",
    "B3": "B3\nlatent, decode-8",
    "B4": "B4\nlatent, W=16",
}

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(order))
bar_vals = [vals[v] for v in order]
bar_cols = [colors[v] for v in order]
bars = ax.bar(x, bar_vals, color=bar_cols, width=0.62, zorder=3)

# Seasonal-naive reference (1.0).
ax.axhline(1.0, color="green", ls="--", lw=1.4, zorder=2)
ax.text(
    len(order) - 0.45, 1.005, "seasonal-naive = 1.0",
    color="green", ha="right", va="bottom", fontsize=9,
)

# ~1.27 plateau band (min..max of the 6 variants).
lo, hi = min(bar_vals), max(bar_vals)
ax.axhspan(lo, hi, color="grey", alpha=0.13, zorder=1)
ax.text(
    -0.45, hi + 0.002,
    f"~1.27 plateau  ({lo:.3f}-{hi:.3f})",
    color="dimgrey", ha="left", va="bottom", fontsize=9,
)

for xi, v in zip(x, order):
    ax.text(xi, vals[v] + 0.0015, f"{vals[v]:.3f}", ha="center", va="bottom",
            fontsize=9, zorder=4)

ax.set_xticks(x)
ax.set_xticklabels([labels[v] for v in order], fontsize=8.5)
ax.set_ylabel("GM-Relative MASE  (lower is better)")
ax.set_ylim(0.98, hi + 0.03)
ax.set_title("All 6 head/rollout variants cluster on the ~1.27 MASE plateau")

# Legend for the two rollout families.
from matplotlib.patches import Patch

ax.legend(
    handles=[
        Patch(facecolor="#c44e52", label="value-space rollout (A)"),
        Patch(facecolor="#4c72b0", label="latent-space rollout (B)"),
    ],
    loc="upper center", frameon=False, fontsize=9, ncol=2,
)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

out = os.path.join(EXP, "plots", "rollout_comparison.png")
fig.savefig(out, dpi=130)
print(f"\nwrote {out}")
print("A1, A2 are cited prior value-space evals (not recomputed here).")
