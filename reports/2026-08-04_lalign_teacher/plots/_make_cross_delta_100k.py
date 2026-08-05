"""The ten retrained cells against the earlier sweep, at backbone 100k.

Same construction as the right panel of `controlled_vs_cross_delta.png`, one
backbone step later: one horizontal bar per cell for
`GM_teacher - GM_earlier_sweep`, with the dataset-level paired cluster
bootstrap 95% interval rescaled from the ratio onto the difference scale.
Reads the `bb_steps = 100000` rows of `results/eval_bootstrap_ci.csv`.

This comparison moves the flag and the code snapshot together.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"

ARMS = ["arm5", "arm5_tr1", "arm5_nse", "arm5_ncpc", "arm5_combab",
        "arm6_v2", "arm6_v2_tr1", "arm6_v2_nse", "arm6_v2_ncpc",
        "arm6_v2_combab"]
LABEL = {"arm5": "arm5 base", "arm5_tr1": "arm5 tr1", "arm5_nse": "arm5 nse",
         "arm5_ncpc": "arm5 ncpc", "arm5_combab": "arm5 combab",
         "arm6_v2": "arm6_v2 base", "arm6_v2_tr1": "arm6_v2 tr1",
         "arm6_v2_nse": "arm6_v2 nse", "arm6_v2_ncpc": "arm6_v2 ncpc",
         "arm6_v2_combab": "arm6_v2 combab"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
TEACHER_LOWER, STUDENT_LOWER = "#2a78d6", "#c04040"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

with open(RESULTS / "eval_bootstrap_ci.csv", newline="") as fh:
    cross = {r["arm_slug"]: r for r in csv.DictReader(fh)
             if r["bb_steps"] == "100000"}

fig, ax = plt.subplots(figsize=(8.2, 5.4))
ys = list(range(len(ARMS)))[::-1]
n_lower = 0
for y, arm in zip(ys, ARMS):
    r = cross.get(arm)
    if not r:
        continue
    earlier = float(r["gm_rel_mase_student"])
    d = float(r["gm_rel_mase_teacher"]) - earlier
    lo = (float(r["ci95_lo_dataset_boot"]) - 1.0) * earlier
    hi = (float(r["ci95_hi_dataset_boot"]) - 1.0) * earlier
    n_lower += d < 0
    c = TEACHER_LOWER if d < 0 else STUDENT_LOWER
    ax.barh(y, d, color=c, alpha=0.45, height=0.6, hatch="//", edgecolor=c)
    ax.plot([lo, hi], [y, y], color=INK, lw=1.4)
    for x in (lo, hi):
        ax.plot([x, x], [y - 0.14, y + 0.14], color=INK, lw=1.4)
    ax.text(min(lo, d) - 0.014 if d < 0 else max(hi, d) + 0.014, y,
            f"{d:+.4f}", va="center", ha="right" if d < 0 else "left",
            fontsize=8)
ax.axvline(0.0, color=INK, lw=1.0)

with open(RESULTS / "eval_paired_tests.csv", newline="") as fh:
    tests = [r for r in csv.DictReader(fh) if r["bb_steps"] == "100000"]
sign_p = float(tests[0]["sign_test_p"]) if tests else float("nan")
ax.text(0.98, 0.03,
        f"{n_lower} of {len(cross)} cells lower under the teacher target,\n"
        f"sign test p = {sign_p:.2f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
        color=INK)

ax.set_yticks(ys)
ax.set_yticklabels([LABEL[a] for a in ARMS])
ax.grid(True, axis="x", color=GRID, alpha=0.6)
ax.set_xlim(-0.36, 0.78)
ax.set_xlabel("GM-Relative MASE(teacher) − GM-Relative MASE(earlier sweep)\n"
              "whiskers = 95% dataset-cluster bootstrap on the ratio, rescaled")
ax.set_title("Teacher − earlier sweep GM-Relative MASE, backbone 100k\n"
             "flag and code snapshot differ together", fontsize=10.5)
fig.tight_layout()
out = HERE / "cross_delta_100k.png"
fig.savefig(out)
print(f"wrote {out}  ({len(cross)} cells, {n_lower} lower, sign p={sign_p:.2f})")
