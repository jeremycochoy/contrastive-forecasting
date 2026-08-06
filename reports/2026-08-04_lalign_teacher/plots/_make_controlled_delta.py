"""Headline figure: the flag measured against a control on the same branch.

Left panel — the controlled comparison. Teacher and student at backbone
step 40k, both trained on this branch, same backbone seed, same head seed,
same code. Bars are `GM_teacher - GM_student` at head seed 20260722, with
the dataset-level paired cluster-bootstrap 95% interval. Negative is the
teacher target scoring lower.

**Two of the ten cells carry three head seeds on both sides.** Their three
per-seed deltas are drawn as open dots with the 3-seed mean as a diamond,
so the reader sees how far the delta itself moves under nothing but the
head seed. `arm5 base` changes sign across the three. The other eight ran
one head seed each; they are labelled `1 seed` and get no spread, because
an unmeasured spread is not a small one.

Right panel — the same ten arms compared instead against the earlier
sweep's student numbers. That comparison moves the flag and the code
snapshot together.

Both read `results/controlled_delta_40k.csv` and `results/eval_bootstrap_ci.csv`.
A cell whose same-branch control is still training is absent from that
file and is drawn as an empty slot labelled "pending".
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
# The report's cell names, so one cell never carries two names.
LABEL = {"arm5": "arm5 base", "arm5_tr1": "arm5 tr1", "arm5_nse": "arm5 nse",
         "arm5_ncpc": "arm5 ncpc", "arm5_combab": "arm5 combab",
         "arm6_v2": "arm6_v2 base", "arm6_v2_tr1": "arm6_v2 tr1",
         "arm6_v2_nse": "arm6_v2 nse", "arm6_v2_ncpc": "arm6_v2 ncpc",
         "arm6_v2_combab": "arm6_v2 combab"}
ARM_COLOR = {"arm5": "#8b1e8b", "arm6_v2": "#b8860b"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
TEACHER_LOWER, STUDENT_LOWER = "#2a78d6", "#c04040"
SEED_INK = "#0f6f6f"   # same teal the head-seed figure uses for 40k cells

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


ctrl = {r["arm_slug"]: r for r in load(RESULTS / "controlled_delta_40k.csv")}
cross = {r["arm_slug"]: r for r in load(RESULTS / "eval_bootstrap_ci.csv")
         if r["bb_steps"] == "40000"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
ys = list(range(len(ARMS)))[::-1]

# --- left: controlled -------------------------------------------------------
ax = axes[0]
n_pending = 0
for y, arm in zip(ys, ARMS):
    r = ctrl.get(arm, {})
    if not r:
        n_pending += 1
        ax.text(0.0, y, "  pending", va="center", ha="left",
                fontsize=9, color=MUTED, style="italic")
        continue
    d = float(r["delta_controlled"])
    lo, hi = float(r["delta_ci95_lo"]), float(r["delta_ci95_hi"])
    c = TEACHER_LOWER if d < 0 else STUDENT_LOWER
    ax.barh(y, d, color=c, alpha=0.85, height=0.6)
    ax.plot([lo, hi], [y, y], color=INK, lw=1.4)
    ax.plot([lo, lo], [y - 0.14, y + 0.14], color=INK, lw=1.4)
    ax.plot([hi, hi], [y - 0.14, y + 0.14], color=INK, lw=1.4)
    ax.text(min(lo, d) - 0.012 if d < 0 else max(hi, d) + 0.012, y,
            f"{d:+.4f}", va="center",
            ha="right" if d < 0 else "left", fontsize=8)
    # The head-seed axis, drawn only where it was measured. Eight cells ran
    # one head seed; they say so rather than borrow arm5's spread.
    if r["head_seed_spread_measured"] == "yes":
        per_seed = [float(v) for v in r["delta_per_seed"].split()]
        ax.plot(per_seed, [y + 0.3] * len(per_seed), "o", mfc="none",
                mec=SEED_INK, mew=1.3, ms=6, zorder=4)
        ax.plot([min(per_seed), max(per_seed)], [y + 0.3, y + 0.3],
                color=SEED_INK, lw=1.0, zorder=3)
        ax.plot([float(r["delta_seed_mean"])], [y + 0.3], "D",
                color=SEED_INK, ms=5, zorder=5)
    else:
        ax.text(-0.355, y + 0.33, "1 head seed, no spread measured",
                va="center", ha="left", fontsize=7, color=MUTED,
                style="italic")
ax.axvline(0.0, color=INK, lw=1.0)

# Aggregate over the ten cells, on both head-seed bases. Neither subsumes
# the other: the first is one seed everywhere, the second folds in the two
# cells that were replicated.
tests = load(RESULTS / "controlled_paired_tests_40k.csv")
agg = next(r for r in tests
           if r["comparison"] == "controlled" and r["bb_steps"] == "40000")
agg_sm = next((r for r in tests
               if r["comparison"] == "controlled_seed_mean"
               and r["bb_steps"] == "40000"), None)
mean_d = float(agg["mean_delta"])
ax.axvline(mean_d, color=INK, lw=1.2, ls="--")
note = (f"dashed line = mean over the 10 cells, {mean_d:+.4f}\n"
        f"sign test p = {float(agg['sign_test_p']):.2f},  "
        f"Wilcoxon p = {float(agg['wilcoxon_p']):.2f}")
if agg_sm is not None:
    ax.axvline(float(agg_sm["mean_delta"]), color=SEED_INK, lw=1.2, ls=":")
    note += (f"\ndotted = same mean with the 3-seed mean on the 2 replicated "
             f"cells, {float(agg_sm['mean_delta']):+.4f}\n"
             f"sign test p = {float(agg_sm['sign_test_p']):.2f},  "
             f"Wilcoxon p = {float(agg_sm['wilcoxon_p']):.2f}")
ax.text(-0.35, len(ARMS) - 0.35, note,
        ha="left", va="bottom", fontsize=8, color=INK)
ax.set_ylim(-0.6, len(ARMS) + 1.5)

ax.set_title("Controlled: only the align-target flag differs\n"
             "teal open dots = the delta under each of 3 head seeds, diamond "
             "= their mean", fontsize=10)
ax.set_xlabel("GM-Relative MASE(teacher) − GM-Relative MASE(student)\n"
              "negative = teacher target lower;  whiskers = 95% dataset-cluster "
              "bootstrap")
ax.set_xlim(-0.36, 0.24)

# --- right: cross-experiment -----------------------------------------------
ax = axes[1]
for y, arm in zip(ys, ARMS):
    r = cross.get(arm)
    if not r:
        continue
    earlier = float(r["gm_rel_mase_student"])
    d = float(r["gm_rel_mase_teacher"]) - earlier
    # The CSV carries the ratio interval; on the difference scale that is
    # (ratio - 1) x earlier, so both panels show a 95% interval per cell.
    lo = (float(r["ci95_lo_dataset_boot"]) - 1.0) * earlier
    hi = (float(r["ci95_hi_dataset_boot"]) - 1.0) * earlier
    c = TEACHER_LOWER if d < 0 else STUDENT_LOWER
    ax.barh(y, d, color=c, alpha=0.45, height=0.6, hatch="//",
            edgecolor=c)
    ax.plot([lo, hi], [y, y], color=INK, lw=1.4)
    ax.plot([lo, lo], [y - 0.14, y + 0.14], color=INK, lw=1.4)
    ax.plot([hi, hi], [y - 0.14, y + 0.14], color=INK, lw=1.4)
    ax.text(min(lo, d) - 0.012 if d < 0 else max(hi, d) + 0.012, y,
            f"{d:+.4f}", va="center",
            ha="right" if d < 0 else "left", fontsize=8)
ax.axvline(0.0, color=INK, lw=1.0)
ax.set_title("Against the earlier sweep", fontsize=10)
ax.set_xlabel("GM-Relative MASE(teacher) − GM-Relative MASE(earlier sweep)\n"
              "whiskers = 95% dataset-cluster bootstrap on the ratio, rescaled")
ax.set_xlim(-0.36, 0.24)

for ax in axes:
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[a] for a in ARMS])
    ax.grid(True, axis="x", color=GRID, alpha=0.6)

fig.suptitle("Teacher − student GM-Relative MASE, backbone 40k",
             fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = HERE / "controlled_vs_cross_delta.png"
fig.savefig(out)
print(f"wrote {out}  ({len(ARMS) - n_pending} controlled, "
      f"{n_pending} pending, {len(cross)} cross-experiment)")
