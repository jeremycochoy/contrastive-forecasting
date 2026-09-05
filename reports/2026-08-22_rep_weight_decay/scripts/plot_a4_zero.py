#!/usr/bin/env python3
"""The L_rep weight at 0.0 on the project's best cell, by align target.

WHY THIS FIGURE EXISTS. The card asks how to beat the project's best model,
the A4 cell `arm6_v2_combab_alignS` at k = 3. Two runs resume its
40,000-step checkpoint with the L_rep weight at 0.0 and train to 200,000
steps, one per align target. This is a DIFFERENT cell from every other run
of this card (k = 3, align on the student; the rest is k = 32, align on the
teacher).

WHAT IT SHOWS. One bar per run at the 200,000-step stop, against the
original A4 as the muted reference bar. The run whose contrastive AUC fell
to chance takes the alarm colour: that is a state, not a series. Each bar
carries its AUC at 200,000 steps.

The two scores are the `score_a4*_bb200k_h30k_student.txt` files under
results/, each the `Aggregate GM-Relative MASE (97 configs)` line of its
run's eval_local.log. The AUC of each run is the mean of the `AUC=` rows of
its `run_*_cf409_a4*.log` over steps 195,000 to 200,000. The reference mean
1.0651 is `reports/2026-08-20_a4_full_pass/a4_full_pass.md`, and its AUC is
the same statistic over the A4 losses CSV,
`reports/2026-08-08_rollout_depth/curves/r3/arm6_v2_combab_alignS__leg_200k__
cf393_arm6_v2_combab_alignS_cf373k3_r2_losses.csv`.

Usage:
  plot_a4_zero.py --results results/ --out plots/a4_zero.png
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("arm_style", HERE / "arm_style.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

# The published A4 at 200,000 steps: the mean of three head seeds of
# `reports/2026-08-20_a4_full_pass/a4_full_pass.md`, and its AUC per
# `reports/2026-08-08_rollout_depth/rollout_depth.md`.
A4_SCORE = 1.0651
A4_AUC = 0.95
# The AUC gate of this card. Below it a run lost the contrastive task.
GATE = 0.55


def read_run(results, tag):
    """(score, AUC over the last 5,000 steps) of one resumed run."""
    score = float((results / f"score_{tag}_bb200k_h30k_student.txt")
                  .read_text().strip())
    log = next(results.glob(f"run_*_cf409_{tag}.log"))
    step, tail = 0, []
    for line in log.read_text().splitlines():
        m = re.match(r"\[\s*(\d+)\]", line)
        if m:
            step = int(m.group(1))
            continue
        m = re.search(r"AUC=([0-9.]+)", line)
        if m and step > 195000:
            tail.append(float(m.group(1)))
    return score, sum(tail) / len(tail)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default=str(HERE.parent / "results"))
    p.add_argument("--out", default=str(HERE.parent / "plots" / "a4_zero.png"))
    args = p.parse_args(argv)
    results = Path(args.results)

    teach = read_run(results, "a4teach")
    zero = read_run(results, "a4zero")
    rows = [
        ("student target, weight 1.0\nthe original A4", A4_SCORE, A4_AUC,
         S.REFERENCE),
        ("teacher target, weight 0.0", teach[0], teach[1], S.SERIES),
        ("student target, weight 0.0", zero[0], zero[1],
         S.LOST if zero[1] < GATE else S.SERIES),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ys = range(len(rows) - 1, -1, -1)
    for y, (label, score, auc, colour) in zip(ys, rows):
        ax.barh(y, score, height=0.62, color=colour, zorder=2)
        ax.text(score + 0.02, y, f"{score:.4f}   AUC {auc:.2f} at 200k",
                va="center", fontsize=8.5, color=S.INK)
    ax.axvline(A4_SCORE, color=S.INK, linestyle=":", linewidth=0.9, zorder=3)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, max(r[1] for r in rows) + 0.55)
    ax.set_xlabel("GM-Relative MASE at the 200,000-step stop "
                  "(lower is better)")
    ax.set_title("The A4 cell (k = 3) resumed at 40,000 steps with the "
                 "L_rep weight at 0.0", color=S.INK, fontsize=10, loc="left")
    S.tidy(ax)
    ax.grid(axis="y", visible=False)
    ax.plot([], [], marker="s", linestyle="none", color=S.REFERENCE,
            markersize=8, label="published reference")
    ax.plot([], [], marker="s", linestyle="none", color=S.SERIES,
            markersize=8, label="held the contrastive task")
    ax.plot([], [], marker="s", linestyle="none", color=S.LOST,
            markersize=8, label="lost it, AUC at chance")
    ax.legend(frameon=False, fontsize=8, labelcolor=S.INK, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.28))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
    print(f"{args.out}: a4teach {teach[0]:.4f} AUC {teach[1]:.3f}, "
          f"a4zero {zero[0]:.4f} AUC {zero[1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
