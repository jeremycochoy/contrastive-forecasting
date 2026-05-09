#!/usr/bin/env python3
"""τ-sweep long-trajectory plot — τ=0.10 and τ=0.20 extended to 50k steps,
compared against the backbone-β_167k single-batch reference values
(horizontal dashed lines).

Each arm's trajectory is the concatenation of its 0–15k initial run and
its 15k–50k resume run:

  τ=0.10 short  : sync_tau_sweep/checkpoints/tau_sweep_0_10_losses.csv
  τ=0.10 long   : sync_tau_sweep/checkpoints/tau_sweep_0_10_50k_losses.csv
  τ=0.20 short  : sync_tau_sweep_arm5_v2/checkpoints/tau_sweep_0_20_v2_losses.csv
  τ=0.20 long   : sync_tau_sweep/checkpoints/tau_sweep_0_20_50k_losses.csv

If a long-run CSV is missing (training not yet finished / launched), the
plot still renders, showing only the short segment for that arm.

backbone-β_167k is shown as a dashed horizontal reference line in each
panel (its per-step trajectory CSV predates the per-batch diagnostic
metrics, so we only have the held-out single-batch value).

Output: experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_long_trajectories.png
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
SYNC = REPO / "sync_tau_sweep/checkpoints"
SYNC_V2 = REPO / "sync_tau_sweep_arm5_v2/checkpoints"
SYNC_010_50K = REPO / "sync_tau_sweep_0_10_50k/checkpoints"
SYNC_020_50K = REPO / "sync_tau_sweep_0_20_50k/checkpoints"
OUT = REPO / "experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_long_trajectories.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# (display_label, color, [csv_segments]).
#
# Each arm's full 0–50k trajectory is the concatenation of:
#   1. original 0–15k run (loaded from the τ-sweep sync dir),
#   2. first preempted 50k attempt (~15k–24k for τ=0.10, ~15k–20k for τ=0.20),
#   3. second resume's _r2 CSV (resumed from best_loss; covers ~24k or
#      ~20k → 50k).
# `load_concat` sorts by step and dedupes — overlapping rows where (2) and
# (3) cover the same step are kept once (latest write wins via stable sort).
# Missing segments are silently skipped, so the plot is robust to partial
# state.
ARMS = [
    ("τ=0.10",  "#d62728",
     [SYNC         / "tau_sweep_0_10_losses.csv",
      SYNC_010_50K / "tau_sweep_0_10_50k_losses.csv",
      SYNC_010_50K / "tau_sweep_0_10_50k_r2_losses.csv"]),
    ("τ=0.20",  "#ff7f0e",
     [SYNC_V2      / "tau_sweep_0_20_v2_losses.csv",
      SYNC_020_50K / "tau_sweep_0_20_50k_losses.csv",
      SYNC_020_50K / "tau_sweep_0_20_50k_r2_losses.csv"]),
]

# backbone-β_167k held-out single-batch reference values (BETA_REF dict
# matches the τ-sweep multisample plot).
BETA = dict(r2_random=0.6839, r2_naive=0.6080, u_temporal=0.0375,
            u_batch=0.0762, auc=0.8966, top1=0.7531)


def smooth(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def load_segment(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    return dict(
        step=np.array([int(r["step"]) for r in rows]),
        r2_random=np.array([float(r["r2_random"]) for r in rows]),
        r2_naive=np.array([float(r["r2_naive"]) for r in rows]),
        u_temporal=np.array([float(r["u_temporal"]) for r in rows]),
        u_batch=np.array([float(r["u_batch"]) for r in rows]),
        auc=np.array([float(r["auc"]) for r in rows]),
        top1=np.array([float(r["top1"]) for r in rows]),
    )


def load_concat(paths: list[Path]) -> dict | None:
    """Concatenate trajectory CSVs by step, sort, and dedupe.

    Where multiple segments cover the same step (e.g. preempted attempt
    + r2 resume), the row from the later-listed segment wins (np.unique
    with return_index keeps the first occurrence; we reverse-sort so the
    first occurrence is the latest one). Missing files silently skipped.
    """
    parts = [load_segment(p) for p in paths]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    keys = list(parts[0])
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    # Stable-sort by step ascending; for duplicates keep the last
    # occurrence (the later-listed segment, which is the most recent
    # training).
    order = np.argsort(out["step"], kind="stable")
    out = {k: v[order] for k, v in out.items()}
    # Dedupe: scan from end keeping last write per step.
    _, keep_rev_idx = np.unique(out["step"][::-1], return_index=True)
    keep_idx = np.sort(len(out["step"]) - 1 - keep_rev_idx)
    out = {k: v[keep_idx] for k, v in out.items()}
    return out


def main() -> None:
    traj: dict[str, dict] = {}
    for label, _color, paths in ARMS:
        t = load_concat(paths)
        if t is not None:
            traj[label] = t

    # Per-metric panels — same zooms as the short-trajectory plot.
    metrics = [
        ("r2_random", "R²_random", (0.60, 0.92), BETA["r2_random"], 500),
        ("r2_naive",  "R²_naive",  (0.45, 0.88), BETA["r2_naive"],  500),
        ("u_temporal", "U_temporal", None,       BETA["u_temporal"], 500),
        ("u_batch",   "U_batch",     None,       BETA["u_batch"],    500),
        ("auc",       "AUC",        (0.882, 0.910), BETA["auc"],    175),
        ("top1",      "Top-1",      (0.72, 0.77),   BETA["top1"],   175),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs = axs.flatten()
    for ax, (key, title, ylim, beta_ref, smooth_w) in zip(axs, metrics):
        for label, color, _ in ARMS:
            t = traj.get(label)
            if t is None:
                continue
            arr = t[key]
            sm = smooth(arr, smooth_w)
            x = t["step"][len(t["step"]) - len(sm):] if len(sm) > 0 else t["step"]
            ax.plot(x, sm if len(sm) > 0 else arr, color=color,
                    label=label, linewidth=1.8)
        # backbone-β reference line. Marked at step 167000 with a dot so
        # the reader sees both the value and the step at which it was
        # measured.
        ax.axhline(beta_ref, color="gray", linestyle="--", linewidth=1.0,
                   label=f"backbone-β 167k = {beta_ref:.4f}")
        ax.scatter([167000], [beta_ref], color="gray", marker="o",
                   s=30, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("step")
        if ylim is not None:
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        "τ-sweep long trajectories — τ=0.10 and τ=0.20 extended to 50k "
        "vs backbone-β_167k reference (R² / U: 500-step MA, AUC / Top-1: "
        "175-step MA).",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
