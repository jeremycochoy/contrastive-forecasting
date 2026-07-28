#!/usr/bin/env python3
"""Panel: latent drift per adjacent-checkpoint pair, per arm (#382).

Two options:
  1. Read the in-training latent-drift CSV each arm writes at
     ``lti_<arm>_latent_drift.csv``. Requires --latent-drift-probe was on
     during training. Columns: step, kind, step_ref, delta_step,
     drift_cos, drift_cos_aligned, rot_gap, cka.
  2. Fall back to the post-hoc probe scripts/latent_drift.py (see
     issue #382 "reuse" note); this script does NOT reimplement drift.

Plots ``drift_cos`` (unaligned) as h_t drift; e_t drift is derived by a
follow-up sub-agent using the same probe on the patch-embedding, so this
scaffold only reads what the training CSV exposes today (h_t).

Usage
-----
    python3 make_latent_movement_per_arm.py --exp-dir experiments/2026-07-28_loss_term_isolation
"""

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["pred", "rep", "align", "pred_moco",
        "rep_moco", "sigreg_e", "sigreg_h", "cpc"]


def load_drift(csv_path):
    """Return list of (step_ref, step, delta_step, drift_cos) for the
    'adjacent' rows only."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path) as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            if row.get("kind") != "adjacent":
                continue
            try:
                rows.append((int(row["step_ref"]), int(row["step"]),
                             int(row["delta_step"]),
                             float(row["drift_cos"])))
            except (KeyError, ValueError):
                continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    args = ap.parse_args()

    per_arm = {}
    for arm in ARMS:
        csv_path = os.path.join(
            args.exp_dir, "artifacts", arm, f"lti_{arm}_latent_drift.csv")
        per_arm[arm] = load_drift(csv_path)

    os.makedirs(os.path.join(args.exp_dir, "results"), exist_ok=True)
    long_csv = os.path.join(args.exp_dir, "results", "latent_movement_per_arm.csv")
    with open(long_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["arm", "step_ref", "step", "delta_step", "drift_cos_h"])
        for arm, rs in per_arm.items():
            for step_ref, step, delta, drift in rs:
                wr.writerow([arm, step_ref, step, delta, f"{drift:.6f}"])

    fig, axs = plt.subplots(3, 3, figsize=(11, 8), sharey=True, sharex=True)
    axs = axs.ravel()
    for i, arm in enumerate(ARMS):
        ax = axs[i]
        rs = per_arm[arm]
        if rs:
            steps = [r[1] for r in rs]
            drifts = [r[3] for r in rs]
            ax.plot(steps, drifts, lw=1.2, marker="o", ms=3)
        ax.set_title(arm)
        ax.set_xlabel("step")
        if i % 3 == 0:
            ax.set_ylabel("drift_cos (h_t)")
        ax.grid(alpha=0.3)
    axs[-1].axis("off")
    fig.suptitle("Latent drift per adjacent checkpoint pair (h_t only, #382)")
    fig.tight_layout()
    plots_dir = os.path.join(args.exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_png = os.path.join(plots_dir, "latent_movement_per_arm.png")
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")
    print(f"wrote {long_csv}")


if __name__ == "__main__":
    sys.exit(main())
