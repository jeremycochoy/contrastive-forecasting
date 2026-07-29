#!/usr/bin/env python3
"""Panel: 1 − ff per arm across training step (#382).

Reads each arm's ``artifacts/<arm>/lti_<arm>_losses.csv`` and plots
``1 − ff`` (i.e. cosine-error between forecaster f_t and encoder h_{t+1})
against ``step`` on a shared axis, one panel per arm.

Usage
-----
    python3 make_cos_error_per_arm.py --exp-dir experiments/2026-07-28_loss_term_isolation

Outputs
-------
    <exp-dir>/results/cos_error_per_arm.csv    long-form per (arm, step)
    <exp-dir>/plots/cos_error_per_arm.png      3×3 grid (8 arms + legend cell)
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["pred", "rep", "align", "pred_moco",
        "rep_moco", "sigreg_e", "sigreg_h", "cpc"]


def load_arm_curve(csv_path):
    """Return (steps, one_minus_ff) as two parallel lists — skips rows with
    blank / non-numeric `ff`."""
    steps, one_minus_ff = [], []
    if not os.path.exists(csv_path):
        return steps, one_minus_ff
    with open(csv_path) as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            try:
                s = int(row["step"])
                ff = float(row["ff"])
            except (KeyError, TypeError, ValueError):
                continue
            steps.append(s)
            one_minus_ff.append(1.0 - ff)
    return steps, one_minus_ff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    args = ap.parse_args()

    curves = {}
    for arm in ARMS:
        csv_path = os.path.join(
            args.exp_dir, "artifacts", arm, f"lti_{arm}_losses.csv")
        curves[arm] = load_arm_curve(csv_path)

    os.makedirs(os.path.join(args.exp_dir, "results"), exist_ok=True)
    long_csv = os.path.join(args.exp_dir, "results", "cos_error_per_arm.csv")
    with open(long_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["arm", "step", "one_minus_ff"])
        for arm, (steps, vals) in curves.items():
            for s, v in zip(steps, vals):
                wr.writerow([arm, s, f"{v:.6f}"])

    fig, axs = plt.subplots(3, 3, figsize=(11, 8), sharey=True, sharex=True)
    axs = axs.ravel()
    for i, arm in enumerate(ARMS):
        ax = axs[i]
        steps, vals = curves[arm]
        if steps:
            ax.plot(steps, vals, lw=1.2)
        ax.set_title(arm)
        ax.set_xscale("log")
        ax.set_xlabel("step")
        if i % 3 == 0:
            ax.set_ylabel("1 − ff")
        ax.grid(alpha=0.3)
    axs[-1].axis("off")
    fig.suptitle("1 − ff per arm across training step")
    fig.tight_layout()
    plots_dir = os.path.join(args.exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_png = os.path.join(plots_dir, "cos_error_per_arm.png")
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")
    print(f"wrote {long_csv}")


if __name__ == "__main__":
    sys.exit(main())
