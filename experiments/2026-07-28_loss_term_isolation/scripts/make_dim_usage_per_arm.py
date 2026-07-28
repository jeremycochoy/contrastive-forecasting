#!/usr/bin/env python3
"""Panel: u_batchtime per arm, h_t solid + e_t dashed (#382).

Reads each arm's ``lti_<arm>_losses.csv`` and plots ``u_batchtime`` (h_t)
and ``u_batchtime_e`` (e_t) against ``step`` on log-x, one panel per arm.

Only the sigreg_e arm writes ``u_batchtime_e`` every step (it captures
e_lat under either SIGReg flag); other arms will show a blank e_t curve.

Usage
-----
    python3 make_dim_usage_per_arm.py --exp-dir experiments/2026-07-28_loss_term_isolation
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


def load_arm(csv_path):
    rows = {"step": [], "u_bt": [], "u_bt_e": []}
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path) as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            try:
                s = int(row["step"])
            except (KeyError, ValueError):
                continue
            rows["step"].append(s)
            for k_src, k_dst in [("u_batchtime", "u_bt"),
                                 ("u_batchtime_e", "u_bt_e")]:
                v = row.get(k_src, "")
                try:
                    rows[k_dst].append(float(v))
                except (ValueError, TypeError):
                    rows[k_dst].append(None)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    args = ap.parse_args()

    per_arm = {}
    for arm in ARMS:
        csv_path = os.path.join(
            args.exp_dir, "artifacts", arm, f"lti_{arm}_losses.csv")
        per_arm[arm] = load_arm(csv_path)

    os.makedirs(os.path.join(args.exp_dir, "results"), exist_ok=True)
    long_csv = os.path.join(args.exp_dir, "results", "dim_usage_per_arm.csv")
    with open(long_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["arm", "step", "u_batchtime", "u_batchtime_e"])
        for arm, r in per_arm.items():
            for s, uh, ue in zip(r["step"], r["u_bt"], r["u_bt_e"]):
                wr.writerow([arm, s,
                             "" if uh is None else f"{uh:.6f}",
                             "" if ue is None else f"{ue:.6f}"])

    fig, axs = plt.subplots(3, 3, figsize=(11, 8), sharey=True)
    axs = axs.ravel()
    for i, arm in enumerate(ARMS):
        ax = axs[i]
        r = per_arm[arm]
        if r["step"]:
            steps = r["step"]
            h_pairs = [(s, v) for s, v in zip(steps, r["u_bt"]) if v is not None]
            e_pairs = [(s, v) for s, v in zip(steps, r["u_bt_e"]) if v is not None]
            if h_pairs:
                xs, ys = zip(*h_pairs)
                ax.plot(xs, ys, lw=1.2, label="u_bt (h_t)")
            if e_pairs:
                xs, ys = zip(*e_pairs)
                ax.plot(xs, ys, lw=1.2, ls="--", label="u_bt (e_t)")
        ax.set_title(arm)
        ax.set_xscale("log")
        ax.set_xlabel("step")
        if i % 3 == 0:
            ax.set_ylabel("u_batchtime")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    axs[-1].axis("off")
    fig.suptitle("u_batchtime per arm — h_t solid, e_t dashed (#382)")
    fig.tight_layout()
    plots_dir = os.path.join(args.exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_png = os.path.join(plots_dir, "dim_usage_per_arm.png")
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")
    print(f"wrote {long_csv}")


if __name__ == "__main__":
    sys.exit(main())
