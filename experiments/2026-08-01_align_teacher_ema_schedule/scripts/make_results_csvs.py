#!/usr/bin/env python3
"""Reduce the four #388 training CSVs to the small tables the report needs.

The raw ``<run>_losses.csv`` is ~30 MB per run (one row per step, 30
columns). Two slices of it matter here, so this writes them out and the
full files stay in the durable run directory:

  alpha_schedule.csv     run, step, ema_tau            (every --stride steps)
  loss_curve.csv         run, step, loss, loss_tau_ref (every --stride steps)
  drift_500.csv          run, arm, alpha, latent, kind, step, step_ref,
                         delta_step, drift_cos, drift_cos_aligned, rot_gap,
                         cka   — the in-training 500-step probe, verbatim
"""

from __future__ import annotations

import argparse
import csv
import os

# (run, arm, alpha) — the four runs #388 trained.
RUNS = [
    ("align_teacher_a09",   "align_teacher", "const_0.9"),
    ("align_teacher_sched", "align_teacher", "sched_0.9_1.0"),
    ("pred_moco_sched",     "pred_moco",     "sched_0.9_1.0"),
    ("rep_moco_sched",      "rep_moco",      "sched_0.9_1.0"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-388", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stride", type=int, default=100,
                   help="Keep every Nth step of the per-step CSVs.")
    return p.parse_args()


def run_paths(root, run):
    d = os.path.join(root, run)
    return (os.path.join(d, f"ats_{run}_losses.csv"),
            os.path.join(d, f"ats_{run}_latent_drift.csv"))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    alpha_path = os.path.join(args.out_dir, "alpha_schedule.csv")
    loss_path = os.path.join(args.out_dir, "loss_curve.csv")
    drift_path = os.path.join(args.out_dir, "drift_500.csv")

    with open(alpha_path, "w", newline="") as fa, \
            open(loss_path, "w", newline="") as fl, \
            open(drift_path, "w", newline="") as fd:
        wa, wl, wd = csv.writer(fa), csv.writer(fl), csv.writer(fd)
        wa.writerow(["run", "step", "ema_tau"])
        wl.writerow(["run", "step", "loss", "loss_tau_ref"])
        wd.writerow(["run", "arm", "alpha", "latent", "kind", "step",
                     "step_ref", "delta_step", "drift_cos",
                     "drift_cos_aligned", "rot_gap", "cka"])
        for run, arm, alpha in RUNS:
            losses, drift = run_paths(args.runs_388, run)
            n_a = n_d = 0
            if os.path.exists(losses):
                with open(losses) as fh:
                    for r in csv.DictReader(fh):
                        step = int(r["step"])
                        if step % args.stride and step != 1:
                            continue
                        if r.get("ema_tau"):
                            wa.writerow([run, step, r["ema_tau"]])
                            n_a += 1
                        wl.writerow([run, step, r["loss"], r["loss_tau_ref"]])
            if os.path.exists(drift):
                with open(drift) as fh:
                    for r in csv.DictReader(fh):
                        wd.writerow([run, arm, alpha, r["latent"], r["kind"],
                                     r["step"], r["step_ref"],
                                     r["delta_step"], r["drift_cos"],
                                     r["drift_cos_aligned"], r["rot_gap"],
                                     r["cka"]])
                        n_d += 1
            print(f"[{run}] alpha rows={n_a} drift rows={n_d}")
    for p in (alpha_path, loss_path, drift_path):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
