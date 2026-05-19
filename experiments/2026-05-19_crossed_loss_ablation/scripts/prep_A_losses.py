#!/usr/bin/env python3
"""Materialise a clean (A) losses CSV for this experiment.

(A)'s #296 backbone run logged its losses CSV once per DDP rank, so
every step appears twice with byte-identical (all-reduced) metrics —
100000 rows for 50000 steps. We do NOT rewrite the #296 artifact (it is
another experiment's record, referenced by #296/#300). Instead we emit a
deduplicated, step-sorted copy into this experiment's results/ and the
plots read that. B/C/A+B's CSVs are already one-row-per-step (this run's
train.py logs rank-0 only) and are used as-is.

Idempotent. Run: python3 prep_A_losses.py
"""
import csv
import os

SRC = ("/home/jupyter/cf-encoder-forecaster-v2/experiments/"
       "2026-05-17_bottleneck_fullfh_ddp/runs/"
       "enc_fcst_bneck128_dk07_fullfh_norminfonce_1L_fp16_ddp128_50k_losses.csv")
DST = ("/home/jupyter/contrastive-forecasting/experiments/"
       "2026-05-19_crossed_loss_ablation/results/A_ref_losses_clean.csv")


def main():
    with open(SRC) as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames
        by_step = {}
        for r in rd:
            try:
                s = int(float(r["step"]))
            except (ValueError, KeyError, TypeError):
                continue
            by_step[s] = r          # identical dups → last wins (no-op)
    rows = [by_step[s] for s in sorted(by_step)]
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    src_n = sum(1 for _ in open(SRC)) - 1
    print(f"src rows={src_n}  unique steps={len(rows)}  "
          f"step range [{rows[0]['step']}, {rows[-1]['step']}]")
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
