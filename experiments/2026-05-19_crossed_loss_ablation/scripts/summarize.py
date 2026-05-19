#!/usr/bin/env python3
"""Extract every number the RESULTS.md table / verdict needs, in one
place, so the report fill-in is mechanical. Prints, per arm:
  triage GM(11) · full GM(97) · final loss · final loss_tau_ref ·
  final 1-AUC · final u_temporal/u_batch · NaN-in-loss? · #csv rows.
Robust to not-yet-finished arms (prints '—')."""
import csv, math, os

A17 = "/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-17_bottleneck_fullfh_ddp"
ART = "/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_ablation"
A_NAME = "enc_fcst_bneck128_dk07_fullfh_norminfonce_1L_fp16_ddp128_50k"

ARMS = [
    ("(A) full_fh_negs", f"{A17}/runs/{A_NAME}_losses.csv",
     f"{A17}/results/gift_eval_triage_{A_NAME}",
     f"{A17}/results/gift_eval_full_{A_NAME}"),
    ("(B) full_hh_negs", f"{ART}/runs/cl_hh_50k_losses.csv",
     f"{ART}/results/gift_eval_triage_cl_hh_50k",
     f"{ART}/results/gift_eval_full_cl_hh_50k"),
    ("(C) full_ff_negs", f"{ART}/runs/cl_ff_50k_losses.csv",
     f"{ART}/results/gift_eval_triage_cl_ff_50k",
     f"{ART}/results/gift_eval_full_cl_ff_50k"),
    ("(A)+(B) full_fh_hh_negs", f"{ART}/runs/cl_fhhh_50k_losses.csv",
     f"{ART}/results/gift_eval_triage_cl_fhhh_50k",
     f"{ART}/results/gift_eval_full_cl_fhhh_50k"),
]


def gm(d):
    p = f"{d}/summary.txt"
    if not os.path.exists(p):
        return None
    for line in open(p):
        if "Aggregate GM-Relative MASE" in line:
            for t in reversed(line.replace(":", " ").split()):
                try:
                    return float(t)
                except ValueError:
                    pass
    return None


def last_row(csv_path):
    if not os.path.exists(csv_path):
        return None, 0, False
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None, 0, False
    nan = False
    for r in rows:
        try:
            v = float(r["loss"])
            if not math.isfinite(v):
                nan = True
        except (ValueError, KeyError):
            nan = True
    return rows[-1], len(rows), nan


def f(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except (ValueError, TypeError):
        return "—"


print(f"{'arm':26s} {'triageGM':>9s} {'fullGM':>8s} {'step':>6s} "
      f"{'loss':>8s} {'tau_ref':>8s} {'1-AUC':>9s} {'u_tmp':>6s} "
      f"{'u_bat':>6s} {'NaN':>4s} {'rows':>6s}")
for lab, lc, td, fd in ARMS:
    row, n, nan = last_row(lc)
    tgm, fgm = gm(td), gm(fd)
    if row:
        step = row.get("step", "—")
        loss = f(row.get("loss"))
        tref = f(row.get("loss_tau_ref"))
        try:
            oneauc = f"{1.0 - float(row['auc']):.2e}"
        except (ValueError, KeyError):
            oneauc = "—"
        ut, ub = f(row.get("u_temporal"), 3), f(row.get("u_batch"), 3)
    else:
        step = loss = tref = oneauc = ut = ub = "—"
    print(f"{lab:26s} {f(tgm):>9s} {f(fgm):>8s} {str(step):>6s} "
          f"{loss:>8s} {tref:>8s} {oneauc:>9s} {ut:>6s} {ub:>6s} "
          f"{('YES' if nan else 'no'):>4s} {n:>6d}")
