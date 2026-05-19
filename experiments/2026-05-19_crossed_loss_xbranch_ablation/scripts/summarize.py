#!/usr/bin/env python3
"""#307 — one place for every number the RESULTS.md tables/verdict need.
Computes the 3 NEW arms from synced artifacts; the #303 A/B/C/A+B lines
are kept verbatim from #303 RESULTS.md (continuation, not recomputed).
Robust to not-yet-finished arms (prints '—')."""
import csv, math, os

SYNC = "/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_xbranch_ablation"

NEW = [
    ("(B)+(C) full_hh_ff_negs",     "hhff"),
    ("(A)+(B)+(C) full_fh_hh_ff",   "fhhhff"),
    ("(B) xbfree full_hh_negs_xbf", "hhxbf"),
]
# #303 published (RESULTS.md), kept as continuation rows:
REF303 = [
    ("(A) full_fh_negs  *#296*", "1.5611", "1.4377"),
    ("(B) full_hh_negs",         "1.4461", "1.3572"),
    ("(C) full_ff_negs",         "1.5185", "1.3822"),
    ("(A)+(B) full_fh_hh_negs",  "1.5426", "1.4517"),
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
            if not math.isfinite(float(r["loss"])):
                nan = True
        except (ValueError, KeyError):
            nan = True
    return rows[-1], len(rows), nan


def f(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except (ValueError, TypeError):
        return "—"


print("== #303 kept (published) ==")
for lab, t, fu in REF303:
    print(f"{lab:30s} triage={t} full={fu}")

print("\n== #307 NEW arms (computed from sync) ==")
print(f"{'arm':30s} {'triageGM':>9s} {'fullGM':>8s} {'step':>7s} "
      f"{'loss':>8s} {'tau_ref':>8s} {'1-AUC':>9s} {'u_tmp':>6s} "
      f"{'u_bat':>6s} {'NaN':>4s} {'rows':>6s}")
for lab, sh in NEW:
    base = f"{SYNC}/sync_{sh}"
    name = f"cl_{sh}_50k"
    row, n, nan = last_row(f"{base}/runs/{name}_losses.csv")
    tgm = gm(f"{base}/results/gift_eval_triage_{name}")
    fgm = gm(f"{base}/results/gift_eval_full_{name}")
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
    print(f"{lab:30s} {f(tgm):>9s} {f(fgm):>8s} {str(step):>7s} "
          f"{loss:>8s} {tref:>8s} {oneauc:>9s} {ut:>6s} {ub:>6s} "
          f"{('YES' if nan else 'no'):>4s} {n:>6d}")
