#!/usr/bin/env python3
"""Stop-grad follow-up — GM-Relative MASE + paired-bootstrap Δ vs the
reference (#328 L3+nobn+triplet, no stop-grad), best AND last checkpoints,
2L and 6L heads, full-97. GM/bootstrap logic identical to #328's analyze.py.

Writes results/gm_table.csv and prints the table.
"""
import csv
import math
import os
import random

EXP = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-10_stopgrad_positive"
REF_RES = f"{EXP}/results/reference"
RES = f"{EXP}/results"
REF_TAG = "allt08_xftrip_nobn_enc3_qk_aon_b1024"
SG_TAG = "allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024"
HEADS = ["2L", "6L"]
CKPTS = [("best", ""), ("last", "_last")]


def relatives(sum_txt):
    out = {}
    if not os.path.exists(sum_txt):
        return out
    for line in open(sum_txt):
        p = line.split()
        if len(p) == 4 and "/" in p[0]:
            try:
                out[p[0]] = float(p[3])
            except ValueError:
                pass
    return out


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def paired_delta_ci(da, db, n=2000, seed=0):
    """Δ = GM(db) − GM(da) with a 90% CI from a paired bootstrap over the
    common task list (resampling tasks with repeats, scoring both models on
    each resample so per-task difficulty cancels)."""
    common = sorted(set(da) & set(db))
    if len(common) < 2:
        return (None, None, None, 0)
    a = [da[c] for c in common]
    b = [db[c] for c in common]
    delta = gm(b) - gm(a)
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        idx = [rng.randrange(len(common)) for _ in common]
        ds.append(gm([b[i] for i in idx]) - gm([a[i] for i in idx]))
    ds.sort()
    return (delta, ds[int(0.05 * n)], ds[int(0.95 * n)], len(common))


def verdict(lo, hi):
    if lo is None:
        return "NA"
    if hi < 0:
        return "BETTER (reliable)"
    if lo > 0:
        return "worse (reliable)"
    return "ns (CI straddles 0)"


rows = []
for head in HEADS:
    for ck_name, ck_suffix in CKPTS:
        ref = relatives(f"{REF_RES}/gift_eval_full_{REF_TAG}{ck_suffix}_{head}/summary.txt")
        sg = relatives(f"{RES}/gift_eval_full_{SG_TAG}{ck_suffix}_{head}/summary.txt")
        if not ref or not sg:
            rows.append({"head": head, "ckpt": ck_name, "ref_gm": gm(list(ref.values())),
                         "sg_gm": None, "delta": None, "ci_lo": None, "ci_hi": None,
                         "n": len(ref), "verdict": "pending"})
            continue
        d, lo, hi, n = paired_delta_ci(ref, sg)
        rows.append({"head": head, "ckpt": ck_name,
                     "ref_gm": gm(list(ref.values())), "sg_gm": gm(list(sg.values())),
                     "delta": d, "ci_lo": lo, "ci_hi": hi, "n": n,
                     "verdict": verdict(lo, hi)})

os.makedirs(RES, exist_ok=True)
with open(f"{RES}/gm_table.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"{'head':<6}{'ckpt':<7}{'ref GM':>8}{'sg GM':>8}{'Δ':>9}{'  90% CI':>20}{'n':>5}   verdict")
for r in rows:
    fmt = lambda v, p="{:.4f}": p.format(v) if v is not None else "--"
    ci = (f"({r['ci_lo']:+.3f},{r['ci_hi']:+.3f})"
          if r["ci_lo"] is not None else "(--)")
    print(f"{r['head']:<6}{r['ckpt']:<7}{fmt(r['ref_gm']):>8}{fmt(r['sg_gm']):>8}"
          f"{fmt(r['delta'], '{:+.4f}'):>9}{ci:>20}{r['n']:>5}   {r['verdict']}")
print(f"\nwrote {RES}/gm_table.csv")
