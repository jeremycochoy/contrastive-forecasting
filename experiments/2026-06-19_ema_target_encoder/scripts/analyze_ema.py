#!/usr/bin/env python3
"""#353 — GM-Relative MASE + paired-bootstrap Δ for the EMA-target enc3+CPC
arm vs the #344 enc3+CPC baseline. GM / bootstrap byte-for-byte from
#341's analyze_sgcap.py (so the comparison cell is directly comparable to
the published #344 numbers).

Writes results/gm_table.csv and results/pairwise_table.csv; prints both.
"""
import csv
import math
import os
import random

W = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
E344 = f"{W}/2026-06-13_cpc_infonce_aux/results"
E353 = f"{W}/2026-06-19_ema_target_encoder/results"

ARMS = {
    "cpc_enc3": (
        "baseline enc3+CPC (#344)", E344,
        "allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_cpc"),
    "ema_enc3": (
        "EMA-target enc3+CPC (NEW)", E353,
        "allt08_xftrip_nobn_enc3_emateach_qk_aon_b1024_cpc"),
}
PAIRS = [
    ("cpc_enc3", "ema_enc3",
     "EMA-target vs --stopgrad-positive-h on enc3+CPC [hypothesis]"),
]
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
        return "B BETTER (reliable)"
    if lo > 0:
        return "B worse (reliable)"
    return "ns (CI straddles 0)"


def cell_rel(key, ck_suffix, head):
    label, d, tag = ARMS[key]
    return relatives(f"{d}/gift_eval_full_{tag}{ck_suffix}_{head}/summary.txt")


gm_rows = []
for key, (label, d, tag) in ARMS.items():
    for head in HEADS:
        for ck_name, ck_suffix in CKPTS:
            rel = cell_rel(key, ck_suffix, head)
            gm_rows.append({"arm": key, "label": label, "head": head,
                            "ckpt": ck_name, "gm": gm(list(rel.values())),
                            "n": len(rel)})

pair_rows = []
for a, b, why in PAIRS:
    for head in HEADS:
        for ck_name, ck_suffix in CKPTS:
            da, db = cell_rel(a, ck_suffix, head), cell_rel(b, ck_suffix, head)
            if not da or not db:
                pair_rows.append({"A": a, "B": b, "why": why, "head": head,
                                  "ckpt": ck_name, "gm_A": gm(list(da.values())),
                                  "gm_B": gm(list(db.values())), "delta": None,
                                  "ci_lo": None, "ci_hi": None, "n": 0,
                                  "verdict": "pending"})
                continue
            d, lo, hi, n = paired_delta_ci(da, db)
            pair_rows.append({"A": a, "B": b, "why": why, "head": head,
                              "ckpt": ck_name, "gm_A": gm(list(da.values())),
                              "gm_B": gm(list(db.values())), "delta": d,
                              "ci_lo": lo, "ci_hi": hi, "n": n,
                              "verdict": verdict(lo, hi)})

os.makedirs(E353, exist_ok=True)
for name, rows in (("gm_table.csv", gm_rows), ("pairwise_table.csv", pair_rows)):
    with open(f"{E353}/{name}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

fmt = lambda v, p="{:.4f}": p.format(v) if v is not None else "--"
print(f"{'arm':<12}{'head':<6}{'ckpt':<7}{'GM':>8}{'n':>5}")
for r in gm_rows:
    print(f"{r['arm']:<12}{r['head']:<6}{r['ckpt']:<7}{fmt(r['gm']):>8}{r['n']:>5}")
print()
print(f"{'A -> B':<24}{'head':<5}{'ckpt':<6}{'GM_A':>7}{'GM_B':>7}{'Δ':>9}"
      f"{'  90% CI':>19}   verdict")
for r in pair_rows:
    ci = (f"({r['ci_lo']:+.3f},{r['ci_hi']:+.3f})"
          if r["ci_lo"] is not None else "(--)")
    print(f"{r['A']+' -> '+r['B']:<24}{r['head']:<5}{r['ckpt']:<6}"
          f"{fmt(r['gm_A']):>7}{fmt(r['gm_B']):>7}"
          f"{fmt(r['delta'], '{:+.4f}'):>9}{ci:>19}   {r['verdict']}")
print(f"\nwrote {E353}/gm_table.csv and {E353}/pairwise_table.csv")
