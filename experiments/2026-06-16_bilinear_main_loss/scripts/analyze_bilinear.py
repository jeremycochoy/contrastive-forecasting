#!/usr/bin/env python3
"""#350 — GM-Relative MASE + paired-bootstrap Δ: the learnable log-bilinear W
main loss vs the τ-scaled-dot-product #348 + CPC baseline.

GM / paired-bootstrap logic is byte-for-byte #348's analyze_noenc.py, so the
numbers are directly comparable and the cpc baseline GMs reproduce #348's
(2L 1.168/1.165, 6L 1.153/1.160).

Δ = GM(bilinear) − GM(cpc); negative ⇒ bilinear better. Per head (2L, 6L) ×
checkpoint (best-loss, last). Writes results/gm_table.csv + pairwise_table.csv.
"""
import csv
import math
import os
import random

W = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
E348 = f"{W}/2026-06-15_no_encoder_redo/results"             # τ baseline (saved)
E350 = f"{W}/2026-06-16_bilinear_main_loss/results"          # bilinear (this work)

# key -> (label, results_dir, tag)
ARMS = {
    "cpc":      ("τ-dot-product + CPC (#348 baseline)", E348,
                 "allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_cpc"),
    "bilinear": ("learnable bilinear W + CPC (this work)", E350,
                 "allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_bilinear"),
}

# (A, B, why) — Δ = GM(B) − GM(A); negative Δ ⇒ B better.
PAIRS = [("cpc", "bilinear", "learnable bilinear W vs τ-scaled dot product [key]")]
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

os.makedirs(E350, exist_ok=True)
for name, rows in (("gm_table.csv", gm_rows), ("pairwise_table.csv", pair_rows)):
    with open(f"{E350}/{name}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

fmt = lambda v, p="{:.4f}": p.format(v) if v is not None else "--"
print(f"{'arm':<12}{'head':<6}{'ckpt':<7}{'GM':>8}{'n':>5}")
for r in gm_rows:
    print(f"{r['arm']:<12}{r['head']:<6}{r['ckpt']:<7}{fmt(r['gm']):>8}{r['n']:>5}")
print()
print(f"{'A -> B':<22}{'head':<5}{'ckpt':<6}{'GM_A':>7}{'GM_B':>7}{'Δ':>9}{'  90% CI':>19}   verdict")
for r in pair_rows:
    ci = (f"({r['ci_lo']:+.3f},{r['ci_hi']:+.3f})" if r["ci_lo"] is not None else "(--)")
    print(f"{r['A']+' -> '+r['B']:<22}{r['head']:<5}{r['ckpt']:<6}"
          f"{fmt(r['gm_A']):>7}{fmt(r['gm_B']):>7}{fmt(r['delta'], '{:+.4f}'):>9}{ci:>19}   {r['verdict']}")
print(f"\nwrote {E350}/gm_table.csv and {E350}/pairwise_table.csv")
