#!/usr/bin/env python3
"""#341 — GM-Relative MASE + paired-bootstrap Δ for the stop-grad × capacity
grid. GM/bootstrap logic identical to #339's analyze.py; generalized to many
arms and pairs. Arms 1/2 (and the no-stop-grad capacity twins, for the sign
of the capacity knob WITHOUT stop-grad) are read from the #336/#339 result
dirs; arms 3/4 from this experiment.

Writes results/gm_table.csv (per-arm GM per head × checkpoint) and
results/pairwise_table.csv (Δ = GM(B) − GM(A) with 90% paired-bootstrap CI),
and prints both.
"""
import csv
import math
import os
import random

W = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
E336 = f"{W}/2026-06-03_crossfade_triplet/results"
E339 = f"{W}/2026-06-10_stopgrad_positive/results"
E341 = f"{W}/2026-06-11_stopgrad_capacity/results"

# key -> (label, results_dir, tag)
ARMS = {
    "a1_bn_enc6":       ("arm1 base+triplet (enc6+bn)",        E336, "allt08_xftrip_bn_enc6_qk_aon_b1024"),
    "a2_sg_enc3_nobn":  ("arm2 stop-grad enc3+nobn (#339)",    E339, "allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024"),
    "a3_sg_enc6_nobn":  ("arm3 stop-grad enc6+nobn (NEW)",     E341, "allt08_xftrip_nobn_enc6_sgpos_qk_aon_b1024"),
    "a4_sg_enc6_bn":    ("arm4 stop-grad enc6+bn (NEW)",       E341, "allt08_xftrip_bn_enc6_sgpos_qk_aon_b1024"),
    # context: the same capacity knob WITHOUT stop-grad (#336 twins)
    "c_nobn_enc3":      ("ctx enc3+nobn no-sg (#328 best)",    E336, "allt08_xftrip_nobn_enc3_qk_aon_b1024"),
    "c_nobn_enc6":      ("ctx enc6+nobn no-sg",                E336, "allt08_xftrip_nobn_enc6_qk_aon_b1024"),
}

# (A, B, why) — Δ = GM(B) − GM(A); negative Δ ⇒ B better.
PAIRS = [
    ("a2_sg_enc3_nobn", "a3_sg_enc6_nobn", "capacity enc3->enc6 (nobn) WITH sg [hypothesis]"),
    ("a1_bn_enc6",      "a4_sg_enc6_bn",   "stop-grad ON base (enc6+bn) [hypothesis]"),
    ("a2_sg_enc3_nobn", "a4_sg_enc6_bn",   "enc6+bn vs enc3+nobn WITH sg [hypothesis]"),
    ("a1_bn_enc6",      "a2_sg_enc3_nobn", "arm2 vs arm1 (completes grid)"),
    ("a1_bn_enc6",      "a3_sg_enc6_nobn", "arm3 vs arm1 (completes grid)"),
    ("a3_sg_enc6_nobn", "a4_sg_enc6_bn",   "bottleneck on/off at enc6 WITH sg (completes grid)"),
    ("c_nobn_enc3",     "c_nobn_enc6",     "capacity enc3->enc6 (nobn) WITHOUT sg [#336 sign]"),
    ("c_nobn_enc3",     "a2_sg_enc3_nobn", "cross-check: #339 published contrast"),
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

os.makedirs(E341, exist_ok=True)
for name, rows in (("gm_table.csv", gm_rows), ("pairwise_table.csv", pair_rows)):
    with open(f"{E341}/{name}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

fmt = lambda v, p="{:.4f}": p.format(v) if v is not None else "--"
print(f"{'arm':<18}{'head':<6}{'ckpt':<7}{'GM':>8}{'n':>5}")
for r in gm_rows:
    print(f"{r['arm']:<18}{r['head']:<6}{r['ckpt']:<7}{fmt(r['gm']):>8}{r['n']:>5}")
print()
print(f"{'A -> B':<36}{'head':<5}{'ckpt':<6}{'GM_A':>7}{'GM_B':>7}{'Δ':>9}{'  90% CI':>19}   verdict")
for r in pair_rows:
    ci = (f"({r['ci_lo']:+.3f},{r['ci_hi']:+.3f})" if r["ci_lo"] is not None else "(--)")
    print(f"{r['A']+' -> '+r['B']:<36}{r['head']:<5}{r['ckpt']:<6}"
          f"{fmt(r['gm_A']):>7}{fmt(r['gm_B']):>7}{fmt(r['delta'], '{:+.4f}'):>9}{ci:>19}   {r['verdict']}")
print(f"\nwrote {E341}/gm_table.csv and {E341}/pairwise_table.csv")
