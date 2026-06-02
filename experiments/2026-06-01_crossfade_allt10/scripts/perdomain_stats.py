#!/usr/bin/env python3
"""Per-domain paired-bootstrap change (crossfade − best recipe), both heads.

Grounds the per-domain table in RESULTS.md: for each data domain, the change in
GM-Relative MASE and its 90% paired-bootstrap interval over that domain's tasks
(same procedure as the benchmark-wide bootstrap, restricted to the domain). A
change is "reliable" when the whole interval is on one side of zero. Writes
results/perdomain_stats.csv and prints the table.
"""
import csv
import math
import os
import random

SYNC = os.environ.get(
    "SYNC", "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/sync_crossfade_allt10")
OUTCSV = os.environ.get(
    "OUTCSV", "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/.claude/"
    "worktrees/crossfade-allt10/experiments/2026-06-01_crossfade_allt10/results/perdomain_stats.csv")
BASE_TAG = "xshh_allt_forked10pct_qk_aon_b1024"
XF_TAG = "xshh_allt_forked10pct_crossfade10pct_qk_aon_b1024"
DOMAINS = ["Healthcare", "Econ/Fin", "Transport", "Energy", "Sales", "Nature", "Web/CloudOps"]


def read_summary(path):
    out = {}
    for line in open(path):
        q = line.split()
        if len(q) == 4 and "/" in q[0]:
            try:
                out[q[0]] = float(q[3])
            except ValueError:
                pass
    return out


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def paired(a, b, n=2000, seed=0):
    common = sorted(set(a) & set(b))
    A, B = [a[c] for c in common], [b[c] for c in common]
    d = gm(B) - gm(A)
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        idx = [rng.randrange(len(common)) for _ in common]
        ds.append(gm([B[i] for i in idx]) - gm([A[i] for i in idx]))
    ds.sort()
    return d, ds[int(0.05 * n)], ds[int(0.95 * n)], len(common)


dmap = {r["dataset"]: r["domain"]
        for r in csv.DictReader(open(f"{SYNC}/baseline/full_{BASE_TAG}_6L_all_results.csv"))}
rows = []
for head in ["2L", "6L"]:
    base = read_summary(f"{SYNC}/analyze/base/gift_eval_full_{BASE_TAG}_{head}/summary.txt")
    xf = read_summary(f"{SYNC}/analyze/xf/gift_eval_full_{XF_TAG}_{head}/summary.txt")
    for dom in DOMAINS:
        a = {c: v for c, v in base.items() if dmap.get(c) == dom}
        b = {c: v for c, v in xf.items() if dmap.get(c) == dom}
        d, lo, hi, n = paired(a, b)
        rows.append(dict(head=head, domain=dom, tasks=n, delta=d, lo=lo, hi=hi,
                         reliable=(hi < 0 or lo > 0)))

os.makedirs(os.path.dirname(OUTCSV), exist_ok=True)
with open(OUTCSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["head", "domain", "tasks", "delta", "ci_lo", "ci_hi", "reliable"])
    for r in rows:
        w.writerow([r["head"], r["domain"], r["tasks"], f"{r['delta']:.4f}",
                    f"{r['lo']:.4f}", f"{r['hi']:.4f}", int(r["reliable"])])
print("wrote", OUTCSV)
print(f"\n{'domain':14s} {'tasks':>5} | {'2L Δ':>7} {'2L reliable':>11} | {'6L Δ':>7} {'6L reliable':>11}")
for dom in DOMAINS:
    r2 = next(r for r in rows if r["head"] == "2L" and r["domain"] == dom)
    r6 = next(r for r in rows if r["head"] == "6L" and r["domain"] == dom)
    print(f"{dom:14s} {r2['tasks']:>5} | {r2['delta']:+7.3f} {('YES' if r2['reliable'] else 'noise'):>11} "
          f"| {r6['delta']:+7.3f} {('YES' if r6['reliable'] else 'noise'):>11}")
