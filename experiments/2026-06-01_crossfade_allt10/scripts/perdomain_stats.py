#!/usr/bin/env python3
"""Per-domain paired-bootstrap change (crossfade − best recipe), both heads.

For each data domain, the change in GM-Relative MASE and its 90% paired-bootstrap
interval over that domain's tasks (the benchmark-wide procedure, restricted to the
domain). A change is "reliable" when the whole interval is on one side of zero.
Writes results/perdomain_stats.csv and plots/perdomain_delta.png, and prints the table.
"""
import csv
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYNC = os.environ.get(
    "SYNC", "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/sync_crossfade_allt10")
ED = os.environ.get(
    "ED", "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/.claude/"
    "worktrees/crossfade-allt10/experiments/2026-06-01_crossfade_allt10")
OUTCSV = os.environ.get("OUTCSV", f"{ED}/results/perdomain_stats.csv")
OUTPNG = os.environ.get("OUTPNG", f"{ED}/plots/perdomain_delta.png")
BASE_TAG = "xshh_allt_forked10pct_qk_aon_b1024"
XF_TAG = "xshh_allt_forked10pct_crossfade10pct_qk_aon_b1024"
# fixed order, most-improved (6-layer) first
DOMAINS = ["Healthcare", "Econ/Fin", "Transport", "Energy", "Sales", "Nature", "Web/CloudOps"]
HLAB = {"2L": "2-layer head", "6L": "6-layer head"}


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
data = {}  # (head, domain) -> dict
for head in ["2L", "6L"]:
    base = read_summary(f"{SYNC}/analyze/base/gift_eval_full_{BASE_TAG}_{head}/summary.txt")
    xf = read_summary(f"{SYNC}/analyze/xf/gift_eval_full_{XF_TAG}_{head}/summary.txt")
    for dom in DOMAINS:
        a = {c: v for c, v in base.items() if dmap.get(c) == dom}
        b = {c: v for c, v in xf.items() if dmap.get(c) == dom}
        d, lo, hi, n = paired(a, b)
        data[(head, dom)] = dict(tasks=n, delta=d, lo=lo, hi=hi, reliable=(hi < 0 or lo > 0))

os.makedirs(os.path.dirname(OUTCSV), exist_ok=True)
with open(OUTCSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["head", "domain", "tasks", "delta", "ci_lo", "ci_hi", "reliable"])
    for head in ["2L", "6L"]:
        for dom in DOMAINS:
            r = data[(head, dom)]
            w.writerow([head, dom, r["tasks"], f"{r['delta']:.4f}", f"{r['lo']:.4f}",
                        f"{r['hi']:.4f}", int(r["reliable"])])
print("wrote", OUTCSV)


def colour(r):
    if not r["reliable"]:
        return "#9aa0a6"                 # inconclusive
    return "#2ca02c" if r["delta"] < 0 else "#d62728"   # reliably better / worse


fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
ypos = list(range(len(DOMAINS)))[::-1]   # first domain at the top
for ax, head in zip(axes, ["2L", "6L"]):
    rs = [data[(head, d)] for d in DOMAINS]
    deltas = [r["delta"] for r in rs]
    errlo = [r["delta"] - r["lo"] for r in rs]
    errhi = [r["hi"] - r["delta"] for r in rs]
    ax.barh(ypos, deltas, color=[colour(r) for r in rs], height=0.6,
            xerr=[errlo, errhi], capsize=3, error_kw=dict(lw=1, ecolor="0.3"))
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{d} ({data[(head, d)]['tasks']})" for d in DOMAINS], fontsize=9)
    for y, r in zip(ypos, rs):
        if r["delta"] >= 0:
            ax.text(r["hi"] + 0.015, y, f"{r['delta']:+.2f}", va="center", ha="left", fontsize=8, color="0.3")
        else:
            ax.text(r["lo"] - 0.015, y, f"{r['delta']:+.2f}", va="center", ha="right", fontsize=8, color="0.3")
    ax.set_title(HLAB[head], fontsize=12)
    ax.set_xlabel("change in error from the crossfade\n(GM-Relative MASE; left = better)")
    ax.grid(axis="x", alpha=0.25)
axes[0].set_xlim(-0.82, 0.22)
fig.suptitle("Per-domain change from the crossfade (whisker = 90% interval; "
             "green = reliably better, red = reliably worse, grey = within noise)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(os.path.dirname(OUTPNG), exist_ok=True)
fig.savefig(OUTPNG, dpi=130)
print("wrote", OUTPNG)
print(f"\n{'domain':14s} {'tasks':>5} | {'2L Δ':>7} {'rel':>5} | {'6L Δ':>7} {'rel':>5}")
for dom in DOMAINS:
    r2, r6 = data[("2L", dom)], data[("6L", dom)]
    print(f"{dom:14s} {r2['tasks']:>5} | {r2['delta']:+7.3f} {('YES' if r2['reliable'] else '·'):>5} "
          f"| {r6['delta']:+7.3f} {('YES' if r6['reliable'] else '·'):>5}")
