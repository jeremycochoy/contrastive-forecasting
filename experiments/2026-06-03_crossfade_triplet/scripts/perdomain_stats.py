#!/usr/bin/env python3
"""Per-domain paired-bootstrap change (triplet arm − allt·0.8% base), both heads.

For each data domain, the change in GM-Relative MASE and its 90% paired-bootstrap
interval over that domain's tasks (the benchmark-wide procedure restricted to the
domain). A change is "reliable" when the whole interval is on one side of zero.
Dataset->domain comes from the eval's all_results.csv (`domain` column). Writes
results/perdomain_stats.csv + plots/perdomain_delta.png, and prints the table.
"""
import csv
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CF = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
BASE_RES = os.environ.get("BASE_RES", f"{CF}/2026-05-29_forked_6Lf_b1024/results")
TRIP_RES = os.environ.get("TRIP_RES", f"{CF}/2026-06-03_crossfade_triplet/results")
ED = os.environ.get("ED", "/tmp/cf-328/experiments/2026-06-03_crossfade_triplet")
OUTCSV = os.environ.get("OUTCSV", f"{ED}/results/perdomain_stats.csv")
OUTPNG = os.environ.get("OUTPNG", f"{ED}/plots/perdomain_delta.png")
BASE_TAG = "xshh_allt_forked2_qk_aon_b1024"
TRIP_TAG = "allt08_xftrip_nobn_enc3_qk_aon_b1024"
# fixed (task-count descending) order
DOMAINS = ["Energy", "Web/CloudOps", "Nature", "Transport", "Econ/Fin", "Healthcare", "Sales"]
HLAB = {"2L": "2-layer head", "6L": "6-layer head"}


def read_summary(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        q = line.split()
        if len(q) == 4 and "/" in q[0]:
            try:
                out[q[0]] = float(q[3])
            except ValueError:
                pass
    return out


def domain_map(res_dir, tag, head):
    """{dataset: domain} from an eval all_results.csv."""
    path = f"{res_dir}/gift_eval_full_{tag}_{head}/all_results.csv"
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        out[r["dataset"]] = r["domain"]
    return out


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def paired(a, b, n=2000, seed=0):
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return float("nan"), float("nan"), float("nan"), len(common)
    A, B = [a[c] for c in common], [b[c] for c in common]
    d = gm(B) - gm(A)
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        idx = [rng.randrange(len(common)) for _ in common]
        ds.append(gm([B[i] for i in idx]) - gm([A[i] for i in idx]))
    ds.sort()
    return d, ds[int(0.05 * n)], ds[int(0.95 * n)], len(common)


data = {}
for head in ["2L", "6L"]:
    base = read_summary(f"{BASE_RES}/gift_eval_full_{BASE_TAG}_{head}/summary.txt")
    trip = read_summary(f"{TRIP_RES}/gift_eval_full_{TRIP_TAG}_{head}/summary.txt")
    dmap = domain_map(TRIP_RES, TRIP_TAG, head) or domain_map(BASE_RES, BASE_TAG, head)
    for dom in DOMAINS:
        a = {c: v for c, v in base.items() if dmap.get(c) == dom}
        b = {c: v for c, v in trip.items() if dmap.get(c) == dom}
        d, lo, hi, n = paired(a, b)
        data[(head, dom)] = dict(tasks=n, delta=d, lo=lo, hi=hi,
                                 reliable=(hi < 0 or lo > 0) if n >= 2 else False)

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
        return "#9aa0a6"
    return "#2ca02c" if r["delta"] < 0 else "#d62728"


if any(data[("6L", d)]["tasks"] >= 2 for d in DOMAINS):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    ypos = list(range(len(DOMAINS)))[::-1]
    for ax, head in zip(axes, ["2L", "6L"]):
        rs = [data[(head, d)] for d in DOMAINS]
        ax.barh(ypos, [r["delta"] for r in rs], color=[colour(r) for r in rs], height=0.6,
                xerr=[[r["delta"] - r["lo"] for r in rs], [r["hi"] - r["delta"] for r in rs]],
                capsize=3, error_kw=dict(lw=1, ecolor="0.3"))
        ax.axvline(0, color="k", lw=0.9)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{d} ({data[(head, d)]['tasks']})" for d in DOMAINS], fontsize=9)
        ax.set_title(HLAB[head], fontsize=12)
        ax.set_xlabel("change in error vs the 0.8%-fork base\n(GM-Relative MASE; left = better)")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Per-domain change from the combined arm (whisker = 90% interval; "
                 "green = reliably better, red = reliably worse, grey = within noise)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(OUTPNG), exist_ok=True)
    fig.savefig(OUTPNG, dpi=130)
    print("wrote", OUTPNG)

print(f"\n{'domain':14s} {'tasks':>5} | {'2L Δ':>7} {'rel':>4} | {'6L Δ':>7} {'rel':>4}")
for dom in DOMAINS:
    r2, r6 = data[("2L", dom)], data[("6L", dom)]
    print(f"{dom:14s} {r2['tasks']:>5} | {r2['delta']:+7.3f} {('Y' if r2['reliable'] else '·'):>4} "
          f"| {r6['delta']:+7.3f} {('Y' if r6['reliable'] else '·'):>4}")
