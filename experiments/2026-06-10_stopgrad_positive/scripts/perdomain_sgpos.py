#!/usr/bin/env python3
"""Per-domain paired-bootstrap change (stop-grad − reference), both heads, both checkpoints.

Same procedure as the benchmark-wide interval (resample tasks with repeats, score
both arms on each resample), restricted to each domain's tasks. A change is
"reliable" when the whole 90% interval is on one side of zero. Writes
results/perdomain_stats.csv and plots/perdomain_delta.png.
"""
import csv
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-10_stopgrad_positive"
WT = "/tmp/cf-sgpos/experiments/2026-06-10_stopgrad_positive"
SG = "allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024"
REF = "allt08_xftrip_nobn_enc3_qk_aon_b1024"
OUTCSV = f"{WT}/results/perdomain_stats.csv"
OUTPNG = f"{WT}/plots/perdomain_delta.png"
DOMAINS = ["Healthcare", "Econ/Fin", "Transport", "Energy", "Sales", "Nature", "Web/CloudOps"]
CKPTS = [("best", "best-loss"), ("last", "last (12.5k)")]
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
    assert len(out) == 97, f"{path}: {len(out)} configs"
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


def cell_dir(arm_tag, ck, head, ref=False):
    sub = "results/reference" if ref else "results"
    mid = "" if ck == "best" else "_last"
    return f"{EXP}/{sub}/gift_eval_full_{arm_tag}{mid}_{head}"


dmap = {r["dataset"]: r["domain"]
        for r in csv.DictReader(open(f"{cell_dir(SG, 'best', '2L')}/all_results.csv"))}
data = {}  # (ck, head, domain) -> dict
for ck, _ in CKPTS:
    for head in ["2L", "6L"]:
        ref = read_summary(f"{cell_dir(REF, ck, head, ref=True)}/summary.txt")
        sg = read_summary(f"{cell_dir(SG, ck, head)}/summary.txt")
        for dom in DOMAINS:
            a = {c: v for c, v in ref.items() if dmap.get(c) == dom}
            b = {c: v for c, v in sg.items() if dmap.get(c) == dom}
            d, lo, hi, n = paired(a, b)
            data[(ck, head, dom)] = dict(tasks=n, delta=d, lo=lo, hi=hi,
                                         reliable=(hi < 0 or lo > 0))

os.makedirs(os.path.dirname(OUTCSV), exist_ok=True)
with open(OUTCSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["checkpoint", "head", "domain", "tasks", "delta", "ci_lo", "ci_hi", "reliable"])
    for ck, _ in CKPTS:
        for head in ["2L", "6L"]:
            for dom in DOMAINS:
                r = data[(ck, head, dom)]
                w.writerow([ck, head, dom, r["tasks"], f"{r['delta']:.4f}",
                            f"{r['lo']:.4f}", f"{r['hi']:.4f}", int(r["reliable"])])
print("wrote", OUTCSV)


def colour(r):
    if not r["reliable"]:
        return "#9aa0a6"
    return "#2ca02c" if r["delta"] < 0 else "#d62728"


fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
ypos = list(range(len(DOMAINS)))[::-1]
for row, (ck, cklab) in enumerate(CKPTS):
    for col, head in enumerate(["2L", "6L"]):
        ax = axes[row][col]
        rs = [data[(ck, head, d)] for d in DOMAINS]
        ax.barh(ypos, [r["delta"] for r in rs], color=[colour(r) for r in rs], height=0.6,
                xerr=[[r["delta"] - r["lo"] for r in rs], [r["hi"] - r["delta"] for r in rs]],
                capsize=3, error_kw=dict(lw=1, ecolor="0.3"))
        ax.axvline(0, color="k", lw=0.9)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{d} ({data[(ck, head, d)]['tasks']})" for d in DOMAINS], fontsize=9)
        for y, r in zip(ypos, rs):
            if r["delta"] >= 0:
                ax.text(r["hi"] + 0.004, y, f"{r['delta']:+.3f}", va="center", ha="left",
                        fontsize=8, color="0.3")
            else:
                ax.text(r["lo"] - 0.004, y, f"{r['delta']:+.3f}", va="center", ha="right",
                        fontsize=8, color="0.3")
        ax.set_title(f"{HLAB[head]} — {cklab}", fontsize=11)
        if row == 1:
            ax.set_xlabel("change in error from the stop-grad\n(GM-Relative MASE; left = better)")
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlim(-0.38, 0.52)
fig.suptitle("Per-domain change from the stop-grad (whisker = 90% paired-bootstrap interval;\n"
             "green = reliably better, red = reliably worse, grey = within noise)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUTPNG, dpi=130)
print("wrote", OUTPNG)
print(f"\n{'domain':14s} | " + " | ".join(f"{ck}-{h}" for ck, _ in CKPTS for h in ("2L", "6L")))
for dom in DOMAINS:
    cells = [data[(ck, h, dom)] for ck, _ in CKPTS for h in ("2L", "6L")]
    print(f"{dom:14s} | " + " | ".join(
        f"{c['delta']:+.3f}{'*' if c['reliable'] else ' '}" for c in cells))
