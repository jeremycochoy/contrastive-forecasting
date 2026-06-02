#!/usr/bin/env python3
"""Per-domain forecast error radar, both heads.

Three profiles: the best recipe (90% real + 10% forked synthetic), the same recipe
with the regime crossfade added, and the parent experiment's per-domain standout
arm (the 0.8% all-time arm) for reference. Parent house style: fixed (alphabetical)
domain order, log radial axis, baseline dashed / new solid, a seasonal-naive ring.
Per-config relative error from the eval summaries; config -> domain from all_results.csv.
"""
import csv
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

SYNC = os.environ.get(
    "SYNC", "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/sync_crossfade_allt10")
OUT = os.environ.get(
    "OUT", "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/.claude/"
    "worktrees/crossfade-allt10/experiments/2026-06-01_crossfade_allt10/plots/perdomain.png")
BASE_TAG = "xshh_allt_forked10pct_qk_aon_b1024"
XF_TAG = "xshh_allt_forked10pct_crossfade10pct_qk_aon_b1024"
DOMAINS = ["Econ/Fin", "Energy", "Healthcare", "Nature", "Sales", "Transport", "Web/CloudOps"]
NAIVE = 1.0
BASE_C, XF_C, ALLT08_C = "#9bb8d3", "#2f6da8", "#e15759"


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


def domain_map():
    p = f"{SYNC}/baseline/full_{BASE_TAG}_6L_all_results.csv"
    return {r["dataset"]: r["domain"] for r in csv.DictReader(open(p))}


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def per_domain(rel, dmap):
    by = {d: [] for d in DOMAINS}
    for cfg, v in rel.items():
        if dmap.get(cfg) in by:
            by[dmap[cfg]].append(v)
    return [gm(by[d]) for d in DOMAINS]


dmap = domain_map()
ang = np.linspace(0, 2 * np.pi, len(DOMAINS), endpoint=False)
ang_c = np.concatenate([ang, ang[:1]])
ring_t = np.linspace(0, 2 * np.pi, 200)

fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), subplot_kw=dict(polar=True))
for ax, head, title in zip(axes, ["2L", "6L"], ["2-layer head", "6-layer head"]):
    base = per_domain(read_summary(f"{SYNC}/analyze/base/gift_eval_full_{BASE_TAG}_{head}/summary.txt"), dmap)
    xf = per_domain(read_summary(f"{SYNC}/analyze/xf/gift_eval_full_{XF_TAG}_{head}/summary.txt"), dmap)
    a08 = per_domain(read_summary(f"{SYNC}/parent_allt08/full_{head}_summary.txt"), dmap)
    ax.plot(ring_t, [NAIVE] * 200, ls="--", color="k", lw=1, alpha=0.55, label="seasonal-naive (1.0)")
    ax.plot(ang_c, base + base[:1], ls="--", color=BASE_C, lw=1.8, marker="o", ms=3, label="best recipe (10% fork)")
    ax.plot(ang_c, xf + xf[:1], ls="-", color=XF_C, lw=2.3, marker="o", ms=3, label="+ regime crossfade")
    ax.plot(ang_c, a08 + a08[:1], ls="-", color=ALLT08_C, lw=1.8, marker="^", ms=3.5, label="parent's 0.8%-fork arm")
    ax.set_rscale("log")
    ax.set_rlim(0.72, 1.75)
    ax.yaxis.set_major_locator(FixedLocator([0.8, 1.0, 1.292, 1.5]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["0.8", "1.0", "1.29", "1.5"], fontsize=7, color="0.45")
    ax.set_rlabel_position(88)
    ax.set_xticks(ang)
    ax.set_xticklabels(DOMAINS, fontsize=9)
    ax.set_title(title, fontsize=12, pad=24)
    ax.grid(alpha=0.3)
    handles = ax.get_legend_handles_labels()

fig.legend(handles[0], handles[1], loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.95))
fig.suptitle("Forecast error by data domain (closer to centre is better; log radial scale)",
             fontsize=12, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.88))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
for head in ["2L", "6L"]:
    base = per_domain(read_summary(f"{SYNC}/analyze/base/gift_eval_full_{BASE_TAG}_{head}/summary.txt"), dmap)
    xf = per_domain(read_summary(f"{SYNC}/analyze/xf/gift_eval_full_{XF_TAG}_{head}/summary.txt"), dmap)
    a08 = per_domain(read_summary(f"{SYNC}/parent_allt08/full_{head}_summary.txt"), dmap)
    print(f"{head}:  {'domain':14s} best   +xfade  0.8%")
    for d, b, x, a in zip(DOMAINS, base, xf, a08):
        print(f"     {d:14s} {b:.3f}  {x:.3f}  {a:.3f}")
