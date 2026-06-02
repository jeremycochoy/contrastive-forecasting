#!/usr/bin/env python3
"""Per-domain forecast error radar: best recipe vs + regime crossfade, both heads.

Styled to match the parent experiment's radar: domains in a fixed (alphabetical)
order, a log radial axis so the near-baseline cluster fills the figure, baseline
dashed / new solid, and two reference rings (seasonal-naive 1.0 and the strongest
prior backbone). Per-config relative error comes from the eval summaries; the
config -> domain map from an all_results.csv.
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
NAIVE, PRIOR = 1.0, 1.292
BASE_C, XF_C, PRIOR_C = "#9bb8d3", "#2f6da8", "#b07aa1"


def domain_map():
    p = f"{SYNC}/baseline/full_{BASE_TAG}_6L_all_results.csv"
    return {r["dataset"]: r["domain"] for r in csv.DictReader(open(p))}


def rels(res_dir, tag, head):
    out = {}
    for line in open(f"{res_dir}/gift_eval_full_{tag}_{head}/summary.txt"):
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

fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), subplot_kw=dict(polar=True))
for ax, head, title in zip(axes, ["2L", "6L"], ["2-layer head", "6-layer head"]):
    base = per_domain(rels(f"{SYNC}/analyze/base", BASE_TAG, head), dmap)
    xf = per_domain(rels(f"{SYNC}/analyze/xf", XF_TAG, head), dmap)
    ax.plot(ring_t, [NAIVE] * 200, ls="--", color="k", lw=1, alpha=0.6, label="seasonal-naive (1.0)")
    ax.plot(ring_t, [PRIOR] * 200, ls="--", color=PRIOR_C, lw=1, alpha=0.8, label="strongest prior backbone")
    ax.plot(ang_c, base + base[:1], ls="--", color=BASE_C, lw=1.8, marker="o", ms=3, label="best recipe")
    ax.plot(ang_c, xf + xf[:1], ls="-", color=XF_C, lw=2.2, marker="o", ms=3, label="+ regime crossfade")
    ax.set_rscale("log")
    ax.set_rlim(0.72, 1.7)
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
