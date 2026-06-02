#!/usr/bin/env python3
"""Per-domain forecast error radar: best recipe vs + regime crossfade, both heads.

For each forecasting head, per-config relative error is read from the eval summaries
and grouped into the benchmark's 7 data domains (config -> domain map from an
all_results.csv). Each axis is a domain; each line a model; inner is better; the grey
ring is the seasonal-naive baseline (1.0).
"""
import csv
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYNC = os.environ.get(
    "SYNC", "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/sync_crossfade_allt10")
OUT = os.environ.get(
    "OUT", "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/.claude/"
    "worktrees/crossfade-allt10/experiments/2026-06-01_crossfade_allt10/plots/perdomain.png")
BASE_TAG = "xshh_allt_forked10pct_qk_aon_b1024"
XF_TAG = "xshh_allt_forked10pct_crossfade10pct_qk_aon_b1024"
DOMAINS = ["Energy", "Web/CloudOps", "Transport", "Nature", "Econ/Fin", "Healthcare", "Sales"]
BASE_C, XF_C = "#9bb8d3", "#2f6da8"


def domain_map():
    p = f"{SYNC}/baseline/full_{BASE_TAG}_6L_all_results.csv"
    return {r["dataset"]: r["domain"] for r in csv.DictReader(open(p))}


def rels(res_dir, tag, head):
    out = {}
    p = f"{res_dir}/gift_eval_full_{tag}_{head}/summary.txt"
    for line in open(p):
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
        d = dmap.get(cfg)
        if d in by:
            by[d].append(v)
    return [gm(by[d]) for d in DOMAINS]


dmap = domain_map()
ang = np.linspace(0, 2 * np.pi, len(DOMAINS), endpoint=False)
ang_closed = np.concatenate([ang, ang[:1]])

fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), subplot_kw=dict(polar=True))
handles = None
for ax, head, title in zip(axes, ["2L", "6L"], ["2-layer head", "6-layer head"]):
    base = per_domain(rels(f"{SYNC}/analyze/base", BASE_TAG, head), dmap)
    xf = per_domain(rels(f"{SYNC}/analyze/xf", XF_TAG, head), dmap)
    vmax = max(max(base), max(xf)) * 1.06
    for vals, c, lab in [(base, BASE_C, "best recipe"), (xf, XF_C, "+ regime crossfade")]:
        v = np.concatenate([vals, vals[:1]])
        ax.plot(ang_closed, v, color=c, lw=2, marker="o", ms=3.5, label=lab)
        ax.fill(ang_closed, v, color=c, alpha=0.06)
    ax.plot(ang_closed, [1.0] * len(ang_closed), color="0.45", lw=1, ls=":")  # seasonal-naive ring
    ax.set_xticks(ang)
    ax.set_xticklabels(DOMAINS, fontsize=9)
    ax.set_ylim(0.7, vmax)                       # floor below the smallest domain (~0.8)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=12, pad=22)
    ax.grid(alpha=0.3)
    handles = ax.get_legend_handles_labels()

fig.legend(handles[0], handles[1], loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.93))
fig.suptitle("Forecast error by data domain (closer to centre is better; dotted ring = seasonal-naive)",
             fontsize=12, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.9))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
for head in ["2L", "6L"]:
    base = per_domain(rels(f"{SYNC}/analyze/base", BASE_TAG, head), dmap)
    xf = per_domain(rels(f"{SYNC}/analyze/xf", XF_TAG, head), dmap)
    print(f"{head}:")
    for d, b, x in zip(DOMAINS, base, xf):
        print(f"   {d:14s} best {b:.3f}  xf {x:.3f}  Δ {x-b:+.3f}")
