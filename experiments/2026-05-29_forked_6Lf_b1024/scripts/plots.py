#!/usr/bin/env python3
"""#322 — plots + scoreboard for the batch-1024 retrain.

Compares #320's 6L-forecaster forked arms at **batch 256** (reference) vs this card's
**batch 1024** retrain (all negatives pooled), per arm × q-head. Bootstrap logic is
lifted verbatim from #320's plots.py so the two cards' CIs are computed identically.

Outputs (into ../plots/ and ../results/):
  gm_summary.png   Full-97 GM-Relative MASE per arm × head, b256 vs b1024.
                   Whisker = bootstrap 90% CI on the GM over its 97 configs.
                   β shown as 2-seed-range bands; v11c + seasonal-naive as lines.
  batch_delta.png  Δ(b1024 − b256) per (arm, head); paired-bootstrap 90% CI over the
                   97 shared configs. Green = whole CI < 0 (b1024 better), red = worse,
                   grey = inconclusive.
  gm_table.csv     The scoreboard (also printed to stdout).
Runs incrementally: cells with missing summaries are skipped (printed as NA).
"""
import csv
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
RES_B256 = f"{ROOT}/2026-05-26_forked_6Lf/results"          # #320 reference
RES_B1024 = f"{ROOT}/2026-05-29_forked_6Lf_b1024/results"   # this card
PLOTS = ("/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/"
         "forked-6Lf-b1024/experiments/2026-05-29_forked_6Lf_b1024/plots")
os.makedirs(PLOTS, exist_ok=True)

# (b256 tag [#320], b1024 eval-dir tag [this card], display label)
# Stabilised b1024 backbones carry the _qk_aon suffix (qk-norm + attn-out-norm collapse fix).
ARMS = [
    ("beta_forked10pct_6Lf_50k",      "beta_forked10pct_qk_aon_b1024",      "β·10%"),
    ("beta_forked2_6Lf_50k",          "beta_forked2_qk_aon_b1024",          "β·0.8%"),
    ("xshh_allt_forked_6Lf_50k",      "xshh_allt_forked_qk_aon_b1024",      "allt·50%"),
    ("xshh_allt_forked10pct_6Lf_50k", "xshh_allt_forked10pct_qk_aon_b1024", "allt·10%"),
    ("xshh_allt_forked2_6Lf_50k",     "xshh_allt_forked2_qk_aon_b1024",     "allt·0.8%"),
]
HEADS = ["2L", "6L"]

# References (full-97 GM-Relative MASE; lower better), from #320 RESULTS.md.
BETA_2L = (1.3272, 1.4591)   # 2-seed range
BETA_6L = (1.3702, 1.4489)
V11C = 1.292
NAIVE = 1.0


def relatives_dict(sum_txt):
    out = {}
    if not os.path.exists(sum_txt):
        return out
    with open(sum_txt) as f:
        for line in f:
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


def gm_ci(xs, n=2000, seed=0):
    xs = [x for x in xs if x and x > 0]
    if len(xs) < 2:
        return (None, None)
    rng = random.Random(seed)
    gms = []
    for _ in range(n):
        s = [xs[rng.randrange(len(xs))] for _ in xs]
        gms.append(gm(s))
    gms.sort()
    return (gms[int(0.05 * n)], gms[int(0.95 * n)])


def paired_delta_ci(da, db, n=2000, seed=0):
    """Δ = gm(b1024) − gm(b256), paired bootstrap over shared configs."""
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


def rels(res_dir, tag, head, sset="full"):
    return relatives_dict(f"{res_dir}/gift_eval_{sset}_{tag}_{head}/summary.txt")


def collect():
    rows = []
    for tag256, tag1024, label in ARMS:
        for head in HEADS:
            r256 = rels(RES_B256, tag256, head, "full")
            r1024 = rels(RES_B1024, tag1024, head, "full")
            t256 = rels(RES_B256, tag256, head, "triage")
            t1024 = rels(RES_B1024, tag1024, head, "triage")
            gm256 = gm(list(r256.values()))
            gm1024 = gm(list(r1024.values()))
            d, lo, hi, npair = paired_delta_ci(r256, r1024)
            rows.append(dict(
                arm=label, head=head,
                b256_full=gm256, b1024_full=gm1024,
                delta=d, ci_lo=lo, ci_hi=hi, n=npair,
                b256_triage=gm(list(t256.values())),
                b1024_triage=gm(list(t1024.values())),
            ))
    return rows


def f(x, p=4):
    return "NA" if x is None else f"{x:.{p}f}"


def print_table(rows):
    print(f"\n{'arm':9} {'head':4} {'b256':>8} {'b1024':>8} {'Δ':>9} "
          f"{'90% CI':>20} {'b256_tri':>9} {'b1024_tri':>9}")
    for r in rows:
        ci = (f"({f(r['ci_lo'],3)}, {f(r['ci_hi'],3)})"
              if r['ci_lo'] is not None else "—")
        print(f"{r['arm']:9} {r['head']:4} {f(r['b256_full']):>8} "
              f"{f(r['b1024_full']):>8} {f(r['delta'],3):>9} {ci:>20} "
              f"{f(r['b256_triage']):>9} {f(r['b1024_triage']):>9}")
    with open(f"{RES_B1024}/gm_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {RES_B1024}/gm_table.csv")


C_256_2L = "#aec7e8"; C_1024_2L = "#1f77b4"   # blue family = 2L head
C_256_6L = "#ffbb78"; C_1024_6L = "#d94801"   # orange family = 6L head
C_V11C = "#9467bd"


def plot_gm_summary(rows):
    by = {(r['arm'], r['head']): r for r in rows}
    arms = [a[2] for a in ARMS]
    series = [
        ("b256 · 2L", C_256_2L, "2L", "b256_full"),
        ("b256 · 6L", C_256_6L, "6L", "b256_full"),
        ("b1024 · 2L", C_1024_2L, "2L", "b1024_full"),
        ("b1024 · 6L", C_1024_6L, "6L", "b1024_full"),
    ]
    x = np.arange(len(arms)); w = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (lab, col, head, key) in enumerate(series):
        vals = [by.get((a, head), {}).get(key) for a in arms]
        ax.bar(x + (i - 1.5) * w, [v or 0 for v in vals], w, label=lab, color=col)
    # references
    ax.axhspan(BETA_2L[0], BETA_2L[1], color=C_1024_2L, alpha=0.10, zorder=0)
    ax.axhspan(BETA_6L[0], BETA_6L[1], color=C_1024_6L, alpha=0.10, zorder=0)
    ax.axhline(V11C, color=C_V11C, ls="--", lw=1.4, label=f"v11c ({V11C})")
    ax.axhline(NAIVE, color="k", ls=":", lw=1.2, label="seasonal-naive (1.0)")
    ax.set_xticks(x); ax.set_xticklabels(arms)
    ax.set_ylabel("Full-97 GM-Relative MASE (lower better)")
    ax.set_title("#322 — forked arms × 6L forecaster: batch 256 (#320) vs batch 1024\n"
                 "(β 2-seed ranges shaded; v11c + seasonal-naive marked)")
    ax.legend(ncol=3, fontsize=8); ax.set_ylim(0.9, None)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/gm_summary.png", dpi=120)
    print("wrote gm_summary.png")


def plot_batch_delta(rows):
    labels, deltas, los, his, cols = [], [], [], [], []
    for r in rows:
        if r['delta'] is None:
            continue
        labels.append(f"{r['arm']}·{r['head']}")
        deltas.append(r['delta']); los.append(r['ci_lo']); his.append(r['ci_hi'])
        cols.append("#2ca02c" if r['ci_hi'] < 0 else
                    "#d62728" if r['ci_lo'] > 0 else "#7f7f7f")
    if not deltas:
        print("batch_delta: no cells yet"); return
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(labels))))
    ax.barh(y, deltas, color=cols,
            xerr=[np.array(deltas) - np.array(los), np.array(his) - np.array(deltas)],
            capsize=3)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Δ GM-Relative MASE  (b1024 − b256);  < 0 = bigger pooled batch helps")
    ax.set_title("#322 — effect of 4× larger all-together negative pool, per arm × head\n"
                 "(paired-bootstrap 90% CI over 97 shared configs; green = reliably better)")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/batch_delta.png", dpi=120)
    print("wrote batch_delta.png")


if __name__ == "__main__":
    rows = collect()
    print_table(rows)
    plot_gm_summary(rows)
    plot_batch_delta(rows)
