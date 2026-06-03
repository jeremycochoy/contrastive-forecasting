#!/usr/bin/env python3
"""#325 — scoreboard + plots: allt·10% + crossfade·10% vs #322's allt·10%.

Per-config relative MASE is parsed from each eval's summary.txt (4-field lines:
`config  modelMASE  snaiveMASE  relativeMASE`; we use the relative). GM and the
paired-bootstrap Δ logic are lifted verbatim from #322's plots.py so the two
cards' CIs are computed identically.

Δ = gm(crossfade) − gm(allt·10%), paired over the shared configs (lower = the
crossfade helps). Outputs gm_table.csv + gm_summary.png + delta.png.

Paths are env-overridable so it runs on elisa (eval outputs live there) or on the
laptop against the synced summaries.
"""
import csv
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Baseline (#322 allt·10%) and this card's crossfade eval results dirs.
BASE_RES = os.environ.get(
    "BASE_RES",
    "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024/results")
XF_RES = os.environ.get(
    "XF_RES",
    "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-01_crossfade_allt10/results")
PLOTS = os.environ.get(
    "PLOTS",
    "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-01_crossfade_allt10/plots")
OUTRES = os.environ.get("OUTRES", XF_RES)
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(OUTRES, exist_ok=True)

BASE_TAG = "xshh_allt_forked10pct_qk_aon_b1024"
XF_TAG = "xshh_allt_forked10pct_crossfade10pct_qk_aon_b1024"
HEADS = ["2L", "6L"]
HLAB = {"2L": "2-layer head", "6L": "6-layer head"}
V11C = 1.292
NAIVE = 1.0
# #322 allt·10% measured GM-Relative MASE (full-97), for the annotation/sanity check.
BASE_REF_FULL = {"2L": 1.222, "6L": 1.191}


def relatives_dict(sum_txt):
    """{config: relative_MASE} from a summary.txt (4-field per-config lines)."""
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
    """Δ = gm(db) − gm(da), paired bootstrap over shared configs (db=crossfade)."""
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


def rels(res_dir, tag, head, sset):
    return relatives_dict(f"{res_dir}/gift_eval_{sset}_{tag}_{head}/summary.txt")


def f(x, p=4):
    return "NA" if x is None else f"{x:.{p}f}"


def collect():
    rows = []
    for head in HEADS:
        base_full = rels(BASE_RES, BASE_TAG, head, "full")
        xf_full = rels(XF_RES, XF_TAG, head, "full")
        base_tri = rels(BASE_RES, BASE_TAG, head, "triage")
        xf_tri = rels(XF_RES, XF_TAG, head, "triage")
        gm_base = gm(list(base_full.values()))
        gm_xf = gm(list(xf_full.values()))
        d, lo, hi, npair = paired_delta_ci(base_full, xf_full)
        rows.append(dict(
            head=head, base_full=gm_base, xf_full=gm_xf,
            base_lo=gm_ci(list(base_full.values()))[0], base_hi=gm_ci(list(base_full.values()))[1],
            xf_lo=gm_ci(list(xf_full.values()))[0], xf_hi=gm_ci(list(xf_full.values()))[1],
            delta=d, ci_lo=lo, ci_hi=hi, n=npair,
            base_triage=gm(list(base_tri.values())), xf_triage=gm(list(xf_tri.values())),
        ))
    return rows


def write_table(rows):
    path = f"{OUTRES}/gm_table.csv"
    cols = ["head", "base_full", "xf_full", "delta", "ci_lo", "ci_hi", "n",
            "base_triage", "xf_triage"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["head"], f(r["base_full"]), f(r["xf_full"]), f(r["delta"]),
                        f(r["ci_lo"]), f(r["ci_hi"]), r["n"], f(r["base_triage"]),
                        f(r["xf_triage"])])
    print(f"wrote {path}")
    print(f"\n{'head':<5}{'allt10 (base)':>16}{'+crossfade':>13}{'Δ':>10}"
          f"{'90% CI on Δ':>22}{'base_tri':>11}{'xf_tri':>9}")
    for r in rows:
        ci = f"({f(r['ci_lo'],3)}, {f(r['ci_hi'],3)})"
        rel = (f"  REF {BASE_REF_FULL[r['head']]:.3f}"
               if r["base_full"] else "")
        print(f"{r['head']:<5}{f(r['base_full']):>16}{f(r['xf_full']):>13}"
              f"{f(r['delta']):>10}{ci:>22}{f(r['base_triage']):>11}"
              f"{f(r['xf_triage']):>9}{rel}")


def plot_summary(rows):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(HEADS))
    w = 0.36
    base = [r["base_full"] for r in rows]
    xf = [r["xf_full"] for r in rows]
    be = [[(r["base_full"] - r["base_lo"]) if r["base_lo"] else 0 for r in rows],
          [(r["base_hi"] - r["base_full"]) if r["base_hi"] else 0 for r in rows]]
    xe = [[(r["xf_full"] - r["xf_lo"]) if r["xf_lo"] else 0 for r in rows],
          [(r["xf_hi"] - r["xf_full"]) if r["xf_hi"] else 0 for r in rows]]
    ax.bar(x - w / 2, base, w, yerr=be, capsize=4, label="best recipe so far", color="#9bb8d3")
    ax.bar(x + w / 2, xf, w, yerr=xe, capsize=4, label="+ 10% regime crossfade", color="#2f6da8")
    ax.axhline(NAIVE, ls=":", c="#aaa", lw=1)
    ax.text(len(HEADS) - 0.5, NAIVE, " seasonal-naive", va="bottom", ha="right", fontsize=8, c="#999")
    for xi, r in zip(x, rows):
        if r["base_full"]:
            ax.text(xi - w / 2, r["base_full"], f"{r['base_full']:.3f}", ha="center", va="bottom", fontsize=8)
        if r["xf_full"]:
            ax.text(xi + w / 2, r["xf_full"], f"{r['xf_full']:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([HLAB[h] for h in HEADS])
    ax.set_ylabel("GM-Relative MASE over 97 tasks (lower better)")
    ax.set_title("Adding a regime crossfade to the best recipe — GIFT-Eval")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/gm_summary.png", dpi=130)
    print(f"wrote {PLOTS}/gm_summary.png")


def plot_delta(rows):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(HEADS))
    d = [r["delta"] for r in rows]
    lo = [(r["delta"] - r["ci_lo"]) if r["ci_lo"] is not None else 0 for r in rows]
    hi = [(r["ci_hi"] - r["delta"]) if r["ci_hi"] is not None else 0 for r in rows]
    colors = []
    for r in rows:
        if r["ci_hi"] is not None and r["ci_hi"] < 0:
            colors.append("#2ca02c")          # reliably better
        elif r["ci_lo"] is not None and r["ci_lo"] > 0:
            colors.append("#d62728")          # reliably worse
        else:
            colors.append("#999999")          # inconclusive
    ax.bar(x, d, 0.5, yerr=[lo, hi], capsize=5, color=colors)
    ax.axhline(0, c="k", lw=0.8)
    for xi, r in zip(x, rows):
        if r["delta"] is not None:
            ax.text(xi, r["delta"], f"{r['delta']:+.3f}", ha="center",
                    va="bottom" if r["delta"] >= 0 else "top", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([HLAB[h] for h in HEADS])
    ax.set_ylabel("change in error with the crossfade\n(GM-Relative MASE, negative = better)")
    ax.set_title("Change in forecast error from adding the crossfade\n"
                 "(90% interval over 97 tasks; grey = interval crosses zero)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/delta.png", dpi=130)
    print(f"wrote {PLOTS}/delta.png")


if __name__ == "__main__":
    rows = collect()
    write_table(rows)
    if any(r["xf_full"] for r in rows):
        plot_summary(rows)
    else:
        print("\n(no crossfade eval summaries yet — table/plots will fill in once evals land)")
