#!/usr/bin/env python3
"""#320 plots — forked arms × 6-layer forecaster vs #318's 1-layer forecaster.

Reads GIFT-Eval summaries from this card's results/ (6Lf) and #318's results/
(1L, reused). Bootstrap CIs are over the per-config Relative-MASE set (compute-
free; captures within-config-set spread, NOT seed spread). Each figure auto-skips
if its inputs are missing, so this is safe to run incrementally as cells land.

Figures:
  gm_summary.png     full-97 + triage GM-Relative MASE, every arm, 1L vs 6Lf
                     forecaster, both q-heads, vs β / v11c / seasonal-naive.
  forecaster_delta.png  Δ(6Lf − 1L) per (arm, head): where the deeper forecaster
                     helps (green, <0) vs hurts (red, >0). Directly answers #320.
"""
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
RES_6LF = f"{ROOT}/2026-05-26_forked_6Lf/results"          # this card (6L forecaster)
RES_1L = f"{ROOT}/2026-05-23_xseries_hh/results"           # #318 (1L forecaster), reused
PLOTS = "/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf/experiments/2026-05-26_forked_6Lf/plots"
os.makedirs(PLOTS, exist_ok=True)

BETA, V11C, NAIVE = 1.3272, 1.292, 1.0  # full-97 references (β·2L #309, v11c, seasonal-naive)

# arm -> (#318 1L tag, #320 6Lf tag, pretty label)
ARMS = [
    ("beta_forked10pct_50k",      "beta_forked10pct_6Lf_50k",      "β·10%"),
    ("beta_forked2_50k",          "beta_forked2_6Lf_50k",          "β·0.8%"),
    ("xshh_allt_forked_50k",      "xshh_allt_forked_6Lf_50k",      "allt·50%"),
    ("xshh_allt_forked10pct_50k", "xshh_allt_forked10pct_6Lf_50k", "allt·10%"),
    ("xshh_allt_forked2_50k",     "xshh_allt_forked2_6Lf_50k",     "allt·0.8%"),
]


def relatives_dict(sum_txt):
    """{config: Relative MASE} parsed from a summary.txt body (or {})."""
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


def relatives(sum_txt):
    """Per-config Relative MASE list (or [])."""
    return list(relatives_dict(sum_txt).values())


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def gm_ci(xs, n=2000, seed=0):
    """Bootstrap 90% CI on the GM over the per-config Relative set."""
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


def cell(res_dir, tag, head, evset):
    """(gm, (lo, hi)) for one eval cell, or (None, ...)."""
    rels = relatives(f"{res_dir}/gift_eval_{evset}_{tag}_{head}/summary.txt")
    if not rels:
        return None, (None, None)
    return gm(rels), gm_ci(rels)


def paired_delta_ci(d1, d6, n=2000, seed=0):
    """Paired bootstrap on Δ = GM(6Lf) − GM(1L) over the configs shared by both
    arms (same configs resampled jointly → cancels config difficulty, isolates the
    forecaster-depth effect). Returns (delta, lo, hi) or (None, None, None)."""
    common = sorted(set(d1) & set(d6))
    if len(common) < 2:
        return (None, None, None)
    a = [d1[c] for c in common]
    b = [d6[c] for c in common]
    delta = gm(b) - gm(a)
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        idx = [rng.randrange(len(common)) for _ in common]
        ds.append(gm([b[i] for i in idx]) - gm([a[i] for i in idx]))
    ds.sort()
    return (delta, ds[int(0.05 * n)], ds[int(0.95 * n)])


# ----------------------------------------------------------- gm_summary
def plot_gm_summary():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # 4 series per arm: (forecaster, head, color, label)
    series = [
        ("1L",  "2L", "#9ecae1", "1L·2L"),
        ("6Lf", "2L", "#1f77b4", "6Lf·2L"),
        ("1L",  "6L", "#fdae6b", "1L·6L"),
        ("6Lf", "6L", "#d94801", "6Lf·6L"),
    ]
    for ax, evset, refs in [
        (axes[0], "full", [("β 1.327", BETA), ("v11c 1.292", V11C), ("naive 1.0", NAIVE)]),
        (axes[1], "triage", [("naive 1.0", NAIVE)]),
    ]:
        n, w = len(ARMS), 0.20
        any_bar = False
        for si, (fc, head, color, lab) in enumerate(series):
            xs, ys, los, his = [], [], [], []
            for ai, (tag1, tag6, _) in enumerate(ARMS):
                res, tag = (RES_1L, tag1) if fc == "1L" else (RES_6LF, tag6)
                g, (lo, hi) = cell(res, tag, head, evset)
                if g is None:
                    continue
                xs.append(ai + (si - 1.5) * w); ys.append(g)
                los.append(g - (lo or g)); his.append((hi or g) - g)
            if ys:
                any_bar = True
                ax.bar(xs, ys, w, color=color, label=lab,
                       yerr=[los, his], capsize=2, ecolor="#555", error_kw={"lw": 0.8})
        for rlab, rv in refs:
            ax.axhline(rv, ls="--", lw=1, color="#888")
            ax.text(n - 0.4, rv, rlab, fontsize=8, va="bottom", ha="right", color="#555")
        ax.set_xticks(range(n)); ax.set_xticklabels([a[2] for a in ARMS])
        ax.set_title(f"{'full-97' if evset == 'full' else 'triage-11'} GM-Relative MASE")
        ax.set_ylabel("GM-Relative MASE (lower better)")
        if any_bar:
            ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", ls=":", alpha=0.4)
    fig.suptitle("#320 — forked arms: 1L vs 6L forecaster (bars = GM; whiskers = bootstrap 90% CI over configs)")
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/gm_summary.png", dpi=110)
    plt.close(fig)
    print("wrote gm_summary.png")


# ----------------------------------------------------------- forecaster delta
def plot_forecaster_delta():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, evset in [(axes[0], "full"), (axes[1], "triage")]:
        labels, deltas, los, his, colors = [], [], [], [], []
        for tag1, tag6, lab in ARMS:
            for head in ("2L", "6L"):
                d1 = relatives_dict(f"{RES_1L}/gift_eval_{evset}_{tag1}_{head}/summary.txt")
                d6 = relatives_dict(f"{RES_6LF}/gift_eval_{evset}_{tag6}_{head}/summary.txt")
                d, lo, hi = paired_delta_ci(d1, d6)
                if d is None:
                    continue
                labels.append(f"{lab}·{head}"); deltas.append(d)
                los.append(d - lo); his.append(hi - d)
                # green only if the whole 90% CI is below 0 (6Lf reliably better)
                colors.append("#2ca02c" if hi < 0 else ("#d62728" if lo > 0 else "#bbbbbb"))
        if not deltas:
            ax.text(0.5, 0.5, f"no {evset} pairs yet", ha="center", transform=ax.transAxes)
            continue
        y = list(range(len(labels)))
        ax.barh(y, deltas, color=colors, xerr=[los, his], capsize=2,
                ecolor="#555", error_kw={"lw": 0.8})
        ax.axvline(0, color="#333", lw=1)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Δ GM-Relative MASE  (6Lf − 1L)")
        ax.set_title(f"{'full-97' if evset == 'full' else 'triage-11'}: green = 6Lf better (90% CI<0), grey = inconclusive")
        ax.grid(axis="x", ls=":", alpha=0.4)
    fig.suptitle("#320 — effect of 1L→6L forecaster on each forked arm "
                 "(paired on seed 20260520; whiskers = paired-bootstrap 90% CI over shared configs)")
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/forecaster_delta.png", dpi=110)
    plt.close(fig)
    print("wrote forecaster_delta.png")


if __name__ == "__main__":
    plot_gm_summary()
    plot_forecaster_delta()
