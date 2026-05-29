#!/usr/bin/env python3
"""#320 plots — forked arms × 6-layer forecaster vs #318's 1-layer forecaster.

Two figures, each fully self-contained and titled:
  gm_summary.png        Full-97 GM-Relative MASE per arm × head, 1L vs 6Lf.
                        β shown as a candlestick spanning the 2-seed range; v11c
                        and seasonal-naive shown as reference lines (black solid
                        for naive; v11c labelled).
  forecaster_delta.png  Δ(6Lf − 1L) per (arm, head), paired-bootstrap 90% CI.
                        Green = whole CI < 0 (6Lf better than 1L); red = whole
                        CI > 0 (worse). Lower is better.
"""
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
RES_6LF = f"{ROOT}/2026-05-26_forked_6Lf/results"
RES_1L = f"{ROOT}/2026-05-23_xseries_hh/results"
PLOTS = "/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf/experiments/2026-05-26_forked_6Lf/plots"
os.makedirs(PLOTS, exist_ok=True)

# β: per-seed full-97 GMs. Both seeds use the #309 recipe verbatim.
#   seed 20260520 (#309): 2L 1.3272, 6L 1.4489
#   seed 20260521 (#318 paired):  2L 1.4591, 6L 1.3702
BETA_2L = [1.3272, 1.4591]
BETA_6L = [1.4489, 1.3702]
V11C = 1.292
NAIVE = 1.0

# arm-tag in (#318 1L), arm-tag in (#320 6Lf), display label
ARMS = [
    ("beta_forked10pct_50k",      "beta_forked10pct_6Lf_50k",      "β·10%"),
    ("beta_forked2_50k",          "beta_forked2_6Lf_50k",          "β·0.8%"),
    ("xshh_allt_forked_50k",      "xshh_allt_forked_6Lf_50k",      "allt·50%"),
    ("xshh_allt_forked10pct_50k", "xshh_allt_forked10pct_6Lf_50k", "allt·10%"),
    ("xshh_allt_forked2_50k",     "xshh_allt_forked2_6Lf_50k",     "allt·0.8%"),
]


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


def cell(res_dir, tag, head):
    """(gm, (lo, hi)) over the 97 per-config relatives. None if missing."""
    rels = list(relatives_dict(f"{res_dir}/gift_eval_full_{tag}_{head}/summary.txt").values())
    if not rels:
        return None, (None, None)
    return gm(rels), gm_ci(rels)


def paired_delta_ci(d1, d6, n=2000, seed=0):
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


# 4 bars per arm. Light = 1L #318 reference, dark = 6Lf this card.
# Blue family = 2L q-head, orange family = 6L q-head.
C_2L_1L = "#aec7e8"   # light blue   — 1L · 2L head (reference)
C_2L_6F = "#1f77b4"   # dark  blue   — 6Lf · 2L head (this card)
C_6L_1L = "#ffbb78"   # light orange — 1L · 6L head (reference)
C_6L_6F = "#d94801"   # dark  orange — 6Lf · 6L head (this card)
C_V11C  = "#9467bd"   # purple for v11c


# ---------------------------------------------------------------- gm_summary
def plot_gm_summary():
    fig, ax = plt.subplots(figsize=(11, 5.8))
    n_arms, w = len(ARMS), 0.20
    # 4 bars per arm, in order: 1L·2L, 1L·6L, 6Lf·2L, 6Lf·6L
    series = [
        ("1L · 2L head (#318 ref)",  C_2L_1L, "1L",  "2L"),
        ("1L · 6L head (#318 ref)",  C_6L_1L, "1L",  "6L"),
        ("6Lf · 2L head (this card)", C_2L_6F, "6Lf", "2L"),
        ("6Lf · 6L head (this card)", C_6L_6F, "6Lf", "6L"),
    ]
    for si, (lab, color, fc, hd) in enumerate(series):
        xs, ys, los, his = [], [], [], []
        for ai, (tag1, tag6, _) in enumerate(ARMS):
            res, tag = (RES_1L, tag1) if fc == "1L" else (RES_6LF, tag6)
            g, (lo, hi) = cell(res, tag, hd)
            if g is None:
                continue
            xs.append(ai + (si - 1.5) * w); ys.append(g)
            los.append(g - lo if lo is not None else 0)
            his.append(hi - g if hi is not None else 0)
        if ys:
            ax.bar(xs, ys, w, color=color, edgecolor="none", label=lab,
                   yerr=[los, his], capsize=2, ecolor="#444",
                   error_kw={"lw": 0.9})

    # β shown as FOUR horizontal dashed lines across the plot (the bounds of the
    # 2-seed range, one pair per head). Each line carries its own short label at
    # the right margin so the four labels never stack on top of each other.
    x_lab = n_arms - 0.4 + 0.04   # just past the right edge of the bars
    for y in BETA_2L:
        ax.axhline(y, color=C_2L_6F, lw=1.3, linestyle=(0, (4, 2)), alpha=0.9)
        ax.text(x_lab, y, f"  β·2L = {y:.3f}",
                ha="left", va="center", fontsize=8.5, color=C_2L_6F, fontweight="bold")
    for y in BETA_6L:
        ax.axhline(y, color=C_6L_6F, lw=1.3, linestyle=(0, (4, 2)), alpha=0.9)
        ax.text(x_lab, y, f"  β·6L = {y:.3f}",
                ha="left", va="center", fontsize=8.5, color=C_6L_6F, fontweight="bold")

    # naive solid black; v11c distinct purple dashed
    ax.axhline(NAIVE, color="black", lw=1.6)
    ax.text(-0.55, NAIVE + 0.02, "seasonal-naive (1.0)",
            ha="left", va="bottom", fontsize=9, color="black")
    ax.axhline(V11C, color=C_V11C, lw=1.6, linestyle=(0, (5, 2)))
    ax.text(-0.55, V11C - 0.02, f"v11c ({V11C})",
            ha="left", va="top", fontsize=9, color=C_V11C, fontweight="bold")

    ax.set_xticks(range(n_arms))
    ax.set_xticklabels([a[2] for a in ARMS])
    ax.set_xlim(-0.6, n_arms + 0.7)   # extra room on the right for β labels
    ax.set_ylabel("Full-97 GM-Relative MASE  (lower = better)")
    ax.set_title("Figure 1 — Full-97 GM-Relative MASE per arm × q-head, "
                 "1L vs 6Lf forecaster\n"
                 "(whisker = bootstrap 90 % CI on the GM over its 97 configs; "
                 "β shown as 4 horizontal dashed lines = the 2 bounds of each head's "
                 "2-seed β range)", fontsize=10)
    ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.92)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.savefig(f"{PLOTS}/gm_summary.png", dpi=120)
    plt.close(fig)
    print("wrote gm_summary.png")


# ---------------------------------------------------------------- delta
def plot_forecaster_delta():
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    labels, deltas, los, his, colors = [], [], [], [], []
    for tag1, tag6, lab in ARMS:
        for head in ("2L", "6L"):
            d1 = relatives_dict(f"{RES_1L}/gift_eval_full_{tag1}_{head}/summary.txt")
            d6 = relatives_dict(f"{RES_6LF}/gift_eval_full_{tag6}_{head}/summary.txt")
            d, lo, hi = paired_delta_ci(d1, d6)
            if d is None:
                continue
            labels.append(f"{lab} · {head}"); deltas.append(d)
            los.append(d - lo); his.append(hi - d)
            colors.append("#2ca02c" if hi < 0 else ("#d62728" if lo > 0 else "#bbbbbb"))
    y = list(range(len(labels)))
    ax.barh(y, deltas, color=colors, xerr=[los, his], capsize=3,
            ecolor="#444", error_kw={"lw": 1.0})
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Δ Full-97 GM-Relative MASE  =  6Lf − 1L   "
                  "(negative = 6Lf better than 1L)")
    ax.set_title("Figure 2 — Effect of 1L → 6L forecaster on each forked arm "
                 "(paired bootstrap 90% CI)", fontsize=11)
    # legend swatches
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#2ca02c", label="6Lf better than 1L (whole CI < 0)"),
        mpatches.Patch(color="#d62728", label="6Lf worse than 1L (whole CI > 0)"),
        mpatches.Patch(color="#bbbbbb", label="inconclusive (CI straddles 0)"),
    ], loc="lower right", fontsize=9, framealpha=0.92)
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/forecaster_delta.png", dpi=120)
    plt.close(fig)
    print("wrote forecaster_delta.png")


if __name__ == "__main__":
    plot_gm_summary()
    plot_forecaster_delta()
