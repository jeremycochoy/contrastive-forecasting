#!/usr/bin/env python3
"""#316 figures — does multi-step (k=12) improve β?

  comparison.png  every arm's GM-MASE ranked, vs β / v11c (the standings).
  k_trend.png     k=1 -> k=12 within each setup (the paired trend).
                  The one cell run with 2 seeds shows its seed range.
"""
import os, csv, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = "/home/jupyter/contrastive-forecasting/experiments"
MAIN = f"{ROOT}/2026-05-23_cpc_multistep_linear"
RES = f"{MAIN}/results"
OUT = "/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound/experiments/2026-05-23_cpc_multistep_linear/plots"
os.makedirs(OUT, exist_ok=True)
V11C = f"{ROOT}/2026-05-11_exp_encoder_forecaster/results/gift_eval_full_v11c"
BETA = f"{ROOT}/2026-05-20_bottleneck_beta2_confound/results/gift_eval_full_bb_beta_50k"

def agg_gm(s):
    if not os.path.exists(s): return None
    for line in open(s):
        if "Aggregate GM-Relative MASE" in line:
            for t in reversed(line.replace(":", " ").split()):
                try: return float(t)
                except ValueError: continue
    return None

def full(tag, seed="20260520"):
    return agg_gm(f"{RES}/gift_eval_full_{tag}_s{seed}_fp32_50k_FINAL_h2L/summary.txt")

def dom_map(c):
    m = {}
    if os.path.exists(c):
        for r in csv.DictReader(open(c)): m[r["dataset"]] = r.get("domain", "?")
    return m

def rel_by_domain(ed):  # ed = eval dir -> {domain: geomean rel-MASE}
    s, dm = f"{ed}/summary.txt", dom_map(f"{ed}/all_results.csv")
    if not os.path.exists(s) or not dm: return {}
    acc = {}
    for line in open(s):
        p = line.split()
        if len(p) < 4: continue
        try: rel = float(p[-1])
        except ValueError: continue
        if p[0] in dm and rel > 0: acc.setdefault(dm[p[0]], []).append(math.log(rel))
    return {d: math.exp(sum(v)/len(v)) for d, v in acc.items()}

v11c, beta = agg_gm(f"{V11C}/summary.txt"), agg_gm(f"{BETA}/summary.txt")

# family -> {k: gm}.  k=1 of the transformer family is β itself.
FAMS = [
    ("transformer head, β-negatives", "#1f77b4", {1: beta, 12: full("bb_cpctrf_k12")}),
    ("linear head, β-negatives",      "#2ca02c", {1: full("bb_linbn_k1"), 12: full("bb_linbn_k12")}),
    ("linear head, CPC-negatives",    "#d62728", {1: full("bb_lincn_k1"), 12: full("bb_cpc_k12")}),
]

def fig_compare():
    arms = [
        ("transformer-1L / β-neg / k=1  (= β)", beta, 1),
        ("transformer-1L / β-neg / k=12",       full("bb_cpctrf_k12"), 12),
        ("linear / β-neg / k=1",                full("bb_linbn_k1"), 1),
        ("linear / β-neg / k=12",               full("bb_linbn_k12"), 12),
        ("linear / CPC-neg / k=1",              full("bb_lincn_k1"), 1),
        ("linear / CPC-neg / k=12  (seed A)",   full("bb_cpc_k12"), 12),
        ("linear / CPC-neg / k=12  (seed B)",   full("bb_cpc_k12", "20260523"), 12),
    ]
    arms = [a for a in arms if a[1] is not None]
    arms.sort(key=lambda r: r[1])                       # best (lowest) first
    labels = [a[0] for a in arms]; vals = [a[1] for a in arms]
    colors = ["#1f77b4" if a[2] == 1 else "#c43b3b" for a in arms]
    fig, ax = plt.subplots(figsize=(9.6, 0.52*len(arms)+1.8))
    y = range(len(arms))
    ax.barh(list(y), vals, color=colors, alpha=0.85)
    for i, v in enumerate(vals): ax.text(v+0.006, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("GIFT-Eval GM-MASE  (97 configs; lower = better)")
    ax.set_xlim(min(vals + [v11c or 1.29])*0.97, max(vals)*1.05)
    if v11c: ax.axvline(v11c, color="#9467bd", ls="--", lw=1.5)
    ax.invert_yaxis()
    handles = [Patch(color="#1f77b4", label="k = 1"), Patch(color="#c43b3b", label="k = 12")]
    if v11c: handles.append(Line2D([], [], color="#9467bd", ls="--", label=f"v11c champion = {v11c:.3f}"))
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title("Every k=12 arm (red) ranks below every k=1 arm (blue) — and below β", fontsize=11.5)
    plt.tight_layout(); plt.savefig(f"{OUT}/comparison.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"wrote comparison.png ({len(arms)} arms)")

def fig_k_trend():
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for lab, col, pts in FAMS:
        ks = sorted(k for k in pts if pts[k] is not None)
        ax.plot(ks, [pts[k] for k in ks], "o-", color=col, lw=2.2, ms=8, label=lab)
    # the one cell run with a 2nd seed: linear/CPC-negs, k=12 -> show its seed range
    sa = FAMS[2][2][12]; sb = full("bb_cpc_k12", "20260523")
    if sa is not None and sb is not None:
        ax.plot([12], [sb], "o", color="#d62728", ms=8, zorder=4)
        ax.plot([12, 12], [sa, sb], color="#d62728", lw=1.6, ls=":", zorder=3)
        ax.annotate(f"this one cell was run twice:\nseeds at {sa:.2f} and {sb:.2f}\n(0.49 apart — bigger than\nany k=1→k=12 gap).\nEvery other point = 1 seed.",
                    xy=(12, (sa+sb)/2), xytext=(6.3, 1.83), fontsize=8.5, color="#b01818",
                    va="center", arrowprops=dict(arrowstyle="->", color="#d62728", alpha=0.7))
    if beta: ax.axhline(beta, color="#1f77b4", ls=":", lw=1.3, label=f"β (= transformer, k=1) = {beta:.3f}")
    if v11c: ax.axhline(v11c, color="#9467bd", ls=":", lw=1.3, label=f"v11c (champion) = {v11c:.3f}")
    ax.set_xticks([1, 12]); ax.set_xlim(0.4, 12.8)
    ax.set_xlabel("number of forecast steps  k"); ax.set_ylabel("GIFT-Eval GM-MASE  (97 configs; lower = better)")
    ax.set_title("Going from k=1 to k=12 makes transfer worse in all three setups\n(but the seed noise is bigger than the effect — trust the direction, not the size)", fontsize=10.5)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8.5, loc="upper left")
    plt.tight_layout(); plt.savefig(f"{OUT}/k_trend.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"wrote k_trend.png (seedA={sa}, seedB={sb})")

def fig_radar():
    arms = [
        ("β  (transformer-1L / β-neg / k=1)", "#333333", BETA, (0, (4, 2)), 1.8),
        ("transformer-1L / β-neg / k=12",     "#1f77b4", f"{RES}/gift_eval_full_bb_cpctrf_k12_s20260520_fp32_50k_FINAL_h2L", "-", 1.8),
        ("linear / β-neg / k=12",             "#2ca02c", f"{RES}/gift_eval_full_bb_linbn_k12_s20260520_fp32_50k_FINAL_h2L", "-", 1.8),
        ("linear / CPC-neg / k=12",           "#d62728", f"{RES}/gift_eval_full_bb_cpc_k12_s20260520_fp32_50k_FINAL_h2L", "-", 1.8),
    ]
    series = [(lab, col, rel_by_domain(ed), ls, lw) for lab, col, ed, ls, lw in arms]
    series = [s for s in series if s[2]]
    if len(series) < 2: print("radar: insufficient data"); return
    domains = sorted({d for _, _, g, _, _ in series for d in g})
    th = np.linspace(0, 2*np.pi, len(domains), endpoint=False); tc = np.concatenate([th, th[:1]])
    vals = [v for _, _, g, _, _ in series for v in g.values()]
    lo, hi = max(0.5, min(vals)*0.9), max(vals)*1.06
    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(th); ax.set_xticklabels(domains, fontsize=10)
    ax.set_rscale("log"); ax.set_ylim(lo, hi)
    rt = [t for t in (0.8, 1, 1.5, 2, 2.5, 3) if lo < t < hi]
    ax.set_yticks(rt); ax.set_yticklabels([f"{t:g}" for t in rt], fontsize=8, color="0.4")
    ax.plot(tc, [1.0]*len(tc), color="k", ls=(0, (2, 2)), lw=1, alpha=0.5)
    for lab, col, g, ls, lw in series:
        v = np.array([g.get(d, np.nan) for d in domains] + [g.get(domains[0], np.nan)])
        ax.plot(tc, v, color=col, ls=ls, lw=lw, marker="o", ms=3, label=lab)
    ax.set_title("Per-domain GM-MASE (log radius; further out = worse)\nevery k=12 arm sits outside β on most domains — single seed, read the broad pattern",
                 fontsize=10, pad=28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=1, fontsize=8.5, frameon=False)
    plt.tight_layout(); plt.savefig(f"{OUT}/perdomain_radar.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"wrote perdomain_radar.png ({len(series)} arms, {len(domains)} domains)")

if __name__ == "__main__":
    fig_compare(); fig_k_trend(); fig_radar()
