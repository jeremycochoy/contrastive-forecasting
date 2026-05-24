#!/usr/bin/env python3
"""#316 figures — does multi-step (k=12) improve β? Robust to missing inputs.

  gm_summary.png       full-97 GM-MASE bars: all CPC arms vs β / v11c / (B).
  k_trend.png          k=1 vs k=12 per forecaster family (the headline trend).
  perdomain_radar.png  per-domain GM-MASE: #1 (transformer k=12) vs β vs v11c.
  training_curves.png  forecast gap & loss vs step: k=1 (β-like) vs k=12.
"""
import csv, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/jupyter/contrastive-forecasting/experiments"
MAIN = f"{ROOT}/2026-05-23_cpc_multistep_linear"
RES, RUNS = f"{MAIN}/results", f"{MAIN}/runs"
OUT = "/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound/experiments/2026-05-23_cpc_multistep_linear/plots"
os.makedirs(OUT, exist_ok=True)
V11C = f"{ROOT}/2026-05-11_exp_encoder_forecaster/results/gift_eval_full_v11c"
BETA = f"{ROOT}/2026-05-20_bottleneck_beta2_confound/results/gift_eval_full_bb_beta_50k"
BB   = f"{ROOT}/2026-05-19_crossed_loss_ablation/results/gift_eval_full_cl_hh_50k"

def agg_gm(s):
    if not os.path.exists(s): return None
    for line in open(s):
        if "Aggregate GM-Relative MASE" in line:
            for t in reversed(line.replace(":", " ").split()):
                try: return float(t)
                except ValueError: continue
    return None

def full(tag):  # small-head full-97 of an arm by backbone-name tag
    return agg_gm(f"{RES}/gift_eval_full_{tag}_s20260520_fp32_50k_FINAL_h2L/summary.txt")

def full_seed(tag, seed):  # small-head full-97 of a specific seed
    return agg_gm(f"{RES}/gift_eval_full_{tag}_s{seed}_fp32_50k_FINAL_h2L/summary.txt")

def dom_map(c):
    m = {}
    if os.path.exists(c):
        for r in csv.DictReader(open(c)): m[r["dataset"]] = r.get("domain", "?")
    return m

def rel_by_domain(s, dm):
    if not os.path.exists(s) or not dm: return {}
    acc = {}
    for line in open(s):
        p = line.split()
        if len(p) < 4: continue
        try: rel = float(p[-1])
        except ValueError: continue
        if p[0] in dm and rel > 0: acc.setdefault(dm[p[0]], []).append(math.log(rel))
    return {d: math.exp(sum(v)/len(v)) for d, v in acc.items()}

# refs
v11c, beta, bb = agg_gm(f"{V11C}/summary.txt"), agg_gm(f"{BETA}/summary.txt"), agg_gm(f"{BB}/summary.txt")

# CPC arms: (label, backbone-tag, family, k). β is family#1's k=1 (= ref beta).
ARMS = [
    ("transformer-head k=12 (#1)",  "bb_cpctrf_k12", "trfm·β-neg",   12),
    ("linear-head k=1 (#2)",        "bb_linbn_k1",   "linear·β-neg", 1),
    ("linear-head k=12 (#2)",       "bb_linbn_k12",  "linear·β-neg", 12),
    ("linear-head k=1 (#3)",        "bb_lincn_k1",   "linear·CPC-neg", 1),
    ("linear-head k=12 (#3)",       "bb_cpc_k12",    "linear·CPC-neg", 12),
]

# ---------------------------------------------------------------- gm_summary
def fig_gm_summary():
    rows = [(lab, full(tag)) for lab, tag, *_ in ARMS]
    sb = full_seed("bb_cpc_k12", "20260523")
    if sb is not None: rows.append(("linear-head k=12 (#3) — seed 2", sb))
    rows = [(l, g) for l, g in rows if g is not None]
    if not rows: print("gm_summary: no arm evals yet"); return
    rows.sort(key=lambda r: r[1], reverse=True)
    labels, vals = [r[0] for r in rows], [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 0.6*len(rows)+2.2))
    y = range(len(rows))
    ax.barh(list(y), vals, color="#1f77b4", alpha=0.85)
    for i, v in enumerate(vals): ax.text(v+0.004, i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("full-97 GM-Relative MASE (lower = better)")
    refs = [r for r in (v11c, beta, bb) if r]
    ax.set_xlim((min(vals+refs))*0.985, max(vals+refs)*1.04)
    if beta: ax.axvline(beta, color="#ff7f0e", ls="--", lw=1.6, label=f"β (k=1) = {beta:.3f}")
    if v11c: ax.axvline(v11c, color="#9467bd", ls="--", lw=1.6, label=f"v11c = {v11c:.3f}")
    if bb:   ax.axvline(bb, color="#7f7f7f", ls=":", lw=1.2, label=f"(B) = {bb:.3f}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("#316 — full GIFT-Eval (97 cfg), single seed each vs β/v11c\n(the two 'seed 2' points span 0.49 — the metric's seed noise)", fontsize=10.5)
    ax.invert_yaxis(); plt.tight_layout(); plt.savefig(f"{OUT}/gm_summary.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"wrote gm_summary.png ({len(rows)} arms)")

# ---------------------------------------------------------------- k_trend
def fig_k_trend():
    fams = {}
    for lab, tag, fam, k in ARMS:
        g = full(tag)
        if g is not None: fams.setdefault(fam, {})[k] = g
    # family #1's k=1 is β itself
    if beta is not None and "trfm·β-neg" in fams: fams["trfm·β-neg"][1] = beta
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cols = {"trfm·β-neg": "#1f77b4", "linear·β-neg": "#2ca02c", "linear·CPC-neg": "#d62728"}
    plotted = False
    for fam, pts in fams.items():
        ks = sorted(pts)
        if len(ks) >= 1:
            ax.plot(ks, [pts[k] for k in ks], "o-", color=cols.get(fam, "#555"), lw=2, ms=8, label=fam)
            plotted = True
    if not plotted: print("k_trend: no data"); plt.close(); return
    # Seed-2 of the linear·CPC-neg k=12 cell — show the spread vs the k-effects.
    seedB = full_seed("bb_cpc_k12", "20260523")
    if seedB is not None and fams.get("linear·CPC-neg", {}).get(12) is not None:
        sa = fams["linear·CPC-neg"][12]
        ax.plot([12], [seedB], "D", color="#d62728", ms=10, mfc="white", mec="#d62728",
                mew=2, label="linear·CPC-neg k=12, seed 2", zorder=5)
        ax.plot([12, 12], [min(sa, seedB), max(sa, seedB)], color="#d62728", lw=4, alpha=0.35, zorder=1)
        ax.annotate(f"2 seeds:\n{abs(seedB-sa):.2f} spread\n(> every k-effect)", (12, (sa+seedB)/2),
                    xytext=(8.6, (sa+seedB)/2), fontsize=8, color="#b01818", va="center",
                    arrowprops=dict(arrowstyle="-", color="#d62728", alpha=0.5))
    if beta: ax.axhline(beta, color="#ff7f0e", ls="--", lw=1.4, label=f"β = {beta:.3f}")
    if v11c: ax.axhline(v11c, color="#9467bd", ls=":", lw=1.4, label=f"v11c = {v11c:.3f}")
    ax.set_xticks([1, 12]); ax.set_xlabel("forecast steps k"); ax.set_ylabel("full-97 GM-MASE (lower=better)")
    ax.set_title("#316 — single-seed trend (k=12 worse), but the 2-seed spread (red bar)\nexceeds every k-effect: suggestive, not significant", fontsize=10.5)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/k_trend.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"wrote k_trend.png ({len(fams)} families, seedB={seedB})")

# ---------------------------------------------------------------- radar
def fig_radar():
    arms = []
    ed = f"{RES}/gift_eval_full_bb_cpctrf_k12_s20260520_fp32_50k_FINAL_h2L"
    g = rel_by_domain(f"{ed}/summary.txt", dom_map(f"{ed}/all_results.csv"))
    if g: arms.append(("CPC k=12 (transformer head)", "#1f77b4", g, full("bb_cpctrf_k12"), "-", 2.0))
    for lab, col, base in (("β (k=1)", "#ff7f0e", BETA), ("v11c", "#9467bd", V11C)):
        gg = rel_by_domain(f"{base}/summary.txt", dom_map(f"{base}/all_results.csv"))
        if gg: arms.append((lab, col, gg, agg_gm(f"{base}/summary.txt"), (0, (4, 2)), 1.6))
    if len(arms) < 2: print("radar: need CPC + ref"); return
    domains = sorted({d for _, _, g, *_ in arms for d in g})
    th = np.linspace(0, 2*np.pi, len(domains), endpoint=False); tc = np.concatenate([th, th[:1]])
    vals = [v for _, _, g, *_ in arms for v in g.values()]; lo, hi = max(0.5, min(vals)*0.92), max(vals)*1.06
    fig, ax = plt.subplots(figsize=(9.5, 9.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1); ax.set_xticks(th); ax.set_xticklabels(domains, fontsize=10)
    ax.set_rscale("log"); ax.set_ylim(lo, hi)
    rt = [t for t in (0.8, 1, 1.5, 2, 2.5, 3) if lo < t < hi]; ax.set_yticks(rt); ax.set_yticklabels([f"{t:g}" for t in rt], fontsize=8, color="0.4")
    ax.plot(tc, [1.0]*len(tc), color="k", ls=(0, (2, 2)), lw=1, alpha=0.6)
    for lab, col, g, gm, ls, lw in arms:
        v = np.array([g.get(d, np.nan) for d in domains] + [g.get(domains[0], np.nan)])
        ax.plot(tc, v, color=col, ls=ls, lw=lw, marker="o", ms=3, label=f"{lab}  GM={gm:.3f}" if gm else lab)
    ax.set_title("#316 — per-domain GM rel-MASE (log): CPC k=12 vs β vs v11c", fontsize=11, pad=24)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8, frameon=False)
    plt.tight_layout(); plt.savefig(f"{OUT}/perdomain_radar.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"wrote perdomain_radar.png ({len(arms)} arms)")

# ---------------------------------------------------------------- training curves
def fig_curves():
    curves = [("k=1 (linear, β-neg)", "#2ca02c", f"{RUNS}/bb_linbn_k1_s20260520_fp32_50k_losses.csv"),
              ("k=12 (transformer, β-neg, #1)", "#1f77b4", f"{RUNS}/bb_cpctrf_k12_s20260520_fp32_50k_losses.csv"),
              ("k=12 (linear, β-neg)", "#2ca02c", f"{RUNS}/bb_linbn_k12_s20260520_fp32_50k_losses.csv")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for title, key in (("loss", "loss"), ("forecast gap (1-step)", "gap")):
        ax = axs[0] if key == "loss" else axs[1]
        for lab, c, p in curves:
            if not os.path.exists(p): continue
            rows = list(csv.DictReader(open(p)))
            xs, ys = [], []
            for r in rows:
                try: s, y = int(r["step"]), float(r[key])
                except (KeyError, ValueError): continue
                if key == "loss" and y <= 0: continue
                xs.append(s); ys.append(y)
            if not xs: continue
            if len(xs) > 800:
                idx = np.linspace(0, len(xs)-1, 800).astype(int); xs=[xs[i] for i in idx]; ys=[ys[i] for i in idx]
            ls = "-" if "k=12" in lab else "--"
            ax.plot(xs, ys, color=c, lw=2, ls=ls, label=lab)
        ax.set_xscale("log");
        if key == "loss": ax.set_yscale("log")
        ax.set_title(title); ax.set_xlabel("step"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("#316 training — k=1 keeps a sharp 1-step gap (~1.09); k=12 collapses it (~0.65)", fontsize=12)
    plt.tight_layout(); plt.savefig(f"{OUT}/training_curves.png", dpi=120, bbox_inches="tight"); plt.close()
    print("wrote training_curves.png")

if __name__ == "__main__":
    fig_gm_summary(); fig_k_trend(); fig_radar(); fig_curves()
