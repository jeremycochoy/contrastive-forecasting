#!/usr/bin/env python3
"""#316 figures — does multi-step (k=12) improve β?

  gm_summary.png   full-97 GM-MASE: every k=12 variant vs β / v11c (the answer).
  k_trend.png      k=1 -> k=12 per forecaster family, with the 2-seed spread bar.
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

v11c, beta = agg_gm(f"{V11C}/summary.txt"), agg_gm(f"{BETA}/summary.txt")

# (label, backbone-tag, family, k). β is the transformer-family k=1 cell.
ARMS = [
    ("transformer head",            "bb_cpctrf_k12", "transformer, β-negs", 12),
    ("linear head, β-negs",         "bb_linbn_k1",   "linear, β-negs",      1),
    ("linear head, β-negs",         "bb_linbn_k12",  "linear, β-negs",      12),
    ("linear head, CPC-negs",       "bb_lincn_k1",   "linear, CPC-negs",    1),
    ("linear head, CPC-negs",       "bb_cpc_k12",    "linear, CPC-negs",    12),
]
COLS = {"transformer, β-negs": "#1f77b4", "linear, β-negs": "#2ca02c", "linear, CPC-negs": "#d62728"}

# ------------------------------------------------------------ gm_summary (answer)
def fig_gm_summary():
    rows = [(f"{lab}, k=12", full(tag)) for lab, tag, fam, k in ARMS if k == 12]
    sb = full("bb_cpc_k12", "20260523")
    if sb is not None: rows.append(("linear head, CPC-negs, k=12 (seed 2)", sb))
    rows = [(l, g) for l, g in rows if g is not None]
    if not rows: print("gm_summary: no evals yet"); return
    rows.sort(key=lambda r: r[1])
    labels, vals = [r[0] for r in rows], [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 0.62*len(rows)+1.8))
    y = range(len(rows))
    ax.barh(list(y), vals, color="#c43b3b", alpha=0.85)
    for i, v in enumerate(vals): ax.text(v+0.006, i, f"{v:.3f}", va="center", fontsize=10)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("full-97 GM-Relative MASE   (lower = better)")
    lo = min(vals + [beta or 1.3, v11c or 1.3]); hi = max(vals)
    ax.set_xlim(lo*0.97, hi*1.05)
    if beta: ax.axvline(beta, color="#1f77b4", ls="-",  lw=2.0, label=f"β (k=1) = {beta:.3f}")
    if v11c: ax.axvline(v11c, color="#9467bd", ls="--", lw=1.6, label=f"v11c (champion) = {v11c:.3f}")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title("Does k=12 improve β?  No — every k=12 variant lands right of β", fontsize=12)
    ax.invert_yaxis(); plt.tight_layout(); plt.savefig(f"{OUT}/gm_summary.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"wrote gm_summary.png ({len(rows)} k=12 arms)")

# ------------------------------------------------------------ k_trend
def fig_k_trend():
    fams = {}
    for lab, tag, fam, k in ARMS:
        g = full(tag)
        if g is not None: fams.setdefault(fam, {})[k] = g
    if beta is not None: fams.setdefault("transformer, β-negs", {})[1] = beta
    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    for fam, pts in fams.items():
        ks = sorted(pts)
        ax.plot(ks, [pts[k] for k in ks], "o-", color=COLS.get(fam, "#555"), lw=2, ms=8, label=fam)
    seedB = full("bb_cpc_k12", "20260523")
    if seedB is not None and fams.get("linear, CPC-negs", {}).get(12) is not None:
        sa = fams["linear, CPC-negs"][12]
        ax.plot([12], [seedB], "D", color="#d62728", ms=10, mfc="white", mec="#d62728", mew=2,
                label="linear, CPC-negs k=12 — 2nd seed", zorder=5)
        ax.plot([12, 12], [min(sa, seedB), max(sa, seedB)], color="#d62728", lw=5, alpha=0.30, zorder=1)
        ax.annotate(f"same arm, 2 seeds:\n{abs(seedB-sa):.2f} apart\n(bigger than every\nk=1→k=12 gap)",
                    (12, (sa+seedB)/2), xytext=(7.4, (sa+seedB)/2+0.04), fontsize=8.5, color="#b01818",
                    va="center", arrowprops=dict(arrowstyle="-", color="#d62728", alpha=0.5))
    if beta: ax.axhline(beta, color="#1f77b4", ls=":", lw=1.4, label=f"β = {beta:.3f}")
    if v11c: ax.axhline(v11c, color="#9467bd", ls=":", lw=1.4, label=f"v11c = {v11c:.3f}")
    ax.set_xticks([1, 12]); ax.set_xlim(0.3, 12.7)
    ax.set_xlabel("forecast steps  k"); ax.set_ylabel("full-97 GM-MASE  (lower = better)")
    ax.set_title("k=12 is worse in all 3 families — but the seed spread (red)\nis larger than the k-effect, so only the direction is reliable", fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout(); plt.savefig(f"{OUT}/k_trend.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"wrote k_trend.png ({len(fams)} families, seed2={seedB})")

if __name__ == "__main__":
    fig_gm_summary(); fig_k_trend()
