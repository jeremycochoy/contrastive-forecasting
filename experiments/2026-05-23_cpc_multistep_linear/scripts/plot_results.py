#!/usr/bin/env python3
"""#316 figure — does multi-step (k=12) improve β?

  k_trend.png   k=1 -> k=12 for each forecaster setup, vs β / v11c.
                The one cell run with 2 seeds shows its seed range.
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

# family -> {k: gm}.  k=1 of the transformer family is β itself.
FAMS = [
    ("transformer head, β-negatives", "#1f77b4", {1: beta, 12: full("bb_cpctrf_k12")}),
    ("linear head, β-negatives",      "#2ca02c", {1: full("bb_linbn_k1"), 12: full("bb_linbn_k12")}),
    ("linear head, CPC-negatives",    "#d62728", {1: full("bb_lincn_k1"), 12: full("bb_cpc_k12")}),
]

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

if __name__ == "__main__":
    fig_k_trend()
