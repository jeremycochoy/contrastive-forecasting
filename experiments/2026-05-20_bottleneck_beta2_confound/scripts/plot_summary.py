#!/usr/bin/env python3
"""#309 summary — full-97 GM-MASE across every arm, sorted bar chart.

The experiment grew into a matrix over {forecaster bottleneck} ×
{precision fp16/fp32} × {AdamW β2} × {τ}, which a radar can't show
cleanly. This horizontal bar chart ranks every arm by full-97
GM-Relative MASE, with v11c and seasonal-naive reference lines.

Auto-skips arms whose summary.txt is missing (preview-safe).
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN = "/home/jupyter/contrastive-forecasting/experiments/2026-05-20_bottleneck_beta2_confound"
CL = "/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_ablation"
V11C = "/home/jupyter/contrastive-forecasting/experiments/2026-05-11_exp_encoder_forecaster/results/gift_eval_full_v11c"
OUT = f"{MAIN}/plots"
os.makedirs(OUT, exist_ok=True)

# Converged backbones only. Two failure classes are EXCLUDED here and
# live in the diagnostic curves + the artifacts/failure section, not the
# results comparison:
#   (i)  fp16 no-bneck arms diverge in training (loss-min ~step 900 is not
#        a converged model);
#   (ii) the fp16 BOTTLENECK arms at τ=0.8 ((B), β) COLLAPSE — their
#        contrastive metrics (top1/gap/auc) collapse after ~step 500, so
#        their best_loss checkpoint (= what FINAL.pth points to and what
#        got eval'd) is a collapse-onset snapshot @ step ~324 / ~500, not
#        a converged 50k backbone. Their sub-1.29 GIFT-Eval numbers are
#        under-training artifacts (see tau08_bneck_collapse.png).
# label, summary_dir, colour-group key
ARMS = [
    ("(B) bneck · fp16 · τ0.1 · β2.95",          f"{CL}/results/gift_eval_full_cl_hh_50k",            "bneck"),
    ("β  bneck · fp16 · τ0.1 · β2.98",            f"{MAIN}/results/gift_eval_full_bb_beta_50k",        "bneck"),
    ("α  no-bneck · fp32 50k · τ0.1 · β2.98",     f"{MAIN}/results/gift_eval_full_bb_alpha_tau01_fp32_50k", "fp32"),
    ("γ  no-bneck · fp32 50k · τ0.1 · β2.95",     f"{MAIN}/results/gift_eval_full_bb_gamma_tau01_fp32_50k", "fp32"),
    ("α  no-bneck · fp32 50k · τ0.8 · β2.98",     f"{MAIN}/results/gift_eval_full_bb_alpha_tau08_fp32_50k", "fp32"),
    ("γ  no-bneck · fp32 50k · τ0.8 · β2.95",     f"{MAIN}/results/gift_eval_full_bb_gamma_tau08_fp32_50k", "fp32"),
]
COLOR = {"bneck": "#1f77b4", "fp32": "#d62728"}


def agg_gm(sum_txt):
    if not os.path.exists(sum_txt):
        return None
    with open(sum_txt) as f:
        for line in f:
            if "Aggregate GM-Relative MASE" in line:
                for t in reversed(line.replace(":", " ").split()):
                    try:
                        return float(t)
                    except ValueError:
                        continue
    return None


rows = []
for lab, edir, grp in ARMS:
    gm = agg_gm(f"{edir}/summary.txt")
    if gm is not None:
        rows.append((lab, gm, COLOR[grp]))
v11c = agg_gm(f"{V11C}/summary.txt")

if not rows:
    print("summary: no arm GMs yet — skipping")
else:
    rows.sort(key=lambda r: r[1], reverse=True)  # worst at top, best at bottom
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 2))
    y = range(len(rows))
    ax.barh(list(y), vals, color=cols, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("full-97 GM-Relative MASE (lower = better)")
    # include the v11c reference in-frame so it's visible that every
    # converged arm sits to its right (worse).
    lo_ref = min([min(vals)] + ([v11c] if v11c else [])) * 0.985
    ax.set_xlim(lo_ref, max(vals) * 1.04)
    if v11c:
        ax.axvline(v11c, color="#9467bd", ls="--", lw=1.5, label=f"v11c (no-bneck all-fp32 dk0.9) = {v11c:.3f}")
    ax.axvline(1.0, color="k", ls=":", lw=1.0, alpha=0.5, label="seasonal naive = 1.0")
    # legend for colour groups
    from matplotlib.patches import Patch
    handles = [Patch(color=COLOR["bneck"], label="bottleneck (fp16, converged)"),
               Patch(color=COLOR["fp32"], label="no-bneck (fp32, converged 50k)")]
    if v11c:
        handles.append(plt.Line2D([0], [0], color="#9467bd", ls="--", lw=1.5, label=f"v11c = {v11c:.3f}"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True)
    ax.set_title("#309 — full GIFT-Eval (97 cfg), converged backbones only: no arm beats v11c "
                 "(best = γ-τ0.1 1.313, +1.6%)", fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{OUT}/gm_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"wrote {OUT}/gm_summary.png — {len(rows)} arms; v11c={v11c}")
