"""The batch of the variant's step 13,755 through four models, one forward and backward each (rerun_step.py, on the
CPU in fp32): the variant at that step (the weights the step read), the variant at its 10k save, and red at its 15k
and 35k saves.

Usage: python3 plot_rerun_compare.py <output png>
A: the largest |gradient| of a token at the output of each of the 32 forecaster calls. B and C: the token norms of
f before the sphere (radius 8) at each call, the largest 0.1 % and the smallest 0.1 %. D: the largest backward gain
(|gradient in| / |gradient out|) of a token through the RMS norm of the attention output of forecaster layer 0.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

D = os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64") + "/diag_variant_nan"
MODELS = [("rerun_cpu_train_s0_dump", "red without SIGReg-F, the weights of step 13,755 (the NaN step)", "#e377c2", "-"),
          ("rerun_cpu_train_s0_resume", "red without SIGReg-F, its 10k save", "#e377c2", "--"),
          ("rerun_red_35k", "red (with SIGReg on f), its 35k save", "#d62728", "-"),
          ("rerun_red_15k", "red (with SIGReg on f), its 15k save", "#d62728", "--")]


def series(rec):
    calls = rec["calls"]
    grad = [float(c["grad_norm"].max()) for c in calls]
    high = [float(torch.quantile(c["raw_norm"].flatten(), 0.999)) for c in calls]
    low = [float(torch.quantile(c["raw_norm"].flatten(), 0.001)) for c in calls]
    layer0 = [r for r in rec["rms"] if r["name"] == "transformer.layers.0.attn_out_rms"]
    gain = [float((r["grad_in"] / r["grad_out"].clamp_min(1e-30)).max()) for r in layer0]
    return grad, high, low, gain


def main(out):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.subplots_adjust(hspace=0.3, wspace=0.25, top=0.9, bottom=0.12)
    calls = np.arange(1, 33)
    panels = [(axes[0, 0], "A. Largest |gradient| of a token at the output of each call", "log", 0),
              (axes[0, 1], "B. Token norm of f before the sphere: the largest 0.1 %", "linear", 1),
              (axes[1, 0], "C. Token norm of f before the sphere: the smallest 0.1 %", "linear", 2),
              (axes[1, 1], "D. Backward gain of the RMS norm on the attention output (forecaster layer 0)", "log", 3)]
    for tag, label, colour, style in MODELS:
        rec = torch.load(f"{D}/{tag}/records.pt", weights_only=False)
        values = series(rec)
        label = f"{label}: gradient norm {rec['grad_total']:,.0f}"
        for ax, _, _, idx in panels:
            ax.plot(calls[:len(values[idx])], values[idx], color=colour, ls=style, lw=2, marker="o", ms=3, label=label)
    for ax, title, scale, idx in panels:
        ax.set_yscale(scale)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("forecaster call in the rollout (1 = depth 0, reads h; 32 = depth 31)")
        ax.grid(alpha=0.3, which="both")
    for ax in (axes[0, 1], axes[1, 0]):
        ax.axhline(8, color="grey", ls=":", lw=1.5)
        ax.text(31.5, 8.3, "sphere radius 8", ha="right", fontsize=8, color="grey")
        ax.set_ylabel("token norm of f before the sphere")
    axes[0, 0].set_ylabel("largest |gradient| of a token (log scale)")
    axes[1, 1].set_ylabel("largest gain of a token (log scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("The same batch (the variant's step 13,755) through four models, one step each in fp32: the gradient "
                 "explodes along the rollout only in the variant at 13,755", fontsize=12)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
