"""The gradient bursts that start a collapse of 'red without SIGReg-F', from the replay of pass 2.

Usage: python3 plot_burst.py <output png> [first step] [last step] [normal step] [burst steps, comma separated]
       [title or -] [legend place of panel A, default 'center right']
A: the gradient norm of each step (log y) with the dimension usage of e and SIGReg on e (right axis).
B: at the burst steps and a normal step, the largest gradient at the output of each of the 32 forecaster calls of
   the rollout (call 1 = depth 0, read on h. Call 32 = depth 31). C: the largest forecaster output of each call.
D: the 6 L_align terms (1 to 32 patches ahead) of each step.
"""
import os
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64") + "/diag_variant_nan"
NAME = "g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10pw2_rollfix"
DISTANCES = (1, 2, 4, 8, 16, 32)
DIST_COLOURS = ("#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4", "#9467bd")
STEP_COLOURS = ("black", "#e377c2", "#d62728", "#ff7f0e", "#8c564b", "#17becf")


def chain(step):
    """Per forecaster call, in depth order: the largest |gradient| at its output, the largest |output|."""
    for line in open(f"{D}/window_probe.jsonl"):
        p = json.loads(line)
        if p["step"] == step:
            grads = [v for t, v in p["bwd"] if t == "transformer.fcst_up_proj"][::-1]  # backward order -> depth
            outs = [v for t, v in p["fwd"] if t == "transformer.fcst_up_proj"]
            return np.array(grads), np.array(outs)
    raise SystemExit(f"no probe record at step {step}")


def step_note(step, normal, grad):
    if step == normal:
        return " (a usual step)"
    return f" (gradient norm {grad:,.0f})" if np.isfinite(grad) else " (non-finite gradient: the NaN step)"


def main(out, first=10470, last=10600, normal=10500, bursts=(10491, 10503, 10505), title=None,
         legend_loc="center right"):
    losses = {int(r["step"]): r for r in csv.DictReader(open(f"{D}/ckpt/{NAME}_losses.csv"))}
    steps = {int(r["step"]): r for r in csv.DictReader(open(f"{D}/steps.csv"))}
    xs = [s for s in range(first, last + 1) if s in steps and s in losses]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5))
    fig.subplots_adjust(hspace=0.32, wspace=0.3, top=0.91)
    ax = axes[0, 0]
    grad = np.array([float(steps[s]["grad_total"]) for s in xs])
    ax.semilogy(xs, grad, color="black", lw=1.5, marker="o", ms=3, label="gradient norm of the step")
    ax.set_ylabel("gradient norm (log scale)")
    ax.set_xlabel("training step")
    ax.set_title("A. Gradient bursts, then the collapse of the latents", loc="left", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    right = ax.twinx()
    right.plot(xs, [float(losses[s]["u_batchtime_e"]) for s in xs], color="#2ca02c", lw=2,
               label="share of the 64 directions used by e")
    right.plot(xs, [float(losses[s]["sigreg_e"]) for s in xs], color="#1f77b4", lw=2, ls="--", label="SIGReg on e")
    right.set_ylabel("share of directions, SIGReg on e")
    for s in bursts:  # on the right axes, so that the labels lie above its lines
        if np.isfinite(float(steps[s]["grad_total"])):
            right.annotate(f"{s:,}", (s, float(steps[s]["grad_total"])), xycoords=ax.transData,
                           textcoords="offset points", xytext=(4, 4), fontsize=8,
                           bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
    lines = ax.get_legend_handles_labels()[0] + right.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + right.get_legend_handles_labels()[1]
    ax.legend(lines, labels, fontsize=8, loc=legend_loc)
    for (ax, panel_title, ylabel, idx) in ((axes[0, 1], "B. Gradient at the output of each forecaster call",
                                      "largest |gradient| (log scale)", 0),
                                     (axes[1, 0], "C. Output of each forecaster call (forward)",
                                      "largest |output| of the call", 1)):
        for s, colour in zip((normal,) + tuple(bursts), STEP_COLOURS):
            values = chain(s)[idx]
            label = f"step {s:,}" + step_note(s, normal, float(steps[s]["grad_total"]))
            ax.plot(np.arange(1, len(values) + 1), values, color=colour, lw=2, marker="o", ms=3, label=label)
        if idx == 0:
            ax.set_yscale("log")
        ax.set_xlabel("forecaster call in the rollout (1 = depth 0, reads h; 32 = depth 31)")
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title, loc="left", fontsize=11)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    ax = axes[1, 1]
    terms = {1: [float(losses[s]["l_align"]) for s in xs]}
    terms.update({d: [float(steps[s][f"align_d{d}"]) for s in xs] for d in DISTANCES[1:]})
    for d, colour in zip(DISTANCES, DIST_COLOURS):
        ax.plot(xs, terms[d], color=colour, lw=2, label=f"L_align {d} patch{'es' if d > 1 else ''} ahead")
    ax.set_xlabel("training step")
    ax.set_ylabel("L_align = 2 - 2 cos (each step)")
    ax.set_title("D. The 6 L_align terms", loc="left", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.suptitle(title or f"Red without SIGReg-F, the replay of pass 2: the start of the collapse near "
                          f"{normal / 1000:.1f}k", fontsize=13)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    if len(a) > 1:
        kw = dict(first=int(a[1]), last=int(a[2]), normal=int(a[3]), bursts=tuple(int(v) for v in a[4].split(",")),
                  title=a[5] if len(a) > 5 and a[5] != "-" else None,
                  legend_loc=a[6] if len(a) > 6 else "center right")
    main(a[0], **kw)
