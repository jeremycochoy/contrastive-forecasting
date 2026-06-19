#!/usr/bin/env python3
"""#350 — per-task Relative MASE scatter: baseline vs each bilinear arm. Makes
the magnitude of the regression concrete on a per-task basis (a GM number alone
hides which tasks degrade and which are unchanged). One panel per bilinear arm;
points above the diagonal are tasks where the bilinear arm is worse.

Reads the three GIFT-Eval `summary.txt` files (per-task Relative MASE).
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
PATHS = {
    "baseline": f"{ROOT}/2026-06-15_no_encoder_redo/results/gift_eval_full_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_cpc_2L/summary.txt",
    "run-1 (W on h)": f"{ROOT}/2026-06-16_bilinear_main_loss/results/gift_eval_full_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_bilinear_2L/summary.txt",
    "run-2 (W on f)": f"{ROOT}/2026-06-16_bilinear_main_loss/results/gift_eval_full_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_bilinear2_2L/summary.txt",
}
COLORS = {"run-1 (W on h)": "#1f77b4", "run-2 (W on f)": "#2ca02c"}
OUT = os.path.join(os.path.dirname(__file__), "..", "plots", "per_task_scatter.png")


def parse(path):
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) < 4: continue
        try:
            out[" ".join(p[:-3])] = float(p[-1])
        except ValueError:
            pass
    return out


def main():
    runs = {k: parse(v) for k, v in PATHS.items()}
    tasks = sorted(set.intersection(*[set(d.keys()) for d in runs.values()]))
    arms = ["run-1 (W on h)", "run-2 (W on f)"]
    base = [runs["baseline"][t] for t in tasks]
    gm = math.exp(sum(math.log(v) for v in base) / len(base))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    lo, hi = 0.1, 200
    for ax, arm in zip(axes, arms):
        y = [runs[arm][t] for t in tasks]
        ax.scatter(base, y, s=18, color=COLORS[arm], alpha=0.75, edgecolor="black", linewidth=0.4)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x (parity)")
        n_worse = sum(1 for b, v in zip(base, y) if v > b)
        n_better = sum(1 for b, v in zip(base, y) if v < b)
        arm_gm = math.exp(sum(math.log(v) for v in y) / len(y))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("τ-baseline + CPC #348  Relative MASE")
        ax.set_ylabel(f"{arm}  Relative MASE")
        ax.set_title(f"{arm}: GM={arm_gm:.3f} (baseline {gm:.3f})\n"
                     f"{n_worse}/{len(tasks)} tasks worse than baseline, "
                     f"{n_better}/{len(tasks)} better")
        ax.grid(alpha=0.3, which="both")
        # Annotate the 3 worst-by-ratio outliers per arm.
        ratios = sorted(((v / b, t, b, v) for t, b, v in zip(tasks, base, y) if b > 0),
                        reverse=True)
        for r, t, b, v in ratios[:3]:
            ax.annotate(t.split("/")[0], (b, v), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("GIFT-Eval per-task Relative MASE: baseline vs bilinear arm "
                 "(log-log; above the diagonal = bilinear worse)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
