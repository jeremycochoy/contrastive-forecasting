#!/usr/bin/env python3
"""#350 — final result plot. Two panels:
  (left)  GM-Relative MASE bar chart with per-arm bootstrap CI;
  (right) paired-bootstrap Δ = GM(arm) − GM(#348 +CPC baseline), 90% CI,
          for each bilinear arm.

Reads the three GIFT-Eval `summary.txt` files (per-task Relative MASE) and
resamples the 97-task list with replacement (seed=0, 2000 draws).
"""
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
PATHS = {
    "τ-baseline + CPC":          f"{ROOT}/2026-06-15_no_encoder_redo/results/gift_eval_full_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_cpc_2L/summary.txt",
    "bilinear W on h (run-1)":   f"{ROOT}/2026-06-16_bilinear_main_loss/results/gift_eval_full_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_bilinear_2L/summary.txt",
    "bilinear W on f (run-2)":   f"{ROOT}/2026-06-16_bilinear_main_loss/results/gift_eval_full_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_bilinear2_2L/summary.txt",
}
COLORS = {"τ-baseline + CPC": "#d62728",
          "bilinear W on h (run-1)": "#1f77b4",
          "bilinear W on f (run-2)": "#2ca02c"}
OUT = os.path.join(os.path.dirname(__file__), "..", "plots", "gm_summary.png")
N_BOOT, CI, SEED = 2000, 0.90, 0


def parse(path):
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) < 4: continue
        try:
            rel = float(p[-1]); _sn = float(p[-2]); _mase = float(p[-3])
            out[" ".join(p[:-3])] = rel
        except ValueError:
            pass
    return out


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def bootstrap_gm(per_task, tasks, n_boot, seed):
    rng = random.Random(seed)
    n = len(tasks); out = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(gm([per_task[tasks[i]] for i in idx]))
    return sorted(out)


def bootstrap_delta(arm, ref, tasks, n_boot, seed):
    rng = random.Random(seed)
    n = len(tasks); out = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(gm([arm[tasks[i]] for i in idx]) - gm([ref[tasks[i]] for i in idx]))
    return sorted(out)


def ci(sorted_vals, ci=CI):
    n = len(sorted_vals)
    return sorted_vals[int(n*(1-ci)/2)], sorted_vals[int(n*(1-(1-ci)/2)) - 1]


def main():
    runs = {k: parse(v) for k, v in PATHS.items()}
    tasks = sorted(set.intersection(*[set(d.keys()) for d in runs.values()]))
    assert len(tasks) == 97, f"expected 97 tasks, got {len(tasks)}"
    base_key = "τ-baseline + CPC"
    arms = [k for k in PATHS if k != base_key]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: GM-Relative MASE with per-arm bootstrap 90% CI.
    ax = axes[0]
    labels = list(PATHS.keys())
    gms = [gm(list(runs[k].values())) for k in labels]
    cis = [ci(bootstrap_gm(runs[k], tasks, N_BOOT, SEED)) for k in labels]
    yerr_lo = [g - lo for g, (lo, _) in zip(gms, cis)]
    yerr_hi = [hi - g for g, (_, hi) in zip(gms, cis)]
    bars = ax.bar(range(len(labels)), gms, yerr=[yerr_lo, yerr_hi],
                  color=[COLORS[k] for k in labels], alpha=0.85, capsize=6)
    for i, (g, lo_hi) in enumerate(zip(gms, cis)):
        ax.text(i, g + (lo_hi[1] - g) + 0.05, f"{g:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("GM-Relative MASE (lower is better)")
    ax.set_title(f"GIFT-Eval full-97, 2L head (best-loss)\n90% CI from {N_BOOT} bootstrap resamples")
    ax.axhline(1.0, color="grey", ls=":", lw=1, label="seasonal-naive (1.0)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, max(g + (h - g) for g, (_, h) in zip(gms, cis)) * 1.15)

    # Right: paired-bootstrap Δ vs baseline.
    ax = axes[1]
    ys = list(range(len(arms)))
    deltas, los, his, points = [], [], [], []
    for k in arms:
        d = bootstrap_delta(runs[k], runs[base_key], tasks, N_BOOT, SEED)
        lo, hi = ci(d)
        mean = sum(d) / len(d)
        point = gm(list(runs[k].values())) - gm(list(runs[base_key].values()))
        deltas.append(mean); los.append(lo); his.append(hi); points.append(point)
    # Point estimate plotted as the marker; whiskers are the bootstrap CI of
    # the paired difference (around the bootstrap mean, which is shown as the
    # whisker centre via the lo/hi half-widths from the point estimate).
    ax.errorbar(points, ys, xerr=[[p - lo for p, lo in zip(points, los)],
                                    [hi - p for p, hi in zip(points, his)]],
                fmt="o", color="black", capsize=6, ms=8)
    for y, k, p, lo, hi in zip(ys, arms, points, los, his):
        ax.scatter([p], [y], s=120, color=COLORS[k], zorder=3)
        ax.text(hi + 0.06, y, f"Δ={p:+.3f}  CI90=[{lo:+.3f}, {hi:+.3f}]", fontsize=9, va="center")
    ax.axvline(0, color="grey", ls="--", lw=1, label="no difference vs baseline")
    ax.set_yticks(ys)
    ax.set_yticklabels(arms, fontsize=9)
    ax.set_xlabel("Δ = GM(bilinear) − GM(τ-baseline + CPC)  (positive ⇒ worse)")
    ax.set_title(f"Paired-bootstrap difference\n{N_BOOT} resamples, 90% CI")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(min(0, min(los) - 0.1), max(his) * 1.5 + 0.1)
    ax.invert_yaxis()

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
