#!/usr/bin/env python3
"""#350 — training-dynamics log-log panels, same layout/scales as #348/#344 so
the report reads the same way. Adds the bilinear-W arm (purple) — the live
run-2 (solid) plus the interrupted run-1 (dotted, W-on-target-of-positive) —
on top of the #348 no-encoder arms and the enc-6 references. `loss_tau_ref` is
the CPC-free, τ=0.07 contrastive reference, computed identically across arms,
so it is directly comparable regardless of each arm's training objective.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
E348 = f"{W}/2026-06-15_no_encoder_redo/runs"
E350 = f"{W}/2026-06-16_bilinear_main_loss/runs"
# Both prior bilinear attempts archived under the experiment ROOT, not /runs.
E350_RUN1_PRIOR = f"{W}/2026-06-16_bilinear_main_loss/_buggy_positive_v1"
E350_RUN2_ABORT = f"{W}/2026-06-16_bilinear_main_loss/_run2_aborted"
BLUE   = "#1f77b4"   # run-1 relaunched (live, fresh — primary)
ORANGE = "#ff7f0e"   # run-1 prior (same form, interrupted at ~9.3k)
GREEN  = "#2ca02c"   # run-2 aborted (W-on-forecast formulation that was abandoned)
RED    = "#d62728"   # τ baseline #348
LOSSES = "bb_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_bilinear_losses.csv"
RUNS = {
    # Drawn FIRST goes UNDER. Bilinear arms last so they sit on top of the baseline.
    "τ-baseline + CPC (#348)":          (f"{E348}/bb_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_cpc_losses.csv", RED, "--"),
    "bilinear W on f (run-2, #350)":    (f"{E350_RUN2_ABORT}/{LOSSES}", GREEN, "-"),
    "bilinear W on h (run-1, #350)":    (f"{E350}/{LOSSES}", BLUE, "-"),
}
OUT = os.path.join(os.path.dirname(__file__), "..", "plots", "training_dynamics.png")
SMOOTH = 25
START_STEP = 100
# Floor constant `log(1 + N·exp(-1/τ))` for τ=0.10, B=1024, T=256, C=1, our
# loss_shape — the same subtracted by `--subtract-contrastive-floor` during
# training. For the τ baseline the floor is the theoretical InfoNCE minimum,
# so the floor-subtracted curve sits ≥ 0 and is meaningful. For the bilinear
# arms W can amplify scores past that floor, so the subtracted curve can go
# negative; add the floor back to both bilinear arms so the curve shown is
# the raw, ≥-0 InfoNCE value and is visible on log-y.
INFONCE_FLOOR = 9.412098
PER_RUN_TRANSFORM = {
    ("bilinear W on h (run-1, #350)", "loss"): lambda v: v + INFONCE_FLOOR,
    ("bilinear W on f (run-2, #350)", "loss"): lambda v: v + INFONCE_FLOOR,
}
PANELS = [
    ("loss",         "total training loss (raw / +floor for bilinears)  (↓)", lambda v: v),
    ("loss_tau_ref", "contrastive reference loss (W-free, τ=0.07)  (↓)", lambda v: v),
    ("cpc_aux",      "CPC InfoNCE term value  (↓)", lambda v: v),
    ("gap_ratio",    "ratio gap (1−ff)/(1−fp)  (↓→0)", lambda v: v),
    ("u_batch",      "U_batch — batch-wise used dims  (↑)", lambda v: v),
    ("r2_naive",     "1 − R²_naive  (↓)", lambda v: 1 - v),
]


def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"losses CSV missing: {path}")
    d = {}
    for r in csv.DictReader(open(path)):
        for k, v in r.items():
            try:
                d.setdefault(k, []).append(float(v) if v not in ("", None) else float("nan"))
            except (ValueError, TypeError):
                d.setdefault(k, []).append(float("nan"))
    return d


def smooth(y, w):
    out, run = [], []
    for v in y:
        run.append(v)
        if len(run) > w:
            run.pop(0)
        win = [x for x in run if x == x]
        out.append(sum(win) / len(win) if win else float("nan"))
    return out


def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (col, title, tf) in zip(axes.flat, PANELS):
        plotted = False
        for lab, (path, c, ls) in RUNS.items():
            d = load(path)
            if not d or col not in d or "step" not in d:
                continue
            y = [tf(v) for v in d[col]]
            per_run = PER_RUN_TRANSFORM.get((lab, col))
            if per_run is not None:
                y = [per_run(v) for v in y]
            sm = smooth(y, SMOOTH)
            step = d["step"]
            xs = [s for s, v in zip(step, sm) if v == v and v > 0 and s >= START_STEP]
            ys = [v for s, v in zip(step, sm) if v == v and v > 0 and s >= START_STEP]
            if not xs:
                continue
            lw = 2.0 if lab.startswith("bilinear W on") else 1.5
            ax.plot(xs, ys, color=c, ls=ls, lw=lw, label=lab)
            plotted = True
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(alpha=0.3, which="both")
        if plotted:
            ax.legend(fontsize=8, loc="best")
    fig.suptitle("Training dynamics (log-log; blue = bilinear W on h, "
                 "green = bilinear W on f, red dashed = τ baseline +CPC #348)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
