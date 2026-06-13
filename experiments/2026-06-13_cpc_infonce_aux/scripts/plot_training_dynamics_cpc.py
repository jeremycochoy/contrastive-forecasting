#!/usr/bin/env python3
"""#344 — training-dynamics log-log panel (same layout/scales as #341's
plot_training_metrics_sgcap.py, so the reports read the same way): the two CPC
arms (solid) against their stop-grad baselines (dashed). 6 panels, all log-log.

The "contrastive loss − floor" panel plots loss MINUS the CPC term
(`loss - cpc_aux`) for the CPC arms, so the contrastive component is
apples-to-apples with the baselines (whose `loss` has no CPC term). The CPC
term itself gets its own panel.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
# Encoding: COLOUR = treatment (blue = no CPC / baseline, red = + CPC);
#           LINESTYLE = architecture (solid = enc3, dashed = enc6).
BLUE, RED = "C0", "C3"
RUNS = {
    "enc3 baseline": (f"{W}/2026-06-10_stopgrad_positive/runs/bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_losses.csv", BLUE, "-"),
    "enc6 baseline": (f"{W}/2026-06-11_stopgrad_capacity/runs/bb_allt08_xftrip_nobn_enc6_sgpos_qk_aon_b1024_losses.csv", BLUE, "--"),
    "enc3 + CPC":    (f"{W}/2026-06-13_cpc_infonce_aux/runs/bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_cpc_losses.csv", RED, "-"),
    "enc6 + CPC":    (f"{W}/2026-06-13_cpc_infonce_aux/runs/bb_allt08_xftrip_nobn_enc6_sgpos_qk_aon_b1024_cpc_losses.csv", RED, "--"),
}
OUT = os.path.join(os.path.dirname(__file__), "..", "plots", "training_dynamics.png")
SMOOTH = 25
START_STEP = 100  # drop the noisy warm-up (and the CPC term's huge init value)

# (column, title, transform). "contrastive" and "cpc_aux" are derived/optional.
PANELS = [
    ("contrastive", "contrastive loss − InfoNCE floor  (↓)", lambda v: v),
    ("cpc_aux",     "CPC InfoNCE auxiliary term  (↓, CPC arms only)", lambda v: v),
    ("gap_ratio",   "ratio gap (1−ff)/(1−fp)  (↓→0)", lambda v: v),
    ("u_batch",     "U_batch — batch-wise used dims  (↑)", lambda v: v),
    ("r2_naive",    "1 − R²_naive  (↓)", lambda v: 1 - v),
    ("r2_random",   "1 − R²_random  (↓)", lambda v: 1 - v),
]


def load(path):
    if not os.path.exists(path):
        return None
    d = {}
    for r in csv.DictReader(open(path)):
        for k, v in r.items():
            try:
                d.setdefault(k, []).append(float(v) if v not in ("", None) else float("nan"))
            except (ValueError, TypeError):
                d.setdefault(k, []).append(float("nan"))
    # derived: contrastive = loss − cpc_aux (cpc_aux is 0/absent for baselines)
    if "loss" in d:
        cpc = d.get("cpc_aux")
        d["contrastive"] = [l - (c if (cpc and c == c) else 0.0)
                            for l, c in zip(d["loss"], (cpc or [0.0] * len(d["loss"])))]
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
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (col, title, tf) in zip(axes.flat, PANELS):
        for lab, (path, c, ls) in RUNS.items():
            d = load(path)
            if not d or col not in d:
                continue
            y = [tf(v) for v in d[col]]
            sm = smooth(y, SMOOTH)
            step = d["step"]
            # start at STEP_START; log-log needs strictly positive y: mask the rest
            xs = [s for s, v in zip(step, sm) if v == v and v > 0 and s >= START_STEP]
            ys = [v for s, v in zip(step, sm) if v == v and v > 0 and s >= START_STEP]
            if xs:
                ax.plot(xs, ys, color=c, ls=ls, lw=1.5, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("#344 CPC InfoNCE auxiliary — training dynamics (log-log; "
                 "blue = no CPC, red = + CPC; solid = enc3, dashed = enc6)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
