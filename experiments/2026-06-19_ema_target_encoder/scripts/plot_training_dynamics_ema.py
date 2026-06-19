#!/usr/bin/env python3
"""#353 — training-dynamics log-log panel. The two enc3 arms in the comparison:
the #344 enc3+CPC baseline (--stopgrad-positive-h) and the EMA-target arm
(--ema-embedding --ema-encoder). Same layout/columns as the #344 dynamics
plot for direct visual comparison.

The "contrastive loss" panel plots `loss - cpc_aux` so the contrastive
component is apples-to-apples; the CPC term itself gets its own panel.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
BLUE, GREEN = "#1f77b4", "#2ca02c"
RUNS = {
    "enc3+CPC baseline (--stopgrad-positive-h)": (
        f"{W}/2026-06-13_cpc_infonce_aux/runs/"
        "bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_cpc_losses.csv",
        BLUE, "-"),
    "enc3+CPC EMA-target (--ema-embedding --ema-encoder)": (
        f"{W}/2026-06-19_ema_target_encoder/runs/"
        "bb_allt08_xftrip_nobn_enc3_emateach_qk_aon_b1024_cpc_losses.csv",
        GREEN, "-"),
}
OUT = os.path.join(os.path.dirname(__file__), "..", "plots",
                   "training_dynamics.png")
SMOOTH = 25
START_STEP = 100

PANELS = [
    ("contrastive", "contrastive loss − floor  (↓)", lambda v: v),
    ("cpc_aux", "CPC InfoNCE auxiliary term  (↓)", lambda v: v),
    ("gap_ratio", "ratio gap (1−ff)/(1−fp)  (↓→0)", lambda v: v),
    ("u_batch", "U_batch — batch-wise used dims  (↑)", lambda v: v),
    ("u_temporal", "U_temporal — time-wise used dims  (↑)", lambda v: v),
    ("r2_naive", "1 − R²_naive  (↓)", lambda v: 1 - v),
    ("r2_random", "1 − R²_random  (↓)", lambda v: 1 - v),
    ("auc", "1 − retrieval AUC  (↓)", lambda v: 1 - v),
]


def load(path):
    if not os.path.exists(path):
        return None
    d = {}
    for r in csv.DictReader(open(path)):
        for k, v in r.items():
            try:
                d.setdefault(k, []).append(
                    float(v) if v not in ("", None) else float("nan"))
            except (ValueError, TypeError):
                d.setdefault(k, []).append(float("nan"))
    if "loss" in d:
        cpc = d.get("cpc_aux")
        d["contrastive"] = [l - (c if (cpc and c == c) else 0.0)
                            for l, c in zip(
                                d["loss"],
                                (cpc or [0.0] * len(d["loss"])))]
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
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax in axes.flat[len(PANELS):]:
        ax.set_visible(False)
    for ax, (col, title, tf) in zip(axes.flat, PANELS):
        for lab, (path, c, ls) in RUNS.items():
            d = load(path)
            if not d or col not in d:
                continue
            y = [tf(v) for v in d[col]]
            sm = smooth(y, SMOOTH)
            step = d["step"]
            xs = [s for s, v in zip(step, sm)
                  if v == v and v > 0 and s >= START_STEP]
            ys = [v for s, v in zip(step, sm)
                  if v == v and v > 0 and s >= START_STEP]
            if xs:
                ax.plot(xs, ys, color=c, ls=ls, lw=1.5, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("#353 EMA-target on enc3+CPC — training dynamics (log-log; "
                 "blue = stop-grad baseline, green = EMA-target)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
