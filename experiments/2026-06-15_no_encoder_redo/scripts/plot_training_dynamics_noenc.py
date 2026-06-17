#!/usr/bin/env python3
"""#348 — training-dynamics log-log panel (same layout/scales as #341/#344, so
the reports read the same way). The two no-encoder arms (solid) against their
strongest encoder'd reference (enc6, dashed). The CPC term itself gets its own
panel; `loss_tau_ref` is the CPC-free contrastive reference, comparable across
all arms. Answers: without the encoder, does the CPC term still reshape the
pretext representation (lower ref loss / ratio-gap, AUC→1) the way it did with it?
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
E348 = f"{W}/2026-06-15_no_encoder_redo/runs"
BLUE, CYAN = "#1f77b4", "#17becf"
RED, ORANGERED, GREEN = "#d62728", "#ff7f0e", "#2ca02c"
RUNS = {
    "no-enc base":       (f"{E348}/bb_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_base_losses.csv", BLUE, "-"),
    "no-enc + CPC":      (f"{E348}/bb_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_cpc_losses.csv", RED, "-"),
    "no-enc + CPC_All":  (f"{E348}/bb_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_cpcall_losses.csv", GREEN, "-"),
    "enc6 base (ref)":   (f"{W}/2026-06-11_stopgrad_capacity/runs/bb_allt08_xftrip_nobn_enc6_sgpos_qk_aon_b1024_losses.csv", CYAN, "--"),
    "enc6 + CPC (ref)":  (f"{W}/2026-06-13_cpc_infonce_aux/runs/bb_allt08_xftrip_nobn_enc6_sgpos_qk_aon_b1024_cpc_losses.csv", ORANGERED, "--"),
}
OUT = os.path.join(os.path.dirname(__file__), "..", "plots", "training_dynamics.png")
SMOOTH = 25
START_STEP = 100

# The four panels the report discusses; kept to 2×2 so each is legible.
PANELS = [
    ("loss_tau_ref", "contrastive reference loss (norm-InfoNCE τ=0.07)  (↓)", lambda v: v),
    ("cpc_aux",      "CPC InfoNCE term value  (CPC / CPC_All arms)", lambda v: v),
    ("r2_naive",     "1 − R²_naive  (↓)", lambda v: 1 - v),
    ("auc",          "1 − retrieval AUC  (↓)", lambda v: 1 - v),
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax in axes.flat[len(PANELS):]:
        ax.set_visible(False)
    for ax, (col, title, tf) in zip(axes.flat, PANELS):
        for lab, (path, c, ls) in RUNS.items():
            d = load(path)
            if not d or col not in d or "step" not in d:
                continue
            y = [tf(v) for v in d[col]]
            sm = smooth(y, SMOOTH)
            step = d["step"]
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
    fig.suptitle("No-encoder training dynamics (log-log; solid = no-encoder base/+CPC/+CPC_All, "
                 "dashed = enc-6 reference)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
