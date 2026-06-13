#!/usr/bin/env python3
"""#344 — headline figure. Left: GM-Relative MASE, baseline vs CPC, grouped by
arm × head × checkpoint. Right: the CPC−baseline paired-bootstrap Δ with 90%
CI per cell (negative ⇒ CPC better). Reads results/{gm_table,pairwise_table}.csv
(written by analyze_cpc.py). Writes plots/gm_summary.png.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-13_cpc_infonce_aux/results"
OUT = os.path.join(os.path.dirname(__file__), "..", "plots", "gm_summary.png")
CLIP = 1.30  # collapsed/last bars can exceed this; clip and label the true value


def rows(name):
    p = f"{RES}/{name}"
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    gm = rows("gm_table.csv")
    pair = rows("pairwise_table.csv")
    gmv = {(r["arm"], r["head"], r["ckpt"]): f(r["gm"]) for r in gm}

    # cells in display order: (head, ckpt)
    cells = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]
    arms = [("enc3", "base_enc3", "cpc_enc3"), ("enc6", "base_enc6", "cpc_enc6")]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # ---- Left: grouped GM bars (baseline vs CPC) ----
    labels, base_vals, cpc_vals = [], [], []
    for arm, bkey, ckey in arms:
        for head, ck in cells:
            labels.append(f"{arm}\n{head} {ck}")
            base_vals.append(gmv.get((bkey, head, ck)))
            cpc_vals.append(gmv.get((ckey, head, ck)))
    x = range(len(labels))
    w = 0.4

    def clipped(vals):
        return [min(v, CLIP) if v is not None else 0 for v in vals]

    axL.bar([i - w / 2 for i in x], clipped(base_vals), w, label="baseline", color="0.6")
    axL.bar([i + w / 2 for i in x], clipped(cpc_vals), w, label="+ CPC", color="C2")
    for i, v in enumerate(base_vals):
        if v is not None and v > CLIP:
            axL.text(i - w / 2, CLIP, f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    for i, v in enumerate(cpc_vals):
        if v is not None and v > CLIP:
            axL.text(i + w / 2, CLIP, f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    axL.axhline(1.0, color="k", lw=0.8, ls=":", label="seasonal-naive")
    axL.set_ylim(1.1, CLIP + 0.02)
    axL.set_xticks(list(x))
    axL.set_xticklabels(labels, fontsize=8)
    axL.set_ylabel("GM-Relative MASE (lower better)")
    axL.set_title(f"GM-Relative MASE: baseline vs +CPC (bars clipped at {CLIP}, true value labelled)", fontsize=10)
    axL.legend(fontsize=9)
    axL.grid(axis="y", alpha=0.3)

    # ---- Right: CPC−baseline paired Δ with 90% CI ----
    pkeys = [("base_enc3", "cpc_enc3", "enc3"), ("base_enc6", "cpc_enc6", "enc6")]
    ylabels, deltas, los, his = [], [], [], []
    for A, B, arm in pkeys:
        for head, ck in cells:
            match = [r for r in pair if r["A"] == A and r["B"] == B and r["head"] == head and r["ckpt"] == ck]
            if not match:
                continue
            r = match[0]
            ylabels.append(f"{arm} {head} {ck}")
            deltas.append(f(r["delta"]))
            los.append(f(r["ci_lo"]))
            his.append(f(r["ci_hi"]))
    y = range(len(ylabels))
    for i, (d, lo, hi) in enumerate(zip(deltas, los, his)):
        if d is None:
            continue
        color = "C2" if hi is not None and hi < 0 else ("C3" if lo is not None and lo > 0 else "0.5")
        axR.plot([lo, hi], [i, i], color=color, lw=2)
        axR.plot(d, i, "o", color=color, ms=6)
    axR.axvline(0, color="k", lw=0.8)
    axR.set_yticks(list(y))
    axR.set_yticklabels(ylabels, fontsize=9)
    axR.invert_yaxis()
    axR.set_xlabel("Δ GM = GM(+CPC) − GM(baseline)   (negative ⇒ CPC better)")
    axR.set_title("Paired-bootstrap Δ, 90% CI\ngreen=CPC reliably better, red=reliably worse, grey=ns", fontsize=10)
    axR.grid(axis="x", alpha=0.3)

    fig.suptitle("#344 CPC InfoNCE auxiliary vs stop-grad baselines (GIFT-Eval full-97)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
