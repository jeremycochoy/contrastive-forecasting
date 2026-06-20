#!/usr/bin/env python3
"""#350 — headline figure (2 panels), GIFT-Eval full-97 GM-Relative MASE.
Left: GM per (head × checkpoint) for the τ-scaled dot product (#348 + CPC
baseline) vs the learnable bilinear W. Right: paired-bootstrap Δ = GM(bilinear)
− GM(τ) with 90% CI per cell (Δ<0 ⇒ bilinear better; band crossing 0 ⇒ ns).
Reads results/gm_table.csv + pairwise_table.csv. Writes plots/gm_summary.png.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-16_bilinear_main_loss/results"
OUT = os.path.join(os.path.dirname(__file__), "..", "plots", "gm_summary.png")
CELLS = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]
BARS = [("τ-dot product + CPC (#348)", "cpc", "0.55"),
        ("learnable bilinear W + CPC", "bilinear", "C0")]


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    gm = {(r["arm"], r["head"], r["ckpt"]): f(r["gm"])
          for r in csv.DictReader(open(f"{RES}/gm_table.csv"))}
    pair = {(r["head"], r["ckpt"]): r
            for r in csv.DictReader(open(f"{RES}/pairwise_table.csv"))}
    x = range(len(CELLS))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.3))

    # Left: grouped GM bars.
    w = 0.8 / len(BARS)
    vals = [gm.get((k, h, c)) for _, k, _ in BARS for h, c in CELLS if gm.get((k, h, c))]
    lo, hi = (min(vals), max(vals)) if vals else (1.1, 1.2)
    for j, (blabel, key, col) in enumerate(BARS):
        ys = [gm.get((key, h, c)) for h, c in CELLS]
        off = (j - (len(BARS) - 1) / 2) * w
        axL.bar([i + off for i in x], [y or 0 for y in ys], w, label=blabel, color=col)
        for i, y in enumerate(ys):
            if y:
                axL.text(i + off, y, f"{y:.3f}", ha="center", va="bottom",
                         fontsize=8, rotation=90)
    axL.axhline(1.0, color="k", lw=0.8, ls=":")
    axL.set_xticks(list(x))
    axL.set_xticklabels([f"{h}\n{c}" for h, c in CELLS], fontsize=9)
    axL.set_ylabel("GM-Relative MASE (lower is better)")
    axL.set_ylim(max(1.0, lo - 0.02), hi + 0.03)
    axL.set_title("GIFT-Eval full-97 GM-Relative MASE", fontsize=11)
    axL.legend(fontsize=9)
    axL.grid(axis="y", alpha=0.3)

    # Right: paired-bootstrap Δ with 90% CI.
    deltas, los, his = [], [], []
    for h, c in CELLS:
        r = pair.get((h, c), {})
        deltas.append(f(r.get("delta")))
        los.append(f(r.get("ci_lo")))
        his.append(f(r.get("ci_hi")))
    yerr_lo = [(d - l) if (d is not None and l is not None) else 0
               for d, l in zip(deltas, los)]
    yerr_hi = [(u - d) if (d is not None and u is not None) else 0
               for d, u in zip(deltas, his)]
    cols = ["C0" if (u is not None and u < 0) else
            ("C3" if (l is not None and l > 0) else "0.5")
            for l, u in zip(los, his)]
    axR.axhline(0.0, color="k", lw=1.0)
    for i, (d, col) in enumerate(zip(deltas, cols)):
        if d is None:
            continue
        axR.errorbar(i, d, yerr=[[yerr_lo[i]], [yerr_hi[i]]], fmt="o",
                     color=col, capsize=5, ms=7)
        axR.text(i + 0.08, d, f"{d:+.3f}", va="center", fontsize=8)
    axR.set_xticks(list(x))
    axR.set_xticklabels([f"{h}\n{c}" for h, c in CELLS], fontsize=9)
    axR.set_xlim(-0.5, len(CELLS) - 0.2)
    axR.set_ylabel("Δ GM = bilinear − τ  (90% CI)")
    axR.set_title("Paired bootstrap: bilinear − τ baseline\n(Δ<0 ⇒ bilinear better; CI crossing 0 ⇒ ns)",
                  fontsize=11)
    axR.grid(axis="y", alpha=0.3)

    fig.suptitle("Learnable bilinear W vs τ-scaled dot product in the main contrastive loss",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
