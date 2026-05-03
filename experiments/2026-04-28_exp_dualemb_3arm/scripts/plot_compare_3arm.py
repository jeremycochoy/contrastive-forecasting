"""Comparison plot for the dualemb 3-arm experiment.

Reads the three gift-eval result CSVs and produces:
  - Per-config MASE scatter (3 arms side-by-side)
  - Per-domain GM-MASE bars
  - CDF of MASE per arm

Run from the repo root:
    python experiments/2026-04-28_exp_dualemb_3arm/scripts/plot_compare_3arm.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent.parent
RESULTS_DIR = HERE / "results"
OUT = HERE / "plots" / "gift_eval_3arm_compare.png"


def _load(arm: str) -> list[dict]:
    path = RESULTS_DIR / f"gift_eval_{arm}" / "all_results.csv"
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            mase = r.get("eval_metrics/MASE[0.5]", "")
            try:
                v = float(mase)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v) or v <= 0:
                continue
            rows.append({
                "config": r["dataset"],
                "domain": r.get("domain", "?"),
                "mase": v,
            })
    return rows


def _gm(values: list[float]) -> float:
    return math.exp(mean(math.log(v) for v in values))


def main() -> None:
    arms = ["revin", "ewma512", "ewma128"]
    labels = ["RevIN", "EWMA span=512", "EWMA span=128"]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]

    data = {a: _load(a) for a in arms}
    # Configs in common (should be all 97)
    config_sets = [{r["config"] for r in data[a]} for a in arms]
    common = set.intersection(*config_sets)
    print(f"Configs in common across all 3 arms: {len(common)}")

    # Aligned per-config arrays for direct comparison
    aligned = {a: [] for a in arms}
    configs_sorted = sorted(common)
    for cfg in configs_sorted:
        for a in arms:
            r = next(x for x in data[a] if x["config"] == cfg)
            aligned[a].append(r["mase"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1: bar of GM-MASE / median per arm
    ax = axes[0, 0]
    gms = [_gm(aligned[a]) for a in arms]
    medians = [float(np.median(aligned[a])) for a in arms]
    x = np.arange(len(arms))
    ax.bar(x - 0.2, gms, width=0.38, label="GM-MASE", color=colors)
    ax.bar(x + 0.2, medians, width=0.38, label="median MASE", alpha=0.6, color=colors)
    for i, (g, m) in enumerate(zip(gms, medians)):
        ax.text(i - 0.2, g + 0.01, f"{g:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.2, m + 0.01, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MASE (lower is better)")
    ax.set_title("Aggregate MASE per arm")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="seasonal-naive (=1)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: CDF of MASE per arm
    ax = axes[0, 1]
    for a, lbl, c in zip(arms, labels, colors):
        v = sorted(aligned[a])
        y = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, y, label=lbl, color=c, linewidth=1.7)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 200)
    ax.set_xlabel("MASE (log)")
    ax.set_ylabel("Fraction of configs ≤ MASE")
    ax.set_title("MASE CDF over 97 configs")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: per-domain GM-MASE
    ax = axes[1, 0]
    domain_to_idx: dict[str, list[int]] = {}
    for i, cfg in enumerate(configs_sorted):
        d = next(x for x in data[arms[0]] if x["config"] == cfg)["domain"]
        domain_to_idx.setdefault(d, []).append(i)
    domains = sorted(domain_to_idx.keys())
    width = 0.27
    for k, (a, lbl, c) in enumerate(zip(arms, labels, colors)):
        gm_per_domain = [_gm([aligned[a][i] for i in domain_to_idx[d]]) for d in domains]
        offs = (k - 1) * width
        ax.bar(np.arange(len(domains)) + offs, gm_per_domain, width=width,
               label=lbl, color=c)
    ax.set_xticks(np.arange(len(domains)))
    ax.set_xticklabels(domains, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("GM-MASE")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("GM-MASE by domain")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: head-to-head EWMA-128 vs EWMA-512 per config (winners)
    ax = axes[1, 1]
    a128 = np.array(aligned["ewma128"])
    a512 = np.array(aligned["ewma512"])
    arev = np.array(aligned["revin"])
    win128_512 = (a128 < a512).sum()
    win128_rev = (a128 < arev).sum()
    win512_rev = (a512 < arev).sum()
    ax.scatter(a512, a128, s=14, alpha=0.7, color="#2ca02c")
    lo, hi = 0.3, 200
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("EWMA-512 MASE")
    ax.set_ylabel("EWMA-128 MASE")
    ax.set_title(f"EWMA-128 vs EWMA-512: {win128_512}/{len(configs_sorted)} wins for -128")
    ax.grid(alpha=0.3)

    fig.suptitle("Dualemb 3-arm GIFT-Eval comparison (97 configs, csb loss, mix=0.5, freq+seasonality emb)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"Saved {OUT}")
    print(f"\nHead-to-head (lower MASE wins):")
    print(f"  EWMA-128 vs EWMA-512:  {win128_512}/{len(configs_sorted)}")
    print(f"  EWMA-128 vs RevIN:     {win128_rev}/{len(configs_sorted)}")
    print(f"  EWMA-512 vs RevIN:     {win512_rev}/{len(configs_sorted)}")


if __name__ == "__main__":
    main()
