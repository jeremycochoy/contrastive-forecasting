"""Comparison plot: composite-v5envboost (env_gain_max=100) vs prior winners.

v5 arms (this exp):
  ewma128: --more-primitives + --env-gain-max 100 (env-bump on top of v3 winner)
  revin:   --enable-pulse + --env-gain-max 100 (env-bump on top of v2pulse winner)

NB: v5-revin segfaulted at config 72/97 — the CSV has 72 configs only. The
plot uses the intersection of configs across all arms, so all comparisons
fall back to those 72 automatically.

Reads:
  experiments/exp_compositesynth_v5envboost_2arm/results/gift_eval_{revin,ewma128}/all_results.csv  (this exp)
  experiments/exp_compositesynth_v3primitives_2arm/results/gift_eval_ewma128/all_results.csv         (prior EWMA winner)
  experiments/exp_compositesynth_v2pulse_2arm/results/gift_eval_revin/all_results.csv                (prior RevIN winner)
  experiments/exp_dualemb_3arm/results/gift_eval_{revin,ewma128}/all_results.csv                     (phase 0 periodic)

Run from the repo root:
    python experiments/exp_compositesynth_v5envboost_2arm/scripts/plot_compare_2arm.py
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
V3_DIR = HERE.parent / "exp_compositesynth_v3primitives_2arm" / "results"
V2P_DIR = HERE.parent / "exp_compositesynth_v2pulse_2arm" / "results"
PHASE0_DIR = HERE.parent / "exp_dualemb_3arm" / "results"
OUT = HERE / "plots" / "gift_eval_v5envboost_compare.png"


def _load(path: Path) -> list[dict]:
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
            rows.append({"config": r["dataset"], "domain": r.get("domain", "?"), "mase": v})
    return rows


def _gm(values: list[float]) -> float:
    return math.exp(mean(math.log(v) for v in values))


def main() -> None:
    arms = [
        ("v5_revin",   RESULTS_DIR / "gift_eval_revin" / "all_results.csv",
            "v5envboost + RevIN (env100)",      "#8B0000"),
        ("v5_ewma128", RESULTS_DIR / "gift_eval_ewma128" / "all_results.csv",
            "v5envboost + EWMA-128 (env100)",   "#FF4500"),
        ("v2pulse_revin",   V2P_DIR / "gift_eval_revin" / "all_results.csv",
            "v2pulse + RevIN (prev RevIN winner)",   "#d62728"),
        ("v3_ewma128",      V3_DIR / "gift_eval_ewma128" / "all_results.csv",
            "v3prim + EWMA-128 (prev EWMA winner)",  "#ff7f0e"),
        ("base_revin",   PHASE0_DIR / "gift_eval_revin" / "all_results.csv",
            "periodic + RevIN (phase 0)",      "#1f77b4"),
        ("base_ewma128", PHASE0_DIR / "gift_eval_ewma128" / "all_results.csv",
            "periodic + EWMA-128 (phase 0)",   "#2ca02c"),
    ]

    data = {}
    for key, path, _, _ in arms:
        if not path.exists():
            print(f"  ! missing {path}, skipping {key}")
            continue
        data[key] = _load(path)
    if not data:
        print("No results found yet.")
        return

    keys = [k for (k, _, _, _) in arms if k in data]
    labels = {k: lbl for (k, _, lbl, _) in arms}
    colors = {k: c for (k, _, _, c) in arms}

    config_sets = [{r["config"] for r in data[k]} for k in keys]
    common = sorted(set.intersection(*config_sets))
    print(f"Configs in common across {len(keys)} arms: {len(common)}")
    print(f"  (v5-revin has {len(data.get('v5_revin', []))} configs — intersection size dominated by this)")

    aligned = {k: [] for k in keys}
    for cfg in common:
        for k in keys:
            r = next(x for x in data[k] if x["config"] == cfg)
            aligned[k].append(r["mase"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    gms = [_gm(aligned[k]) for k in keys]
    medians = [float(np.median(aligned[k])) for k in keys]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, gms, width=0.38, label="GM-MASE",
           color=[colors[k] for k in keys])
    ax.bar(x + 0.2, medians, width=0.38, label="median MASE", alpha=0.6,
           color=[colors[k] for k in keys])
    for i, (g, m) in enumerate(zip(gms, medians)):
        ax.text(i - 0.2, g + 0.02, f"{g:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, m + 0.02, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in keys], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("MASE (lower better)")
    ax.set_title(f"Aggregate MASE per arm ({len(common)} configs)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8,
               label="seasonal-naive (=1)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    for k in keys:
        v = sorted(aligned[k])
        y = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, y, label=labels[k], color=colors[k], linewidth=1.5)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 200)
    ax.set_xlabel("MASE (log)")
    ax.set_ylabel("Fraction of configs ≤ MASE")
    ax.set_title(f"MASE CDF over {len(common)} configs")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    domain_to_idx: dict[str, list[int]] = {}
    for i, cfg in enumerate(common):
        d = next(x for x in data[keys[0]] if x["config"] == cfg)["domain"]
        domain_to_idx.setdefault(d, []).append(i)
    domains = sorted(domain_to_idx.keys())
    width = 0.85 / max(len(keys), 1)
    for ki, k in enumerate(keys):
        gm_per_domain = [_gm([aligned[k][i] for i in domain_to_idx[d]]) for d in domains]
        offs = (ki - (len(keys) - 1) / 2.0) * width
        ax.bar(np.arange(len(domains)) + offs, gm_per_domain, width=width,
               label=labels[k], color=colors[k])
    ax.set_xticks(np.arange(len(domains)))
    ax.set_xticklabels(domains, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("GM-MASE")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("GM-MASE by domain")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    pairs = []
    if "v5_ewma128" in data and "v3_ewma128" in data:
        pairs.append(("v5envboost vs v3prim — EWMA-128 (env-bump test)",
                      "v5_ewma128", "v3_ewma128", "#FF4500"))
    if "v5_revin" in data and "v2pulse_revin" in data:
        pairs.append(("v5envboost vs v2pulse — RevIN (env-bump test)",
                      "v5_revin", "v2pulse_revin", "#8B0000"))
    if pairs:
        lo, hi = 0.3, 200
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=0.8)
        for label, comp_key, base_key, c in pairs:
            xs = np.array(aligned[base_key])
            ys = np.array(aligned[comp_key])
            wins = int((ys < xs).sum())
            ax.scatter(xs, ys, s=14, alpha=0.7, color=c,
                       label=f"{label}: v5 wins {wins}/{len(xs)}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Reference (prior winner) MASE")
        ax.set_ylabel("v5envboost MASE")
        ax.set_title("v5 head-to-head: env-bump effect (below diagonal = v5 wins)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Not enough arms to plot head-to-head",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle(
        "Composite-v5envboost (env_gain_max=100) vs prior winners on GIFT-Eval — "
        f"{len(common)} configs (v5-revin partial — eval segfaulted at config 72)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"Saved {OUT}")
    print("\nPer-arm aggregate (lower better):")
    for k in keys:
        v = aligned[k]
        print(f"  {labels[k]:>40s}  GM={_gm(v):.3f}  median={np.median(v):.3f}  "
              f"max={max(v):.1f}  configs<1.5={(np.array(v) < 1.5).sum()}/{len(v)}")
    print("\nHead-to-head (lower MASE wins):")
    for label, ck, bk, _ in pairs:
        xs = np.array(aligned[bk])
        ys = np.array(aligned[ck])
        print(f"  {label}: v5 beats prior winner on {(ys < xs).sum()}/{len(xs)} configs")


if __name__ == "__main__":
    main()
