"""Comparison plot: composite synth (2 arms) vs the periodic-synth baseline
from exp_dualemb_3arm (matching norms only — RevIN and EWMA span=128).

Reads four CSVs:
  experiments/exp_compositesynth_2arm/results/gift_eval_{revin,ewma128}/all_results.csv  (this exp)
  experiments/exp_dualemb_3arm/results/gift_eval_{revin,ewma128}/all_results.csv          (baseline)

Produces a single PNG with four panels:
  - Aggregate (GM, median) MASE per arm
  - MASE CDF
  - Per-domain GM-MASE
  - Head-to-head: composite vs periodic at the same norm

Run from the repo root:
    python experiments/exp_compositesynth_2arm/scripts/plot_compare_2arm.py
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
BASELINE_DIR = HERE.parent / "exp_dualemb_3arm" / "results"
OUT = HERE / "plots" / "gift_eval_compositesynth_compare.png"


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
            rows.append({
                "config": r["dataset"],
                "domain": r.get("domain", "?"),
                "mase": v,
            })
    return rows


def _gm(values: list[float]) -> float:
    return math.exp(mean(math.log(v) for v in values))


def main() -> None:
    arms = [
        ("comp_revin",  RESULTS_DIR / "gift_eval_revin"   / "all_results.csv", "Composite + RevIN",     "#d62728"),
        ("comp_ewma128", RESULTS_DIR / "gift_eval_ewma128" / "all_results.csv", "Composite + EWMA-128",  "#ff7f0e"),
        ("base_revin",  BASELINE_DIR / "gift_eval_revin"   / "all_results.csv", "Periodic + RevIN",      "#1f77b4"),
        ("base_ewma128", BASELINE_DIR / "gift_eval_ewma128" / "all_results.csv", "Periodic + EWMA-128",   "#2ca02c"),
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
    common = set.intersection(*config_sets)
    print(f"Configs in common across {len(keys)} arms: {len(common)}")

    aligned = {k: [] for k in keys}
    configs_sorted = sorted(common)
    for cfg in configs_sorted:
        for k in keys:
            r = next(x for x in data[k] if x["config"] == cfg)
            aligned[k].append(r["mase"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: bar of GM-MASE / median per arm
    ax = axes[0, 0]
    gms = [_gm(aligned[k]) for k in keys]
    medians = [float(np.median(aligned[k])) for k in keys]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, gms, width=0.38, label="GM-MASE",
           color=[colors[k] for k in keys])
    ax.bar(x + 0.2, medians, width=0.38, label="median MASE", alpha=0.6,
           color=[colors[k] for k in keys])
    for i, (g, m) in enumerate(zip(gms, medians)):
        ax.text(i - 0.2, g + 0.01, f"{g:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.2, m + 0.01, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in keys], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("MASE (lower is better)")
    ax.set_title("Aggregate MASE per arm")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8,
               label="seasonal-naive (=1)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: CDF of MASE per arm
    ax = axes[0, 1]
    for k in keys:
        v = sorted(aligned[k])
        y = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, y, label=labels[k], color=colors[k], linewidth=1.7)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 200)
    ax.set_xlabel("MASE (log)")
    ax.set_ylabel("Fraction of configs ≤ MASE")
    ax.set_title(f"MASE CDF over {len(configs_sorted)} configs")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: per-domain GM-MASE
    ax = axes[1, 0]
    domain_to_idx: dict[str, list[int]] = {}
    for i, cfg in enumerate(configs_sorted):
        d = next(x for x in data[keys[0]] if x["config"] == cfg)["domain"]
        domain_to_idx.setdefault(d, []).append(i)
    domains = sorted(domain_to_idx.keys())
    width = 0.8 / max(len(keys), 1)
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
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: head-to-head — composite vs periodic at matched norm
    ax = axes[1, 1]
    pairs = []
    if "comp_revin" in data and "base_revin" in data:
        pairs.append(("RevIN", "comp_revin", "base_revin", "#d62728"))
    if "comp_ewma128" in data and "base_ewma128" in data:
        pairs.append(("EWMA-128", "comp_ewma128", "base_ewma128", "#ff7f0e"))

    if pairs:
        lo, hi = 0.3, 200
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=0.8)
        for label, comp_key, base_key, c in pairs:
            xs = np.array(aligned[base_key])
            ys = np.array(aligned[comp_key])
            wins = int((ys < xs).sum())
            ax.scatter(xs, ys, s=14, alpha=0.7, color=c,
                       label=f"{label}: {wins}/{len(xs)} wins for composite")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Periodic-synth baseline MASE")
        ax.set_ylabel("Composite-synth MASE")
        ax.set_title("Composite vs Periodic at matched norm")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No matched arm pairs available yet",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle(
        "Composite-synth (TimesFM-style) vs periodic-synth on GIFT-Eval — "
        f"{len(configs_sorted)} configs, csb loss, mix=0.5, freq+seasonality emb",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"Saved {OUT}")
    print("\nPer-arm aggregate:")
    for k in keys:
        v = aligned[k]
        print(f"  {labels[k]:>22s}  GM={_gm(v):.3f}  median={np.median(v):.3f}  "
              f"max={max(v):.1f}  configs<1.5={(np.array(v) < 1.5).sum()}/{len(v)}")
    print("\nHead-to-head (lower MASE wins):")
    for label, comp_key, base_key, _ in pairs:
        xs = np.array(aligned[base_key])
        ys = np.array(aligned[comp_key])
        print(f"  {label:>10s}: composite beats periodic on {(ys < xs).sum()}/{len(xs)}")


if __name__ == "__main__":
    main()
