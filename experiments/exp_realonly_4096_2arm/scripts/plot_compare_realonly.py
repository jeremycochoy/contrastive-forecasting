"""Comparison plot: realonly_4096 (T=4096, C=1, mix_ratio=0.0) vs the
phase-3-and-2 best-of-breed and the phase-0 periodic baseline.

The headline question this plot answers:
  Does pure-real-data pretraining at T=4096 / C=1 match or beat the
  half-synth phase 1–5 recipes at T=1024 / C=4?

Reads:
  experiments/exp_realonly_4096_2arm/results/gift_eval_{revin,ewma128}/all_results.csv (this exp)
  experiments/exp_compositesynth_v3primitives_2arm/results/gift_eval_ewma128/all_results.csv (best EWMA from phases)
  experiments/exp_compositesynth_v2pulse_2arm/results/gift_eval_revin/all_results.csv         (best RevIN from phases)
  experiments/exp_dualemb_3arm/results/gift_eval_{revin,ewma128}/all_results.csv              (phase 0 periodic)

Run from the repo root:
    python experiments/exp_realonly_4096_2arm/scripts/plot_compare_realonly.py
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
OUT = HERE / "plots" / "gift_eval_realonly_compare.png"


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
            sn_mape = r.get("eval_metrics/SN_MAPE_ratio", "")
            sn_wql = r.get("eval_metrics/SN_WQL_ratio", "")
            try:
                smape = float(sn_mape) if sn_mape else float("nan")
            except (TypeError, ValueError):
                smape = float("nan")
            try:
                swql = float(sn_wql) if sn_wql else float("nan")
            except (TypeError, ValueError):
                swql = float("nan")
            rows.append({
                "config": r["dataset"], "domain": r.get("domain", "?"),
                "mase": v, "sn_mape": smape, "sn_wql": swql,
            })
    return rows


def _gm(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v) and v > 0]
    if not finite:
        return float("nan")
    return math.exp(mean(math.log(v) for v in finite))


def main() -> None:
    arms = [
        ("real_revin",   RESULTS_DIR / "gift_eval_revin" / "all_results.csv",
            "realonly_4096 + RevIN (mix=0.0)",     "#8B0000"),
        ("real_ewma128", RESULTS_DIR / "gift_eval_ewma128" / "all_results.csv",
            "realonly_4096 + EWMA-128 (mix=0.0)",  "#FF4500"),
        ("v2pulse_revin",   V2P_DIR / "gift_eval_revin" / "all_results.csv",
            "v2pulse + RevIN (best phase RevIN)",   "#d62728"),
        ("v3_ewma128",      V3_DIR / "gift_eval_ewma128" / "all_results.csv",
            "v3prim + EWMA-128 (best phase EWMA)",  "#ff7f0e"),
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

    aligned_mase = {k: [] for k in keys}
    aligned_sn_mape = {k: [] for k in keys}
    aligned_sn_wql = {k: [] for k in keys}
    for cfg in common:
        for k in keys:
            r = next(x for x in data[k] if x["config"] == cfg)
            aligned_mase[k].append(r["mase"])
            aligned_sn_mape[k].append(r["sn_mape"])
            aligned_sn_wql[k].append(r["sn_wql"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    gms = [_gm(aligned_mase[k]) for k in keys]
    medians = [float(np.median(aligned_mase[k])) for k in keys]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, gms, width=0.38, label="GM-MASE",
           color=[colors[k] for k in keys])
    ax.bar(x + 0.2, medians, width=0.38, label="median MASE", alpha=0.6,
           color=[colors[k] for k in keys])
    for i, (g, m) in enumerate(zip(gms, medians)):
        if math.isfinite(g):
            ax.text(i - 0.2, g + 0.02, f"{g:.3f}", ha="center", fontsize=8)
        if math.isfinite(m):
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
        v = sorted(aligned_mase[k])
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
        gm_per_domain = [_gm([aligned_mase[k][i] for i in domain_to_idx[d]]) for d in domains]
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
    if "real_ewma128" in data and "v3_ewma128" in data:
        pairs.append(("realonly vs v3prim — EWMA-128 (synth-vs-realonly test)",
                      "real_ewma128", "v3_ewma128", "#FF4500"))
    if "real_revin" in data and "v2pulse_revin" in data:
        pairs.append(("realonly vs v2pulse — RevIN (synth-vs-realonly test)",
                      "real_revin", "v2pulse_revin", "#8B0000"))
    if pairs:
        lo, hi = 0.3, 200
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=0.8)
        for label, comp_key, base_key, c in pairs:
            xs = np.array(aligned_mase[base_key])
            ys = np.array(aligned_mase[comp_key])
            wins = int((ys < xs).sum())
            ax.scatter(xs, ys, s=14, alpha=0.7, color=c,
                       label=f"{label}: realonly wins {wins}/{len(xs)}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Phase-best MASE")
        ax.set_ylabel("realonly_4096 MASE")
        ax.set_title("realonly vs phase winners (below diagonal = realonly wins)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Not enough arms to plot head-to-head",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle(
        "realonly_4096 (T=4096, C=1, mix=0.0) vs phase 1-5 winners on GIFT-Eval — "
        f"{len(common)} configs",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"Saved {OUT}")
    print("\nPer-arm aggregate (lower better):")
    for k in keys:
        v = aligned_mase[k]
        gm = _gm(v)
        gm_str = f"{gm:.3f}" if math.isfinite(gm) else "  N/A"
        gm_mape = _gm(aligned_sn_mape[k])
        gm_wql = _gm(aligned_sn_wql[k])
        sn_str = ""
        if math.isfinite(gm_mape) or math.isfinite(gm_wql):
            mape_s = f"{gm_mape:.3f}" if math.isfinite(gm_mape) else "N/A"
            wql_s = f"{gm_wql:.3f}" if math.isfinite(gm_wql) else "N/A"
            sn_str = f"  GM-MAPE_SN={mape_s}  GM-CRPS_SN={wql_s}"
        print(f"  {labels[k]:>40s}  GM={gm_str}  median={np.median(v):.3f}  "
              f"max={max(v):.1f}  configs<1.5={(np.array(v) < 1.5).sum()}/{len(v)}{sn_str}")
    print("\nHead-to-head (lower MASE wins):")
    for label, ck, bk, _ in pairs:
        xs = np.array(aligned_mase[bk])
        ys = np.array(aligned_mase[ck])
        print(f"  {label}: realonly beats phase winner on {(ys < xs).sum()}/{len(xs)} configs")


if __name__ == "__main__":
    main()
