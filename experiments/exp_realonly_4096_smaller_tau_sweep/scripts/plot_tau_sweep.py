"""τ sweep plot for the smaller-arch + EWMA-128 realonly_4096 setup.

τs: 0.05 (this exp), 0.07 (= #20 EWMA-smaller, reference), 0.20 (this exp).

Reads:
  experiments/exp_realonly_4096_smaller_tau_sweep/results/gift_eval_tau005/all_results.csv
  experiments/exp_realonly_4096_smaller_tau_sweep/results/gift_eval_tau020/all_results.csv
  experiments/exp_realonly_4096_smaller_2arm/results/gift_eval_ewma128/all_results.csv  (τ=0.07 reference)

Caveat: τ=0.05 / 0.20 use bs=96; τ=0.07 reference (#20 EWMA-smaller) used bs=24. Direct
comparison is approximate due to bs change. Within {0.05, 0.20} the comparison is clean.

Run:
    python experiments/exp_realonly_4096_smaller_tau_sweep/scripts/plot_tau_sweep.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent.parent
SWEEP_DIR = HERE / "results"
TAU007_DIR = HERE.parent / "exp_realonly_4096_smaller_2arm" / "results"
OUT = HERE / "plots" / "tau_sweep.png"


def _load(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
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
    taus = [0.05, 0.07, 0.20]
    paths = {
        0.05: SWEEP_DIR / "gift_eval_tau005" / "all_results.csv",
        0.07: TAU007_DIR / "gift_eval_ewma128" / "all_results.csv",
        0.20: SWEEP_DIR / "gift_eval_tau020" / "all_results.csv",
    }
    data = {t: _load(paths[t]) for t in taus}
    available = [t for t in taus if data[t]]
    print(f"τs with data: {available}")

    if not available:
        print("No data yet.")
        return

    config_sets = [{r["config"] for r in data[t]} for t in available]
    common = sorted(set.intersection(*config_sets))
    print(f"Common configs across {len(available)} τs: {len(common)}")

    aligned_mase: dict[float, list[float]] = {t: [] for t in available}
    aligned_sn_mape: dict[float, list[float]] = {t: [] for t in available}
    aligned_sn_wql: dict[float, list[float]] = {t: [] for t in available}
    for cfg in common:
        for t in available:
            r = next(x for x in data[t] if x["config"] == cfg)
            aligned_mase[t].append(r["mase"])
            aligned_sn_mape[t].append(r["sn_mape"])
            aligned_sn_wql[t].append(r["sn_wql"])

    gms = {t: _gm(aligned_mase[t]) for t in available}
    gm_mape = {t: _gm(aligned_sn_mape[t]) for t in available}
    gm_wql = {t: _gm(aligned_sn_wql[t]) for t in available}
    counts_under_1_5 = {t: int((np.array(aligned_mase[t]) < 1.5).sum()) for t in available}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    xs = available
    ax.plot(xs, [gms[t] for t in xs], 'o-', color='#FF4500', markersize=10, label='GM-MASE')
    ax.plot(xs, [gm_mape[t] for t in xs], 's-', color='#1f77b4', markersize=10, label='GM-MAPE_SN')
    ax.plot(xs, [gm_wql[t] for t in xs], '^-', color='#2ca02c', markersize=10, label='GM-CRPS_SN')
    ax.axhline(0.882, color='#1f77b4', linestyle=':', alpha=0.5, label='Aksu MAPE 0.882')
    ax.axhline(0.642, color='#2ca02c', linestyle=':', alpha=0.5, label='Aksu CRPS 0.642')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.7)
    for t in xs:
        ax.annotate(f"{gms[t]:.3f}", (t, gms[t]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    ax.set_xscale('log')
    ax.set_xticks(taus)
    ax.set_xticklabels([str(t) for t in taus])
    ax.set_xlabel('τ (contrastive temperature)')
    ax.set_ylabel('GM (lower better)')
    ax.set_title(f'τ sweep — {len(common)} configs (smaller arch + EWMA-128)')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.bar(range(len(xs)), [counts_under_1_5[t] for t in xs],
           color=['#FF4500' if t == min(gms, key=gms.get) else '#888' for t in xs])
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f'τ={t}' for t in xs])
    ax.set_xlabel('τ')
    ax.set_ylabel(f'configs with MASE < 1.5 (out of {len(common)})')
    ax.set_title('"Near-or-better-than-SN" config count per τ')
    for i, t in enumerate(xs):
        ax.text(i, counts_under_1_5[t] + 0.5, str(counts_under_1_5[t]),
                ha='center', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 0]
    domain_to_idx: dict[str, list[int]] = {}
    for i, cfg in enumerate(common):
        d = next(x for x in data[available[0]] if x["config"] == cfg)["domain"]
        domain_to_idx.setdefault(d, []).append(i)
    domains = sorted(domain_to_idx.keys())
    width = 0.85 / max(len(available), 1)
    cmap = plt.cm.viridis
    for ki, t in enumerate(available):
        gm_per_domain = [_gm([aligned_mase[t][i] for i in domain_to_idx[d]]) for d in domains]
        offs = (ki - (len(available) - 1) / 2.0) * width
        ax.bar(np.arange(len(domains)) + offs, gm_per_domain, width=width,
               label=f"τ={t}", color=cmap(available.index(t) / max(len(available)-1, 1)))
    ax.set_xticks(np.arange(len(domains)))
    ax.set_xticklabels(domains, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('GM-MASE')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.7)
    ax.set_title('Per-domain GM-MASE by τ')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 1]
    for t in available:
        v = sorted(aligned_mase[t])
        y = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, y, label=f"τ={t}", linewidth=1.5,
                color=cmap(available.index(t) / max(len(available)-1, 1)))
    ax.axvline(1.0, color='gray', linestyle='--', linewidth=0.7)
    ax.set_xscale('log')
    ax.set_xlim(0.3, 200)
    ax.set_xlabel('MASE (log)')
    ax.set_ylabel('Fraction of configs ≤ MASE')
    ax.set_title(f"MASE CDF by τ ({len(common)} configs)")
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "τ sweep on smaller arch (L=6 H=384 nhead=6) + EWMA-128 — realonly_4096",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"Saved {OUT}")
    print()
    print(f"{'τ':>6}  {'GM-MASE':>10}  {'GM-MAPE_SN':>12}  {'GM-CRPS_SN':>12}  {'<1.5':>6}")
    for t in xs:
        print(f"{t:>6}  {gms[t]:>10.4f}  {gm_mape[t]:>12.4f}  {gm_wql[t]:>12.4f}  "
              f"{counts_under_1_5[t]:>3}/{len(common)}")


if __name__ == "__main__":
    main()
