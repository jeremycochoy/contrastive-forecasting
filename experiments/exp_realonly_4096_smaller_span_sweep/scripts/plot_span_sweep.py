"""Span sweep plot for the smaller-arch EWMA realonly_4096 setup.

Spans: 32, 64, 128 (from exp_realonly_4096_smaller_2arm), 256, 512.

Reads:
  experiments/exp_realonly_4096_smaller_span_sweep/results/gift_eval_ewma_span{32,64,256,512}/all_results.csv
  experiments/exp_realonly_4096_smaller_2arm/results/gift_eval_ewma128/all_results.csv  (span=128)

Plots:
  - GM-MASE / GM-MAPE_SN / GM-CRPS_SN as a function of span (line plot)
  - Per-domain GM at each span (heatmap-ish bars)
  - configs<1.5 count per span

Run:
    python experiments/exp_realonly_4096_smaller_span_sweep/scripts/plot_span_sweep.py
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
SPAN128_DIR = HERE.parent / "exp_realonly_4096_smaller_2arm" / "results"
OUT = HERE / "plots" / "span_sweep.png"


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
    spans = [32, 64, 128, 256, 512]
    paths = {
        32:  SWEEP_DIR / "gift_eval_ewma_span32" / "all_results.csv",
        64:  SWEEP_DIR / "gift_eval_ewma_span64" / "all_results.csv",
        128: SPAN128_DIR / "gift_eval_ewma128" / "all_results.csv",
        256: SWEEP_DIR / "gift_eval_ewma_span256" / "all_results.csv",
        512: SWEEP_DIR / "gift_eval_ewma_span512" / "all_results.csv",
    }
    data = {s: _load(paths[s]) for s in spans}
    available = [s for s in spans if data[s]]
    print(f"Spans with data: {available}")

    if not available:
        print("No data yet.")
        return

    config_sets = [{r["config"] for r in data[s]} for s in available]
    common = sorted(set.intersection(*config_sets))
    print(f"Common configs across {len(available)} spans: {len(common)}")

    aligned_mase: dict[int, list[float]] = {s: [] for s in available}
    aligned_sn_mape: dict[int, list[float]] = {s: [] for s in available}
    aligned_sn_wql: dict[int, list[float]] = {s: [] for s in available}
    for cfg in common:
        for s in available:
            r = next(x for x in data[s] if x["config"] == cfg)
            aligned_mase[s].append(r["mase"])
            aligned_sn_mape[s].append(r["sn_mape"])
            aligned_sn_wql[s].append(r["sn_wql"])

    gms = {s: _gm(aligned_mase[s]) for s in available}
    gm_mape = {s: _gm(aligned_sn_mape[s]) for s in available}
    gm_wql = {s: _gm(aligned_sn_wql[s]) for s in available}
    counts_under_1_5 = {s: int((np.array(aligned_mase[s]) < 1.5).sum()) for s in available}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    xs = available
    ax.plot(xs, [gms[s] for s in xs], 'o-', color='#FF4500', markersize=8, label='GM-MASE')
    ax.plot(xs, [gm_mape[s] for s in xs], 's-', color='#1f77b4', markersize=8, label='GM-MAPE_SN')
    ax.plot(xs, [gm_wql[s] for s in xs], '^-', color='#2ca02c', markersize=8, label='GM-CRPS_SN')
    ax.axhline(0.882, color='#1f77b4', linestyle=':', alpha=0.5, label='Aksu MAPE target 0.882')
    ax.axhline(0.642, color='#2ca02c', linestyle=':', alpha=0.5, label='Aksu CRPS target 0.642')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.7)
    ax.set_xscale('log', base=2)
    ax.set_xticks(spans)
    ax.set_xticklabels([str(s) for s in spans])
    ax.set_xlabel('span')
    ax.set_ylabel('GM (lower better)')
    ax.set_title(f'Span sweep — {len(common)} configs')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.bar(range(len(xs)), [counts_under_1_5[s] for s in xs],
           color=['#FF4500' if s == min(gms, key=gms.get) else '#888' for s in xs])
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(s) for s in xs])
    ax.set_xlabel('span')
    ax.set_ylabel(f'configs with MASE < 1.5 (out of {len(common)})')
    ax.set_title('Count of "near-or-better-than-SN" configs per span')
    for i, s in enumerate(xs):
        ax.text(i, counts_under_1_5[s] + 0.5, str(counts_under_1_5[s]),
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
    for ki, s in enumerate(available):
        gm_per_domain = [_gm([aligned_mase[s][i] for i in domain_to_idx[d]]) for d in domains]
        offs = (ki - (len(available) - 1) / 2.0) * width
        ax.bar(np.arange(len(domains)) + offs, gm_per_domain, width=width,
               label=f"span={s}", color=cmap(ki / max(len(available)-1, 1)))
    ax.set_xticks(np.arange(len(domains)))
    ax.set_xticklabels(domains, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('GM-MASE')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.7)
    ax.set_title('Per-domain GM-MASE by span')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 1]
    for s in available:
        v = sorted(aligned_mase[s])
        y = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, y, label=f"span={s}", linewidth=1.5,
                color=cmap(available.index(s) / max(len(available)-1, 1)))
    ax.axvline(1.0, color='gray', linestyle='--', linewidth=0.7)
    ax.set_xscale('log')
    ax.set_xlim(0.3, 200)
    ax.set_xlabel('MASE (log)')
    ax.set_ylabel('Fraction of configs ≤ MASE')
    ax.set_title(f"MASE CDF by span ({len(common)} configs)")
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "EWMA span sweep on smaller arch (L=6 H=384 nhead=6) — realonly_4096",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"Saved {OUT}")
    print()
    print(f"{'span':>5}  {'GM-MASE':>10}  {'GM-MAPE_SN':>12}  {'GM-CRPS_SN':>12}  {'<1.5':>6}")
    for s in xs:
        print(f"{s:>5}  {gms[s]:>10.4f}  {gm_mape[s]:>12.4f}  {gm_wql[s]:>12.4f}  "
              f"{counts_under_1_5[s]:>3}/{len(common)}")


if __name__ == "__main__":
    main()
