#!/usr/bin/env python3
"""Compute MASE and WQL skill scores against the local Seasonal-Naive baseline.

Reads a directory of GIFT-Eval ``all_results.csv`` files (one per arm)
and produces:

- A wide-format ``comparison_with_sn.csv`` joining all arms on the
  ``dataset`` column. Includes ``sn_mase`` and ``sn_wql`` from the
  per-config seasonal-naive baseline computed in
  ``experiments/periodic-synth-mix/scripts/seasonal_naive_check.py``.
  (Multivariate datasets — bitbrains_*, bizitobs_*, ett1/ett2 multi-channel —
  drop out because the local SN script chokes on them; see freq-embedding
  REPORT.md "caveats".)
- A plain-text ``skill_scores.txt`` with the GM-MASE / MASE-skill / GM-WQL /
  WQL-skill row per arm.

Skill score convention (higher is better, SN = 0):

    skill = 1 - GM-Rel   where GM-Rel = exp(mean(log(arm_metric / sn_metric)))

Usage:
    python experiments/freq-embedding/scripts/skill_scores.py \\
        --results-dir results/ \\
        --arms 'fe+mu_qh:R1q_femu fe+mu_revin_qh:R1q_femu_revin ...' \\
        --sn-csv experiments/freq-embedding/results/comparison_with_sn.csv \\
        --out-dir experiments/freq-embedding/results/

The --arms flag is a space-separated list of ``label:dirname`` pairs;
``label`` is what shows in the skill-scores table, ``dirname`` is the
subdirectory under ``--results-dir`` whose ``all_results.csv`` we read.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MASE_COL = "eval_metrics/MASE[0.5]"
WQL_COL = "eval_metrics/mean_weighted_sum_quantile_loss"


def _read_arm(csv_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if MASE_COL not in df.columns or WQL_COL not in df.columns:
        raise KeyError(
            f"Missing required columns in {csv_path}; "
            f"have {df.columns.tolist()}")
    return df[["dataset", MASE_COL, WQL_COL]].rename(columns={
        MASE_COL: f"mase_{label}",
        WQL_COL: f"wql_{label}",
    })


def _read_sn_baseline(sn_csv: Path) -> pd.DataFrame:
    """Pull the per-config sn_mase / sn_wql columns from an existing
    `comparison_with_sn.csv`. They were computed locally for 43 univariate
    configs (the rest fall through `NaN`)."""
    df = pd.read_csv(sn_csv)
    keep = ["dataset"]
    if "sn_mase" in df.columns:
        keep.append("sn_mase")
    if "sn_wql" in df.columns:
        keep.append("sn_wql")
    return df[keep]


def _gm(values: np.ndarray) -> float:
    """Geometric mean over finite-positive values."""
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(values))))


def _aggregate(joined: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    """Per-arm GM-MASE, GM-WQL, MASE-skill, WQL-skill on the rows where
    sn_mase + sn_wql are both available."""
    sn_mask = joined["sn_mase"].notna() & joined["sn_wql"].notna()
    sub = joined[sn_mask].copy()
    n = len(sub)
    sn_gm_mase = _gm(sub["sn_mase"].to_numpy())
    sn_gm_wql = _gm(sub["sn_wql"].to_numpy())

    rows = []
    for label in arms:
        m = sub[f"mase_{label}"].to_numpy()
        w = sub[f"wql_{label}"].to_numpy()
        gm_mase = _gm(m)
        gm_wql = _gm(w)
        # Per-config relative ratios then GM (avoids issues when one of
        # sn_mase / arm_mase is NaN for a row).
        rel_mase = m / sub["sn_mase"].to_numpy()
        rel_wql = w / sub["sn_wql"].to_numpy()
        gm_rel_mase = _gm(rel_mase)
        gm_rel_wql = _gm(rel_wql)
        mase_skill = (1.0 - gm_rel_mase) * 100.0
        wql_skill = (1.0 - gm_rel_wql) * 100.0
        rows.append({
            "arm": label,
            "n_configs": n,
            "gm_mase": gm_mase,
            "mase_skill_pct": mase_skill,
            "gm_wql": gm_wql,
            "wql_skill_pct": wql_skill,
        })

    rows.append({
        "arm": "Seasonal Naive",
        "n_configs": n,
        "gm_mase": sn_gm_mase,
        "mase_skill_pct": 0.0,
        "gm_wql": sn_gm_wql,
        "wql_skill_pct": 0.0,
    })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="Directory containing per-arm subdirectories with all_results.csv")
    ap.add_argument("--arms", required=True,
                    help="Space-separated 'label:dirname' pairs.")
    ap.add_argument("--sn-csv", required=True,
                    help="Existing comparison_with_sn.csv with sn_mase / sn_wql columns.")
    ap.add_argument("--out-dir", required=True,
                    help="Where to write comparison_with_sn.csv + skill_scores.txt")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_dir)

    # Parse arms
    pairs = []
    for entry in args.arms.split():
        if ":" not in entry:
            raise SystemExit(f"--arms entry '{entry}' must be 'label:dirname'")
        label, dirname = entry.split(":", 1)
        pairs.append((label, dirname))

    # Build joined frame on dataset
    joined = _read_sn_baseline(Path(args.sn_csv))
    for label, dirname in pairs:
        path = results_root / dirname / "all_results.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping arm '{label}'")
            continue
        df_arm = _read_arm(path, label)
        joined = joined.merge(df_arm, on="dataset", how="outer")

    # Drop arms missing entirely
    arms_present = [
        label for label, _ in pairs
        if f"mase_{label}" in joined.columns
    ]
    print(f"  arms found: {arms_present}")

    # Save the wide CSV
    csv_out = out_dir / "comparison_with_sn.csv"
    joined.sort_values("dataset").to_csv(csv_out, index=False)
    print(f"  wrote {csv_out} ({len(joined)} configs, "
          f"{len(arms_present) * 2} arm columns)")

    # Aggregate skill table
    table = _aggregate(joined, arms_present)
    txt_out = out_dir / "skill_scores.txt"
    n_configs = int(table.iloc[0]["n_configs"])
    sn_row = table[table["arm"] == "Seasonal Naive"].iloc[0]
    with open(txt_out, "w") as f:
        f.write(f"Skill scores vs Seasonal Naive on {n_configs} configs "
                f"(univariate, SN baseline computed)\n")
        f.write("Higher is better. SN = 0.\n\n")
        f.write(f"{'Arm':<25} {'GM-MASE':>10} {'MASE skill':>12} "
                f"{'GM-WQL':>10} {'WQL skill':>12}\n")
        f.write("-" * 75 + "\n")
        for _, r in table.iterrows():
            if r["arm"] == "Seasonal Naive":
                continue
            f.write(f"{r['arm']:<25} {r['gm_mase']:>10.3f} "
                    f"{r['mase_skill_pct']:>11.1f}% {r['gm_wql']:>10.4f} "
                    f"{r['wql_skill_pct']:>11.1f}%\n")
        f.write("\n")
        f.write(f"SN baseline (n={n_configs}): "
                f"GM-MASE = {sn_row['gm_mase']:.3f}  "
                f"GM-WQL = {sn_row['gm_wql']:.4f}\n")
    print(f"  wrote {txt_out}")
    print()
    print(open(txt_out).read())


if __name__ == "__main__":
    main()
