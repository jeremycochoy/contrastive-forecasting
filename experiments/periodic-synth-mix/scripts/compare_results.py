#!/usr/bin/env python3
"""Compare CONTROL vs MIX GIFT-Eval results for the periodic-synth-mix experiment.

Reads ``results/R1v3c_ctrl/all_results.csv`` and ``results/R1v3c_mix/all_results.csv``
(and the existing ``results/R1v3b/summary.txt`` for a third reference row),
and produces:

- ``experiments/periodic-synth-mix/results/comparison.txt`` — aggregate GM-Rel
  MASE + per-dataset MASE / seasonal-naive MASE / relative, for both arms plus
  the reference v3b-120k run, plus the ``Δ MIX - CTRL`` column.
- ``experiments/periodic-synth-mix/results/periodic_datasets.txt`` — same
  tables restricted to the 6 "periodic" configs identified in HANDOFF.md.
- ``experiments/periodic-synth-mix/plots/mase_compare.png`` — bar chart of the
  periodic datasets across all three arms.

Usage:
    python experiments/periodic-synth-mix/scripts/compare_results.py \\
        --ctrl-dir results/R1v3c_ctrl \\
        --mix-dir  results/R1v3c_mix
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd


# 6 periodic datasets we are specifically targeting (from HANDOFF.md).
PERIODIC_DATASETS = [
    "m4_hourly/H/short",
    "solar/10T/medium",
    "solar/10T/long",
    "solar/H/short",
    "ett2/W/short",
    "ett1/15T/short",
]


def _read_all_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Official format stores point MASE as 'eval_metrics/MASE[0.5]'.
    mase_col = "eval_metrics/MASE[0.5]"
    if mase_col not in df.columns:
        raise KeyError(f"MASE column '{mase_col}' missing in {csv_path}; "
                       f"columns = {df.columns.tolist()}")
    # Keep a clean frame: dataset / MASE / (domain if present)
    keep = {"dataset": "dataset", mase_col: "mase"}
    if "domain" in df.columns:
        keep["domain"] = "domain"
    out = df[list(keep.keys())].rename(columns=keep)
    return out


def _read_summary_seasonal(summary_path: str) -> pd.DataFrame:
    """Parse the SN_MASE / Relative table from a summary.txt."""
    if not os.path.exists(summary_path):
        return pd.DataFrame(columns=["dataset", "sn_mase", "relative"])
    rows = []
    with open(summary_path) as f:
        for line in f:
            # Rows look like:   "dataset_name/freq/len        MASE   SN_MASE   Relative"
            parts = line.split()
            if len(parts) >= 4:
                try:
                    rel = float(parts[-1])
                    sn = float(parts[-2])
                    mase = float(parts[-3])
                    ds = " ".join(parts[:-3])
                    # Heuristic: first field is our dataset slug if it contains
                    # a slash ("dataset/freq/len").
                    if "/" in ds:
                        rows.append((ds, mase, sn, rel))
                except ValueError:
                    continue
    df = pd.DataFrame(rows, columns=["dataset", "mase", "sn_mase", "relative"])
    return df


def _gm(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    if len(a) == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(a))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl-dir", required=True,
                    help="Dir containing R1v3c_ctrl all_results.csv")
    ap.add_argument("--mix-dir", required=True,
                    help="Dir containing R1v3c_mix all_results.csv")
    ap.add_argument("--reference-summary",
                    default=os.path.abspath(
                        os.path.join(os.path.dirname(__file__),
                                     "..", "..", "..",
                                     "results", "R1v3b", "summary.txt")),
                    help="Path to R1v3b summary.txt used as reference row")
    ap.add_argument("--out-dir",
                    default=os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "..", "results")),
                    help="Directory for output artefacts")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -- Load three arms -------------------------------------------------------
    ctrl = _read_all_results(os.path.join(args.ctrl_dir, "all_results.csv"))
    mix = _read_all_results(os.path.join(args.mix_dir, "all_results.csv"))
    ctrl = ctrl.rename(columns={"mase": "mase_ctrl"})
    mix = mix.rename(columns={"mase": "mase_mix"})

    # Reference R1v3b (includes SN_MASE column we need for Relative)
    ref = _read_summary_seasonal(args.reference_summary)
    if not len(ref):
        print(f"WARNING: no reference rows parsed from {args.reference_summary}")
    ref = ref.rename(columns={"mase": "mase_v3b"})

    # Merge
    df = ctrl.merge(mix, on="dataset", how="outer", suffixes=("", ""))
    df = df.merge(ref[["dataset", "mase_v3b", "sn_mase"]], on="dataset",
                  how="left", suffixes=("", ""))

    # Relative MASE per arm (vs SN)
    df["rel_ctrl"] = df["mase_ctrl"] / df["sn_mase"]
    df["rel_mix"] = df["mase_mix"] / df["sn_mase"]
    df["rel_v3b"] = df["mase_v3b"] / df["sn_mase"]
    df["mix_minus_ctrl"] = df["mase_mix"] - df["mase_ctrl"]

    # -- Aggregate (GM of Relative MASE) --------------------------------------
    agg = {
        "GM-Rel MASE CTRL": _gm(df["rel_ctrl"].values),
        "GM-Rel MASE MIX":  _gm(df["rel_mix"].values),
        "GM-Rel MASE v3b (ref)": _gm(df["rel_v3b"].values),
    }

    # -- Write full comparison -------------------------------------------------
    cmp_path = out_dir / "comparison.txt"
    with open(cmp_path, "w") as f:
        f.write("=" * 96 + "\n")
        f.write("Periodic-synth-mix — GIFT-Eval comparison (CONTROL vs MIX vs v3b-120k reference)\n")
        f.write("=" * 96 + "\n\n")
        f.write(f"{'Dataset':<40s} {'CTRL':>8s} {'MIX':>8s} {'v3b':>8s} "
                f"{'SN':>8s} {'rel_C':>7s} {'rel_M':>7s} {'rel_v':>7s} {'Δ M-C':>8s}\n")
        f.write("-" * 96 + "\n")
        for _, r in df.sort_values("dataset").iterrows():
            def _f(v, w=8, prec=3):
                return f"{v:>{w}.{prec}f}" if pd.notna(v) else " " * w
            f.write(
                f"{str(r['dataset'])[:40]:<40s} "
                f"{_f(r['mase_ctrl'])} {_f(r['mase_mix'])} "
                f"{_f(r['mase_v3b'])} {_f(r['sn_mase'])} "
                f"{_f(r['rel_ctrl'], 7)} {_f(r['rel_mix'], 7)} "
                f"{_f(r['rel_v3b'], 7)} {_f(r['mix_minus_ctrl'])}\n"
            )
        f.write("-" * 96 + "\n\n")
        for k, v in agg.items():
            f.write(f"  {k:<30s} {v:.4f}\n")
        f.write("\n")
        f.write("Interpretation:\n")
        f.write("  Δ M-C  < 0  means MIX beats CONTROL at that config.\n")
        f.write("  rel    < 1  means we beat seasonal-naive.\n")
        f.write(f"  GM-Rel MIX improvement vs CTRL: "
                f"{100 * (agg['GM-Rel MASE CTRL'] - agg['GM-Rel MASE MIX']) / agg['GM-Rel MASE CTRL']:+.2f}%\n")
    print(f"wrote {cmp_path}")
    for k, v in agg.items():
        print(f"  {k:<30s} {v:.4f}")

    # -- Periodic-only subset --------------------------------------------------
    per = df[df["dataset"].isin(PERIODIC_DATASETS)].copy()
    per_path = out_dir / "periodic_datasets.txt"
    with open(per_path, "w") as f:
        f.write("Periodic-dataset focus (HANDOFF.md worst configs)\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Dataset':<24s} {'CTRL':>8s} {'MIX':>8s} {'v3b':>8s} "
                f"{'SN':>8s} {'rel_C':>7s} {'rel_M':>7s} {'rel_v':>7s}\n")
        f.write("-" * 90 + "\n")
        for _, r in per.iterrows():
            def _f(v, w=8, prec=3):
                return f"{v:>{w}.{prec}f}" if pd.notna(v) else " " * w
            f.write(
                f"{str(r['dataset']):<24s} "
                f"{_f(r['mase_ctrl'])} {_f(r['mase_mix'])} "
                f"{_f(r['mase_v3b'])} {_f(r['sn_mase'])} "
                f"{_f(r['rel_ctrl'], 7)} {_f(r['rel_mix'], 7)} "
                f"{_f(r['rel_v3b'], 7)}\n"
            )
        f.write("-" * 90 + "\n")
        if len(per):
            f.write(f"\nGM-Rel over periodic subset only:\n")
            f.write(f"  CTRL: {_gm(per['rel_ctrl'].values):.4f}\n")
            f.write(f"  MIX:  {_gm(per['rel_mix'].values):.4f}\n")
            f.write(f"  v3b:  {_gm(per['rel_v3b'].values):.4f}\n")
    print(f"wrote {per_path}")

    # -- Bar chart -------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(per):
            x = np.arange(len(per))
            width = 0.25
            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
            ax.bar(x - width, per["mase_ctrl"].values, width, label="CONTROL 30k")
            ax.bar(x,         per["mase_mix"].values, width, label="MIX 30k")
            ax.bar(x + width, per["mase_v3b"].values, width, label="v3b 120k (ref)")
            ax.plot(x, per["sn_mase"].values, "k.-", label="Seasonal-naive")
            ax.set_xticks(x)
            ax.set_xticklabels(per["dataset"].tolist(), rotation=30, ha="right")
            ax.set_ylabel("MASE (lower is better)")
            ax.set_title("Periodic datasets — CONTROL vs MIX vs v3b")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            p_out = plots_dir / "mase_compare.png"
            plt.savefig(p_out, bbox_inches="tight")
            plt.close()
            print(f"wrote {p_out}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
