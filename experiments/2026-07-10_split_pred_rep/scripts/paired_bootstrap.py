"""Paired-bootstrap CI on GM-Rel MASE ratio between two GIFT-Eval runs.

Both inputs are the ``all_results.csv`` files produced by
``experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py``.
For each of the tasks present in both files, computes
``rel_MASE = MASE / SN_MASE`` (Seasonal-Naive reference), then bootstraps
the ratio ``GM_rel(a) / GM_rel(b)`` by resampling tasks with
replacement, ``n_boot`` times. Emits point estimates + a 95% CI.

Usage:

  python3 paired_bootstrap.py --a <split arm all_results.csv> \\
                              --b <arm C reference all_results.csv> \\
                              --sn ~/workspaces/gift-eval/results/seasonal_naive/all_results.csv \\
                              [--n-boot 10000] [--seed 42]

Ratio < 1.0 → arm A beats arm B on GM-Rel MASE; > 1.0 → reverse.
"""

import argparse
import numpy as np
import pandas as pd

MASE = "eval_metrics/MASE[0.5]"


def _gm(values):
    arr = pd.to_numeric(values, errors="coerce").dropna()
    arr = arr[arr > 0]
    return float(np.exp(np.log(arr).mean())) if len(arr) else float("nan")


def _load_rel_mase(all_results_csv, sn_df):
    df = pd.read_csv(all_results_csv).set_index("dataset")
    common = df.index.intersection(sn_df.index)
    return (df.loc[common, MASE] / sn_df.loc[common, MASE]).dropna()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="all_results.csv for arm A (numerator)")
    p.add_argument("--b", required=True, help="all_results.csv for arm B (denominator)")
    p.add_argument("--sn", required=True, help="Seasonal-Naive all_results.csv")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sn = pd.read_csv(args.sn).set_index("dataset")
    ra = _load_rel_mase(args.a, sn)
    rb = _load_rel_mase(args.b, sn)
    tasks = ra.index.intersection(rb.index)
    ra, rb = ra.loc[tasks], rb.loc[tasks]
    n = len(tasks)

    gm_a, gm_b = _gm(ra), _gm(rb)
    ratio = gm_a / gm_b
    diff = gm_a - gm_b

    rng = np.random.default_rng(args.seed)
    boot_ratios, boot_diffs = np.empty(args.n_boot), np.empty(args.n_boot)
    log_a, log_b = np.log(ra.values), np.log(rb.values)
    for i in range(args.n_boot):
        idx = rng.integers(0, n, n)
        ga = np.exp(log_a[idx].mean())
        gb = np.exp(log_b[idx].mean())
        boot_ratios[i] = ga / gb
        boot_diffs[i]  = ga - gb

    lo_r, hi_r = np.quantile(boot_ratios, [0.025, 0.975])
    lo_d, hi_d = np.quantile(boot_diffs,  [0.025, 0.975])
    p_a_beats_b = float((boot_ratios < 1.0).mean())

    print(f"n_tasks={n}")
    print(f"GM-Rel MASE  A={gm_a:.4f}  B={gm_b:.4f}")
    print(f"ratio A/B    {ratio:.4f}   95% CI [{lo_r:.4f}, {hi_r:.4f}]")
    print(f"diff  A-B    {diff:+.4f}   95% CI [{lo_d:+.4f}, {hi_d:+.4f}]")
    print(f"P(GM(A) < GM(B))  {p_a_beats_b:.4f}   ({args.n_boot} bootstraps)")


if __name__ == "__main__":
    main()
