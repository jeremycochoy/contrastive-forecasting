"""Compute headline aggregates for one all_results.csv cell.

Emits a single CSV line: ``gm_rel_mase,gm_mase,gm_mape_sn,gm_crps_sn,n``.

Aggregation matches the rest of the project (geometric mean over the 97
GIFT-Eval configs):
- GM-Rel MASE: geomean of ``MASE / SN_MASE`` per task (the same number
  ``Aggregate GM-Relative MASE`` printed in summary.txt).
- GM-MASE:     geomean of raw ``MASE``.
- GM-MAPE_SN:  geomean of ``MAPE / SN_MAPE``.
- GM-CRPS_SN:  geomean of ``mean_weighted_sum_quantile_loss /
  SN(mean_weighted_sum_quantile_loss)``.

Seasonal-Naive reference is read from
``~/workspaces/gift-eval/results/seasonal_naive/all_results.csv`` (same
path used by ``experiments/2026-04-13_gift-eval/scripts/
eval_gift_eval_official.py``).
"""

import os
import sys

import numpy as np
import pandas as pd

MASE = "eval_metrics/MASE[0.5]"
MAPE = "eval_metrics/MAPE[0.5]"
WQL = "eval_metrics/mean_weighted_sum_quantile_loss"

SN_CANDIDATES = [
    os.path.expanduser(
        "~/workspaces/gift-eval/results/seasonal_naive/all_results.csv"),
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "gift-eval-ref", "seasonal_naive", "all_results.csv"),
]


def _gm(values):
    arr = pd.to_numeric(values, errors="coerce").dropna()
    arr = arr[arr > 0]
    if len(arr) == 0:
        return float("nan")
    return float(np.exp(np.log(arr).mean()))


def _load_sn():
    for cand in SN_CANDIDATES:
        if os.path.exists(cand):
            return pd.read_csv(cand).set_index("dataset")
    raise FileNotFoundError(
        f"Seasonal-Naive reference not found in: {SN_CANDIDATES}")


def compute(cell_csv):
    df = pd.read_csv(cell_csv).set_index("dataset")
    sn = _load_sn()
    common = df.index.intersection(sn.index)
    df = df.loc[common]
    sn = sn.loc[common]

    rel_mase = df[MASE] / sn[MASE]
    rel_mape = df[MAPE] / sn[MAPE]
    rel_wql = df[WQL] / sn[WQL]

    return {
        "gm_rel_mase": _gm(rel_mase),
        "gm_mase": _gm(df[MASE]),
        "gm_mape_sn": _gm(rel_mape),
        "gm_crps_sn": _gm(rel_wql),
        "n": int(len(common)),
    }


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: _compute_gm.py <all_results.csv>")
    r = compute(sys.argv[1])
    print(f"{r['gm_rel_mase']:.4f},{r['gm_mase']:.4f},"
          f"{r['gm_mape_sn']:.4f},{r['gm_crps_sn']:.4f},{r['n']}")


if __name__ == "__main__":
    main()
