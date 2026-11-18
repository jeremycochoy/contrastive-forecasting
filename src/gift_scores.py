"""The GM-Relative MASE of a GIFT-Eval per-config table (#462).

A per-config table is the ``all_results.csv`` of one eval: one row for each
config, with the config name ``<dataset>/<freq>/<term>`` in the column
``dataset``. The GM-Relative MASE is the geometric mean, over the configs, of
the MASE[0.5] of the model divided by the MASE[0.5] of Seasonal Naive. Over
the 97 configs, it is the aggregate of the eval's ``summary.txt``.
"""

import csv
import math

MASE_COLUMN = "eval_metrics/MASE[0.5]"
HORIZONS = ("short", "medium", "long")


def read_mase(path) -> dict:
    """``{config: MASE[0.5]}`` of one per-config table."""
    with open(path) as f:
        return {row["dataset"]: float(row[MASE_COLUMN])
                for row in csv.DictReader(f)}


def gm_relative_mase(mase: dict, naive: dict, horizon: str | None = None) -> float:
    """The geometric mean of ``mase[c] / naive[c]`` over the configs ``c`` of
    ``mase``, or over the configs of one horizon (``short``, ``medium`` or
    ``long``, the last part of the config name). ``naive`` holds the MASE of
    Seasonal Naive. ValueError when a config has no Seasonal Naive value, or
    when the horizon has no config."""
    configs = [c for c in mase if horizon is None or c.endswith("/" + horizon)]
    if not configs:
        raise ValueError(f"no config of the horizon {horizon!r}")
    missing = [c for c in configs if c not in naive]
    if missing:
        raise ValueError(f"no Seasonal Naive MASE for {missing}")
    logs = [math.log(mase[c] / naive[c]) for c in configs]
    return math.exp(sum(logs) / len(logs))
