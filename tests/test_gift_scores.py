"""src/gift_scores.py: the GM-Relative MASE of a GIFT-Eval per-config table,
over every config or over the configs of one horizon (#462)."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.gift_scores import gm_relative_mase, read_mase  # noqa: E402

COLUMNS = ["dataset", "model", "eval_metrics/MSE[mean]",
           "eval_metrics/MASE[0.5]", "domain", "num_variates"]


def table(path, mase):
    """A per-config table in the layout of all_results.csv."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for config, value in mase.items():
            writer.writerow([config, "m", 9.0, value, "Energy", 1])
    return path


NAIVE = {"a/H/short": 2.0, "b/D/short": 1.0, "c/H/long": 4.0,
         "c/H/medium": 1.0}
MODEL = {"a/H/short": 1.0, "b/D/short": 2.0, "c/H/long": 2.0,
         "c/H/medium": 1.0}


def test_read_mase_reads_the_mase_column_by_config(tmp_path):
    assert read_mase(table(tmp_path / "t.csv", MODEL)) == MODEL


def test_the_gm_over_every_config():
    # The ratios 0.5, 2, 0.5 and 1.
    assert gm_relative_mase(MODEL, NAIVE) == pytest.approx(0.5 ** 0.25)


@pytest.mark.parametrize("horizon,expected", [
    ("short", 1.0), ("medium", 1.0), ("long", 0.5)])
def test_the_gm_over_one_horizon(horizon, expected):
    assert gm_relative_mase(MODEL, NAIVE, horizon) == pytest.approx(expected)


def test_the_gm_is_the_geometric_mean_of_the_ratios():
    mase = {f"d{i}/H/short": 1.0 + i for i in range(5)}
    naive = {c: 2.0 for c in mase}
    logs = [math.log(v / 2.0) for v in mase.values()]
    assert gm_relative_mase(mase, naive) == pytest.approx(
        math.exp(sum(logs) / len(logs)))


def test_a_config_with_no_seasonal_naive_value_is_an_error():
    with pytest.raises(ValueError, match="x/H/short"):
        gm_relative_mase({"x/H/short": 1.0}, NAIVE)


def test_a_horizon_with_no_config_is_an_error():
    with pytest.raises(ValueError, match="long"):
        gm_relative_mase({"a/H/short": 1.0}, NAIVE, "long")
