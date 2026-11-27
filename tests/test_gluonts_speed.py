"""src.gluonts_speed: two exact speedups of the gluonts evaluation (#425).

The eval divides the MASE of each window by the seasonal error, so the slice
version must give the same float as gluonts, bit for bit: on series with
missing values, in 1 and 2 dimensions, and when the seasonality is longer
than the series. Each kept quantile parse must equal a new parse. The last
test runs the gluonts evaluation of a small test set before and after
use_fast_gluonts and compares the metrics.
"""
import sys
from pathlib import Path

import gluonts.model.evaluation as evaluation
import numpy as np
import pandas as pd
import pytest
from gluonts.dataset.common import ListDataset
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MSIS, MeanWeightedSumQuantileLoss
from gluonts.ev.ts_stats import seasonal_error as gluonts_seasonal_error
from gluonts.model.forecast import Quantile, QuantileForecast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gluonts_speed import (  # noqa: E402
    parse_quantile, quantile_from_str, seasonal_error, use_fast_gluonts)

QUANTILES = [0.1, 0.5, 0.9]
GLUONTS_PARSE = Quantile.parse
GLUONTS_FROM_STR = Quantile.from_str


@pytest.fixture
def restore_gluonts(monkeypatch):
    """The functions of gluonts come back at the end of the test."""
    monkeypatch.setattr(evaluation, "seasonal_error", evaluation.seasonal_error)
    for name in ("parse", "from_str"):
        monkeypatch.setattr(Quantile, name, Quantile.__dict__[name])


def series_with_gaps(shape, seed):
    """float32 values of ``shape`` with about 3% of them missing (NaN)."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=shape).cumsum(axis=-1).astype(np.float32)
    values[rng.random(shape) < 0.03] = np.nan
    return values


def assert_same(expected, actual):
    """Same values, same mask and same dtype, bit for bit."""
    assert type(expected) is type(actual)
    assert np.asarray(expected).dtype == np.asarray(actual).dtype
    assert np.array_equal(np.ma.getdata(expected), np.ma.getdata(actual),
                          equal_nan=True)
    assert np.array_equal(np.ma.getmaskarray(expected),
                          np.ma.getmaskarray(actual))


@pytest.mark.parametrize("seasonality", [1, 24, 288, 5000])
@pytest.mark.parametrize("masked", [True, False])
def test_seasonal_error_of_one_series(seasonality, masked):
    values = series_with_gaps((4000,), seed=seasonality)
    if masked:
        values = np.ma.masked_invalid(values)
    assert_same(gluonts_seasonal_error(values, seasonality, time_axis=-1),
                seasonal_error(values, seasonality, time_axis=-1))


@pytest.mark.parametrize("time_axis", [0, 1, -1])
def test_seasonal_error_in_two_dimensions(time_axis):
    values = np.ma.masked_invalid(series_with_gaps((3, 700), seed=7))
    assert_same(gluonts_seasonal_error(values, 12, time_axis=time_axis),
                seasonal_error(values, 12, time_axis=time_axis))


@pytest.mark.parametrize("length", [1, 2])
def test_seasonal_error_of_short_series(length):
    values = np.ma.masked_invalid(np.arange(length, dtype=np.float32))
    assert_same(gluonts_seasonal_error(values, 1, time_axis=-1),
                seasonal_error(values, 1, time_axis=-1))


@pytest.mark.parametrize("quantile",
                         [0.1, 0.5, "0.5", "0.50", "p90", 0.025, 0.0, -0.0])
def test_parse_quantile(quantile):
    for _ in range(2):  # a new parse, then the kept one
        assert parse_quantile(Quantile, quantile) == GLUONTS_PARSE(quantile)
    if isinstance(quantile, str):
        assert quantile_from_str(Quantile, quantile) == GLUONTS_FROM_STR(quantile)


def test_parse_quantile_keeps_the_sign_of_zero():
    assert parse_quantile(Quantile, 0.0).name == "0.0"
    assert parse_quantile(Quantile, -0.0).name == "-0.0"


def test_parse_quantile_errors_and_objects():
    with pytest.raises(ValueError):
        parse_quantile(Quantile, "median")
    quantile = Quantile.from_float(0.3)
    assert parse_quantile(Quantile, quantile) is quantile


def test_use_fast_gluonts(restore_gluonts):
    use_fast_gluonts()
    assert evaluation.seasonal_error is seasonal_error
    assert Quantile.parse("0.5") == GLUONTS_PARSE("0.5")
    assert Quantile.from_str("p10") == GLUONTS_FROM_STR("p10")


def evaluate_small_test_set():
    """The metrics of fixed forecasts of 3 hourly series with gaps."""
    dataset = ListDataset(
        [{"start": pd.Period("2024-01-01 00:00", freq="h"),
          "target": series_with_gaps((600,), seed=i)} for i in range(3)],
        freq="h")
    _, template = split(dataset, offset=-48)
    test_data = template.generate_instances(prediction_length=24, windows=2)
    rng = np.random.default_rng(0)
    forecasts = [
        QuantileForecast(
            forecast_arrays=np.sort(rng.normal(size=(3, 24)), axis=0),
            forecast_keys=[str(q) for q in QUANTILES],
            start_date=entry["start"] + len(entry["target"]))
        for entry in test_data.input]
    return evaluation.evaluate_forecasts_raw(
        forecasts, test_data=test_data, batch_size=4, seasonality=24,
        metrics=[MASE(), MSIS(), MeanWeightedSumQuantileLoss(QUANTILES)])


def test_evaluation_metrics_unchanged(restore_gluonts):
    expected = evaluate_small_test_set()
    use_fast_gluonts()
    actual = evaluate_small_test_set()
    assert expected.keys() == actual.keys()
    for name in expected:
        assert_same(expected[name], actual[name])
