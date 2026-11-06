"""--forecast-batch-size N of the official GIFT-Eval script.

With N > 1 the windows of one config go through the backbone, the rollout
and the head N at a time (forecast_B4_windows). Each window must get the
forecast of the per-window path (forecast_B4, predict_item), up to
floating-point noise: the EWMA statistics, the normalised rollout feedback,
the latent sphere, the left padding and the unscaling are per window.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.forecasting_head import (  # noqa: E402
    LinearForecastingHead, LinearQuantileForecastingHead,
    TransformerQuantileForecastingHead, forecast_B4, forecast_B4_windows,
    windows_per_pass)
from src.freq_embedding import FREQ_NAMES_V2  # noqa: E402
from src.models import ConfigurableModel  # noqa: E402

EVAL_PY = (REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
           / "eval_gift_eval_official.py")
CPU = torch.device("cpu")
T_RAW, W, H = 256, 16, 16
HORIZON = 40                       # three rolled patches, the last one cut
TOL = dict(rtol=1e-4, atol=1e-3)   # the tolerance of the R batch test (#425)


def backbone(zero_pad=True, sphere=False):
    """A tiny EWMA backbone: zero padding and vocabulary v2 (the current
    runs), or no padding and vocabulary v1 (the runs before #419)."""
    torch.manual_seed(0)
    return ConfigurableModel(
        C=1, H=H, W=W, encoder_type="gru", num_layers=2, nhead=2,
        ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
        rev_norm_kind="ewma", rev_norm_span=128, num_encoder_layers=1,
        freq_emb_dim=3, seasonality_emb_dim=3,
        num_freqs=len(FREQ_NAMES_V2) if zero_pad else 10,
        rev_norm_skip_leading_zeros=zero_pad, latent_sphere=sphere).eval()


HEADS = {
    "transformer-q": lambda: TransformerQuantileForecastingHead(
        H=H, num_layers=1, nhead=2, forecast_len=W),
    "linear-q": lambda: LinearQuantileForecastingHead(H=H, forecast_len=W),
    "linear-point": lambda: LinearForecastingHead(H=H, forecast_len=W),
}


def head(kind):
    torch.manual_seed(1)
    return HEADS[kind]().eval()


def walk(n, seed, level=60.0, scale=1.0):
    rng = np.random.default_rng(seed)
    return (level + scale * rng.standard_normal(n).cumsum()).astype(np.float32)


def contexts():
    """``(5, T_RAW, 1)`` raw windows: a plain walk, another level and scale,
    left zero padding that ends inside a patch, a window of one real patch,
    and a window of padding only (an all-NaN series)."""
    rows = [walk(T_RAW, 0), walk(T_RAW, 1, level=1e3, scale=20.0),
            walk(T_RAW, 2), walk(T_RAW, 3), np.zeros(T_RAW, np.float32)]
    rows[2][:100] = 0.0
    rows[3][:T_RAW - W] = 0.0
    return torch.from_numpy(np.stack(rows))[:, :, None]


@pytest.mark.parametrize("kind", list(HEADS))
@pytest.mark.parametrize("sphere", [False, True])
def test_a_batch_of_windows_is_the_windows_one_by_one(kind, sphere):
    m, h = backbone(sphere=sphere), head(kind)
    batch = forecast_B4_windows(m, h, contexts(), HORIZON, CPU)
    for i, context in enumerate(contexts()):
        one = forecast_B4(m, h, context, HORIZON, CPU)
        assert batch[i].shape == one.shape and np.isfinite(one).all()
        np.testing.assert_allclose(batch[i], one, **TOL)


def test_one_window_is_the_per_window_forecast():
    m, h = backbone(), head("transformer-q")
    context = contexts()[:1]
    batch = forecast_B4_windows(m, h, context, HORIZON, CPU)
    assert batch.shape == (1, 9, HORIZON, 1)
    np.testing.assert_allclose(batch[0], forecast_B4(m, h, context[0],
                                                     HORIZON, CPU), **TOL)


def test_fewer_windows_per_pass_for_a_horizon_longer_than_the_context():
    # GIFT-Eval at the 1024 context: 64 context patches, and the longest
    # horizon (900 values) has 57 patches. Every config runs the full batch.
    assert windows_per_pass(64, 64, 57) == 64
    assert windows_per_pass(64, 64, 64) == 64
    # A horizon of three contexts: a pass holds half the windows, so it
    # holds as many latents as the full batch with a horizon of one context.
    assert windows_per_pass(8, 4, 12) == 4
    assert windows_per_pass(8, 4, 1000) == 1
    assert windows_per_pass(1, 4, 1000) == 1


# ---------------------------------------------------------------------------
# The predictor of the eval script
# ---------------------------------------------------------------------------

def eval_module(tag):
    pytest.importorskip("gift_eval")
    spec = importlib.util.spec_from_file_location(f"eval_batched_{tag}",
                                                  EVAL_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse(module, monkeypatch, *flags):
    monkeypatch.setattr(sys, "argv", ["eval", "--backbone-path", "bb.pth",
                                      "--head-path", "head.pth", *flags])
    return module.parse_args()


def items():
    """Seven gluonts items, the edge cases of a config: a series longer
    than the context, one shorter (left padding), missing values inside and
    at the end (forward fill), a missing start (back fill, or dropped under
    'zeros'), an all-NaN series, another level and scale, and a seventh so
    that the last batch of four is a partial one."""
    start = pd.Period("2020-01-01 00:00", freq="h")
    series = [walk(T_RAW + 100, 0), walk(100, 1), walk(T_RAW + 100, 2),
              walk(T_RAW + 50, 3), np.full(T_RAW, np.nan, np.float32),
              walk(T_RAW + 100, 4, level=1e3, scale=20.0), walk(T_RAW, 5)]
    series[2][50:80] = series[2][-3:] = np.nan
    series[3][:T_RAW - 40] = np.nan
    return [{"target": s, "start": start, "item_id": f"s{i}"}
            for i, s in enumerate(series)]


def predictors(module, m, h, batch_size, horizon=HORIZON, t_raw=T_RAW,
               context_pad="zeros"):
    kwargs = dict(prediction_length=horizon, device=CPU, t_raw=t_raw,
                  strategy="B4", context_pad=context_pad)
    return (module.ContrastiveForecasterPredictor(m, h, **kwargs),
            module.BatchedForecastPredictor(m, h, batch_size=batch_size,
                                            **kwargs))


def assert_same_forecasts(got, want):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert (g.start_date, g.item_id, g.forecast_keys) == (
            w.start_date, w.item_id, w.forecast_keys)
        assert np.isfinite(w.forecast_array).all()
        np.testing.assert_allclose(g.forecast_array, w.forecast_array, **TOL)


@pytest.mark.parametrize("kind", ["transformer-q", "linear-q"])
@pytest.mark.parametrize("context_pad", ["first", "zeros"])
def test_the_batched_predictor_gives_each_window_its_own_forecast(
        kind, context_pad):
    module = eval_module(f"items_{kind}_{context_pad}")
    one_by_one, batched = predictors(
        module, backbone(zero_pad=context_pad == "zeros"), head(kind),
        batch_size=4, context_pad=context_pad)
    windows = items()
    assert_same_forecasts(list(batched.predict(windows)),
                          [one_by_one.predict_item(it) for it in windows])


def test_a_config_of_one_window():
    module = eval_module("one")
    one_by_one, batched = predictors(module, backbone(), head("linear-q"),
                                     batch_size=4)
    (item,) = items()[1:2]
    assert_same_forecasts(list(batched.predict([item])),
                          [one_by_one.predict_item(item)])


def test_a_long_horizon_runs_smaller_passes(monkeypatch):
    """A context of 64 values (4 patches) and a horizon of 160 (10 patches):
    4 windows per pass in place of 8, and the same forecasts."""
    module = eval_module("long")
    seen, forecast = [], module.forecast_B4_windows

    def spy(backbone, head, contexts, horizon, device):
        seen.append(len(contexts))
        return forecast(backbone, head, contexts, horizon, device)
    monkeypatch.setattr(module, "forecast_B4_windows", spy)
    one_by_one, batched = predictors(module, backbone(), head("linear-q"),
                                     batch_size=8, horizon=160, t_raw=64)
    windows = items() + items()[:3]
    assert batched.batch_size == 4
    assert_same_forecasts(list(batched.predict(windows)),
                          [one_by_one.predict_item(it) for it in windows])
    assert seen == [4, 4, 2]


def test_the_flag_defaults_to_one_window_and_batches_b4_only(monkeypatch):
    module = eval_module("flag")
    assert parse(module, monkeypatch).forecast_batch_size == 1
    args = parse(module, monkeypatch, "--strategy", "B4",
                 "--forecast-batch-size", "64")
    assert args.forecast_batch_size == 64
    for flags in (["--strategy", "A2", "--forecast-batch-size", "64"],
                  ["--strategy", "B4", "--forecast-batch-size", "0"]):
        with pytest.raises(SystemExit):
            parse(module, monkeypatch, *flags)


def test_build_predictor_batches_above_one_window(monkeypatch):
    module = eval_module("build")
    m, h = backbone(), head("linear-q")
    dataset = SimpleNamespace(prediction_length=48,
                              test_data=SimpleNamespace(label=[]))
    shape = ["--strategy", "B4", "--t-raw", str(T_RAW),
             "--context-len", str(T_RAW)]
    args = parse(module, monkeypatch, *shape)
    plain = module.build_predictor(args, m, h, dataset, CPU)
    assert type(plain) is module.ContrastiveForecasterPredictor
    args = parse(module, monkeypatch, *shape, "--forecast-batch-size", "16")
    batched = module.build_predictor(args, m, h, dataset, CPU)
    assert isinstance(batched, module.BatchedForecastPredictor)
    assert batched.batch_size == 16 and batched.t_raw == T_RAW
    with pytest.raises(ValueError, match="B4 only"):
        module.BatchedForecastPredictor(m, h, 48, CPU, strategy="A2",
                                        batch_size=16)


def gluonts_test_data(h, windows=3, n=3):
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.split import split
    items = [{"target": walk(T_RAW + 200, i),
              "start": pd.Period("2020-01-01 00:00", freq="h"),
              "item_id": f"s{i}"} for i in range(n)]
    _, template = split(ListDataset(items, freq="h"), offset=-h * windows)
    return template.generate_instances(prediction_length=h, windows=windows)


def test_the_batched_predictor_scores_the_mase_of_the_per_window_path():
    """Nine windows in passes of four, through gluonts evaluate_model."""
    from gluonts.ev.metrics import MASE
    from gluonts.model import evaluate_model
    module = eval_module("mase")
    test_data = gluonts_test_data(h=24)

    def mase(predictor):
        res = evaluate_model(predictor, test_data=test_data, metrics=[MASE()],
                             axis=None, mask_invalid_label=True,
                             allow_nan_forecast=False, seasonality=24)
        return float(res["MASE[0.5]"][0])
    one_by_one, batched = predictors(module, backbone(), head("transformer-q"),
                                     batch_size=4, horizon=24)
    want, got = mase(one_by_one), mase(batched)
    assert np.isfinite(want) and want > 0
    assert abs(got - want) <= 1e-5 * want
