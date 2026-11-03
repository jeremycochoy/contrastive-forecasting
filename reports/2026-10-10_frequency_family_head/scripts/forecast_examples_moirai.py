"""Forecasts of a few GIFT-Eval test windows by Moirai small 1.1-R, for the plots
of plot_forecast_examples.py (the user, 11-03): the settings of the GIFT-Eval
Moirai notebook (context 4000, patch size 32, 100 samples; the median of the
samples). Each window is CONFIG:INDEX, INDEX in the test windows of the config.

It needs uni2ts (its source folder on the path) and gift_eval:
  PYTHONPATH=<uni2ts>/src python3 forecast_examples_moirai.py OUT.npz m4_hourly/H/short:0,150 ...
"""
import sys

import numpy as np

from forecast_examples_ours import parse_windows

MODEL = "Salesforce/moirai-1.1-R-small"


def config_dataset(config):
    """The gift_eval dataset of a config name (dataset/freq/term or dataset/term)."""
    from gift_eval.data import Dataset
    parts = config.split("/")
    name, term = "/".join(parts[:-1]), parts[-1]
    for candidate in (name, parts[0]):
        try:
            multivariate = Dataset(name=candidate, term=term, to_univariate=False).target_dim > 1
            return Dataset(name=candidate, term=term, to_univariate=multivariate)
        except Exception:  # m4_hourly/H is the dataset m4_hourly
            continue
    raise SystemExit(f"no GIFT-Eval dataset for {config}")


def moirai_predictor(module, prediction_length):
    from uni2ts.model.moirai import MoiraiForecast
    model = MoiraiForecast(module=module, prediction_length=prediction_length, context_length=4000,
                           patch_size=32, num_samples=100, target_dim=1,
                           feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    return model.create_predictor(batch_size=8)


def main():
    from uni2ts.model.moirai import MoiraiModule
    out, windows = sys.argv[1], parse_windows(sys.argv[2:])
    module = MoiraiModule.from_pretrained(MODEL)
    saved = {}
    for config, indices in windows.items():
        dataset = config_dataset(config)
        predictor = moirai_predictor(module, dataset.prediction_length)
        inputs = list(dataset.test_data.input)
        for i in indices:
            forecast = next(iter(predictor.predict([inputs[i]])))
            saved[f"{config}|{i}|median"] = np.asarray(forecast.quantile(0.5), dtype=float)
            print(f"{config}|{i}: horizon {len(saved[f'{config}|{i}|median'])}")
    np.savez(out, **saved)


if __name__ == "__main__":
    main()
