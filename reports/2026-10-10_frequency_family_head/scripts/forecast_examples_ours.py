"""Forecasts of a few GIFT-Eval test windows by one of our models, for the plots
of plot_forecast_examples.py (the user, 11-03: our forecasts next to Moirai's).
It runs the model loading and the predictor of eval_gift_eval_official.py with
the flags of the forecast score (the median of the B4 head); nothing of the
eval changes. Each window is CONFIG:INDEX, INDEX in the test windows of the
config (gift_eval test_data.input).

Run from the eval scripts folder, with PYTHONPATH at the code checkout:
  python3 forecast_examples_ours.py OUT.npz m4_hourly/H/short:0,150 ... -- <eval flags>
"""
import sys

import numpy as np
import torch


def parse_windows(specs):
    """{config: [index, ...]} from CONFIG:INDEX[,INDEX...]."""
    windows = {}
    for spec in specs:
        config, indices = spec.rsplit(":", 1)
        windows[config] = [int(i) for i in indices.split(",")]
    return windows


def config_dataset(ev, config):
    """The gift_eval dataset of a config name, as the eval builds it."""
    for ds_name, term in ev.get_all_dataset_configs():
        if ev.get_ds_config_name(ds_name, term)[0] == config:
            multivariate = ev.GiftDataset(name=ds_name, term=term, to_univariate=False).target_dim > 1
            return ev.GiftDataset(name=ds_name, term=term, to_univariate=multivariate)
    raise SystemExit(f"no GIFT-Eval config {config}")


def predictor_for(ev, args, backbone, head, dataset, device):
    """The eval's predictor, with the frequency and seasonality ids of the config."""
    from src.freq_embedding import freq_to_id, seasonality_to_id
    backbone._eval_freq_id = freq_to_id(dataset.freq, backbone._freq_vocab)
    backbone._eval_seasonality_id = seasonality_to_id(ev.get_seasonality(dataset.freq))
    return ev.ContrastiveForecasterPredictor(
        backbone=backbone, head=head, prediction_length=dataset.prediction_length, device=device,
        quantile_levels=ev.QUANTILE_LEVELS, strategy=args.strategy, context_pad=args.context_pad)


def main():
    split = sys.argv.index("--")
    out, windows = sys.argv[1], parse_windows(sys.argv[2:split])
    sys.argv = [sys.argv[0]] + sys.argv[split + 1:]
    import eval_gift_eval_official as ev
    args = ev.parse_args()
    device = torch.device(args.device)
    backbone, head = ev.load_models(args, device)
    saved = {}
    for config, indices in windows.items():
        dataset = config_dataset(ev, config)
        predictor = predictor_for(ev, args, backbone, head, dataset, device)
        inputs, labels = list(dataset.test_data.input), list(dataset.test_data.label)
        for i in indices:
            forecast = next(iter(predictor.predict([inputs[i]])))
            key = f"{config}|{i}"
            saved[f"{key}|context"] = np.asarray(inputs[i]["target"], dtype=float)
            saved[f"{key}|truth"] = np.asarray(labels[i]["target"], dtype=float)
            saved[f"{key}|median"] = np.asarray(forecast.quantile(0.5), dtype=float)
            print(f"{key}: context {len(saved[f'{key}|context'])}, horizon {len(saved[f'{key}|truth'])}")
    np.savez(out, **saved)


if __name__ == "__main__":
    main()
