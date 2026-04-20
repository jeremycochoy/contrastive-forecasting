#!/usr/bin/env python3
"""Compare predictions from two backbone+head pairs on the same plots.

Each subplot shows: context (gray) + ground truth (orange) +
prediction A (blue) + prediction B (red).

Usage:
    GIFT_EVAL=~/gift-eval-data PYTHONPATH=. python plot_predictions_compare.py \
        --backbone-a checkpoints/tiny_v2_best_gap.pth \
        --head-a checkpoints/R1_forecaster_recon_w16_best.pth \
        --label-a "R1 (GRU v2)" \
        --backbone-b checkpoints/tiny_v3_best_gap.pth \
        --head-b checkpoints/R1v3_best.pth \
        --label-b "R1v3 (GRU v3)" \
        --encoder-a gru --encoder-b gru \
        --device cuda
"""

import os
import sys
import warnings
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from src.models import ConfigurableModel
from src.forecasting_head import ForecastingHead, forecast_with_strategy
from gift_eval.data import Dataset as GiftDataset

BACKBONE_CONFIG = dict(
    C=4, H=512, W=16, encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
    rev_norm_span=32,
)
T_RAW = 1024

WORST_CONFIGS = [
    ("covid_deaths", "short", "covid_deaths D short"),
    ("bizitobs_application", "long", "bizitobs_app 10S long"),
    ("bizitobs_application", "medium", "bizitobs_app 10S medium"),
    ("bitbrains_rnd/H", "short", "bitbrains_rnd H short"),
    ("bizitobs_application", "short", "bizitobs_app 10S short"),
    ("bitbrains_rnd/5T", "medium", "bitbrains_rnd 5T medium"),
    ("m4_yearly", "short", "m4_yearly A short"),
    ("bitbrains_rnd/5T", "long", "bitbrains_rnd 5T long"),
    ("bizitobs_service", "long", "bizitobs_service 10S long"),
    ("bizitobs_service", "medium", "bizitobs_service 10S medium"),
    ("saugeenday/D", "short", "saugeen D short"),
    ("solar/10T", "medium", "solar 10T medium"),
]

BEST_CONFIGS = [
    ("bizitobs_l2c/5T", "short", "bizitobs_l2c 5T short"),
    ("jena_weather/10T", "short", "jena_weather 10T short"),
    ("sz_taxi/H", "short", "sz_taxi H short"),
    ("sz_taxi/15T", "medium", "sz_taxi 15T medium"),
    ("sz_taxi/15T", "long", "sz_taxi 15T long"),
    ("restaurant", "short", "restaurant D short"),
    ("us_births/D", "short", "us_births D short"),
    ("loop_seattle/5T", "short", "loop_seattle 5T short"),
    ("us_births/M", "short", "us_births M short"),
    ("bizitobs_l2c/5T", "medium", "bizitobs_l2c 5T medium"),
    ("m_dense/D", "short", "m_dense D short"),
    ("saugeenday/M", "short", "saugeen M short"),
]


def load_model(backbone_path, head_path, encoder_type, forecast_len, device):
    config = dict(BACKBONE_CONFIG)
    config['encoder_type'] = encoder_type
    backbone = ConfigurableModel(**config)
    backbone.load_state_dict(torch.load(backbone_path, map_location=device, weights_only=True))
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    head = ForecastingHead(H=512, hidden_dim=128, num_gru_layers=2,
                           forecast_len=forecast_len, dropout=0.1)
    head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))
    head = head.to(device).eval()
    return backbone, head


def predict_item(backbone, head, target, prediction_length, device, strategy='B4'):
    target = np.asarray(target, dtype=np.float32)
    if np.isnan(target).any():
        target = target.copy()
        mask = np.isnan(target)
        if mask.all():
            target[:] = 0.0
        else:
            first_valid = np.where(~mask)[0][0]
            target[:first_valid] = target[first_valid]
            for i in range(1, len(target)):
                if np.isnan(target[i]):
                    target[i] = target[i - 1]

    n = len(target)
    if n >= T_RAW:
        context = target[-T_RAW:]
    else:
        context = np.concatenate([np.full(T_RAW - n, target[0], dtype=np.float32), target])

    # (1, T_RAW, 1) — single channel
    context_t = torch.from_numpy(context).float().unsqueeze(0).unsqueeze(-1).to(device)

    with torch.no_grad():
        forecast = forecast_with_strategy(
            strategy, backbone, head, context_t,
            horizon=prediction_length, device=device)

    return forecast[:, 0]


def plot_comparison(configs, model_a, model_b, label_a, label_b,
                    device, output_path, title_prefix, strategy='B4'):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(title_prefix, fontsize=16, fontweight='bold')

    backbone_a, head_a = model_a
    backbone_b, head_b = model_b

    for idx, (ds_name, term, label) in enumerate(configs[:6]):
        ax = axes[idx // 2, idx % 2]
        try:
            to_univ_check = GiftDataset(name=ds_name, term=term, to_univariate=False)
            to_univariate = to_univ_check.target_dim > 1
            dataset = GiftDataset(name=ds_name, term=term, to_univariate=to_univariate)
            pred_len = dataset.prediction_length

            test_items = list(dataset.test_data.input)
            item = test_items[0]
            target_context = np.asarray(item["target"], dtype=np.float32)

            label_items = list(dataset.test_data.label)
            label_item = label_items[0]
            target_full = np.asarray(label_item["target"], dtype=np.float32)
            ground_truth = target_full[-pred_len:]

            # Predict with both models
            forecast_a = predict_item(backbone_a, head_a, target_context, pred_len, device, strategy)
            forecast_b = predict_item(backbone_b, head_b, target_context, pred_len, device, strategy)

            # Plot
            ctx_len = min(len(target_context), pred_len * 3)
            ctx_to_show = target_context[-ctx_len:]
            t_ctx = np.arange(len(ctx_to_show))
            t_pred = np.arange(len(ctx_to_show), len(ctx_to_show) + pred_len)

            ax.plot(t_ctx, ctx_to_show, color='gray', alpha=0.6, linewidth=1, label='Context')
            ax.plot(t_pred, ground_truth, color='#2196F3', linewidth=2, label='Ground truth')
            ax.plot(t_pred, forecast_a, color='#E53935', linewidth=2, linestyle='--', label=label_a)
            ax.plot(t_pred, forecast_b, color='#4CAF50', linewidth=2, linestyle=':', label=label_b)
            ax.set_title(label, fontsize=10)
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)[:60]}", transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, color='red')
            ax.set_title(label, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone-a', required=True)
    p.add_argument('--head-a', required=True)
    p.add_argument('--label-a', default='Model A')
    p.add_argument('--encoder-a', default='gru')
    p.add_argument('--backbone-b', required=True)
    p.add_argument('--head-b', required=True)
    p.add_argument('--label-b', default='Model B')
    p.add_argument('--encoder-b', default='gru')
    p.add_argument('--device', default='cuda')
    p.add_argument('--forecast-len', type=int, default=16)
    p.add_argument('--strategy', default='B4')
    args = p.parse_args()

    device = torch.device(args.device)

    print("Loading model A...")
    model_a = load_model(args.backbone_a, args.head_a, args.encoder_a, args.forecast_len, device)
    print("Loading model B...")
    model_b = load_model(args.backbone_b, args.head_b, args.encoder_b, args.forecast_len, device)

    output_dir = os.path.join(project_root, 'experiments', 'reconstruction-head', 'prediction_plots')
    os.makedirs(output_dir, exist_ok=True)

    print("Plotting worst configs (1/2)...")
    plot_comparison(WORST_CONFIGS[:6], model_a, model_b, args.label_a, args.label_b,
                    device, os.path.join(output_dir, 'compare_worst_1.png'),
                    f'{args.label_a} vs {args.label_b} — Worst Configs (1/2)',
                    strategy=args.strategy)

    print("Plotting worst configs (2/2)...")
    plot_comparison(WORST_CONFIGS[6:12], model_a, model_b, args.label_a, args.label_b,
                    device, os.path.join(output_dir, 'compare_worst_2.png'),
                    f'{args.label_a} vs {args.label_b} — Worst Configs (2/2)',
                    strategy=args.strategy)

    print("Plotting best configs (1/2)...")
    plot_comparison(BEST_CONFIGS[:6], model_a, model_b, args.label_a, args.label_b,
                    device, os.path.join(output_dir, 'compare_best_1.png'),
                    f'{args.label_a} vs {args.label_b} — Best Configs (1/2)',
                    strategy=args.strategy)

    print("Plotting best configs (2/2)...")
    plot_comparison(BEST_CONFIGS[6:12], model_a, model_b, args.label_a, args.label_b,
                    device, os.path.join(output_dir, 'compare_best_2.png'),
                    f'{args.label_a} vs {args.label_b} — Best Configs (2/2)',
                    strategy=args.strategy)

    print("Done!")


if __name__ == "__main__":
    main()
