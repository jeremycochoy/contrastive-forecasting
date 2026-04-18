#!/usr/bin/env python3
"""Plot R1 predictions vs ground truth for selected GIFT-Eval configs.

Creates 4 images with 6 subplots each:
- Image 1-2: Best configs (low MASE)
- Image 3-4: Worst configs (high MASE)

Each subplot shows: context (gray) + prediction (blue) + ground truth (orange).

Usage:
    GIFT_EVAL=~/gift-eval-data PYTHONPATH=. python plot_predictions.py \
        --backbone-path checkpoints/tiny_v2_best_gap.pth \
        --head-path checkpoints/R1_forecaster_recon_w16_best.pth \
        --label R1 --device cuda
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from src.models import ConfigurableModel
from src.forecasting_head import ForecastingHead, forecast_with_strategy
from gift_eval.data import Dataset as GiftDataset
from gluonts.time_feature import get_seasonality

BACKBONE_CONFIG = dict(
    C=4, H=512, W=16, encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
    rev_norm_span=32,
)
T_RAW = 1024
BACKBONE_C = 4

# Best and worst configs (from R1 MASE, post-alignment fix)
BEST_CONFIGS = [
    ("bizitobs_l2c/5T", "short", "bizitobs_l2c 5T short (MASE=0.35)"),
    ("jena_weather/10T", "short", "jena_weather 10T short (MASE=0.42)"),
    ("sz_taxi/H", "short", "sz_taxi H short (MASE=0.62)"),
    ("sz_taxi/15T", "medium", "sz_taxi 15T medium (MASE=0.64)"),
    ("sz_taxi/15T", "long", "sz_taxi 15T long (MASE=0.64)"),
    ("restaurant", "short", "restaurant D short (MASE=0.78)"),
    ("us_births/D", "short", "us_births D short (MASE=0.79)"),
    ("loop_seattle/5T", "short", "loop_seattle 5T short (MASE=0.82)"),
    ("us_births/M", "short", "us_births M short (MASE=0.88)"),
    ("bizitobs_l2c/5T", "medium", "bizitobs_l2c 5T medium (MASE=0.90)"),
    ("m_dense/D", "short", "m_dense D short (MASE=0.90)"),
    ("saugeenday/M", "short", "saugeen M short (MASE=0.91)"),
]

WORST_CONFIGS = [
    ("covid_deaths", "short", "covid_deaths D short (MASE=48.8)"),
    ("bizitobs_application", "long", "bizitobs_app 10S long (MASE=7.5)"),
    ("bizitobs_application", "medium", "bizitobs_app 10S medium (MASE=6.8)"),
    ("bitbrains_rnd/H", "short", "bitbrains_rnd H short (MASE=6.2)"),
    ("bizitobs_application", "short", "bizitobs_app 10S short (MASE=4.9)"),
    ("bitbrains_rnd/5T", "medium", "bitbrains_rnd 5T medium (MASE=4.8)"),
    ("m4_yearly", "short", "m4_yearly A short (MASE=4.1)"),
    ("bitbrains_rnd/5T", "long", "bitbrains_rnd 5T long (MASE=4.0)"),
    ("bizitobs_service", "long", "bizitobs_service 10S long (MASE=3.9)"),
    ("bizitobs_service", "medium", "bizitobs_service 10S medium (MASE=3.9)"),
    ("saugeenday/D", "short", "saugeen D short (MASE=3.6)"),
    ("solar/10T", "medium", "solar 10T medium (MASE=3.4)"),
]


def load_models(backbone_path, head_path, forecast_len, device):
    backbone = ConfigurableModel(**BACKBONE_CONFIG)
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
    """Run prediction on a single time series."""
    target = np.asarray(target, dtype=np.float32)

    # Handle NaN
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

    # Prepare context
    n = len(target)
    if n >= T_RAW:
        context = target[-T_RAW:]
    else:
        context = np.concatenate([np.full(T_RAW - n, target[0], dtype=np.float32), target])

    # (1, T_RAW, 1) — single channel, no replication needed
    context_t = torch.from_numpy(context).float().unsqueeze(0).unsqueeze(-1).to(device)

    with torch.no_grad():
        forecast = forecast_with_strategy(
            strategy, backbone, head, context_t,
            horizon=prediction_length, device=device)

    return forecast[:, 0]  # channel 0


def plot_configs(configs, backbone, head, device, output_path, title_prefix, strategy='B4'):
    """Plot 6 configs in a 3x2 grid."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(title_prefix, fontsize=16, fontweight='bold')

    for idx, (ds_name, term, label) in enumerate(configs[:6]):
        ax = axes[idx // 2, idx % 2]
        try:
            # Load dataset
            to_univ_check = GiftDataset(name=ds_name, term=term, to_univariate=False)
            to_univariate = to_univ_check.target_dim > 1
            dataset = GiftDataset(name=ds_name, term=term, to_univariate=to_univariate)
            pred_len = dataset.prediction_length

            # Get first test item
            test_items = list(dataset.test_data.input)
            item = test_items[0]
            target_context = np.asarray(item["target"], dtype=np.float32)

            # Get ground truth (from label)
            label_items = list(dataset.test_data.label)
            label_item = label_items[0]
            target_full = np.asarray(label_item["target"], dtype=np.float32)
            ground_truth = target_full[-pred_len:]

            # Predict
            forecast = predict_item(backbone, head, target_context, pred_len, device, strategy)

            # Plot
            ctx_len = min(len(target_context), pred_len * 3)  # show 3x prediction length of context
            ctx_to_show = target_context[-ctx_len:]
            t_ctx = np.arange(len(ctx_to_show))
            t_pred = np.arange(len(ctx_to_show), len(ctx_to_show) + pred_len)

            ax.plot(t_ctx, ctx_to_show, color='gray', alpha=0.6, linewidth=1, label='Context')
            ax.plot(t_pred, ground_truth, color='orange', linewidth=2, label='Ground truth')
            ax.plot(t_pred, forecast, color='blue', linewidth=2, linestyle='--', label='Prediction')
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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--backbone-path', required=True)
    p.add_argument('--head-path', required=True)
    p.add_argument('--label', default='R1', help='Label for plot titles')
    p.add_argument('--device', default='cuda')
    p.add_argument('--strategy', default='B4')
    p.add_argument('--forecast-len', type=int, default=16)
    args = p.parse_args()

    device = torch.device(args.device)

    print("Loading models...")
    backbone, head = load_models(args.backbone_path, args.head_path,
                                 forecast_len=args.forecast_len, device=device)
    print("Models loaded")

    output_dir = os.path.join(project_root, 'experiments', 'reconstruction-head', 'prediction_plots')
    os.makedirs(output_dir, exist_ok=True)

    label = args.label
    print("Plotting best configs (1/2)...")
    plot_configs(BEST_CONFIGS[:6], backbone, head, device,
                 os.path.join(output_dir, f'{label}_best_1.png'),
                 f'{label} Forecaster Reconstruction — Best Configs (1/2)',
                 strategy=args.strategy)

    print("Plotting best configs (2/2)...")
    plot_configs(BEST_CONFIGS[6:12], backbone, head, device,
                 os.path.join(output_dir, f'{label}_best_2.png'),
                 f'{label} Forecaster Reconstruction — Best Configs (2/2)',
                 strategy=args.strategy)

    print("Plotting worst configs (1/2)...")
    plot_configs(WORST_CONFIGS[:6], backbone, head, device,
                 os.path.join(output_dir, f'{label}_worst_1.png'),
                 f'{label} Forecaster Reconstruction — Worst Configs (1/2)',
                 strategy=args.strategy)

    print("Plotting worst configs (2/2)...")
    plot_configs(WORST_CONFIGS[6:12], backbone, head, device,
                 os.path.join(output_dir, f'{label}_worst_2.png'),
                 f'{label} Forecaster Reconstruction — Worst Configs (2/2)',
                 strategy=args.strategy)

    print("Done! Plots saved to:", output_dir)


if __name__ == "__main__":
    main()
