"""
Forecasting head for contrastive time-series backbone.

Decodes backbone forecaster latents into future normalized values.
Mirrors the GRURecoveryHead architecture from src/recovery.py but outputs
forecast_len values instead of ARMA parameters.

At patch position t, f_flat[:, t, :] represents what comes NEXT after patch t.
The ForecastingHead decodes this into the next forecast_len actual (normalized)
values: x_norm[(t+1)*W : (t+1)*W + forecast_len].
"""

import math

import numpy as np
import torch
import torch.nn as nn


W = 16  # Patch size (must match backbone)
FORECAST_LEN = 128  # Output horizon per patch


class ForecastingHead(nn.Module):
    """GRU-based head: decodes backbone forecaster latents into future values.

    Input: (B*C, T, H) - sequence of forecaster latents per channel
    Output: (B*C, T, forecast_len) - predicted next forecast_len normalized values

    At position t, predicts x_norm[(t+1)*W : (t+1)*W + forecast_len].
    """

    def __init__(self, H=512, hidden_dim=128, num_gru_layers=2,
                 forecast_len=128, dropout=0.1):
        super().__init__()
        self.forecast_len = forecast_len

        # Project down from H to hidden_dim first (saves GRU parameters)
        self.input_proj = nn.Sequential(
            nn.Linear(H, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            bidirectional=True,
        )

        gru_out_dim = hidden_dim * 2  # bidirectional

        self.output_layers = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.forecast_head = nn.Linear(hidden_dim, forecast_len)

    def forward(self, x):
        """x: (B*C, T, H) -> (B*C, T, forecast_len)"""
        x = self.input_proj(x)
        gru_out, _ = self.gru(x)
        features = self.output_layers(gru_out)
        return self.forecast_head(features)


def extract_forecaster_latents(backbone, x):
    """Extract f_lat from backbone for forecasting head input.

    Applies RevEWMNorm, patches the input, and runs the transformer to get
    the forecaster latents (f_flat). The normalized input is also returned
    for target extraction.

    Args:
        backbone: frozen ConfigurableModel
        x: (B, T_raw, C) raw input tensor

    Returns:
        f_bc: (B*C, T, H) forecaster latents (detached)
        x_norm: (B, T_raw, C) normalized input (for target extraction)
    """
    W_bb = backbone.W
    H_bb = backbone.H

    with torch.no_grad():
        # Apply reversible normalization
        if backbone.rev_norm is not None:
            x_norm = backbone.rev_norm(x, mode='norm')
        else:
            x_norm = x

        B, T_raw, C = x_norm.shape
        T = T_raw // W_bb

        # Reshape to patches: (B, T, C, W)
        xr = x_norm.view(B, T, W_bb, C).permute(0, 1, 3, 2)

        # Get forecaster latents from transformer
        f_flat, _ = backbone.transformer(xr)
        # f_flat: (B*C, T, H)

    return f_flat.detach(), x_norm.detach()


def compute_valid_targets(x_norm, W=16, forecast_len=128):
    """Extract targets for training: normalized values after each valid patch.

    For patch t, the target is x_norm[:, (t+1)*W : (t+1)*W + forecast_len, :].
    Valid patches are those where (t+1)*W + forecast_len <= T_raw.

    Args:
        x_norm: (B, T_raw, C) normalized input
        W: patch size
        forecast_len: number of future values to predict

    Returns:
        targets: (B*C, T_valid, forecast_len) target values
        T_valid: number of patches with valid targets
    """
    B, T_raw, C = x_norm.shape
    T = T_raw // W

    # (t+1)*W + forecast_len <= T_raw  =>  t <= T_raw/W - 1 - forecast_len/W
    # t_max = floor((T_raw - forecast_len) / W) - 1
    T_valid = (T_raw - forecast_len) // W  # number of valid patches (t=0..T_valid-1)

    # Extract targets for each valid patch
    # targets[b, c, t, :] = x_norm[b, (t+1)*W : (t+1)*W + forecast_len, c]
    targets = []
    for t in range(T_valid):
        start = (t + 1) * W
        end = start + forecast_len
        # x_norm[:, start:end, :] is (B, forecast_len, C)
        targets.append(x_norm[:, start:end, :])

    # Stack: (T_valid, B, forecast_len, C) -> permute to (B, C, T_valid, forecast_len)
    targets = torch.stack(targets, dim=0)  # (T_valid, B, forecast_len, C)
    targets = targets.permute(1, 3, 0, 2)  # (B, C, T_valid, forecast_len)
    targets = targets.reshape(B * C, T_valid, forecast_len)

    return targets, T_valid


def forecast_autoregressive(backbone, head, x_context, horizon, device):
    """Produce a forecast of length `horizon` given context x_context.

    Uses sliding window: encode context, predict 128 steps, shift window
    by 128, repeat until horizon is covered.

    Args:
        backbone: frozen ConfigurableModel
        head: trained ForecastingHead
        x_context: (1, T_context, C) or (T_context, C) context series
        horizon: total number of future steps to predict
        device: torch device

    Returns:
        forecast: (horizon, C) numpy array of predicted values (in original scale)
    """
    forecast_len = head.forecast_len
    W_bb = backbone.W
    T_raw = x_context.shape[-2] if x_context.dim() >= 2 else x_context.shape[0]

    # Ensure 3D: (1, T_context, C)
    if x_context.dim() == 2:
        x_context = x_context.unsqueeze(0)
    if x_context.dim() != 3:
        raise ValueError(f"Expected 2D or 3D input, got {x_context.dim()}D")

    x_context = x_context.to(device).float()
    B, T_ctx, C = x_context.shape
    assert B == 1, "Autoregressive forecast only supports batch size 1"

    all_preds = []
    remaining = horizon
    current_context = x_context.clone()

    while remaining > 0:
        with torch.no_grad():
            # Extract forecaster latents
            f_bc, x_norm = extract_forecaster_latents(backbone, current_context)
            # f_bc: (C, T, H) since B=1

            # Run forecasting head
            pred_norm = head(f_bc)  # (C, T, forecast_len)

            # Use the LAST patch position's prediction (most informed)
            pred_last = pred_norm[:, -1, :]  # (C, forecast_len)

            # Denormalize: use EMA stats at the end of context
            # backbone.rev_norm.mean/stdev are (1, T_ctx, C) after the norm call
            if backbone.rev_norm is not None and backbone.rev_norm.mean is not None:
                last_mean = backbone.rev_norm.mean[:, -1:, :]  # (1, 1, C)
                last_stdev = backbone.rev_norm.stdev[:, -1:, :]  # (1, 1, C)
                # pred_last is (C, forecast_len), stats are (1, 1, C)
                # Reshape stats to (C, 1) for broadcasting
                last_mean_c = last_mean.squeeze(0).squeeze(0).unsqueeze(1)  # (C, 1)
                last_stdev_c = last_stdev.squeeze(0).squeeze(0).unsqueeze(1)  # (C, 1)
                pred_actual = pred_last * last_stdev_c.clamp(min=1e-5) + last_mean_c
            else:
                pred_actual = pred_last

            # pred_actual: (C, forecast_len) -> take what we need
            n_take = min(forecast_len, remaining)
            chunk = pred_actual[:, :n_take]  # (C, n_take)
            all_preds.append(chunk.cpu())

            remaining -= n_take

            if remaining > 0:
                # Shift context window: drop first forecast_len points,
                # append predicted values
                # chunk is (C, n_take) -> reshape to (1, n_take, C)
                new_values = pred_actual[:, :forecast_len].T.unsqueeze(0)  # (1, forecast_len, C)
                # Slide context: drop oldest forecast_len steps, append new
                current_context = torch.cat([
                    current_context[:, forecast_len:, :],
                    new_values
                ], dim=1)

    # Concatenate all chunks: list of (C, n_i) -> (C, horizon)
    forecast = torch.cat(all_preds, dim=1)  # (C, horizon)
    forecast = forecast.T.numpy()  # (horizon, C)

    return forecast
