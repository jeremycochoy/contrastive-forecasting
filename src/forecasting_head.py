"""
Forecasting head for contrastive time-series backbone.

Decodes backbone forecaster latents into future normalized values.
Mirrors the GRURecoveryHead architecture from src/recovery.py but outputs
forecast_len values instead of ARMA parameters.

At patch position t, f_flat[:, t, :] represents what comes NEXT after patch t.
The ForecastingHead decodes this into the next forecast_len actual (normalized)
values: x_norm[(t+1)*W : (t+1)*W + forecast_len].

Supports multiple rollout strategies:
  A1: value-space rollout with 128-value head (original baseline)
  A2: value-space rollout with W-value head
  B1: latent-space rollout, decode all at end
  B2: latent-space rollout, decode each step with 128-head, crop to W
  B3: latent-space rollout, decode every 8 tokens (non-overlapping 128)
  B4: latent-space rollout, decode each step with W-head
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

    This is the PREDICTION target: the head predicts future values.

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


def compute_reconstruction_targets(x_norm, W=16, output_len=16, mode='forecaster'):
    """Extract RECONSTRUCTION targets: the values a latent represents.

    Unlike compute_valid_targets (which extracts PREDICTION targets),
    this extracts the values that each latent position actually encodes:

    - mode='forecaster': f[t] ≈ e[t+1], so target is patch t+1 values.
      target[t] = x_norm[(t+1)*W : (t+1)*W + output_len]
    - mode='encoder': e[t] encodes patch t, so target is patch t values.
      target[t] = x_norm[t*W : t*W + output_len]

    Args:
        x_norm: (B, T_raw, C) normalized input
        W: patch size
        output_len: values per position (W=16 for single-patch, 128 for multi-patch)
        mode: 'forecaster' (target=patch t+1) or 'encoder' (target=patch t)

    Returns:
        targets: (B*C, T_valid, output_len) target values
        T_valid: number of valid positions
    """
    B, T_raw, C = x_norm.shape
    T = T_raw // W

    if mode == 'forecaster':
        # f[t] represents patch t+1: target starts at (t+1)*W
        # Valid: (t+1)*W + output_len <= T_raw => t <= T_raw/W - 1 - output_len/W
        T_valid = (T_raw - output_len) // W  # patches t=0..T_valid-1
        targets = []
        for t in range(T_valid):
            start = (t + 1) * W
            end = start + output_len
            targets.append(x_norm[:, start:end, :])
    elif mode == 'encoder':
        # e[t] represents patch t: target starts at t*W
        # Valid: t*W + output_len <= T_raw => t <= T_raw/W - output_len/W
        T_valid = (T_raw - output_len) // W + 1  # patches t=0..T_valid-1
        # But we also need t < T (total patches)
        T_valid = min(T_valid, T)
        targets = []
        for t in range(T_valid):
            start = t * W
            end = start + output_len
            targets.append(x_norm[:, start:end, :])
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'forecaster' or 'encoder'.")

    targets = torch.stack(targets, dim=0)  # (T_valid, B, output_len, C)
    targets = targets.permute(1, 3, 0, 2)  # (B, C, T_valid, output_len)
    targets = targets.reshape(B * C, T_valid, output_len)

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


# ============================================================================
# Latent-space rollout infrastructure
# ============================================================================

def extract_encoder_latents(backbone, x):
    """Extract encoder latents e[t] (before transformer) and RevEWMNorm stats.

    Args:
        backbone: frozen ConfigurableModel
        x: (B, T_raw, C) raw input tensor

    Returns:
        e_bc: (B*C, T, H) encoder latents (detached)
        x_norm: (B, T_raw, C) normalized input
    """
    W_bb = backbone.W

    with torch.no_grad():
        if backbone.rev_norm is not None:
            x_norm = backbone.rev_norm(x, mode='norm')
        else:
            x_norm = x

        B, T_raw, C = x_norm.shape
        T = T_raw // W_bb

        # Reshape to patches: (B, T, C, W)
        xr = x_norm.view(B, T, W_bb, C).permute(0, 1, 3, 2)

        # Run encoder only (input_to_latent), not transformer
        e = backbone.transformer.input_to_latent(xr)  # (B, T, C, H)
        B, T, C, H = e.size()
        e_bc = e.permute(0, 2, 1, 3).reshape(B * C, T, H)

    return e_bc.detach(), x_norm.detach()


def rollout_latent(backbone, encoder_latents, n_future_tokens):
    """Generate future forecaster latents via autoregressive rollout.

    Uses the backbone's causal transformer: since contrastive training
    makes f[t] ≈ e[t+1], we feed f[-1] back as the next encoder latent.

    Args:
        backbone: frozen ConfigurableModel (on correct device)
        encoder_latents: (B*C, T, H) encoder latents from context
        n_future_tokens: how many future latent tokens to generate

    Returns:
        future_f: (B*C, n_future_tokens, H) generated forecaster latents
    """
    device = encoder_latents.device
    seq = encoder_latents  # (B*C, T, H)
    generated = []

    with torch.no_grad():
        for _ in range(n_future_tokens):
            # Build causal mask for current sequence length
            T_cur = seq.size(1)
            causal_mask = torch.triu(
                torch.ones(T_cur, T_cur, device=device), diagonal=1
            ).bool()
            causal_mask = causal_mask.float().masked_fill(causal_mask, float('-inf'))

            # Run transformer layers (bypass encoder — we already have latents)
            x = seq
            for layer in backbone.transformer.layers:
                x = layer(x, tgt_mask=causal_mask, tgt_is_causal=True)

            # x[:, -1, :] is f[-1] ≈ e[next]
            new_token = x[:, -1:, :]  # (B*C, 1, H)
            generated.append(new_token)

            # Append as next encoder latent
            seq = torch.cat([seq, new_token], dim=1)

    return torch.cat(generated, dim=1)  # (B*C, n_future_tokens, H)


def _get_denorm_stats(backbone, C):
    """Extract last-timestep RevEWMNorm stats for denormalization."""
    if backbone.rev_norm is not None and backbone.rev_norm.mean is not None:
        last_mean = backbone.rev_norm.mean[:, -1:, :]  # (1, 1, C)
        last_stdev = backbone.rev_norm.stdev[:, -1:, :]
        mean_c = last_mean.squeeze(0).squeeze(0).unsqueeze(1)  # (C, 1)
        stdev_c = last_stdev.squeeze(0).squeeze(0).unsqueeze(1)
        return mean_c, stdev_c
    return None, None


def _denormalize(pred_norm, mean_c, stdev_c):
    """Denormalize predictions: pred_norm * stdev + mean."""
    if mean_c is not None:
        return pred_norm * stdev_c.clamp(min=1e-5) + mean_c
    return pred_norm


# ============================================================================
# Forecast strategies
# ============================================================================

def forecast_A1(backbone, head, x_context, horizon, device):
    """A1: Value-space rollout with 128-value head (original baseline).

    Identical to forecast_autoregressive — kept for API consistency.
    """
    return forecast_autoregressive(backbone, head, x_context, horizon, device)


def forecast_A2(backbone, head, x_context, horizon, device):
    """A2: Value-space rollout with W-value head.

    Same as A1 but head outputs W=16 values. Slides by W each step.
    """
    forecast_len = head.forecast_len  # should be W (16)
    W_bb = backbone.W

    if x_context.dim() == 2:
        x_context = x_context.unsqueeze(0)
    x_context = x_context.to(device).float()
    B, T_ctx, C = x_context.shape

    all_preds = []
    remaining = horizon
    current_context = x_context.clone()

    while remaining > 0:
        with torch.no_grad():
            f_bc, x_norm = extract_forecaster_latents(backbone, current_context)
            pred_norm = head(f_bc)  # (C, T, forecast_len)
            pred_last = pred_norm[:, -1, :]  # (C, forecast_len)

            mean_c, stdev_c = _get_denorm_stats(backbone, C)
            pred_actual = _denormalize(pred_last, mean_c, stdev_c)

            n_take = min(forecast_len, remaining)
            all_preds.append(pred_actual[:, :n_take].cpu())
            remaining -= n_take

            if remaining > 0:
                new_values = pred_actual[:, :forecast_len].T.unsqueeze(0)
                current_context = torch.cat([
                    current_context[:, forecast_len:, :],
                    new_values
                ], dim=1)

    forecast = torch.cat(all_preds, dim=1)
    return forecast.T.numpy()


def _b_variant_setup(backbone, x_context, device):
    """Common setup for all B-variants: extract latents and denorm stats."""
    if x_context.dim() == 2:
        x_context = x_context.unsqueeze(0)
    x_context = x_context.to(device).float()
    B, T_ctx, C = x_context.shape

    with torch.no_grad():
        # Get BOTH encoder latents (for rollout) and forecaster latents (for head context)
        e_bc, x_norm = extract_encoder_latents(backbone, x_context)
        f_bc, _ = extract_forecaster_latents(backbone, x_context)
        mean_c, stdev_c = _get_denorm_stats(backbone, C)

    T_ctx_patches = e_bc.size(1)
    return e_bc, f_bc, mean_c, stdev_c, T_ctx_patches, C


def _b_variant_decode(head, f_ctx, future_f, T_ctx_patches):
    """Feed [context_f, rolled_f] as one sequence to the head.

    The bidirectional GRU processes real context latents first, then rolled
    latents. Predictions at future positions are returned.

    Args:
        head: ForecastingHead
        f_ctx: (B*C, T, H) context forecaster latents
        future_f: (B*C, N, H) rolled forecaster latents
        T_ctx_patches: int, number of context patches (= T)

    Returns:
        future_preds: (B*C, N, forecast_len) predictions at future positions
    """
    full_f = torch.cat([f_ctx, future_f], dim=1)  # (B*C, T+N, H)
    all_preds = head(full_f)  # (B*C, T+N, forecast_len)
    return all_preds[:, T_ctx_patches:, :]  # (B*C, N, forecast_len)


def forecast_B1(backbone, head, x_context, horizon, device):
    """B1: Latent-space rollout, decode all at end.

    Generate future latent tokens, decode with full-sequence head.
    Use all forecast_len values from last chunk, first W from others.
    """
    W_bb = backbone.W
    forecast_len = head.forecast_len

    e_bc, f_ctx, mean_c, stdev_c, T_ctx, C = _b_variant_setup(
        backbone, x_context, device)

    with torch.no_grad():
        n_tokens = math.ceil(horizon / W_bb)
        future_f = rollout_latent(backbone, e_bc, n_tokens)

        # Decode full sequence [context + rolled]
        future_preds = _b_variant_decode(head, f_ctx, future_f, T_ctx)
        # future_preds: (C, n_tokens, forecast_len)

        all_preds = []
        remaining = horizon
        for i in range(n_tokens):
            pred_norm = future_preds[:, i, :]  # (C, forecast_len)

            if remaining <= forecast_len:
                n_take = remaining
            else:
                n_take = W_bb

            pred_actual = _denormalize(pred_norm[:, :n_take], mean_c, stdev_c)
            all_preds.append(pred_actual.cpu())
            remaining -= n_take
            if remaining <= 0:
                break

    forecast = torch.cat(all_preds, dim=1)
    return forecast.T.numpy()


def forecast_B2(backbone, head, x_context, horizon, device):
    """B2: Latent-space rollout, decode with 128-head, crop to W.

    At each latent step, decode with full-sequence head but keep only first W.
    """
    W_bb = backbone.W

    e_bc, f_ctx, mean_c, stdev_c, T_ctx, C = _b_variant_setup(
        backbone, x_context, device)

    with torch.no_grad():
        n_tokens = math.ceil(horizon / W_bb)
        future_f = rollout_latent(backbone, e_bc, n_tokens)
        future_preds = _b_variant_decode(head, f_ctx, future_f, T_ctx)

        all_preds = []
        remaining = horizon
        for i in range(n_tokens):
            pred_norm = future_preds[:, i, :]

            n_take = min(W_bb, remaining)
            pred_actual = _denormalize(pred_norm[:, :n_take], mean_c, stdev_c)
            all_preds.append(pred_actual.cpu())
            remaining -= n_take
            if remaining <= 0:
                break

    forecast = torch.cat(all_preds, dim=1)
    return forecast.T.numpy()


def forecast_B3(backbone, head, x_context, horizon, device):
    """B3: Latent-space rollout, non-overlapping 128 decode.

    Roll forward 8 latent tokens (= 128 values), use the 8th token's
    full prediction. Repeat.
    """
    W_bb = backbone.W
    forecast_len = head.forecast_len
    tokens_per_chunk = forecast_len // W_bb  # 8

    e_bc, f_ctx, mean_c, stdev_c, T_ctx, C = _b_variant_setup(
        backbone, x_context, device)

    with torch.no_grad():
        n_chunks = math.ceil(horizon / forecast_len)
        n_tokens = n_chunks * tokens_per_chunk
        future_f = rollout_latent(backbone, e_bc, n_tokens)
        future_preds = _b_variant_decode(head, f_ctx, future_f, T_ctx)

        all_preds = []
        remaining = horizon
        for chunk_i in range(n_chunks):
            token_idx = (chunk_i + 1) * tokens_per_chunk - 1
            pred_norm = future_preds[:, token_idx, :]

            n_take = min(forecast_len, remaining)
            pred_actual = _denormalize(pred_norm[:, :n_take], mean_c, stdev_c)
            all_preds.append(pred_actual.cpu())
            remaining -= n_take
            if remaining <= 0:
                break

    forecast = torch.cat(all_preds, dim=1)
    return forecast.T.numpy()


def forecast_B4(backbone, head, x_context, horizon, device):
    """B4: Latent-space rollout with W-value head.

    Generate latent tokens, decode each with full-sequence W-head.
    """
    W_bb = backbone.W

    e_bc, f_ctx, mean_c, stdev_c, T_ctx, C = _b_variant_setup(
        backbone, x_context, device)

    with torch.no_grad():
        n_tokens = math.ceil(horizon / W_bb)
        future_f = rollout_latent(backbone, e_bc, n_tokens)
        future_preds = _b_variant_decode(head, f_ctx, future_f, T_ctx)

        all_preds = []
        remaining = horizon
        for i in range(n_tokens):
            pred_norm = future_preds[:, i, :]

            n_take = min(head.forecast_len, remaining)
            pred_actual = _denormalize(pred_norm[:, :n_take], mean_c, stdev_c)
            all_preds.append(pred_actual.cpu())
            remaining -= n_take
            if remaining <= 0:
                break

    forecast = torch.cat(all_preds, dim=1)
    return forecast.T.numpy()


# Strategy dispatch
FORECAST_STRATEGIES = {
    'A1': forecast_A1,
    'A2': forecast_A2,
    'B1': forecast_B1,
    'B2': forecast_B2,
    'B3': forecast_B3,
    'B4': forecast_B4,
}


def forecast_with_strategy(strategy, backbone, head, x_context, horizon, device):
    """Dispatch to the appropriate forecast strategy.

    Args:
        strategy: one of 'A1', 'A2', 'B1', 'B2', 'B3', 'B4'
        backbone, head, x_context, horizon, device: same as forecast_autoregressive
    """
    fn = FORECAST_STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(FORECAST_STRATEGIES)}")
    return fn(backbone, head, x_context, horizon, device)
