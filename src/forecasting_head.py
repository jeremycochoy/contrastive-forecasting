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


# Standard 9 quantile levels used by GIFT-Eval.
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class QuantileForecastingHead(nn.Module):
    """Same GRU trunk as ForecastingHead but outputs Q quantile predictions.

    Input: (B*C, T, H)
    Output: (B*C, T, num_quantiles, forecast_len)

    Trained with pinball loss averaged over quantiles. Median (q=0.5) gives
    the point forecast comparable to the MSE head; other quantiles let the
    model express uncertainty (which the MSE head couldn't, leading to the
    amplitude-damping failure noted in the 2026-04-27_periodic-synth-mix report).
    """

    def __init__(self, H=512, hidden_dim=128, num_gru_layers=2,
                 forecast_len=128, dropout=0.1,
                 quantile_levels=QUANTILE_LEVELS):
        super().__init__()
        self.forecast_len = forecast_len
        self.quantile_levels = list(quantile_levels)
        self.num_quantiles = len(self.quantile_levels)

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
        gru_out_dim = hidden_dim * 2

        self.output_layers = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output: forecast_len * num_quantiles values per (B*C, T) position.
        self.forecast_head = nn.Linear(
            hidden_dim, forecast_len * self.num_quantiles)

    def forward(self, x):
        """x: (B*C, T, H) -> (B*C, T, num_quantiles, forecast_len)"""
        x = self.input_proj(x)
        gru_out, _ = self.gru(x)
        features = self.output_layers(gru_out)
        flat = self.forecast_head(features)                  # (..., Q*L)
        BC, T, _ = flat.shape
        return flat.view(BC, T, self.num_quantiles, self.forecast_len)


class TransformerQuantileForecastingHead(nn.Module):
    """N-layer transformer head on backbone latents.

    Mirrors the backbone's depth+width by default (H=384, num_layers=6,
    nhead=6, ffn_mult=4) — same representational surface as the backbone
    but trained from scratch with Moirai-style HP and cosine LR. Pre-norm,
    GELU, dropout=0.1.

    Causality (``causal=True``, default) imposes a triangular mask so the
    head can't peek at future tokens — matches the backbone's
    training-time causality. With ``causal=False`` the head sees the full
    sequence both directions; for predictions further than W=16 ahead
    (``forecast_len > W``), the bidir variant is necessary so the head
    can use ``f_t, f_{t+1}, …, f_{t+k}`` to reconstruct multiple future
    patches from position t (the eval-time B1/B2/B3 strategies feed the
    head a [encoder_ctx, rolled_f] sequence to maintain this access).

    Input:  (B*C, T, H) backbone latents
    Output: (B*C, T, num_quantiles, forecast_len)
    """

    def __init__(self, H=384, num_layers=6, nhead=6, ffn_mult=4.0,
                 forecast_len=16, dropout=0.1, causal=True,
                 quantile_levels=QUANTILE_LEVELS, **_unused):
        super().__init__()
        self.forecast_len = forecast_len
        self.causal = causal
        self.quantile_levels = list(quantile_levels)
        self.num_quantiles = len(self.quantile_levels)

        layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=nhead,
            dim_feedforward=int(ffn_mult * H),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(H)
        self.forecast_head = nn.Linear(
            H, forecast_len * self.num_quantiles)

    def forward(self, x, src_mask=None):
        """Run the head's transformer + output projection.

        ``src_mask``: optional `(T, T)` bool mask (True = block). When
        provided, overrides ``self.causal`` and is used as-is. Used by
        the trainer's ``e_then_f`` path to build a mixed mask that
        prevents the head from peeking at future encoder latents.
        """
        if src_mask is not None:
            x = self.transformer(x, mask=src_mask, is_causal=False)
        elif self.causal:
            T = x.size(1)
            causal = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            x = self.transformer(x, mask=causal, is_causal=True)
        else:
            x = self.transformer(x)
        x = self.norm(x)
        flat = self.forecast_head(x)
        BC, T_out, _ = flat.shape
        return flat.view(BC, T_out, self.num_quantiles, self.forecast_len)


class TransformerGaussianForecastingHead(nn.Module):
    """Same trunk as TransformerQuantileForecastingHead but a Gaussian
    per-position output (μ, log σ²) trained with NLL.

    Round 5 + 7 showed the 9-bin pinball quantile loss plateaus at
    ~0.192 ema across every transformer variant — strong evidence the
    loss surface itself has hit a noise floor. This head outputs only
    2 numbers per forecast step (mean and log-variance) and trains with
    Gaussian NLL: smooth gradient everywhere, closed-form quantiles via
    inverse normal CDF at eval time, dramatically fewer outputs.

    Input:  (B*C, T, H) backbone latents
    Output: (mu, log_var) each shape (B*C, T, forecast_len)
    """

    def __init__(self, H=384, num_layers=6, nhead=6, ffn_mult=4.0,
                 forecast_len=16, dropout=0.1, causal=True,
                 quantile_levels=QUANTILE_LEVELS, **_unused):
        super().__init__()
        self.forecast_len = forecast_len
        self.causal = causal
        # Gaussian heads still expose `quantile_levels` so eval-side
        # `_gaussian_to_quantiles` knows which inverse-CDF points to emit.
        self.quantile_levels = list(quantile_levels)
        self.num_quantiles = len(self.quantile_levels)

        layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=nhead,
            dim_feedforward=int(ffn_mult * H),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(H)
        # 2 outputs per forecast step: μ and log σ².
        self.forecast_head = nn.Linear(H, forecast_len * 2)

    def forward(self, x, src_mask=None):
        if src_mask is not None:
            x = self.transformer(x, mask=src_mask, is_causal=False)
        elif self.causal:
            T = x.size(1)
            causal = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            x = self.transformer(x, mask=causal, is_causal=True)
        else:
            x = self.transformer(x)
        x = self.norm(x)
        flat = self.forecast_head(x)                          # (BC, T, 2*L)
        BC, T_out, _ = flat.shape
        out = flat.view(BC, T_out, 2, self.forecast_len)
        mu = out[:, :, 0, :]                                  # (BC, T, L)
        log_var = out[:, :, 1, :]
        return mu, log_var

    @torch.no_grad()
    def to_quantiles(self, mu, log_var):
        """Convert (μ, log σ²) → 9 quantile values via inverse normal CDF.

        Returns shape (BC, T, num_quantiles, forecast_len). Same layout as
        the quantile heads' forward output, so existing forecast_B4 code
        can consume it after this conversion.
        """
        sigma = torch.exp(0.5 * log_var).clamp(min=1e-6)
        # Inverse standard-normal CDF at each quantile level.
        levels = torch.tensor(
            self.quantile_levels, device=mu.device, dtype=mu.dtype)
        z = torch.special.ndtri(levels)                       # (Q,)
        # broadcast: (BC,T,1,L) + (BC,T,1,L) * (Q,1) → (BC,T,Q,L)
        return mu.unsqueeze(-2) + sigma.unsqueeze(-2) * z.view(-1, 1)


def build_e_then_f_mask(T_e: int, T_f: int, device=None) -> torch.Tensor:
    """Mask for the [e_0..e_{T_e-1}, f_0..f_{T_f-1}] training-time input.

    The head's output at position ``T_e + p_f`` (f-block at index p_f,
    decoding patch p_f+1) must not see ``e_{p_f+1}`` — the encoder
    latent that encodes the *target* patch — otherwise it can copy the
    answer directly. With multi-layer transformers, blocking only the
    direct attention path is insufficient: layer-1 outputs at e-rows
    that ARE allowed to see future e-cols would absorb that info, and
    f-rows in layer 2+ would receive it indirectly. So we need every
    row to be causal:

      - e-block rows: causal within e (e-row r attends to e-cols 0..r),
        no attention to the f-block.
      - f-block rows at relative pos p_f = r - T_e:
          * e-block cols c < T_e: allowed iff c <= p_f (block c >= p_f+1
            including transitive routes — see e-block causality above).
          * f-block cols c >= T_e: causal within f (c <= r).

    Returns a `(T_e+T_f, T_e+T_f)` bool tensor where True = block.
    """
    total = T_e + T_f
    rows = torch.arange(total, device=device).unsqueeze(1)            # (total, 1)
    cols = torch.arange(total, device=device).unsqueeze(0)            # (1, total)
    is_e_row = rows < T_e
    is_e_col = cols < T_e
    e_to_e = is_e_row & is_e_col & (cols <= rows)         # causal within e
    f_to_e = (~is_e_row) & is_e_col & (cols <= rows - T_e)
    f_to_f = (~is_e_row) & (~is_e_col) & (cols <= rows)
    allowed = e_to_e | f_to_e | f_to_f
    return ~allowed


def gaussian_nll_loss(mu, log_var, target):
    """Per-step Gaussian NLL: 0.5 * (log(2π) + log σ² + (y-μ)² / σ²).

    Args:
        mu:      (..., forecast_len)  predicted means
        log_var: (..., forecast_len)  predicted log-variances
        target:  (..., forecast_len)  ground-truth values

    Returns:
        Scalar mean NLL.
    """
    inv_var = (-log_var).exp()
    return 0.5 * (math.log(2 * math.pi) + log_var
                  + (target - mu).pow(2) * inv_var).mean()


class LinearForecastingHead(nn.Module):
    """Single Linear over backbone latents (no GRU/MLP). Diagnostic baseline.

    If a single Linear matches the GRU head's GIFT-Eval score, head capacity
    isn't the bottleneck and effort should go to schedule + HP.

    Input:  (B*C, T, H) backbone latents
    Output: (B*C, T, forecast_len)
    """

    def __init__(self, H=512, forecast_len=128, **_unused):
        super().__init__()
        self.forecast_len = forecast_len
        self.forecast_head = nn.Linear(H, forecast_len)

    def forward(self, x):
        return self.forecast_head(x)


class LinearQuantileForecastingHead(nn.Module):
    """Linear-probe variant of QuantileForecastingHead.

    Input:  (B*C, T, H)
    Output: (B*C, T, num_quantiles, forecast_len)
    """

    def __init__(self, H=512, forecast_len=128,
                 quantile_levels=QUANTILE_LEVELS, **_unused):
        super().__init__()
        self.forecast_len = forecast_len
        self.quantile_levels = list(quantile_levels)
        self.num_quantiles = len(self.quantile_levels)
        self.forecast_head = nn.Linear(
            H, forecast_len * self.num_quantiles)

    def forward(self, x):
        flat = self.forecast_head(x)
        BC, T, _ = flat.shape
        return flat.view(BC, T, self.num_quantiles, self.forecast_len)


def quantile_loss(predicted, target, quantile_levels=QUANTILE_LEVELS):
    """Pinball loss summed over quantiles, averaged over the rest.

    Args:
        predicted: (..., num_quantiles, forecast_len)
        target:    (..., forecast_len) or (..., 1, forecast_len)
        quantile_levels: iterable of Q levels in (0, 1).

    Returns:
        scalar loss = mean over (..., forecast_len) of average over Q of
        ``q * relu(target - pred)  +  (1-q) * relu(pred - target)``.
    """
    if target.dim() == predicted.dim() - 1:
        target = target.unsqueeze(-2)                       # (..., 1, L)
    err = target - predicted                                 # (..., Q, L)
    q = predicted.new_tensor(list(quantile_levels)).view(
        *([1] * (predicted.dim() - 2)), -1, 1)               # (..., Q, 1)
    # max(q*err, (q-1)*err) is the standard pinball form; equivalent to
    # q*relu(err) + (1-q)*relu(-err) when err ∈ R.
    return torch.maximum(q * err, (q - 1) * err).mean()


def extract_forecaster_latents(backbone, x, freq_ids=None,
                                seasonality_ids=None):
    """Extract f_lat from backbone for forecasting head input.

    Applies RevEWMNorm, patches the input (incl. patch-stats, freq-emb,
    and seasonality-emb if configured), and runs the transformer to get
    the forecaster latents (``f_flat``).

    Args:
        backbone: frozen ConfigurableModel
        x: (B, T_raw, C) raw input tensor
        freq_ids: LongTensor (B,) of freq class ids when the backbone has a
            freq embedding. Defaults to class 0 (unknown) if missing.
        seasonality_ids: LongTensor (B,) of seasonality class ids when the
            backbone has a seasonality embedding. Defaults to class 0
            (unknown) if missing.
    """
    with torch.no_grad():
        if backbone.rev_norm is not None:
            x_norm = backbone.rev_norm(x, mode='norm')
        else:
            x_norm = x

        B = x_norm.shape[0]
        if (getattr(backbone, 'freq_embedding', None) is not None
                and freq_ids is None):
            default = int(getattr(backbone, '_eval_freq_id', 0))
            freq_ids = torch.full((B,), default, dtype=torch.long, device=x.device)
        if (getattr(backbone, 'seasonality_embedding', None) is not None
                and seasonality_ids is None):
            default = int(getattr(backbone, '_eval_seasonality_id', 0))
            seasonality_ids = torch.full((B,), default, dtype=torch.long, device=x.device)

        xr = backbone.prepare_encoder_input(
            x_norm, freq_ids=freq_ids, seasonality_ids=seasonality_ids)

        f_flat, _ = backbone.transformer(xr)

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

def extract_encoder_latents(backbone, x, freq_ids=None, seasonality_ids=None):
    """Extract encoder latents e[t] and RevEWMNorm stats.

    Returns the same tensor used as the contrastive target ``o_lat``
    during training: patch encoder output + any causal encoder layers
    (``backbone.transformer.encoder_layers``). When ``encoder_layers`` is
    empty (baseline runs with ``num_encoder_layers=0``), this collapses
    to the original patch-encoder-only behaviour and is bit-identical to
    pre-encoder-stage checkpoints.

    Args:
        backbone: frozen ConfigurableModel
        x: (B, T_raw, C) raw input tensor
        freq_ids, seasonality_ids: optional LongTensors (B,). When the
            backbone has the corresponding embedding configured but the
            id tensor is missing, falls back to class 0 (unknown).

    Returns:
        e_bc: (B*C, T, H) encoder latents (detached)
        x_norm: (B, T_raw, C) normalized input
    """
    with torch.no_grad():
        if backbone.rev_norm is not None:
            x_norm = backbone.rev_norm(x, mode='norm')
        else:
            x_norm = x

        B = x_norm.shape[0]
        if (getattr(backbone, 'freq_embedding', None) is not None
                and freq_ids is None):
            default = int(getattr(backbone, '_eval_freq_id', 0))
            freq_ids = torch.full((B,), default, dtype=torch.long, device=x.device)
        if (getattr(backbone, 'seasonality_embedding', None) is not None
                and seasonality_ids is None):
            default = int(getattr(backbone, '_eval_seasonality_id', 0))
            seasonality_ids = torch.full((B,), default, dtype=torch.long, device=x.device)

        xr = backbone.prepare_encoder_input(
            x_norm, freq_ids=freq_ids, seasonality_ids=seasonality_ids)

        e = backbone.transformer.input_to_latent(xr)  # (B, T, C, H)
        B, T, C, H = e.size()
        e_bc = e.permute(0, 2, 1, 3).reshape(B * C, T, H)

        encoder_layers = getattr(backbone.transformer, 'encoder_layers', None)
        if encoder_layers is not None and len(encoder_layers) > 0:
            mask = backbone.transformer.causal_mask
            if mask is None or mask.size(0) != T:
                mask = backbone.transformer._generate_square_subsequent_mask(T).to(e_bc.device)
                backbone.transformer.causal_mask = mask
            for layer in encoder_layers:
                e_bc = layer(e_bc, tgt_mask=mask, tgt_is_causal=True)

    return e_bc.detach(), x_norm.detach()


def extract_teacher_encoder_latents(backbone, x, freq_ids=None,
                                    seasonality_ids=None):
    """EMA-teacher counterpart of :func:`extract_encoder_latents` (#388).

    Same output layout, and the same input preparation: normalisation,
    patching and the freq / seasonality embedding tables are the student's,
    because the teacher covers only ``input_to_latent`` and the encoder
    stack. ``backbone.teacher_forward`` then runs those two from the EMA
    copies, falling back to the student's copy of whichever one is not under
    EMA. Use it to measure the teacher's ``h_t`` on the same probe batch as
    the student's, so the two drift curves are directly comparable.

    Raises RuntimeError when the backbone carries no teacher (built without
    ``ema_embedding`` / ``ema_encoder``).

    Returns:
        h_bc: (B*C, T, H) teacher encoder latents (detached)
        x_norm: (B, T_raw, C) normalized input
    """
    with torch.no_grad():
        if backbone.rev_norm is not None:
            x_norm = backbone.rev_norm(x, mode='norm')
        else:
            x_norm = x

        B = x_norm.shape[0]
        if (getattr(backbone, 'freq_embedding', None) is not None
                and freq_ids is None):
            default = int(getattr(backbone, '_eval_freq_id', 0))
            freq_ids = torch.full((B,), default, dtype=torch.long, device=x.device)
        if (getattr(backbone, 'seasonality_embedding', None) is not None
                and seasonality_ids is None):
            default = int(getattr(backbone, '_eval_seasonality_id', 0))
            seasonality_ids = torch.full((B,), default, dtype=torch.long, device=x.device)

        xr = backbone.prepare_encoder_input(
            x_norm, freq_ids=freq_ids, seasonality_ids=seasonality_ids)

        h = backbone.teacher_forward(xr)                 # (B, T, C, H)
        B, T, C, H = h.size()
        h_bc = h.permute(0, 2, 1, 3).reshape(B * C, T, H)

    return h_bc.detach(), x_norm.detach()


def rollout_latent(backbone, encoder_latents, n_future_tokens):
    """Generate future forecaster latents via autoregressive rollout.

    Uses the backbone's causal transformer: since contrastive training
    makes f[t] ≈ e[t+1], we feed f[-1] back as the next encoder latent.

    Each token runs `TransformerBlock.forecaster_forward` — the same
    forecaster operator the training forward and the training rollout depth
    (#373) compose, bottleneck projections included.

    Args:
        backbone: frozen ConfigurableModel (on correct device)
        encoder_latents: (B*C, T, H) encoder latents from context, in
            encoder-dim space (`dimension_e`).
        n_future_tokens: how many future latent tokens to generate

    Returns:
        future_f: (B*C, n_future_tokens, H) generated forecaster latents
            in encoder-dim space (post-up-proj), so they share the
            residual-stream dim with `encoder_latents` and can feed the
            decoding head unchanged.
    """
    seq = encoder_latents  # (B*C, T, H) in dimension_e
    generated = []

    with torch.no_grad():
        for _ in range(n_future_tokens):
            # Run the forecaster on the current sequence (bypass the encoder
            # — we already have latents). x[:, -1, :] is f[-1] ≈ e[next].
            x = backbone.transformer.forecaster_forward(seq)
            new_token = x[:, -1:, :]  # (B*C, 1, H)
            generated.append(new_token)

            # Append as next encoder latent (in dimension_e — matches `seq`).
            seq = torch.cat([seq, new_token], dim=1)

    return torch.cat(generated, dim=1)  # (B*C, n_future_tokens, H)


def rollout_forecaster_latents(backbone, forecasted_latent, depth):
    """f^(1) .. f^(k) — re-apply the forecaster to its own output (#373).

    The training loss ties f^(0)_t to h_{t+1} only, while the eval rolls the
    forecaster out on its own output. This composes the same operator
    (`TransformerBlock.forecaster_forward`, the one :func:`rollout_latent`
    composes) over the WHOLE sequence at once, so copy j gives f^(j)_t, which
    the rollout-depth objective ties to h_{t+1+j}.

    Feeding f^(j-1) back in place of the encoder prefix is the fixed-point
    approximation of the eval rollout: it costs one dropped key per depth at
    the front of the attention context and buys a fully parallel pass over
    all t. Gradient flows through the chain — no detach between passes.

    Args:
        backbone: ConfigurableModel whose `transformer` holds the forecaster.
        forecasted_latent: (B, T, C, H) f^(0), the training forward's output.
        depth: k, how many further applications to compose.

    Returns:
        list of k tensors (B, T, C, H); entry j−1 is f^(j).
    """
    if depth <= 0:
        return []
    B, T, C, H = forecasted_latent.shape
    seq = forecasted_latent.permute(0, 2, 1, 3).reshape(B * C, T, H)
    depths = []
    for _ in range(depth):
        seq = backbone.transformer.forecaster_forward(seq)
        depths.append(seq.reshape(B, C, T, H).permute(0, 2, 1, 3))
    return depths


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
    """Extract encoder latents and denorm stats for latent-space rollout.

    Returns e[0], ..., e[k] (encoder latents for context patches)
    and denormalization statistics.
    """
    if x_context.dim() == 2:
        x_context = x_context.unsqueeze(0)
    x_context = x_context.to(device).float()
    B, T_ctx, C = x_context.shape

    with torch.no_grad():
        e_ctx, x_norm = extract_encoder_latents(backbone, x_context)  # (BC, n_ctx, H)
        mean_c, stdev_c = _get_denorm_stats(backbone, C)

    n_ctx = e_ctx.size(1)
    return e_ctx, mean_c, stdev_c, n_ctx, C


def _b_variant_decode(head, e_ctx, rolled_f, n_ctx):
    """Feed [e[0], ..., e[k], f[k+1], ..., f[k+m]] to the head.

    The head is a content-based decoder (bidirectional GRU, no position
    encoding). It reconstructs the patch each latent represents:
    output at position i → values of p[i].

    Args:
        head: ForecastingHead
        e_ctx: (BC, n_ctx, H) encoder latents e[0..k]
        rolled_f: (BC, m, H) rolled forecaster latents f[k+1..k+m]
        n_ctx: k+1, number of context patches

    Returns:
        rolled_out: (BC, m, forecast_len) head output at rolled positions
    """
    seq = torch.cat([e_ctx, rolled_f], dim=1)   # (BC, n_ctx+m, H)
    all_out = head(seq)
    # Gaussian heads return (mu, log_var); convert to quantile shape so
    # downstream forecast_* code can treat them like quantile heads.
    if isinstance(all_out, tuple):
        mu, log_var = all_out
        all_out = head.to_quantiles(mu, log_var)  # (BC, n_ctx+m, Q, L)
    return all_out[:, n_ctx:, ...]               # (BC, m, ...)


def forecast_B1(backbone, head, x_context, horizon, device):
    """B1: Latent rollout, per-token decode, last chunk uses full output_len.

    Sequence: [e[0], ..., e[k], f[k+1], ..., f[k+m]]
    Forecast: W values per rolled position, except the last position
              uses up to output_len values to cover remaining horizon.
    """
    W_bb = backbone.W
    output_len = head.forecast_len

    e_ctx, mean_c, stdev_c, n_ctx, C = _b_variant_setup(
        backbone, x_context, device)

    with torch.no_grad():
        n_tokens = math.ceil(horizon / W_bb)                        # m
        rolled_f = rollout_latent(backbone, e_ctx, n_tokens)        # (BC, m, H)
        rolled_out = _b_variant_decode(head, e_ctx, rolled_f, n_ctx)  # (BC, m, output_len)

        preds = []
        remaining = horizon
        for i in range(rolled_out.size(1)):
            pred_norm = rolled_out[:, i, :]                         # (BC, output_len)
            if remaining <= output_len:
                n_take = remaining   # last chunk: use full capacity
            else:
                n_take = W_bb
            preds.append(_denormalize(pred_norm[:, :n_take], mean_c, stdev_c).cpu())
            remaining -= n_take
            if remaining <= 0:
                break

    return torch.cat(preds, dim=1).T.numpy()


def forecast_B2(backbone, head, x_context, horizon, device):
    """B2: Latent rollout, per-token decode, crop each to W.

    Sequence: [e[0], ..., e[k], f[k+1], ..., f[k+m]]
    Like B4 but uses a head with output_len=128, cropped to W per position.
    """
    W_bb = backbone.W

    e_ctx, mean_c, stdev_c, n_ctx, C = _b_variant_setup(
        backbone, x_context, device)

    with torch.no_grad():
        n_tokens = math.ceil(horizon / W_bb)                        # m
        rolled_f = rollout_latent(backbone, e_ctx, n_tokens)        # (BC, m, H)
        rolled_out = _b_variant_decode(head, e_ctx, rolled_f, n_ctx)  # (BC, m, output_len)

        preds = []
        remaining = horizon
        for i in range(rolled_out.size(1)):
            pred_norm = rolled_out[:, i, :]                         # (BC, output_len)
            n_take = min(W_bb, remaining)
            preds.append(_denormalize(pred_norm[:, :n_take], mean_c, stdev_c).cpu())
            remaining -= n_take
            if remaining <= 0:
                break

    return torch.cat(preds, dim=1).T.numpy()


def forecast_B3(backbone, head, x_context, horizon, device, recon_mode=None):
    """B3/B3R: Latent rollout, non-overlapping block decode.

    Sequence: [e[0], ..., e[k], f[k+1], ..., f[k+m]]
    Take FIRST position in each group of (output_len/W) tokens.
    Position 0 → p[k+1..k+output_len/W], position stride → next block, etc.
    """
    W_bb = backbone.W
    output_len = head.forecast_len
    stride = output_len // W_bb                                     # 8 for output_len=128

    e_ctx, mean_c, stdev_c, n_ctx, C = _b_variant_setup(
        backbone, x_context, device)

    with torch.no_grad():
        n_blocks = math.ceil(horizon / output_len)
        n_tokens = n_blocks * stride                                # m
        rolled_f = rollout_latent(backbone, e_ctx, n_tokens)        # (BC, m, H)
        rolled_out = _b_variant_decode(head, e_ctx, rolled_f, n_ctx)  # (BC, m, output_len)

        preds = []
        remaining = horizon
        for block_start in range(0, rolled_out.size(1), stride):
            if remaining <= 0:
                break
            pred_norm = rolled_out[:, block_start, :]               # (BC, output_len)
            n_take = min(output_len, remaining)
            preds.append(_denormalize(pred_norm[:, :n_take], mean_c, stdev_c).cpu())
            remaining -= n_take

    return torch.cat(preds, dim=1).T.numpy()


def forecast_B4(backbone, head, x_context, horizon, device):
    """B4: Latent rollout, per-token decode with W-value head.

    Sequence: [e[0], ..., e[k], f[k+1], ..., f[k+m]]
    W values from each rolled position. Simplest latent rollout strategy.

    For an MSE head, returns ``(horizon, C)``.
    For a QuantileForecastingHead, returns ``(num_quantiles, horizon, C)``
    so the eval predictor can build a real probabilistic forecast.
    """
    W_bb = backbone.W

    e_ctx, mean_c, stdev_c, n_ctx, C = _b_variant_setup(
        backbone, x_context, device)

    is_quantile = isinstance(
        head, (QuantileForecastingHead, LinearQuantileForecastingHead,
               TransformerQuantileForecastingHead,
               TransformerGaussianForecastingHead))

    with torch.no_grad():
        n_tokens = math.ceil(horizon / W_bb)                        # m
        rolled_f = rollout_latent(backbone, e_ctx, n_tokens)        # (BC, m, H)
        rolled_out = _b_variant_decode(head, e_ctx, rolled_f, n_ctx)
        # MSE head     : rolled_out = (BC, m, L)
        # Quantile head: rolled_out = (BC, m, Q, L)

        preds = []
        remaining = horizon
        for i in range(rolled_out.size(1)):
            pred_norm = rolled_out[:, i, ...]                       # (BC, L) or (BC, Q, L)
            n_take = min(head.forecast_len, remaining)
            if is_quantile:
                pred_norm = pred_norm[..., :n_take]                 # (BC, Q, n_take)
                # Denormalize per-quantile (broadcasting over Q):
                # mean_c, stdev_c are (BC, 1) — reshape to (BC, 1, 1) for Q-broadcast.
                m = mean_c.unsqueeze(-1)
                s = stdev_c.unsqueeze(-1).clamp(min=1e-5)
                pred_actual = pred_norm * s + m                     # (BC, Q, n_take)
                preds.append(pred_actual.cpu())
            else:
                pred_norm = pred_norm[:, :n_take]
                preds.append(_denormalize(pred_norm, mean_c, stdev_c).cpu())
            remaining -= n_take
            if remaining <= 0:
                break

    if is_quantile:
        out = torch.cat(preds, dim=-1)                              # (BC, Q, horizon)
        # (BC, Q, horizon) → (Q, horizon, C). BC = B*C, B=1 in eval.
        Q = out.size(1)
        BC = out.size(0)
        # Predictor calls eval with B=1 in our gluonts wrapper, so BC == C.
        return out.permute(1, 2, 0).numpy()                         # (Q, horizon, C)
    return torch.cat(preds, dim=1).T.numpy()                        # (horizon, C)


def forecast_B3R(backbone, head, x_context, horizon, device):
    """B3R: Same as B3, kept for backward compatibility."""
    return forecast_B3(backbone, head, x_context, horizon, device)


# Strategy dispatch
FORECAST_STRATEGIES = {
    'A1': forecast_A1,
    'A2': forecast_A2,
    'B1': forecast_B1,
    'B2': forecast_B2,
    'B3': forecast_B3,
    'B3R': forecast_B3R,
    'B4': forecast_B4,
}


def forecast_with_strategy(strategy, backbone, head, x_context, horizon, device):
    """Dispatch to the appropriate forecast strategy.

    Args:
        strategy: one of 'A1', 'A2', 'B1', 'B2', 'B3', 'B3R', 'B4'
        backbone, head, x_context, horizon, device: same as forecast_autoregressive
    """
    fn = FORECAST_STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(FORECAST_STRATEGIES)}")
    return fn(backbone, head, x_context, horizon, device)
