"""
Reversible Exponential Weighted Moving Normalization (RevEWMNorm).

Inspired by RevIN (ICLR 2022), adapted for contrastive forecasting
with first-patch initialization to avoid cold-start spikes.

Input/output shape: [B, T, C] (batch, time, channels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import _autocast_ctx


def compute_patch_stats(mean: torch.Tensor, stdev: torch.Tensor,
                        W: int, kind: str = 'diff',
                        eps: float = 1e-5) -> torch.Tensor | None:
    """Compute per-patch summary statistics for the encoder.

    The reversible normalisation strips off the running mean/std from
    the input before patching, so the per-patch encoder loses the
    *absolute* level/scale information. This helper produces a small
    fixed-size feature per patch that the encoder can use to recover
    that information without breaking the reversibility property.

    Args:
        mean: per-step EMA mean of shape ``[B, T_raw, C]``. Typically
            ``rev_norm.mean`` after the forward pass.
        stdev: per-step EMA std of the same shape.
        W: patch size (16 in the canonical config).
        kind: which feature to emit.

            ``'none'``
                Return ``None`` — the model should not concat any stats.

            ``'diff'``
                Two scale-free per-patch features (returned shape
                ``[B, T_patches, C, 2]``):

                - ``dmean[t] = (mean_p[t] - mean_p[t-1]) / std_p[t-1]``
                  — how the local level shifted between adjacent patches,
                  measured in standard-deviation units. Bounded under
                  reasonable assumptions and naturally stationary.
                - ``dlogstd[t] = log(std_p[t]) - log(std_p[t-1])``
                  — log-ratio of the per-patch volatility. Symmetric
                  around 0 and stationary.

                Both features are zero at ``t = 0`` (no previous patch).

            ``'raw'``
                Two centred per-patch features (mean and ``log(std)``,
                each centred per-(batch, channel) over the time axis).
                Useful as an ablation against ``'diff'``. Shape
                ``[B, T_patches, C, 2]``.

        eps: stability constant for log/division.

    Returns:
        ``None`` when ``kind == 'none'``, else a tensor shaped
        ``[B, T_patches, C, 2]`` ready to concat to the patch features.

    Notes:
        - We average the per-step EMA stats across each patch's W
          timesteps. For the 16-step EWMA span=32 setting the EMA
          barely changes within a patch, so this is essentially the
          mid-patch value, but it is well-defined for any span.
        - When ``rev_norm`` is RevIN (single per-series mean/std),
          ``mean`` and ``stdev`` are constant along T after broadcast,
          so the resulting diffs/centred values are all 0. This module
          therefore carries zero information for RevIN — callers should
          guard against that combination.
    """
    if kind == 'none':
        return None
    if kind not in {'diff', 'raw'}:
        raise ValueError(f"Unknown patch_stats kind {kind!r}: expected one of "
                         "'none', 'diff', 'raw'")
    B, T_raw, C = mean.shape
    if T_raw % W != 0:
        raise ValueError(f"T_raw={T_raw} must be a multiple of W={W}")
    T_p = T_raw // W

    # Per-patch averages of the EMA stats.
    mean_p = mean.view(B, T_p, W, C).mean(dim=2)             # [B, T_p, C]
    std_p = stdev.view(B, T_p, W, C).mean(dim=2)             # [B, T_p, C]

    if kind == 'diff':
        # dmean[t] = (mean_p[t] - mean_p[t-1]) / std_p[t-1]
        dmean = (mean_p[:, 1:] - mean_p[:, :-1]) / std_p[:, :-1].clamp(min=eps)
        # F.pad pads the trailing dims. For [B, T_p-1, C] we want to pad
        # the T axis (dim=1) on the LEFT with one zero row, leaving the
        # C axis untouched. The pad spec for dim=-2 is (left, right).
        dmean = F.pad(dmean, (0, 0, 1, 0), 'constant', 0.0)  # [B, T_p, C]

        log_std = torch.log(std_p.clamp(min=eps))
        dlogstd = log_std[:, 1:] - log_std[:, :-1]
        dlogstd = F.pad(dlogstd, (0, 0, 1, 0), 'constant', 0.0)
        feat = torch.stack([dmean, dlogstd], dim=-1)         # [B, T_p, C, 2]
    else:  # 'raw'
        # Centre per-(B, C) so the absolute level (which is unbounded)
        # doesn't dominate. log_std is naturally bounded for sane data
        # but we still centre for symmetry.
        mean_centered = mean_p - mean_p.mean(dim=1, keepdim=True)
        log_std = torch.log(std_p.clamp(min=eps))
        log_std_centered = log_std - log_std.mean(dim=1, keepdim=True)
        feat = torch.stack([mean_centered, log_std_centered], dim=-1)

    return feat


# Number of features added per patch when patch-stats is enabled.
PATCH_STATS_DIM = 2


# ── Left zero padding (#419) ────────────────────────────────────────────────
#
# The GiftEvalPretrain loader pads a series shorter than the window with
# zeros on the left. On such a window the plain EWMA starts from a first
# patch of zeros (mean 0, variance 0), stays at 0 over the padding, and then
# meets the first real value with a spread built from zeros. At span 128 the
# first value of any level normalises to sqrt((1 - a) / a), about +8, and a
# 20-point series stays far above 0 to its end. The helpers below let the
# statistics start at the first nonzero value instead.


def patch_padding(pad_mask: torch.Tensor, W: int) -> torch.Tensor:
    """``[B, T_raw, C]`` padded values to ``[B, T_raw // W, C]`` patches:
    True where the whole patch is padding. A patch that holds the first real
    value is real."""
    B, T_raw, C = pad_mask.shape
    return pad_mask.reshape(B, T_raw // W, W, C).all(dim=2)


def leading_zero_count(x: torch.Tensor) -> torch.Tensor:
    """Exact zeros before the first nonzero value: ``[B, T, C]`` → ``[B, 1, C]``.

    An all-zero series counts ``T``.
    """
    nonzero = x != 0
    first = nonzero.float().argmax(dim=1, keepdim=True)
    full = torch.full_like(first, x.shape[1])
    return torch.where(nonzero.any(dim=1, keepdim=True), first, full)


def zero_union_padding(values: torch.Tensor,
                       *sources: torch.Tensor) -> torch.Tensor:
    """``values`` with the leading padding of each row set to exactly 0.
    The padding of a row is the longest padding of that row in ``sources``,
    their union (#421).

    A transform that builds a row from several rows (mixup, a crossfade)
    calls it, so it never writes values into padding: its values exist only
    where every source row holds real values. All tensors are ``[B, T, C]``.
    A row with no padding in any source keeps its values.
    """
    z = torch.stack([leading_zero_count(s) for s in sources]).amax(dim=0)
    t = torch.arange(values.shape[1], device=values.device).view(1, -1, 1)
    return values.masked_fill(t < z, 0.0)


def _masked_first_patch(xs: torch.Tensor, n_real: torch.Tensor, W: int):
    """Mean and variance of the first ``min(W, n_real)`` values of ``xs``.

    ``xs`` is ``[B, T, C]`` with the real values first, and ``n_real`` is
    ``[B, 1, C]``. A series with no real value gets mean 0 and variance 0,
    as the plain first patch of an all-zero series does.
    """
    head = xs[:, :W]
    at = torch.arange(head.shape[1], device=xs.device).view(1, -1, 1)
    keep = (at < n_real).to(xs.dtype)
    count = keep.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (head * keep).sum(dim=1, keepdim=True) / count
    var = ((head - mean) ** 2 * keep).sum(dim=1, keepdim=True) / count
    return mean, var


def _ewm_statistics(x64: torch.Tensor, init_mean: torch.Tensor,
                    init_var: torch.Tensor, alpha: float):
    """The EWMA mean and variance of :class:`RevEWMNorm` from a given start,
    by the same cumulative-sum form, in float64."""
    T = x64.shape[1]
    steps = torch.arange(T, device=x64.device, dtype=torch.float64)
    decay = ((1.0 - alpha) ** steps).view(1, T, 1)
    inv_decay, shift = 1.0 / decay, decay * (1.0 - alpha)
    mean = alpha * torch.cumsum(x64 * inv_decay, dim=1) * decay
    mean = mean + shift * init_mean
    var = alpha * torch.cumsum((x64 - mean) ** 2 * inv_decay, dim=1) * decay
    return mean, var + shift * init_var


class RevEWMNorm(nn.Module):
    """Reversible EWM normalization with first-patch initialization.

    Computes per-channel, per-timestep exponential weighted moving mean and
    standard deviation. The EMA is initialized from the statistics of the
    first patch (first ``patch_size`` timesteps) to avoid the cold-start
    problem where starting from zeros causes an initial spike.

    Args:
        num_features: Number of channels (C).
        span: EMA span parameter. ``alpha = 2 / (span + 1)``.
        patch_size: Number of timesteps in the first patch used for
            initializing EMA statistics.
        eps: Small constant for numerical stability.
        affine: If True, adds learnable scale and bias after normalization.
        skip_leading_zeros: If True (#419), the zeros before the first
            nonzero value count as left padding. The statistics start at the
            first nonzero value, as if the padding were not there, and the
            padded positions stay 0 after normalisation. ``pad_mask`` then
            marks them. The mode is recorded in the state dict as the
            buffer ``leading_zero_pad``, so a loader can rebuild it.
    """

    def __init__(self, num_features: int, span: float, patch_size: int,
                 eps: float = 1e-5, affine: bool = False,
                 patch_emb_dtype: str = "fp32",
                 skip_leading_zeros: bool = False):
        super().__init__()
        self.num_features = num_features
        self.span = span
        self.patch_size = patch_size
        self.eps = eps
        self.alpha = 2.0 / (span + 1.0)
        self.affine = affine
        # Dtype for the output cast of this module. The internal cumsum
        # statistics math is always fp64 (see _compute_statistics); only the
        # final cast and downstream arithmetic in the normalise/denormalise
        # paths are governed by this knob.
        self.patch_emb_dtype = patch_emb_dtype

        # Stored statistics (set during 'norm', used during 'denorm')
        self.mean = None
        self.stdev = None
        # #419: the left-padding mode, and the padded positions of the last
        # 'norm' call. None when the mode is off.
        self.skip_leading_zeros = bool(skip_leading_zeros)
        self.pad_mask = None
        if self.skip_leading_zeros:
            self.register_buffer("leading_zero_pad",
                                 torch.ones((), dtype=torch.bool))

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape ``[B, T, C]``.
            mode: ``'norm'`` to normalize, ``'denorm'`` to denormalize.

        Returns:
            Tensor of the same shape as ``x``.
        """
        # Run the normalise/denormalise math under the chosen patch-emb
        # autocast context. fp32 = disabled autocast (no-op);
        # fp16/bf16 = enabled at that dtype. The fp64 cumsum inside
        # `_compute_statistics` is unaffected (autocast doesn't downcast
        # explicit `.to(float64)` operations).
        with _autocast_ctx(self.patch_emb_dtype):
            if mode == 'norm' and self.skip_leading_zeros:
                self._compute_statistics_skip_zeros(x)
                return self._normalize(x).masked_fill(self.pad_mask, 0.0)
            if mode == 'norm':
                self._compute_statistics(x)
                return self._normalize(x)
            elif mode == 'denorm':
                if self.mean is None or self.stdev is None:
                    raise RuntimeError(
                        "Cannot denormalize before normalizing. Call with mode='norm' first.")
                return self._denormalize(x)
            else:
                raise ValueError(f"Unknown mode '{mode}'. Expected 'norm' or 'denorm'.")

    def _compute_statistics(self, x: torch.Tensor):
        """Compute EMA mean and std for each timestep, initialized from first patch.

        Args:
            x: ``[B, T, C]``
        """
        B, T, C = x.shape
        alpha = self.alpha
        W = min(self.patch_size, T)

        # Initialize EMA from first patch statistics (float64 to handle extreme values)
        first_patch_64 = x[:, :W, :].to(torch.float64)  # [B, W, C]
        ema_mean_init = first_patch_64.mean(dim=1, keepdim=True)  # [B, 1, C]
        ema_var_init = first_patch_64.var(dim=1, keepdim=True, unbiased=False)  # [B, 1, C]

        # Build EMA weights: weights[t] = (1-alpha)^(T-1-t) for cumsum trick
        # We compute in float64 for numerical stability then cast back
        device = x.device
        dtype = x.dtype

        arange = torch.arange(T, device=device, dtype=torch.float64)
        # For the cumsum approach:
        # ema[t] = alpha * sum_{k=0}^{t} (1-alpha)^{t-k} * x[k] + (1-alpha)^{t+1} * init
        # = alpha * [(1-alpha)^t * x[0] + (1-alpha)^{t-1} * x[1] + ... + x[t]] + (1-alpha)^{t+1} * init

        # decay[t] = (1-alpha)^t
        decay = (1.0 - alpha) ** arange  # [T]
        # decay_shift[t] = (1-alpha)^{t+1} (for init term)
        decay_shift = decay * (1.0 - alpha)  # [T]

        # Weighted cumsum for mean
        x_64 = x.to(torch.float64)  # [B, T, C]
        init_mean_64 = ema_mean_init  # already float64

        # weights_for_x[t, k] = (1-alpha)^{t-k} for k <= t
        # sum = cumsum of (x[k] * (1-alpha)^{-k}) * (1-alpha)^t
        # Rewrite: weighted_x[k] = x[k] / decay[k], then cumsum * decay gives the sum
        # NOTE: inv_decay can overflow for very long T or large spans.
        # For T=4096, span=300 the peak is ~5e11 (safe in float64).
        # If needed, add chunking for longer sequences.
        inv_decay = 1.0 / decay  # [T]
        inv_decay = inv_decay.view(1, T, 1)  # [1, T, 1]
        decay_bc = decay.view(1, T, 1)  # [1, T, 1]
        decay_shift_bc = decay_shift.view(1, T, 1)  # [1, T, 1]

        # EMA mean
        weighted_x = x_64 * inv_decay
        cumsum_wx = torch.cumsum(weighted_x, dim=1)
        ema_sum = alpha * cumsum_wx * decay_bc  # alpha * sum_{k=0}^{t} (1-a)^{t-k} * x[k]
        ema_mean = ema_sum + decay_shift_bc * init_mean_64  # add init contribution

        # EMA variance
        init_var_64 = ema_var_init  # already float64
        residuals_sq = (x_64 - ema_mean) ** 2  # [B, T, C]
        weighted_rsq = residuals_sq * inv_decay
        cumsum_wrsq = torch.cumsum(weighted_rsq, dim=1)
        ema_var_sum = alpha * cumsum_wrsq * decay_bc
        ema_var = ema_var_sum + decay_shift_bc * init_var_64

        self.mean = ema_mean.to(dtype).detach()  # [B, T, C]
        self.stdev = torch.sqrt(ema_var).to(dtype).detach()  # [B, T, C]

    def _compute_statistics_skip_zeros(self, x: torch.Tensor):
        """Statistics from the real values only (#419).

        Each series is shifted left past its leading zeros, the plain
        first-patch EWMA runs on what remains, and the result is shifted
        back. A padded position keeps the statistics of the first real
        value. Its normalised value is 0 whatever they are.
        """
        T = x.shape[1]
        z = leading_zero_count(x)                                # [B, 1, C]
        t = torch.arange(T, device=x.device).view(1, T, 1)
        xs = x.gather(1, (t + z).clamp(max=T - 1)).to(torch.float64)
        init_mean, init_var = _masked_first_patch(xs, T - z, self.patch_size)
        mean, var = _ewm_statistics(xs, init_mean, init_var, self.alpha)
        back = (t - z).clamp(min=0)
        self.mean = mean.gather(1, back).to(x.dtype).detach()
        self.stdev = torch.sqrt(var.gather(1, back)).to(x.dtype).detach()
        self.pad_mask = t < z

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        x = x / self.stdev.clamp(min=self.eps)
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev.clamp(min=self.eps)
        x = x + self.mean
        return x


class RevIN(nn.Module):
    """Standard reversible instance normalisation (Kim and others, ICLR 2022).

    A single per-instance, per-channel mean+std is computed over the entire
    context window, then subtracted/divided away before the backbone and
    re-applied during denormalisation.

    Unlike :class:`RevEWMNorm`, the normalisation is *static* across the
    time axis (one mean/std per (B, C)) — there's no rolling-mean
    interaction with the periodic signal that can dampen amplitude.

    Args:
        num_features: Number of channels (C).
        eps: Stability constant.
        affine: If True, adds learnable per-channel scale/bias.

    Shape conventions (matching RevEWMNorm so it's a drop-in replacement):
        Input/output: ``[B, T, C]``.
        Stored stats: ``mean``, ``stdev`` are ``[B, 1, C]`` after ``norm``.
    """

    def __init__(self, num_features: int, eps: float = 1e-5,
                 affine: bool = False, **_unused):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.mean = None
        self.stdev = None
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            # Per-(B, C) statistics over the full T axis.
            mean = x.mean(dim=1, keepdim=True)                  # [B, 1, C]
            var = x.var(dim=1, keepdim=True, unbiased=False)    # [B, 1, C]
            self.mean = mean.detach()
            self.stdev = torch.sqrt(var + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.mean is None or self.stdev is None:
                raise RuntimeError(
                    "Cannot denormalize before normalizing. Call with mode='norm' first.")
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            return x * self.stdev + self.mean
        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'norm' or 'denorm'.")


# ── The mean/std scaling of Moirai 1.0 (#421) ───────────────────────────────
#
# uni2ts `PackedStdScaler` (uni2ts/module/packed_scaler.py) gives each series
# window one loc and one scale, from its observed context values, in float64:
#
#   n     = the count of observed context values
#   loc   = sum(x) / n
#   var   = sum((x - loc)^2) / (n - 1)        (correction 1)
#   scale = sqrt(var + 1e-5)                  (minimum_scale 1e-5)
#
# Its `safe_div` divides by 1 where a count is 0. So a context with no
# observed value gets loc 0 and scale sqrt(1e-5), and a context with one
# observed value gets loc = that value and scale sqrt(1e-5). Moirai reads the
# observed values outside the prediction range, and scales the whole window
# with the result.

MEAN_STD_CORRECTION = 1
MEAN_STD_MINIMUM_SCALE = 1e-5


def safe_div(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """uni2ts ``safe_div``: ``numer / denom``, with 1 where ``denom`` is 0."""
    return numer / torch.where(denom == 0, torch.ones_like(denom), denom)


def mean_std_statistics(x: torch.Tensor, observed: torch.Tensor,
                        correction: int = MEAN_STD_CORRECTION,
                        minimum_scale: float = MEAN_STD_MINIMUM_SCALE):
    """``(loc, scale)`` of uni2ts ``PackedStdScaler``, one pair per series.

    ``x`` is ``[B, T, C]``, and ``observed`` (bool, the same shape) marks
    the values the statistics read. The sums run in float64, and the result
    is float32, as in uni2ts. Each output is ``[B, 1, C]``.
    """
    x64 = x.to(torch.float64)
    obs = observed.to(torch.float64)
    n = obs.sum(dim=1, keepdim=True)
    loc = safe_div((x64 * obs).sum(dim=1, keepdim=True), n)
    var = safe_div((((x64 - loc) ** 2) * obs).sum(dim=1, keepdim=True),
                   n - correction)
    return loc.float(), torch.sqrt(var + minimum_scale).float()


class RevMeanStdNorm(nn.Module):
    """The mean/std scaling of Moirai 1.0 (#421).

    One loc and one scale per series window (per row and channel), from the
    observed values of its context (:func:`mean_std_statistics`). The whole
    window is normalised with them, so a value after the context changes
    neither the statistics nor a normalised context value.

    ``forward(x, 'norm', context_end)``: ``context_end`` (``[B, 1, C]``, a
    value index) ends the context of each window. None: the whole window is
    context, as at inference.

    Args:
        num_features: Number of channels (C).
        skip_leading_zeros: As in :class:`RevEWMNorm` (#419). The zeros
            before the first nonzero value are left padding: the statistics
            skip them, they stay 0 after normalisation, and ``pad_mask``
            marks them. The buffer ``leading_zero_pad`` records the mode.

    The buffer ``mean_std_scaling`` names the kind in the state dict, so the
    eval and the backbone loader rebuild it from a checkpoint.
    """

    def __init__(self, num_features: int, skip_leading_zeros: bool = False):
        super().__init__()
        self.num_features = num_features
        self.skip_leading_zeros = bool(skip_leading_zeros)
        # The statistics and the padding of the last 'norm' call.
        self.mean = self.stdev = self.pad_mask = None
        self.register_buffer("mean_std_scaling",
                             torch.ones((), dtype=torch.bool))
        if self.skip_leading_zeros:
            self.register_buffer("leading_zero_pad",
                                 torch.ones((), dtype=torch.bool))

    def forward(self, x: torch.Tensor, mode: str,
                context_end: torch.Tensor | None = None) -> torch.Tensor:
        if mode == 'norm':
            return self._normalize(x, context_end)
        if mode == 'denorm':
            if self.mean is None or self.stdev is None:
                raise RuntimeError(
                    "Cannot denormalize before normalizing. Call with mode='norm' first.")
            return x * self.stdev + self.mean
        raise ValueError(f"Unknown mode '{mode}'. Expected 'norm' or 'denorm'.")

    def _normalize(self, x: torch.Tensor, context_end):
        t = torch.arange(x.shape[1], device=x.device).view(1, -1, 1)
        pad = (t < leading_zero_count(x) if self.skip_leading_zeros
               else torch.zeros_like(x, dtype=torch.bool))
        observed = ~pad if context_end is None else ~pad & (t < context_end)
        loc, scale = mean_std_statistics(x, observed)
        self.mean, self.stdev = loc.detach(), scale.detach()
        self.pad_mask = pad if self.skip_leading_zeros else None
        return ((x - self.mean) / self.stdev).masked_fill(pad, 0.0)
