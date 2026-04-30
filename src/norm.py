"""
Reversible Exponential Weighted Moving Normalization (RevEWMNorm).

Inspired by RevIN (ICLR 2022), adapted for contrastive forecasting
with first-patch initialization to avoid cold-start spikes.

Input/output shape: [B, T, C] (batch, time, channels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
          ``mean`` and ``stdev`` are constant along T after broadcast;
          the resulting diffs/centred values are all 0. This module
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
        # F.pad pads the trailing dims; for [B, T_p-1, C] we want to pad
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
    """

    def __init__(self, num_features: int, span: float, patch_size: int,
                 eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.span = span
        self.patch_size = patch_size
        self.eps = eps
        self.alpha = 2.0 / (span + 1.0)
        self.affine = affine

        # Stored statistics (set during 'norm', used during 'denorm')
        self.mean = None
        self.stdev = None

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

        Vectorised via the cumsum trick: rewrite the EMA recurrence as a
        prefix sum of ``x[k] / (1-alpha)^k`` rescaled by ``(1-alpha)^t``.
        Stays in ``x.dtype`` (typically float32) for ~3-4× GPU speedup
        over the previous float64 implementation, with cached
        decay/inv_decay/decay_shift buffers per (T, alpha, device).

        Float32 is safe for ``span >= 64`` and ``T <= 1024`` with bounded
        inputs (typical |x| < ~100). For T >= 2048 OR when |x| can exceed
        ~1000, the cumsum-trick intermediates overflow float32: the term
        ``residuals_sq[k] / (1-alpha)^k`` grows like ``|x|^2 *
        exp(2*alpha*T)`` and a sum of T such terms can overshoot 3.4e38
        (float32 max). The 4096-window gift-pretrain-small-4096 dataset
        has rows with |x| up to ~1e5; running the float32 path on it
        NaN'd the loss at step 1697 of a 30k run. So we promote to
        float64 inside the cumsum when ``T > 2048``, paying a small
        speed cost (one extra cast + 8-byte-wide reduction) for
        numerical stability. The float32 path is preserved for the
        more common T=1024 case used throughout phases 1–5.

        Args:
            x: ``[B, T, C]``
        """
        B, T, C = x.shape
        alpha = self.alpha
        W = min(self.patch_size, T)
        device, dtype = x.device, x.dtype

        # Initialise from first-patch statistics (same dtype as x).
        first_patch = x[:, :W, :]
        ema_mean_init = first_patch.mean(dim=1, keepdim=True)
        ema_var_init = first_patch.var(dim=1, keepdim=True, unbiased=False)

        # Cache decay/inv_decay/decay_shift buffers keyed on
        # (T, alpha, device, work_dtype). Recomputed only when the key
        # changes. work_dtype is float64 for long sequences to bound
        # cumsum-trick intermediates safely; otherwise the input dtype.
        work_dtype = torch.float64 if T > 2048 else dtype
        cache_key = (T, alpha, device, dtype, work_dtype)
        if getattr(self, "_cache_key", None) != cache_key:
            arange = torch.arange(T, device=device, dtype=work_dtype)
            decay = (1.0 - alpha) ** arange                          # [T]
            self._cache_decay = decay.view(1, T, 1)
            self._cache_inv_decay = (1.0 / decay).view(1, T, 1)
            self._cache_decay_shift = (decay * (1.0 - alpha)).view(1, T, 1)
            self._cache_key = cache_key
        decay = self._cache_decay
        inv_decay = self._cache_inv_decay
        decay_shift = self._cache_decay_shift

        # Promote x to work_dtype only for the cumsum trick. Cast back
        # before storing self.mean / self.stdev so downstream stays in
        # the input dtype.
        if work_dtype != dtype:
            x_w = x.to(work_dtype)
            ema_mean_init_w = ema_mean_init.to(work_dtype)
            ema_var_init_w = ema_var_init.to(work_dtype)
        else:
            x_w = x
            ema_mean_init_w = ema_mean_init
            ema_var_init_w = ema_var_init

        # EMA mean via cumsum trick.
        weighted_x = x_w * inv_decay
        cumsum_wx = torch.cumsum(weighted_x, dim=1)
        ema_mean = alpha * cumsum_wx * decay + decay_shift * ema_mean_init_w

        # EMA variance via the same trick on (x - mean)^2.
        residuals_sq = (x_w - ema_mean) ** 2
        weighted_rsq = residuals_sq * inv_decay
        cumsum_wrsq = torch.cumsum(weighted_rsq, dim=1)
        ema_var = alpha * cumsum_wrsq * decay + decay_shift * ema_var_init_w

        # Cast back to input dtype before storing.
        self.mean = ema_mean.to(dtype).detach()              # [B, T, C]
        self.stdev = torch.sqrt(ema_var.clamp(min=1e-12)).to(dtype).detach()

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
    """Standard reversible instance normalisation (Kim et al., ICLR 2022).

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
