"""
Reversible Exponential Weighted Moving Normalization (RevEWMNorm).

Inspired by RevIN (ICLR 2022) and RevEWMSTDN, adapted for contrastive
forecasting with first-patch initialization to avoid cold-start spikes.

Input/output shape: [B, T, C] (batch, time, channels).
"""

import torch
import torch.nn as nn


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

        Args:
            x: ``[B, T, C]``
        """
        B, T, C = x.shape
        alpha = self.alpha
        W = min(self.patch_size, T)

        # Initialize EMA from first patch statistics
        first_patch = x[:, :W, :]  # [B, W, C]
        ema_mean_init = first_patch.mean(dim=1, keepdim=True)  # [B, 1, C]
        ema_var_init = first_patch.var(dim=1, keepdim=True, unbiased=False)  # [B, 1, C]

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
        init_mean_64 = ema_mean_init.to(torch.float64)  # [B, 1, C]

        # weights_for_x[t, k] = (1-alpha)^{t-k} for k <= t
        # sum = cumsum of (x[k] * (1-alpha)^{-k}) * (1-alpha)^t
        # Rewrite: weighted_x[k] = x[k] / decay[k], then cumsum * decay gives the sum
        # NOTE: inv_decay can overflow for very long T or large spans.
        # For T=4096, span=300 the peak is ~5e11 (safe in float64).
        # If needed, add chunking as in RevEWMSTDN.
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
        init_var_64 = ema_var_init.to(torch.float64)  # [B, 1, C]
        residuals_sq = (x_64 - ema_mean) ** 2  # [B, T, C]
        weighted_rsq = residuals_sq * inv_decay
        cumsum_wrsq = torch.cumsum(weighted_rsq, dim=1)
        ema_var_sum = alpha * cumsum_wrsq * decay_bc
        ema_var = ema_var_sum + decay_shift_bc * init_var_64

        self.mean = ema_mean.to(dtype).detach()  # [B, T, C]
        self.stdev = torch.sqrt(ema_var).to(dtype).detach()  # [B, T, C]

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
