"""Diagnostic metrics for the contrastive-forecasting backbone.

All metrics are computed without gradient. They probe whether the
forecaster has learned a useful representation by comparing its
predictions against simple baselines (random pair, naive last-step,
past-window negatives) and by checking how much of the latent
hidden-dim space is actually used.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def q_random(f: Tensor, h_target: Tensor) -> Tensor:
    """Forecast cosine-error vs random-pair baseline; averaged over (c, t).

    Caller passes already-shifted target (e.g. f = f[:, :-1], h_target = h[:, 1:]).
    """
    assert f.shape == h_target.shape, (f.shape, h_target.shape)
    B = f.shape[0]
    # Cyclic-shift derangement: guarantees perm[b] != b (avoids the
    # zero-pair denominator bias when B is small).
    shift = int(torch.randint(1, B, (1,)).item()) if B > 1 else 0
    perm = (torch.arange(B, device=f.device) + shift) % B
    num = (1.0 - F.cosine_similarity(f, h_target, dim=-1, eps=1e-8)).mean(dim=0)
    denom = (1.0 - F.cosine_similarity(h_target, h_target[perm], dim=-1, eps=1e-8)).mean(dim=0)
    return (num / denom.clamp_min(1e-8)).mean()


@torch.no_grad()
def q_naive_latent(f: Tensor, h_target: Tensor, h_prev: Tensor) -> Tensor:
    """Same numerator as q_random; denominator is naive 'latent doesn't change' baseline."""
    assert f.shape == h_target.shape == h_prev.shape
    num = (1.0 - F.cosine_similarity(f, h_target, dim=-1, eps=1e-8)).mean(dim=0)
    denom = (1.0 - F.cosine_similarity(h_target, h_prev, dim=-1, eps=1e-8)).mean(dim=0)
    return (num / denom.clamp_min(1e-8)).mean()


@torch.no_grad()
def dim_usage(z: Tensor, axis: int) -> Tensor:
    """Dimension-usage U = 1 / (d * mean_{i!=j} cos²(z_i, z_j)), clipped to [0, 1].

    ``axis`` is the n axis (samples to compare); last axis is the feature
    dim d. Other axes are slice axes — U is computed per slice and averaged.
    """
    if axis < 0:
        axis += z.ndim
    assert axis != z.ndim - 1, "axis cannot be the feature dim"

    z = z.movedim(axis, 0)
    n = z.shape[0]
    d = z.shape[-1]

    z_norm = F.normalize(z, p=2, dim=-1, eps=1e-12)
    fixed_shape = z_norm.shape[1:-1]
    flat = z_norm.reshape(n, -1, d).permute(1, 0, 2)   # (S, n, d)
    sim = torch.matmul(flat, flat.transpose(-1, -2))   # (S, n, n)
    sq = sim.pow(2)
    # mean over off-diagonal: (sum_all - sum_diag) / (n*(n-1))
    sum_all = sq.sum(dim=(-1, -2))
    sum_diag = torch.diagonal(sq, dim1=-2, dim2=-1).sum(dim=-1)
    off_mean = (sum_all - sum_diag) / (n * (n - 1))
    u_per_slice = (1.0 / (d * off_mean.clamp_min(1e-12))).clamp_max(1.0)
    if fixed_shape:
        u_per_slice = u_per_slice.reshape(fixed_shape)
    return u_per_slice.mean()


@torch.no_grad()
def retrieval_auc_top1(
    f: Tensor,
    h_full: Tensor,
    lookback_lags: tuple[int, ...] = (1, 2, 4, 8),
) -> tuple[Tensor, Tensor]:
    """Retrieval AUC + Top-1 against past-window negatives.

    For each query (b, t, c) with t >= max(lookback_lags):
        positive  = h_full[b, t+1, c, :]
        negatives = [ h_full[b, t-k, c, :] for k in lookback_lags ]
    Score by cosine sim against f[b, t, c, :]. Per-query AUC = (# negs
    the positive beats) / len(lags); Top1 = 1 iff positive beats all.

    Args:
        f: ``(B, T, C, H)``.
        h_full: ``(B, T+1, C, H)`` — must include position t+1.
        lookback_lags: lags k for past-window negatives.

    Returns:
        (auc, top1), both scalar tensors. NaN if no valid queries.
    """
    B, T, C, H = f.shape
    max_lag = max(lookback_lags)
    if T <= max_lag or T + 1 > h_full.shape[1]:
        nan = torch.full((), float('nan'), device=f.device, dtype=f.dtype)
        return nan, nan

    f_v = f[:, max_lag:T, :, :]                               # (B, T_v, C, H)
    pos = h_full[:, max_lag + 1:T + 1, :, :]
    sim_pos = F.cosine_similarity(f_v, pos, dim=-1)           # (B, T_v, C)

    sims_neg = [
        F.cosine_similarity(f_v, h_full[:, max_lag - k:T - k, :, :], dim=-1)
        for k in lookback_lags
    ]
    sim_neg = torch.stack(sims_neg, dim=-1)                   # (B, T_v, C, n_neg)

    beats = (sim_pos.unsqueeze(-1) > sim_neg).float()
    auc = beats.mean(dim=-1).mean()
    top1 = beats.prod(dim=-1).mean()
    return auc, top1
