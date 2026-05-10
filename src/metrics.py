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


def _err(a: Tensor, b: Tensor, eps: float) -> Tensor:
    # 1 - cosine, computed on the last (feature) axis.
    return 1.0 - F.cosine_similarity(a, b, dim=-1, eps=eps)


@torch.no_grad()
def q_random(f: Tensor, h_target: Tensor, eps: float = 1e-8) -> Tensor:
    """Q_random = mean_{c,t} [ mean_b e(f, h_target) / mean_b e(h_target, h_target[perm]) ].

    e(a, b) = 1 - cos_sim(a, b). Caller passes the already-shifted
    target — i.e. for forecaster latents f at position t, pass
    h_target = h[:, 1:T+1, ...] so f[..., t, ...] aligns with
    h_target[..., t, ...] = h[..., t+1, ...].

    Args:
        f, h_target: tensors with the same shape ``(B, ..., H)``. Batch
            is the first axis; H is the last. Common shapes are
            ``(B, T, C, H)`` and ``(B*C, T, H)`` — both work.

    Returns:
        Scalar tensor.

    Example:
        >>> # f, h: (B, T, C, H) backbone outputs
        >>> q = q_random(f[:, :-1], h[:, 1:])
    """
    assert f.shape == h_target.shape, (f.shape, h_target.shape)
    B = f.shape[0]
    perm = torch.randperm(B, device=f.device)
    num = _err(f, h_target, eps).mean(dim=0)                          # (...,)
    denom = _err(h_target, h_target[perm], eps).mean(dim=0).clamp_min(eps)
    return (num / denom).mean()


@torch.no_grad()
def q_naive_latent(
    f: Tensor, h_target: Tensor, h_prev: Tensor, eps: float = 1e-8
) -> Tensor:
    """Q_naive_latent: same numerator as q_random, denominator uses
    naive 'latent doesn't change' baseline e(h_target, h_prev).

    For backbone latents h of shape ``(B, T, C, H)``:
        h_target = h[:, 1:T+1, ...]
        h_prev   = h[:, 0:T,   ...]
    """
    assert f.shape == h_target.shape == h_prev.shape
    num = _err(f, h_target, eps).mean(dim=0)
    denom = _err(h_target, h_prev, eps).mean(dim=0).clamp_min(eps)
    return (num / denom).mean()


@torch.no_grad()
def dim_usage(z: Tensor, axis: int) -> Tensor:
    """Dimension-usage U = 1 / (d * mean_{i!=j} cos²(z_i, z_j)), clipped to [0, 1].

    The chosen ``axis`` is the n axis (samples to compare); last axis is
    the feature dim d. All remaining axes are 'fixed slice' axes — U is
    computed per slice and the slices are averaged into a scalar.

    For isotropic random z, mean cos² ≈ 1/d so U → 1.
    For collinear z, cos² = 1 so U → 1/d.
    """
    if axis < 0:
        axis += z.ndim
    assert axis != z.ndim - 1, "axis cannot be the feature dim"

    # Move the n-axis to position 0, leaving (n, *fixed, d).
    z = z.movedim(axis, 0)
    n = z.shape[0]
    d = z.shape[-1]
    if n < 2:
        return torch.ones((), device=z.device, dtype=z.dtype)

    z_norm = F.normalize(z, p=2, dim=-1, eps=1e-12)
    # Flatten fixed axes for batched matmul; restore at end.
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


def u_batch(z: Tensor) -> Tensor:
    """U with batch as the n axis (default: axis 0)."""
    return dim_usage(z, axis=0)


def u_temporal(z: Tensor, time_axis: int) -> Tensor:
    """U with time as the n axis. ``time_axis`` is caller-supplied
    because different call sites store time in different positions."""
    return dim_usage(z, axis=time_axis)


@torch.no_grad()
def retrieval_auc_top1_legacy(
    f: Tensor,
    h_full: Tensor,
    lookback_lags: tuple[int, ...] = (1, 2, 4, 8),
) -> tuple[Tensor, Tensor]:
    """LEGACY: AUC + Top-1 against TEMPORAL-ONLY negatives, kept for
    historical-CSV reproduction only.

    New eval pipelines must use :func:`retrieval_auc_topk` instead.
    This variant only tests same-sample, same-channel, recent-past
    negatives, which a causal-attention position counter aces without
    encoding forecasting content.

    For each query (b, t, c) with t >= max(lookback_lags):
        positive  = h_full[b, t+1, c, :]
        negatives = [ h_full[b, t-k, c, :] for k in lookback_lags ]
    Score by cosine sim against f[b, t, c, :]. Per-query AUC = (# negs
    the positive beats) / len(lags); Top1 = 1 iff positive beats all.

    TODO: original spec has 3 extra "sub-window shift" negatives that
    require additional encoder passes — not implemented here.

    Args:
        f: ``(B, T, C, H)``.
        h_full: ``(B, T+1, C, H)`` — must include position t+1.
        lookback_lags: lags k for past-window negatives.

    Returns:
        (auc, top1), both scalar tensors. NaN if no valid queries
        (e.g. T <= max_lag).
    """
    B, T, C, H = f.shape
    max_lag = max(lookback_lags)
    if T <= max_lag or T + 1 > h_full.shape[1]:
        nan = torch.full((), float('nan'), device=f.device, dtype=f.dtype)
        return nan, nan

    f_v = f[:, max_lag:T, :, :]                               # (B, T_v, C, H)
    pos = h_full[:, max_lag + 1:T + 1, :, :]
    sim_pos = F.cosine_similarity(f_v, pos, dim=-1)           # (B, T_v, C)

    n_neg = len(lookback_lags)
    sims_neg = []
    for k in lookback_lags:
        neg = h_full[:, max_lag - k:T - k, :, :]
        sims_neg.append(F.cosine_similarity(f_v, neg, dim=-1))
    sim_neg = torch.stack(sims_neg, dim=-1)                   # (B, T_v, C, n_neg)

    beats = (sim_pos.unsqueeze(-1) > sim_neg).float()
    auc = beats.mean(dim=-1).mean()
    top1 = beats.prod(dim=-1).mean()
    return auc, top1


@torch.no_grad()
def retrieval_auc_topk(
    f: Tensor,
    h_full: Tensor,
    lookback_lags: tuple[int, ...] = (1, 2, 4, 8),
    n_batch_negs: int = 8,
    top_k: tuple[int, ...] = (1, 3),
    seed: int | None = 0,
) -> dict[str, Tensor]:
    """Retrieval AUC + top-k against TEMPORAL **AND** BATCH negatives.

    The default retrieval metric for contrastive-backbone diagnostics.
    A causal encoder can learn an implicit position counter via
    attention depth that distinguishes "next time step" from "recent
    past time steps" inside a single window without encoding content.
    Adding negatives drawn from *other batch samples* at the positive
    time step forces the model to also distinguish between different
    time series, not just different time positions — something a pure
    counter cannot do.

    Negatives per query (b, t, c) with t ≥ max(lookback_lags):
        positive       = h_full[b, t+1, c, :]
        temporal negs  = [ h_full[b, t-k, c, :]  for k in lookback_lags ]
                         (same sample, same channel, past — len = |lags|)
        batch negs     = [ h_full[b', t+1, c, :] for b' ∈ rand subset of
                          {0..B-1} \\ {b}, size = min(n_batch_negs, B-1) ]
                         (different sample, same channel, same target time)

    Score = cosine sim vs ``f[b, t, c, :]``. AUC = mean fraction of
    negatives the positive beats. top-k = fraction of queries where the
    positive beats at least ``n_neg - (k - 1)`` negatives (i.e. rank ≤ k
    among 1 positive + n_neg negatives, ties counted against under
    strict ``>``).

    Channel handling: asserts ``C == 1`` for now. Proper cross-channel
    negatives need sampling other channels at the same (b, t) — should
    be added when the backbone supports multi-channel inputs. Until
    then the assert is the contract.

    Args:
        f: ``(B, T, C, H)``.
        h_full: ``(B, T+1, C, H)`` — must include position t+1.
        lookback_lags: lags k for past-window negatives.
        n_batch_negs: number of cross-batch negatives per query. Clamped
            to ``B - 1``.
        top_k: which top-k cutoffs to compute. Default ``(1, 3)``.
        seed: int for the per-batch random permutation. ``None`` uses
            the global RNG; default ``0`` is reproducible across calls.

    Returns:
        dict ``{'auc', 'top1', 'top3', ...}`` of scalar tensors. NaN
        when the eval window has no valid queries (T ≤ max_lag) or
        when B ≤ 1 (no cross-batch negatives available).
    """
    B, T, C, H = f.shape
    assert C == 1, (
        f"retrieval_auc_topk: got C={C}; only C=1 implemented. "
        "Proper cross-channel negatives need sampling other channels at the "
        "same (b, t) — implement when the backbone supports multi-channel."
    )

    out_keys = ["auc"] + [f"top{k}" for k in top_k]
    nan = torch.full((), float("nan"), device=f.device, dtype=f.dtype)

    max_lag = max(lookback_lags)
    if T <= max_lag or T + 1 > h_full.shape[1] or B <= 1:
        return {k: nan for k in out_keys}

    f_v = f[:, max_lag:T, :, :]                          # (B, T_v, C, H)
    pos = h_full[:, max_lag + 1:T + 1, :, :]             # (B, T_v, C, H)
    sim_pos = F.cosine_similarity(f_v, pos, dim=-1)      # (B, T_v, C)

    # Temporal negatives.
    sims_neg_t = []
    for k in lookback_lags:
        neg = h_full[:, max_lag - k:T - k, :, :]
        sims_neg_t.append(F.cosine_similarity(f_v, neg, dim=-1))
    sim_neg_t = torch.stack(sims_neg_t, dim=-1)          # (B, T_v, C, n_t)

    # Cross-batch negatives: pick n_b = min(n_batch_negs, B-1) other
    # batches at the positive time. Sample WITH replacement from
    # {0..B-2}, then shift to skip self-index. Vectorised, no Python
    # loop over batch.
    n_b = min(n_batch_negs, B - 1)
    g = torch.Generator(device=f.device) if seed is not None else None
    if seed is not None:
        g.manual_seed(int(seed))
    raw = torch.randint(0, B - 1, (B, n_b), generator=g, device=f.device)
    b_idx = torch.arange(B, device=f.device).unsqueeze(1)   # (B, 1)
    rand_b = raw + (raw >= b_idx).long()                    # (B, n_b)

    # pos[rand_b]: (B, n_b, T_v, C, H) — gather rows of `pos` by index.
    neg_b = pos[rand_b]
    sim_neg_b = F.cosine_similarity(
        f_v.unsqueeze(1),                                   # (B, 1, T_v, C, H)
        neg_b,                                              # (B, n_b, T_v, C, H)
        dim=-1,
    )                                                       # (B, n_b, T_v, C)
    sim_neg_b = sim_neg_b.permute(0, 2, 3, 1)               # (B, T_v, C, n_b)

    sim_neg = torch.cat([sim_neg_t, sim_neg_b], dim=-1)     # (B, T_v, C, n_t + n_b)
    n_neg = sim_neg.size(-1)
    beats = (sim_pos.unsqueeze(-1) > sim_neg).float()       # strict; ties = miss

    out: dict[str, Tensor] = {"auc": beats.mean(dim=-1).mean()}
    n_beats = beats.sum(dim=-1)                             # (B, T_v, C)
    for k in top_k:
        threshold = n_neg - (k - 1)
        out[f"top{k}"] = (n_beats >= threshold).float().mean()
    return out
