"""Diagnostic metrics for the contrastive-forecasting backbone.

All metrics are computed without gradient. They probe whether the
forecaster has learned a useful representation by comparing its
predictions against simple baselines (random pair, naive last-step,
past-window negatives) and by checking how much of the latent
hidden-dim space is actually used.
"""

from __future__ import annotations

import os

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
def q_hard_neg(sim_pos: Tensor, sim_neg: Tensor, eps: float = 1e-8) -> Tensor:
    """Q_hard_neg = mean over queries of e(f, target) / e(f, hardest_neg).

    e(a, b) = 1 - cos_sim(a, b). For each query, the "hardest negative"
    is the negative with the largest cosine similarity to the forecast
    (i.e. the worst confusable). The ratio is < 1 when the forecast is
    closer to the target than to its hardest distractor (top-1 win), > 1
    when it loses top-1. Sign-matched complement to ``top1``: continuous
    margin where top1 is binary.

    Decoupled from how negatives were sampled — caller supplies the
    pre-computed cosine similarities.

    Args:
        sim_pos: ``(...,)`` — cos(f, target) per query.
        sim_neg: ``(..., N)`` — cos(f, neg_i) for the N negatives per
            query. The last axis is the negative-pool axis.

    Returns:
        Scalar tensor (mean over query dims).
    """
    assert sim_neg.shape[:-1] == sim_pos.shape, (sim_pos.shape, sim_neg.shape)
    err_p = 1.0 - sim_pos
    err_h = (1.0 - sim_neg.max(dim=-1).values).clamp_min(eps)
    return (err_p / err_h).mean()


@torch.no_grad()
def m_z(sim_pos: Tensor, sim_neg: Tensor, eps: float = 1e-8) -> Tensor:
    """M_z = mean over queries of (sim_pos - μ_neg) / σ_neg.

    Standardized margin (discriminability index d′). The positive's
    cosine similarity is reported in units of the negative pool's
    standard deviation. Under isotropic negatives in dim ``d``, σ ≈
    1/√d, so for a forecast with cos(f, target) = ρ this is ≈ ρ·√d.
    Linear in ρ — does not saturate.

    Decoupled from pool definition like ``q_hard_neg``; caller supplies
    similarities.

    Args:
        sim_pos: ``(...,)``.
        sim_neg: ``(..., N)``.

    Returns:
        Scalar tensor.
    """
    assert sim_neg.shape[:-1] == sim_pos.shape, (sim_pos.shape, sim_neg.shape)
    mu = sim_neg.mean(dim=-1)
    sd = sim_neg.std(dim=-1, unbiased=False).clamp_min(eps)
    return ((sim_pos - mu) / sd).mean()


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
    # Materialising sim = flat @ flat.T is (S, n, n) — 16 GiB at n=65k
    # (#369, B=1024). Accumulate Σ sim² row-chunk-wise so peak allocation
    # is (S, chunk, n) instead. Chunk size settable via env for tuning.
    chunk = int(os.environ.get("DIM_USAGE_CHUNK", "4096"))
    if chunk <= 0 or chunk >= n:
        sim = torch.matmul(flat, flat.transpose(-1, -2))   # (S, n, n)
        sq = sim.pow(2)
        sum_all = sq.sum(dim=(-1, -2))
        sum_diag = torch.diagonal(sq, dim1=-2, dim2=-1).sum(dim=-1)
    else:
        S = flat.shape[0]
        sum_all = torch.zeros(S, device=z.device, dtype=flat.dtype)
        sum_diag = torch.zeros(S, device=z.device, dtype=flat.dtype)
        flat_T = flat.transpose(-1, -2)  # (S, d, n)
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            sim_chunk = torch.matmul(flat[:, i:j, :], flat_T)   # (S, c, n)
            sq_chunk = sim_chunk.pow(2)
            sum_all = sum_all + sq_chunk.sum(dim=(-1, -2))
            # Diagonal entries at (k, i+k) within the chunk.
            idx = torch.arange(i, j, device=z.device)
            sum_diag = sum_diag + sq_chunk[:, torch.arange(j - i, device=z.device), idx].sum(dim=-1)
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


def u_batchtime(z: Tensor) -> Tensor:
    """U with batch and time pooled into a single n axis of size ``B*T``.

    Mirrors SIGReg's pooling — ``z.reshape(-1, K)`` flattens every non-
    feature axis into one sample axis. Here we pool the two leading
    axes (batch, time) so the n-axis sees ``B*T`` samples per channel
    slice. ``z`` must have shape ``(B, T, ..., H)``.

    Complements ``u_batch`` (cross-batch only, per-time) and
    ``u_temporal`` (cross-time only, per-batch): the (batch × time)
    product is the single statistic SIGReg actually regularises, so
    pinning the matching dimension-usage version closes the loop on the
    dim-usage panel.
    """
    assert z.ndim >= 3, (
        f"u_batchtime expects (B, T, ..., H); got shape {tuple(z.shape)}")
    B, T = z.shape[0], z.shape[1]
    pooled = z.reshape(B * T, *z.shape[2:])
    return dim_usage(pooled, axis=0)


@torch.no_grad()
def drift_pair(A: Tensor, B: Tensor, eps: float = 1e-8) -> dict[str, Tensor]:
    """Latent-drift metric between two representations of the SAME probe
    batch at two training steps.

    Splits per-token movement into an informative component and a pure
    feature-axis rotation component. Given two tensors ``A`` and ``B``
    of matching shape ``(..., H)``, all non-feature axes are flattened
    into an ``N``-row sample axis; rows are then L2-normalised (the
    loss geometry lives on the unit sphere, so drift measured there is
    the drift the objective actually moves) before three quantities
    are computed:

    ``drift_cos`` (``1 − mean_i cos(A_i, B_i)``) — raw per-token
    movement. Zero when ``A_i = c_i B_i`` for any positive scalar; ``1``
    at orthogonality; ``2`` for antipodal pairs. Includes both
    informative geometric change and uninformative feature-axis
    rotation.

    ``drift_cos_aligned`` (``1 − (1/N) Σ_k σ_k(A^T B)``) — same
    per-token movement AFTER a Procrustes rotation ``R* = argmin_R
    ||A R − B||_F`` (``R`` orthogonal on the feature axis) removes the
    best global feature-axis rotation. This is the movement a
    downstream linear head cannot absorb — real information change.
    Derivation: on unit rows, ``mean_i cos(R* A_i, B_i) = (1/N)
    trace(A R* B^T)``; the maximum over orthogonal ``R`` is ``Σ_k σ_k``
    (Von-Neumann trace inequality applied to the SVD of ``A^T B``).

    ``rot_gap`` (``drift_cos − drift_cos_aligned``) — the raw movement
    that is pure feature-axis rotation. Non-zero late in training
    means the encoder keeps rotating its output frame without changing
    the token geometry a linear head sees.

    ``cka`` (linear centered CKA) — cross-check. Ranges ``[0, 1]``.
    Rotation-invariant on the feature axis and scale-invariant. High
    ``cka`` with ``drift_cos > 0`` confirms the movement is
    reparameterization, not geometric change.

    Args:
        A, B: same-shape latent tensors ``(..., H)``. Typical shapes:
            ``(B, T, C, H)`` and ``(B*C, T, H)`` — both work.
        eps: L2-normalization and clamp floor for stability.

    Returns:
        dict with scalar tensors ``drift_cos``, ``drift_cos_aligned``,
        ``rot_gap``, ``cka``. All computed in fp32 regardless of input
        dtype (the trace and Gram products are small).
    """
    assert A.shape == B.shape, (A.shape, B.shape)
    H = A.shape[-1]
    A = A.reshape(-1, H).float()
    B = B.reshape(-1, H).float()
    N = A.shape[0]
    An = F.normalize(A, p=2, dim=-1, eps=eps)
    Bn = F.normalize(B, p=2, dim=-1, eps=eps)
    drift_cos = 1.0 - (An * Bn).sum(dim=-1).mean()
    M = An.transpose(0, 1) @ Bn                                # (H, H)
    svd = torch.linalg.svdvals(M)                              # (H,)
    drift_cos_aligned = 1.0 - svd.sum() / N
    rot_gap = drift_cos - drift_cos_aligned
    Ac = A - A.mean(dim=0, keepdim=True)
    Bc = B - B.mean(dim=0, keepdim=True)
    Mc = Ac.transpose(0, 1) @ Bc
    hsic_xy = (Mc * Mc).sum()
    Mxx = Ac.transpose(0, 1) @ Ac
    Myy = Bc.transpose(0, 1) @ Bc
    hsic_xx = (Mxx * Mxx).sum().clamp_min(eps)
    hsic_yy = (Myy * Myy).sum().clamp_min(eps)
    cka = hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt())
    return {
        "drift_cos": drift_cos,
        "drift_cos_aligned": drift_cos_aligned,
        "rot_gap": rot_gap,
        "cka": cka,
    }


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
    n_batch_negs: int = 128,
    top_k: tuple[int, ...] = (1, 3),
) -> dict[str, Tensor]:
    """Retrieval metrics against a CROSS-BATCH × TEMPORAL-OFFSET product
    of negatives.

    The default retrieval metric for contrastive-backbone diagnostics.
    A causal encoder can learn an implicit position counter via
    attention depth that distinguishes "next time step" from "recent
    past time steps" inside a single window without encoding content.
    Forcing the model to discriminate against negatives drawn from
    *other* time series at *offset* times defeats both the position-
    counter and the persistence baseline in a single pool.

    Negatives per query (b, t, c) with t ≥ max(lookback_lags) form the
    product of two axes:
        positive  = h_full[b, t+1, c, :]
        negatives = h_full[b', t+1-k, c, :]
                    for b' ∈ first n_b indices of {0..B-1} \\ {b},
                        n_b = min(n_batch_negs, B - 1)
                        k   ∈ lookback_lags
        total     = n_b × len(lookback_lags) per query

    Defaults give 128 × 4 = 512 negatives per query — tighter retrieval
    signal than the legacy 4-temporal pool, harder to saturate.

    The "first n_b indices" trick (rather than a random shuffle) gives
    deterministic, lower-variance metric estimates: the HF streaming
    dataloader already shuffles the batch dimension upstream, so the
    distribution of negatives is iid-equivalent.

    Score = cosine sim vs ``f[b, t, c, :]``. Top-k counts ties as miss
    via strict ``>``.

    Channel handling: asserts ``C == 1`` for now. Proper cross-channel
    negatives need sampling other channels at the same (b, t) — should
    be added when the backbone supports multi-channel inputs. Until
    then the assert is the contract.

    Args:
        f: ``(B, T, C, H)``.
        h_full: ``(B, T+1, C, H)`` — must include position t+1.
        lookback_lags: temporal offsets k (subtracted from t+1).
        n_batch_negs: number of cross-batch negatives per query. Clamped
            to ``B - 1``.
        top_k: which top-k cutoffs to compute. Default ``(1, 3)``.

    Returns:
        dict with scalar-tensor values for keys:
            ``auc``       — mean fraction of negatives beaten
            ``mrr``       — mean reciprocal rank of the positive among
                            (1 + n_neg) candidates (ties count as miss)
            ``top{k}``    — fraction of queries with rank ≤ k, one per
                            entry in ``top_k``
            ``q_hard_neg`` — mean err(f, target) / err(f, hardest_neg)
            ``m_z``       — mean (sim_pos − μ_neg) / σ_neg

        All values are NaN when the eval window has no valid queries
        (T ≤ max_lag) or when B ≤ 1.
    """
    B, T, C, H = f.shape
    assert C == 1, (
        f"retrieval_auc_topk: got C={C}; only C=1 implemented. "
        "Proper cross-channel negatives need sampling other channels at the "
        "same (b, t) — implement when the backbone supports multi-channel."
    )

    out_keys = (
        ["auc", "mrr"] + [f"top{k}" for k in top_k] + ["q_hard_neg", "m_z"]
    )
    nan = torch.full((), float("nan"), device=f.device, dtype=f.dtype)

    max_lag = max(lookback_lags)
    if T <= max_lag or T + 1 > h_full.shape[1] or B <= 1:
        return {k: nan for k in out_keys}

    f_v = f[:, max_lag:T, :, :]                              # (B, T_v, C, H)
    pos = h_full[:, max_lag + 1:T + 1, :, :]                 # (B, T_v, C, H)
    sim_pos = F.cosine_similarity(f_v, pos, dim=-1)          # (B, T_v, C)

    # Cross-batch indices: first n_b of {0..B-1} \ {b}, deterministic.
    n_b = min(n_batch_negs, B - 1)
    raw = torch.arange(n_b, device=f.device).unsqueeze(0).expand(B, n_b)
    b_idx = torch.arange(B, device=f.device).unsqueeze(1)    # (B, 1)
    rand_b = raw + (raw >= b_idx).long()                     # (B, n_b)

    # Cross-batch × temporal-offset product. For each lag k: cosine sim
    # of f_v vs every other batch row's lagged slice, then keep the n_b
    # negatives selected by rand_b.
    #
    # Memory-lean form: the old code did h_slice_k[rand_b] →
    # (B, n_b, T_v, C, H) and F.cosine_similarity over the last dim,
    # materialising a (B, n_b, T_v, C, H) tensor (~12 GB fp32 at B=256,
    # n_b=128, T_v≈248, H=384 — OOM'd every training step). Instead,
    # L2-normalise once and contract H *inside* an einsum to get the full
    # (B, B, T_v, C) cosine matrix (~65 MB), then gather the n_b columns
    # per row. Algebraically identical to F.cosine_similarity (same
    # eps=1e-8 denominator floor via F.normalize); the einsum routes to a
    # batched matmul and never forms the (·, ·, ·, ·, H) intermediate.
    f_n = F.normalize(f_v, dim=-1, eps=1e-8)                      # (B, T_v, C, H)
    sims_per_lag = []
    for k in lookback_lags:
        h_slice_k = h_full[:, max_lag + 1 - k:T + 1 - k, :, :]   # (B, T_v, C, H)
        h_n_k = F.normalize(h_slice_k, dim=-1, eps=1e-8)         # (B, T_v, C, H)
        full_k = torch.einsum('btch,Btch->bBtc', f_n, h_n_k)     # (B, B, T_v, C)
        sim_k = full_k[b_idx, rand_b]                             # (B, n_b, T_v, C)
        sims_per_lag.append(sim_k.permute(0, 2, 3, 1))            # (B, T_v, C, n_b)

    sim_neg = torch.cat(sims_per_lag, dim=-1)                     # (B, T_v, C, n_b * n_lags)
    n_neg = sim_neg.size(-1)
    beats = (sim_pos.unsqueeze(-1) > sim_neg).float()             # strict; ties = miss

    n_beats = beats.sum(dim=-1)                                   # (B, T_v, C)
    rank = (n_neg - n_beats + 1.0)                                # in [1, n_neg+1]

    out: dict[str, Tensor] = {
        "auc": beats.mean(dim=-1).mean(),
        "mrr": (1.0 / rank).mean(),
    }
    for k in top_k:
        threshold = n_neg - (k - 1)
        out[f"top{k}"] = (n_beats >= threshold).float().mean()
    out["q_hard_neg"] = q_hard_neg(sim_pos, sim_neg)
    out["m_z"] = m_z(sim_pos, sim_neg)
    return out
