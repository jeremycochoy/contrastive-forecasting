"""Regime-crossfade synthetic stream (#325, follow-up to #322).

Blend two *distinct real windows* A, B (drawn from the same step's real
sub-batch) with a per-sample monotone crossfade s(t):

    C(t) = (1 - s(t))·A(t) + s(t)·B(t)
    s(t) = 0              t <= l
           (t-l)/(l'-l)   l < t < l'
           1              t >= l'

A and B are z-normalised per series (channel) before blending, with one s(t)
per sample, shared across channels. Because A and B stay in the batch as their own
rows, C shares A's past and B's future with batch-mates — a hard-negative
signal for the contrastive loss that position alone cannot satisfy.

Sampling (T = window length):
    midpoint  m  ~ U(0, T)
    width     w  ~ LogUniform(T/128, T)    (sharp -> gradual transition)
    l = m - w/2, l' = m + w/2, both clipped to [0, T]

A blend of two real windows has no single canonical frequency/seasonality, so
the labels are the sentinel 0 ("unknown"), matching the forked-ARMA stream.

With ``zero_padding`` (the GiftEvalPretrain stream, #419), a parent z-scores
over its real values only and keeps its left padding at exactly 0, and the
blend keeps the union of the two paddings (#421). Without it, nothing
changes.
"""
from __future__ import annotations

import numpy as np
import torch
from numpy.random import Generator

from .norm import leading_zero_count, zero_union_padding

_EPS = 1e-6


def _zscore_per_series(window: torch.Tensor) -> torch.Tensor:
    """z-normalise a ``[T, C]`` window per channel over the time axis."""
    mean = window.mean(dim=0, keepdim=True)
    std = window.std(dim=0, unbiased=False, keepdim=True)
    return (window - mean) / std.clamp_min(_EPS)


def _zscore_real(window: torch.Tensor) -> torch.Tensor:
    """:func:`_zscore_per_series` over the real values of each channel of a
    ``[T, C]`` window, the values after its left zero padding (#419, #421).

    The padding stays exactly 0. A channel whose real values are all equal
    keeps them: their z-score would be all 0, and a 0 at the start reads as
    padding.
    """
    out = torch.zeros_like(window)
    first = leading_zero_count(window.unsqueeze(0))[0, 0].tolist()
    for c, z in enumerate(first):
        real = window[z:, c:c + 1]
        varies = len(real) > 0 and bool((real != real[0]).any())
        out[z:, c:c + 1] = _zscore_per_series(real) if varies else real
    return out


def _parents(real, ia, ib, zero_padding):
    """The two z-normalised parents of a blend."""
    zscore = _zscore_real if zero_padding else _zscore_per_series
    return zscore(real[ia]), zscore(real[ib])


def _blend(a, b, s, real, ia, ib, zero_padding):
    """``(1 - s)·a + s·b``. With ``zero_padding`` its padding is the union
    of the two parents' paddings, at exactly 0."""
    c = (1.0 - s) * a + s * b
    if zero_padding:
        c = zero_union_padding(c[None], real[ia][None], real[ib][None])[0]
    return c


def _sample_crossfade_weight(T: int, rng: Generator) -> np.ndarray:
    """One monotone crossfade s(t) in [0, 1] of length ``T`` (float32).

    midpoint m ~ U(0, T); width w ~ LogUniform(T/128, T); the ramp spans the
    clipped [l, l'] and is 0 before it, 1 after it.
    """
    m = float(rng.uniform(0.0, T))
    w = float(np.exp(rng.uniform(np.log(T / 128.0), np.log(T))))
    l = min(max(m - w / 2.0, 0.0), float(T))
    lp = min(max(m + w / 2.0, 0.0), float(T))
    t = np.arange(T, dtype=np.float32)
    s = (t - l) / max(lp - l, _EPS)
    return np.clip(s, 0.0, 1.0).astype(np.float32)


def generate_crossfade_batch(
    real: torch.Tensor,
    n_out: int,
    *,
    rng: Generator,
    return_labels: bool = False,
    zero_padding: bool = False,
):
    """Build ``n_out`` regime-crossfade rows from a real sub-batch.

    Args:
        real: ``[N, T, C]`` float32 real windows (``N >= 2``). For each output
            two distinct rows A != B are drawn from these (and left in place by
            the caller, so they remain batch-mates of the blend).
        n_out: number of crossfade rows to produce.
        rng: numpy ``Generator`` (drives source choice, midpoint, width).
        return_labels: also return ``(freq_ids, seas_ids)``, both the sentinel
            ``0`` as ``[n_out]`` int64.
        zero_padding: the real rows carry left zero padding (#419). The
            padding stays 0 (#421, see the module docstring).

    Returns:
        ``[n_out, T, C]`` float32 (plus labels if requested). An empty tensor
        when ``n_out <= 0``.
    """
    N, T, C = real.shape
    if n_out <= 0:
        out = real.new_zeros((0, T, C))
        if return_labels:
            z = torch.zeros(0, dtype=torch.int64)
            return out, z, z
        return out
    if N < 2:
        raise ValueError(f"crossfade needs >=2 real rows, got {N}")

    out = real.new_zeros((n_out, T, C))
    for k in range(n_out):
        ia = int(rng.integers(0, N))
        ib = int(rng.integers(0, N - 1))   # draw B != A uniformly over the rest
        if ib >= ia:
            ib += 1
        a, b = _parents(real, ia, ib, zero_padding)
        s = torch.from_numpy(_sample_crossfade_weight(T, rng)).unsqueeze(-1)  # [T, 1]
        out[k] = _blend(a, b, s, real, ia, ib, zero_padding)

    if return_labels:
        z = torch.zeros(n_out, dtype=torch.int64)
        return out, z, z
    return out


def generate_crossfade_triplets(
    real: torch.Tensor,
    n_triplets: int,
    *,
    rng: Generator,
    return_labels: bool = False,
    zero_padding: bool = False,
):
    """Build ``n_triplets`` explicit (A_norm, B_norm, C) crossfade triplets (#328).

    For each triplet, draw two distinct real rows A != B, z-normalise each per
    series, and blend ``C = (1 - s(t))·A_norm + s(t)·B_norm`` with the same
    monotone ramp s(t) as :func:`generate_crossfade_batch`. Unlike that function
    (#326, which emits only C and leaves the raw real rows as its in-batch
    parents), here BOTH z-normalised parents are emitted alongside the blend, so
    the contrastive loss sees C with its two exact parents as batch-mates: A_norm
    shares C's past, B_norm shares C's future, all on the same normalised scale.

    Args:
        real: ``[N, T, C]`` float32 real windows (``N >= 2``).
        n_triplets: number of (A_norm, B_norm, C) triplets to produce.
        rng: numpy ``Generator`` (drives source choice, midpoint, width).
        return_labels: also return ``(freq_ids, seas_ids)``, both the sentinel 0.
        zero_padding: the real rows carry left zero padding (#419). The
            padding stays 0 (#421, see the module docstring).

    Returns:
        ``[3 * n_triplets, T, C]`` float32, rows ordered
        ``[A0, B0, C0, A1, B1, C1, ...]`` (plus sentinel labels if requested).
    """
    N, T, C = real.shape
    if n_triplets <= 0:
        out = real.new_zeros((0, T, C))
        if return_labels:
            z = torch.zeros(0, dtype=torch.int64)
            return out, z, z
        return out
    if N < 2:
        raise ValueError(f"crossfade triplets need >=2 real rows, got {N}")

    out = real.new_zeros((3 * n_triplets, T, C))
    for k in range(n_triplets):
        ia = int(rng.integers(0, N))
        ib = int(rng.integers(0, N - 1))   # draw B != A uniformly over the rest
        if ib >= ia:
            ib += 1
        a, b = _parents(real, ia, ib, zero_padding)
        s = torch.from_numpy(_sample_crossfade_weight(T, rng)).unsqueeze(-1)  # [T, 1]
        out[3 * k] = a
        out[3 * k + 1] = b
        out[3 * k + 2] = _blend(a, b, s, real, ia, ib, zero_padding)

    if return_labels:
        z = torch.zeros(3 * n_triplets, dtype=torch.int64)
        return out, z, z
    return out
