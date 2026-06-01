"""Regime-crossfade synthetic stream (#325, follow-up to #322).

Blend two *distinct real windows* A, B (drawn from the same step's real
sub-batch) with a per-sample monotone crossfade s(t):

    C(t) = (1 - s(t))·A(t) + s(t)·B(t)
    s(t) = 0              t <= l
           (t-l)/(l'-l)   l < t < l'
           1              t >= l'

A and B are z-normalised per series (channel) before blending; one s(t) per
sample, shared across channels. Because A and B stay in the batch as their own
rows, C shares A's past and B's future with batch-mates — a hard-negative
signal for the contrastive loss that position alone cannot satisfy.

Sampling (T = window length):
    midpoint  m  ~ U(0, T)
    width     w  ~ LogUniform(T/128, T)    (sharp -> gradual transition)
    l = m - w/2, l' = m + w/2, both clipped to [0, T]

A blend of two real windows has no single canonical frequency/seasonality, so
the labels are the sentinel 0 ("unknown"), matching the forked-ARMA stream.
"""
from __future__ import annotations

import numpy as np
import torch
from numpy.random import Generator

_EPS = 1e-6


def _zscore_per_series(window: torch.Tensor) -> torch.Tensor:
    """z-normalise a ``[T, C]`` window per channel over the time axis."""
    mean = window.mean(dim=0, keepdim=True)
    std = window.std(dim=0, unbiased=False, keepdim=True)
    return (window - mean) / std.clamp_min(_EPS)


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
        a = _zscore_per_series(real[ia])
        b = _zscore_per_series(real[ib])
        s = torch.from_numpy(_sample_crossfade_weight(T, rng)).unsqueeze(-1)  # [T, 1]
        out[k] = (1.0 - s) * a + s * b

    if return_labels:
        z = torch.zeros(n_out, dtype=torch.int64)
        return out, z, z
    return out
