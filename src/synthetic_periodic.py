"""
Clean periodic synthetic data generator.

Designed to teach the contrastive backbone that some series have a fixed
period that seasonal-naive can trivially exploit. One primitive per series,
no additive noise. See experiments/periodic-synth-mix/DESIGN.md for context.

Primitives
----------
- Sinusoid: ``sin(2π (t/P + φ))``
- Square: ``sign(sin(2π (t/P + φ)))`` with a 50% random global sign flip
  (i.e. random up/down start).
- Saw: ``2 * frac(t/P + φ) - 1`` with a 50% random sign flip (i.e. random
  slope direction — ramp up vs ramp down).

Parameters (sampled independently per series)
---------------------------------------------
- Samples-per-period ``P`` — log-uniform in ``[8, 256]``.
- Phase ``φ`` — uniform in ``[0, 1)``.
- Optional envelope ``exp(λ t)`` with probability ``p_env`` (default 0.3),
  where the *total gain* ``exp(λ (T-1))`` is log-uniform in ``[0.1, 10]``.
- Scale — log-uniform in ``[0.1, 1000]`` applied after the envelope.

All operations are vectorized over the full ``batch_size * C`` axis so a
batch of 48 series costs a single pass of ``np.sin`` / ``np.where``.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.random import Generator


# Primitive codes (internal use)
_PRIM_SIN = 0
_PRIM_SQUARE = 1
_PRIM_SAW = 2
_N_PRIMITIVES = 3

# Defaults per DESIGN.md
_DEFAULT_SPP_MIN = 8.0
_DEFAULT_SPP_MAX = 256.0
_DEFAULT_P_ENV = 0.3
_DEFAULT_ENV_GAIN_MIN = 0.1
_DEFAULT_ENV_GAIN_MAX = 10.0
_DEFAULT_SCALE_MIN = 0.1
_DEFAULT_SCALE_MAX = 1000.0


def _as_rng(seed_or_rng) -> Generator:
    if isinstance(seed_or_rng, Generator):
        return seed_or_rng
    return np.random.default_rng(seed_or_rng)


def generate_periodic_batch(
    batch_size: int,
    T_raw: int = 1024,
    C: int = 4,
    seed: int | None = None,
    *,
    rng: Generator | None = None,
    spp_range: tuple[float, float] = (_DEFAULT_SPP_MIN, _DEFAULT_SPP_MAX),
    p_env: float = _DEFAULT_P_ENV,
    env_gain_range: tuple[float, float] = (_DEFAULT_ENV_GAIN_MIN, _DEFAULT_ENV_GAIN_MAX),
    scale_range: tuple[float, float] = (_DEFAULT_SCALE_MIN, _DEFAULT_SCALE_MAX),
    return_meta: bool = False,
    return_freq_ids: bool = False,
    return_labels: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Generate a batch of clean periodic time series.

    One primitive per ``(batch, channel)`` slot, independently parameterised.
    Returns a ``torch.float32`` tensor shaped ``[batch_size, T_raw, C]``
    matching the main data pipeline's convention.

    Parameters
    ----------
    batch_size : int
        Number of multi-channel samples.
    T_raw : int
        Length of each series.
    C : int
        Channels per sample (each channel is an independent primitive).
    seed : int or None
        Seed for the internal Generator (ignored if ``rng`` is provided).
    rng : np.random.Generator or None
        Explicit generator to thread RNG state through a training loop.
    spp_range : (float, float)
        Log-uniform range of samples-per-period.
    p_env : float
        Probability each series receives an exponential envelope.
    env_gain_range : (float, float)
        Log-uniform range of the *total* envelope gain across ``T_raw``.
    scale_range : (float, float)
        Log-uniform range of the final multiplicative scale.
    return_meta : bool
        If True, also return a dict of per-series metadata (shape ``[N]``
        each, where ``N = batch_size * C``). Useful for plotting / sanity
        checks.
    return_freq_ids : bool
        Legacy single-axis path: returns one freq id per batch row,
        derived from the row's minimum spp via ``spp_to_freq_id``.
    return_labels : bool
        Dual-axis path: returns ``(freq_ids, seasonality_ids)`` per batch
        row. ``seasonality_ids`` come from ``seasonality_to_id`` of the
        row's minimum spp; ``freq_ids`` are sampled uniformly from
        ``{1..NUM_FREQS-1}``, independently of the waveform — the model
        learns the freq–seasonality coupling from real data, while synth
        provides uniform coverage of all combinations. Adding this flag
        does not change ``X`` for a fixed seed (the freq draw is appended
        after the data sampling).

    Returns
    -------
    X : torch.Tensor
        ``[batch_size, T_raw, C]`` float32 tensor.
    meta : dict (optional)
        Per-series arrays with keys ``primitive`` (0=sin 1=sq 2=saw),
        ``spp`` (samples-per-period), ``phase`` ([0,1)), ``sign_flip`` (bool),
        ``use_env`` (bool), ``env_gain`` (total gain), ``scale``.
    """
    rng = _as_rng(rng if rng is not None else seed)

    N = batch_size * C

    # -- Joint (freq, seasonality) sampling for the dual-axis path -----------
    # When return_labels=True we sample seasonality_id per BATCH ROW from
    # {0..NUM_SEASONALITIES-1} uniformly, then draw spp per channel within
    # that bucket's range. Bucket 0 is the "no period info" sentinel and
    # uses very long spp (>1024) so the visible signal in a 1024-window
    # looks aperiodic. freq_id is sampled per row INDEPENDENTLY of seas
    # so every (freq, seas) combination is covered with non-zero density.
    row_seasonality_ids = None
    row_freq_ids = None
    if return_labels:
        from src.freq_embedding import (  # local to avoid circular
            NUM_FREQS, NUM_SEASONALITIES, SEASONALITY_BUCKET_SPP_RANGES,
        )
        row_seasonality_ids = rng.integers(
            0, NUM_SEASONALITIES, size=batch_size).astype(np.int64)
        row_freq_ids = rng.integers(
            1, NUM_FREQS, size=batch_size).astype(np.int64)
        spp_2d = np.empty((batch_size, C), dtype=np.float64)
        for b in range(batch_size):
            lo, hi = SEASONALITY_BUCKET_SPP_RANGES[int(row_seasonality_ids[b])]
            spp_2d[b] = np.exp(rng.uniform(np.log(lo), np.log(hi), size=C))
        spp = spp_2d.reshape(N)
    else:
        # Legacy path — uniform spp draw across the whole [spp_range] interval.
        spp = np.exp(rng.uniform(np.log(spp_range[0]), np.log(spp_range[1]), size=N))
    phase = rng.uniform(0.0, 1.0, size=N)
    primitive = rng.integers(0, _N_PRIMITIVES, size=N)
    sign_flip = rng.random(size=N) < 0.5
    use_env = rng.random(size=N) < p_env
    env_gain = np.exp(rng.uniform(np.log(env_gain_range[0]),
                                  np.log(env_gain_range[1]),
                                  size=N))
    # λ chosen so exp(λ*(T-1)) = env_gain exactly
    lam = np.log(env_gain) / max(T_raw - 1, 1)
    scale = np.exp(rng.uniform(np.log(scale_range[0]),
                               np.log(scale_range[1]),
                               size=N))

    # -- Broadcast-friendly shapes --------------------------------------------
    t = np.arange(T_raw, dtype=np.float64).reshape(1, T_raw)  # [1, T]
    spp_b = spp.reshape(N, 1)
    phase_b = phase.reshape(N, 1)

    # Fractional position within each period
    u = t / spp_b + phase_b                 # [N, T]
    angle = 2.0 * np.pi * u                 # [N, T]

    # -- Compute all primitives, then select ----------------------------------
    sin_y = np.sin(angle)                   # [N, T]
    # np.sign(0) = 0 which is fine; by construction integer-sample zeros are
    # measure-zero events on a continuous phase draw.
    square_y = np.sign(np.sin(angle))       # [N, T]
    # Sawtooth in [-1, 1) with np.mod handling negative phases cleanly.
    saw_y = 2.0 * np.mod(u, 1.0) - 1.0       # [N, T]

    prim_b = primitive.reshape(N, 1)
    # np.select is a clean way to dispatch over multiple conditions.
    y = np.select(
        [prim_b == _PRIM_SIN, prim_b == _PRIM_SQUARE, prim_b == _PRIM_SAW],
        [sin_y, square_y, saw_y],
    )

    # Random sign flip applies only to square & saw per spec
    flip_mask = (primitive != _PRIM_SIN) & sign_flip      # [N]
    y = np.where(flip_mask.reshape(N, 1), -y, y)

    # -- Optional exponential envelope ----------------------------------------
    # Compute envelope unconditionally but only apply where use_env is True.
    env = np.exp(lam.reshape(N, 1) * t)      # [N, T]
    y = np.where(use_env.reshape(N, 1), y * env, y)

    # -- Final scale ----------------------------------------------------------
    y = y * scale.reshape(N, 1)

    # -- Reshape to [B, T, C] -------------------------------------------------
    #   N was organised as batch-major over channels: N = b*C + c
    X = y.reshape(batch_size, C, T_raw).transpose(0, 2, 1).astype(np.float32, copy=False)

    X_t = torch.from_numpy(np.ascontiguousarray(X))

    # Per-BATCH-row labels.
    spp_per_row = spp.reshape(batch_size, C).min(axis=1)

    freq_ids_t = None
    seasonality_ids_t = None
    if return_labels:
        # Joint-coverage path: row_freq_ids and row_seasonality_ids were
        # sampled up-front (so all (freq, seas) bucket pairs are covered
        # uniformly), and spp was drawn from each row's bucket.
        seasonality_ids_t = torch.from_numpy(row_seasonality_ids)
        freq_ids_t = torch.from_numpy(row_freq_ids)
    elif return_freq_ids:
        # Legacy single-axis path: freq_id is the seasonality bucket
        # derived from spp via the historical mapping.
        from src.freq_embedding import spp_to_freq_id  # local
        freq_ids_np = np.array(
            [spp_to_freq_id(float(s)) for s in spp_per_row],
            dtype=np.int64)
        freq_ids_t = torch.from_numpy(freq_ids_np)

    if not return_meta:
        if return_labels:
            return X_t, freq_ids_t, seasonality_ids_t
        if return_freq_ids:
            return X_t, freq_ids_t
        return X_t

    meta = dict(
        primitive=primitive,
        spp=spp,
        phase=phase,
        sign_flip=sign_flip,
        use_env=use_env,
        env_gain=env_gain,
        scale=scale,
    )
    if return_labels:
        return X_t, freq_ids_t, seasonality_ids_t, meta
    if return_freq_ids:
        return X_t, meta, freq_ids_t
    return X_t, meta


def primitive_name(code: int) -> str:
    """Map integer primitive code to human-readable name."""
    return {_PRIM_SIN: "sinusoid", _PRIM_SQUARE: "square", _PRIM_SAW: "saw"}.get(
        int(code), f"unknown({code})"
    )
