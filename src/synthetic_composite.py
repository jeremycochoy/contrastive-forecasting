"""
TimesFM-style composite synthetic generator with dual-axis labels.

Stacks: piecewise linear trend (always on) + ARMA(p,q) (optionally
integrated → ARIMA(1,p,q)) + two free-spp waves + one seasonality-tied
wave whose period falls inside the row's seasonality bucket. Each
non-trend component is gated by an independent Bernoulli; at least one
non-trend is forced on per channel; the trend is applied either
multiplicatively (heteroskedastic) or additively (50/50). Per-(b,c)
exponential envelope (p=0.3) and log-uniform final scale.

Per-row (freq_id, seas_id) labels follow the dual-axis embedding's
vocabulary. The seas_id is emitted as the row's drawn bucket only when
the seasonality-tied wave is on for that row; otherwise the row is
labelled with seas_id=0 (no period info), so the embedding row 0
remains a faithful "label says: no useful period in this signal" prior.

Differences vs the existing pieces in this repo:
    src/synthetic.py (TimesFM-original): adds dual-axis labels, three
        waveform primitives instead of two, no per-component pre-mix
        unit-std normalisation. ARIMA component is internally normalised
        to a random target-std so it mixes against [-1,1] waves without
        cumsum-blowing-up; the rest of the recipe is range-tuned.
    src/synthetic_periodic.py (current on-the-fly): adds ARMA, ARIMA
        integration, piecewise trend, multiplicative trend; allows
        multiple components stacked per channel; the seas-tied wave is
        gated, so seas_id=0 is sometimes emitted intentionally.

ARMA sampling reuses sample_arma_from_roots / generate_piecewise_linear
from src.synthetic verbatim — same polynomial-root method as the
training_data_prep pipeline (../rnd) writes into the bundle synth
parquets.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.random import Generator

from src.freq_embedding import (
    NUM_FREQS,
    NUM_SEASONALITIES,
    SEASONALITY_BUCKET_SPP_RANGES,
)
from src.synthetic import (
    generate_piecewise_linear,
    sample_arma_from_roots,
)


# Primitive codes — same convention as src.synthetic_periodic plus PULSE.
_PRIM_SIN = 0
_PRIM_SQUARE = 1
_PRIM_SAW = 2
_PRIM_PULSE = 3      # sparse pulse train (low-duty unit-amplitude)
_N_PRIMITIVES = 3
_N_PRIMITIVES_WITH_PULSE = 4

# Default pulse width in samples. The pulse train fires at every multiple
# of spp; each fire is a contiguous run of `pulse_width` samples at ±1.
# Width 1 is a Dirac comb, 2-3 represents a "burst" with brief decay.
_DEFAULT_PULSE_WIDTH = 1

# Defaults chosen so component natural amplitudes sit in roughly the
# same band as the [-1, 1] waves before the U[0,1] mixing weights.
# Trend: piecewise-linear cumsum with slope_std=0.003 gives end-of-T
# drift std ≈ slope_std·T/√n ≈ 0.003·1024/√5 ≈ 1.4 — comparable to a
# unit-amplitude sine. ARIMA: sample_arma_from_roots returns ~unit-std
# innovations through a stable filter (≈1.5–3 std before integration);
# we *do* normalise it post-cumsum to a random target-std (per user OK)
# so cumsum at T=1024 doesn't dominate the mix by an order of magnitude.
_DEFAULT_SLOPE_STD = 0.003
_DEFAULT_ARMA_DIMENSION = 8
_DEFAULT_ARMA_TARGET_STD_RANGE = (0.5, 3.0)
_DEFAULT_P_ARMA = 0.5
_DEFAULT_P_FREE = 0.5
_DEFAULT_P_SEAS_TIED = 0.5
_DEFAULT_P_INTEGRATE = 0.5
_DEFAULT_P_TREND_MULT = 0.5
_DEFAULT_P_ENV = 0.3
_DEFAULT_ENV_GAIN_RANGE = (0.1, 10.0)
_DEFAULT_SCALE_RANGE = (0.1, 1000.0)
_DEFAULT_FREE_SPP_RANGE = (4.0, 512.0)  # log-uniform; T_raw=1024 → up to T/2

# Cap on how many times we retry a NaN-producing ARMA draw before
# falling back to white noise. statsmodels can emit NaN on some
# pathological polynomials; in practice the unit-disk + 1/(1+ε) shrink
# keeps this rare.
_ARMA_RETRY_LIMIT = 4


def _as_rng(seed_or_rng) -> Generator:
    if isinstance(seed_or_rng, Generator):
        return seed_or_rng
    return np.random.default_rng(seed_or_rng)


def _sample_wave(T: int, rng: Generator,
                 spp_range: tuple[float, float],
                 *,
                 enable_pulse: bool = False,
                 pulse_width: int = _DEFAULT_PULSE_WIDTH) -> np.ndarray:
    """Single wave at random primitive ∈ {sin, square, saw, [pulse]} with
    log-uniform spp.

    Square and saw get a 50% sign flip per spec (random "up first vs
    down first" / "ramp up vs ramp down"). When ``enable_pulse=True``,
    a 4th primitive (sparse pulse train) is drawn with equal probability
    — sparse-amplitude burst content that EWMA normalisers don't smooth
    away (each burst lasts `pulse_width` samples at ±1 followed by 0
    until the next period).

    Returns float64 array of length T with peak amplitude in [-1, 1].
    """
    n_prim = _N_PRIMITIVES_WITH_PULSE if enable_pulse else _N_PRIMITIVES
    spp = float(np.exp(rng.uniform(np.log(spp_range[0]), np.log(spp_range[1]))))
    phase = rng.uniform(0.0, 1.0)
    prim = int(rng.integers(0, n_prim))
    t = np.arange(T, dtype=np.float64)
    u = t / spp + phase
    if prim == _PRIM_SIN:
        return np.sin(2.0 * np.pi * u)
    if prim == _PRIM_SQUARE:
        y = np.sign(np.sin(2.0 * np.pi * u))
        if rng.random() < 0.5:
            y = -y
        return y
    if prim == _PRIM_SAW:
        y = 2.0 * np.mod(u, 1.0) - 1.0
        if rng.random() < 0.5:
            y = -y
        return y
    # PULSE: sparse train. The wave is 0 except for a contiguous run of
    # `pulse_width` samples (with sign s ∈ ±1) at every period. Phase
    # offsets where the burst lands within each cycle. Implementation:
    #   distance to the next pulse start, in fractional cycles, is
    #     frac = mod(t/spp + phase, 1)
    #   the burst is "on" when frac falls in [0, pulse_width/spp).
    pw_frac = max(pulse_width, 1) / max(spp, 1.0)
    pw_frac = min(pw_frac, 0.5)  # cap at 50% duty so it stays "sparse"
    frac = np.mod(u, 1.0)
    y = np.where(frac < pw_frac, 1.0, 0.0)
    if rng.random() < 0.5:
        y = -y
    return y


def _safe_sample_arma(T: int, rng: Generator, dimension: int) -> np.ndarray:
    """ARMA sample with retry on NaN. Uses the rnd polynomial-root method."""
    for _ in range(_ARMA_RETRY_LIMIT):
        y = sample_arma_from_roots(T, rng, dimension=dimension)
        if np.isfinite(y).all():
            return y
    # Fallback: standard Gaussian noise — keeps the run going on
    # pathological seeds. Same effect on the contrastive head as a
    # tiny innovation-only ARMA(0,0).
    return rng.standard_normal(T)


def _build_one_channel(rng: Generator, T: int,
                        drawn_seas_id: int,
                        seas_tied_on: bool,
                        *,
                        slope_std: float,
                        arma_dimension: int,
                        p_arma: float,
                        p_free: float,
                        p_integrate: float,
                        p_trend_mult: float,
                        arma_target_std_range: tuple[float, float],
                        free_spp_range: tuple[float, float],
                        enable_pulse: bool = False,
                        pulse_width: int = _DEFAULT_PULSE_WIDTH,
                        n_free_waves: int = 2,
                        n_seas_tied_waves: int = 1,
                        ) -> tuple[np.ndarray, dict]:
    """Build one composite channel of length T per the recipe.

    Returns the float64 series and a small per-channel meta dict. Per-row
    state (drawn_seas_id, seas_tied_on) is passed in by the caller; the
    other coinflips are independent per channel. Trend is always on.
    Enforces "at least one non-trend on" by force-on'ing a uniform pick
    from {ARMA, free-wave, free-wave, ...} when all per-channel coinflips
    give "off" and the row's seas-tied wave is also off.

    ``n_free_waves`` and ``n_seas_tied_waves`` control how many of each
    wave kind appear (each independently coinflipped against ``p_free`` /
    presence of ``seas_tied_on``). Defaults preserve phase-1 behaviour
    (2 free + 1 seas-tied). Phase 2B uses (1 free + 2 seas-tied) to
    boost periodic-signal coverage in seas-tied-on rows.
    """
    arma_on = rng.random() < p_arma
    # Each free wave coinflips independently. List of bools sized n_free_waves.
    free_on = [rng.random() < p_free for _ in range(n_free_waves)]
    any_free_on = any(free_on)

    if not (arma_on or any_free_on or seas_tied_on):
        # Force on: uniform pick from {ARMA, free_0, free_1, ...}.
        n_choices = 1 + n_free_waves
        choice = int(rng.integers(0, n_choices))
        if choice == 0:
            arma_on = True
        else:
            free_on[choice - 1] = True
            any_free_on = True

    parts: list[np.ndarray] = []
    weights: list[float] = []
    integrate_used = False

    if arma_on:
        y = _safe_sample_arma(T, rng, dimension=arma_dimension)
        if rng.random() < p_integrate:
            y = np.cumsum(y)
            integrate_used = True
        std = float(y.std())
        if std > 1e-12:
            target = float(np.exp(rng.uniform(
                np.log(arma_target_std_range[0]),
                np.log(arma_target_std_range[1]))))
            y = y * (target / std)
        parts.append(y)
        weights.append(float(rng.uniform(0.0, 1.0)))

    for is_on in free_on:
        if is_on:
            parts.append(_sample_wave(T, rng, spp_range=free_spp_range,
                                      enable_pulse=enable_pulse,
                                      pulse_width=pulse_width))
            weights.append(float(rng.uniform(0.0, 1.0)))

    if seas_tied_on:
        spp_lo, spp_hi = SEASONALITY_BUCKET_SPP_RANGES[drawn_seas_id]
        for _ in range(n_seas_tied_waves):
            parts.append(_sample_wave(T, rng, spp_range=(spp_lo, spp_hi),
                                      enable_pulse=enable_pulse,
                                      pulse_width=pulse_width))
            weights.append(float(rng.uniform(0.0, 1.0)))

    # The "≥1 non-trend on" force above guarantees parts is non-empty.
    non_trend = np.zeros(T, dtype=np.float64)
    for p, w in zip(parts, weights):
        non_trend += p * w

    trend = generate_piecewise_linear(T, rng, slope_std=slope_std)
    if rng.random() < p_trend_mult and not np.allclose(non_trend, 0.0):
        sample = non_trend * trend
    else:
        trend_weight = float(rng.uniform(0.0, 1.0))
        sample = non_trend + trend_weight * trend

    meta = dict(
        arma_on=arma_on,
        # Back-compat aliases for free1/free2 — first two flags. With
        # n_free_waves != 2 these are best-effort.
        free1_on=bool(free_on[0]) if n_free_waves >= 1 else False,
        free2_on=bool(free_on[1]) if n_free_waves >= 2 else False,
        free_on=tuple(free_on),
        seas_tied_on=seas_tied_on,
        n_seas_tied_waves=n_seas_tied_waves if seas_tied_on else 0,
        integrate_used=integrate_used,
    )
    return sample, meta


def generate_composite_batch(
    batch_size: int,
    T_raw: int = 1024,
    C: int = 4,
    seed: int | None = None,
    *,
    rng: Generator | None = None,
    # Per-component coinflip probabilities
    p_arma: float = _DEFAULT_P_ARMA,
    p_free: float = _DEFAULT_P_FREE,
    p_seas_tied: float = _DEFAULT_P_SEAS_TIED,
    p_integrate: float = _DEFAULT_P_INTEGRATE,
    p_trend_mult: float = _DEFAULT_P_TREND_MULT,
    p_env: float = _DEFAULT_P_ENV,
    # Sampling ranges
    slope_std: float = _DEFAULT_SLOPE_STD,
    arma_dimension: int = _DEFAULT_ARMA_DIMENSION,
    arma_target_std_range: tuple[float, float] = _DEFAULT_ARMA_TARGET_STD_RANGE,
    env_gain_range: tuple[float, float] = _DEFAULT_ENV_GAIN_RANGE,
    scale_range: tuple[float, float] = _DEFAULT_SCALE_RANGE,
    free_spp_range: tuple[float, float] = _DEFAULT_FREE_SPP_RANGE,
    # Pulse-train primitive (phase-2 spike-deficit fix)
    enable_pulse: bool = False,
    pulse_width: int = _DEFAULT_PULSE_WIDTH,
    # Wave count knobs (phase-2B periodic-coverage fix)
    n_free_waves: int = 2,
    n_seas_tied_waves: int = 1,
    # API knobs
    return_labels: bool = False,
    return_meta: bool = False,
) -> torch.Tensor | tuple:
    """Generate a batch of composite synthetic series with dual-axis labels.

    Per-row state (constant across channels):
      * drawn ``seas_id ~ U{0..NUM_SEASONALITIES-1}``,
        ``freq_id ~ U{1..NUM_FREQS-1}``
      * a Bernoulli(``p_seas_tied``) deciding whether every channel of
        the row carries a seasonality-tied wave whose spp falls in the
        bucket's range. The emitted ``seas_id`` is the drawn bucket only
        when the row's seas-tied wave is on; otherwise it is 0 (the
        embedding's no-period-info sentinel).

    Per-channel state (independent per (b, c)):
      * trend (always on); ARMA on/off; two free-spp waves on/off
      * if all three per-channel non-trend coinflips give "off" AND the
        row-level seas-tied wave is also off, one of {ARMA, free1, free2}
        is force-on'd uniformly so every channel has ≥1 non-trend
        component (matches TimesFM's "ensure at least one" rule)
      * trend application: 50% multiplicative else additive with U[0,1] weight
      * exponential envelope on the final per-(b,c) signal with
        probability ``p_env``, total gain log-uniform in ``env_gain_range``
      * log-uniform final ``scale_range`` multiplier

    Parameters
    ----------
    batch_size, T_raw, C, seed, rng
        Standard. ``seed`` is ignored when ``rng`` is given.
    p_arma, p_free, p_seas_tied, p_integrate, p_trend_mult, p_env
        Bernoulli probabilities for each component / decision. Defaults
        match the recipe spec; tunable for the HP-search phase.
    slope_std
        Piecewise-linear-trend slope std. Default 0.003 — low because the
        trend is not pre-normalised; chosen so cumsum'd typical magnitude
        is comparable to a [-1, 1] wave.
    arma_dimension
        Max AR/MA order (uniform draw on [1, dimension]). Default 8 —
        same as TimesFM / ../rnd.
    arma_target_std_range
        Log-uniform target std the ARMA / ARIMA component is rescaled to
        post-(optional)integration so it mixes against bounded waves
        without dominating an order of magnitude.
    env_gain_range, scale_range, free_spp_range
        Log-uniform ranges. ``free_spp_range=(4, T_raw/2)`` follows TimesFM.
    return_labels
        If True, returns ``(X, freq_ids, seasonality_ids)`` matching the
        synthetic_periodic API.
    return_meta
        If True, also returns a per-row meta dict of bookkeeping flags
        (drawn vs emitted seas, per-row seas_tied_on, etc.). Useful for
        coverage tests and plotting.

    Returns
    -------
    torch.Tensor of shape ``[batch_size, T_raw, C]``, float32, or a
    tuple including labels and/or meta per the flags above.
    """
    rng = _as_rng(rng if rng is not None else seed)
    N = batch_size * C

    drawn_seas = rng.integers(0, NUM_SEASONALITIES, size=batch_size).astype(np.int64)
    drawn_freq = rng.integers(1, NUM_FREQS, size=batch_size).astype(np.int64)
    seas_tied_on_row = (rng.random(size=batch_size) < p_seas_tied)

    out = np.empty((batch_size, C, T_raw), dtype=np.float64)
    per_channel_meta: list[dict] = []

    for b in range(batch_size):
        for c in range(C):
            channel, ch_meta = _build_one_channel(
                rng, T_raw,
                drawn_seas_id=int(drawn_seas[b]),
                seas_tied_on=bool(seas_tied_on_row[b]),
                slope_std=slope_std,
                arma_dimension=arma_dimension,
                p_arma=p_arma,
                p_free=p_free,
                p_integrate=p_integrate,
                p_trend_mult=p_trend_mult,
                arma_target_std_range=arma_target_std_range,
                free_spp_range=free_spp_range,
                enable_pulse=enable_pulse,
                pulse_width=pulse_width,
                n_free_waves=n_free_waves,
                n_seas_tied_waves=n_seas_tied_waves,
            )
            out[b, c] = channel
            per_channel_meta.append(ch_meta)

    # -- Final exponential envelope (per-(b,c)) -----------------------------
    use_env = rng.random(size=N) < p_env
    env_gain = np.exp(rng.uniform(np.log(env_gain_range[0]),
                                  np.log(env_gain_range[1]),
                                  size=N))
    lam = np.log(env_gain) / max(T_raw - 1, 1)
    t = np.arange(T_raw, dtype=np.float64).reshape(1, T_raw)
    env = np.exp(lam.reshape(N, 1) * t)
    flat = out.reshape(N, T_raw)
    flat = np.where(use_env.reshape(N, 1), flat * env, flat)

    # -- Final log-uniform scale (per-(b,c)) --------------------------------
    scale = np.exp(rng.uniform(np.log(scale_range[0]),
                               np.log(scale_range[1]),
                               size=N))
    flat = flat * scale.reshape(N, 1)

    X = flat.reshape(batch_size, C, T_raw).transpose(0, 2, 1)
    X = X.astype(np.float32, copy=False)
    X_t = torch.from_numpy(np.ascontiguousarray(X))

    # -- Labels: emit drawn seas only when seas-tied wave was on ------------
    out_tuple: list = [X_t]
    if return_labels:
        emitted_seas = np.where(seas_tied_on_row, drawn_seas, 0).astype(np.int64)
        out_tuple.append(torch.from_numpy(drawn_freq))
        out_tuple.append(torch.from_numpy(emitted_seas))

    if return_meta:
        meta = dict(
            drawn_seas_id=drawn_seas,
            drawn_freq_id=drawn_freq,
            seas_tied_on_row=seas_tied_on_row,
            emitted_seas_id=np.where(seas_tied_on_row, drawn_seas, 0).astype(np.int64),
            use_env=use_env,
            env_gain=env_gain,
            scale=scale,
            per_channel=per_channel_meta,
        )
        out_tuple.append(meta)

    if len(out_tuple) == 1:
        return X_t
    return tuple(out_tuple)
