"""
TimesFM-style composite synthetic data generation.

Generates time series by randomly combining four components:
  (I)   Piecewise linear trend
  (II)  ARMA(p, q) via polynomial root sampling (guaranteed stable)
  (III) Sinusoid with log-uniform period
  (IV)  Sinusoid with log-uniform period

Each component is independently enabled/disabled. The trend can be
applied multiplicatively (50% chance) to create heteroskedastic series.
The final series is integrated (cumsum) to produce ARIMA(1, p, q) output.

References:
  - Das et al., "A decoder-only foundation model for time-series forecasting",
    arXiv:2310.10688, Appendix A.8
  - RevIN (Kim et al., ICLR 2022) for normalization of non-stationary input
"""

import numpy as np
from numpy.random import Generator
import torch


def generate_piecewise_linear(T: int, rng: Generator,
                              n_segments: tuple[int, int] = (3, 8),
                              slope_std: float = 0.5) -> np.ndarray:
    """Generate a piecewise linear curve of length T.

    Args:
        T: Length of the output series.
        rng: NumPy random generator.
        n_segments: (min, max) number of segments.
        slope_std: Standard deviation of Gaussian-sampled slopes.

    Returns:
        Array of shape (T,).
    """
    n = rng.integers(n_segments[0], n_segments[1] + 1)
    breakpoints = np.sort(rng.integers(0, T, size=n - 1))

    slopes = rng.standard_normal(n) * slope_std
    curve = np.zeros(T)

    prev = 0
    for i, bp in enumerate(breakpoints):
        curve[prev:bp] = slopes[i]
        prev = bp
    curve[prev:] = slopes[-1]

    return np.cumsum(curve)


def generate_sinusoid(T: int, rng: Generator) -> np.ndarray:
    """Generate a single sinusoid with log-uniform period and random phase.

    Period is sampled log-uniformly from [4, T/2], giving equal weight
    per octave of the frequency spectrum.

    Args:
        T: Length of the output series.
        rng: NumPy random generator.

    Returns:
        Array of shape (T,) with values in [-1, 1].
    """
    period = np.exp(rng.uniform(np.log(4), np.log(max(T / 2, 5))))
    phase = rng.uniform(0, 1)
    t = np.arange(T)
    return np.sin(2 * np.pi * (t / period + phase))


def _uniform_complex_roots(n: int, rng: Generator) -> np.ndarray:
    """Sample n complex roots uniformly inside the unit disk."""
    # Uniform in disk: radius = sqrt(uniform), angle = uniform
    radius = np.sqrt(rng.uniform(0, 1, n))
    angle = rng.uniform(0, 2 * np.pi, n)
    return radius * np.exp(1j * angle)


def sample_arma_from_roots(T: int, rng: Generator,
                           dimension: int = 8,
                           eps: float = 0.1) -> np.ndarray:
    """Generate an ARMA(p, q) sample using polynomial root sampling.

    AR and MA characteristic polynomial roots are sampled uniformly
    inside the unit disk, guaranteeing stability (AR) and invertibility
    (MA) by construction. Complex roots come in conjugate pairs to keep
    coefficients real.

    Args:
        T: Length of the output series.
        rng: NumPy random generator.
        dimension: Max order for p and q (sampled from [1, dimension]).
        eps: Shrink factor to push roots away from the unit circle.

    Returns:
        Array of shape (T,).
    """
    from statsmodels.tsa.arima_process import arma_generate_sample

    n_ar = rng.integers(1, dimension + 1)
    n_ma = rng.integers(1, dimension + 1)

    def _sample_poly(n_coeffs: int) -> np.ndarray:
        """Sample polynomial with roots inside the unit disk."""
        if n_coeffs == 0:
            return np.array([1.0])
        # Complex roots come in conjugate pairs; odd order gets one real root
        if n_coeffs % 2 == 0:
            roots = _uniform_complex_roots(n_coeffs // 2, rng)
            roots = np.concatenate([roots, np.conj(roots)])
        else:
            real_root = rng.uniform(-1, 1, 1)
            if n_coeffs > 1:
                complex_roots = _uniform_complex_roots((n_coeffs - 1) // 2, rng)
                roots = np.concatenate([real_root, complex_roots, np.conj(complex_roots)])
            else:
                roots = real_root
        # Shrink toward origin for numerical stability
        roots = roots / (1 + eps)
        return np.real(np.poly(roots))

    ar_poly = _sample_poly(n_ar)
    ma_poly = _sample_poly(n_ma)

    return arma_generate_sample(ar_poly, ma_poly, T,
                                distrvs=rng.standard_normal)


def _generate_single_series(T: int, rng: Generator,
                            dimension: int = 8,
                            integrate: bool = True) -> np.ndarray:
    """Generate one composite synthetic series (TimesFM recipe).

    Randomly enables/disables 4 components, combines with random
    weights, optionally applies trend multiplicatively, then integrates.

    Args:
        T: Length of the output.
        rng: NumPy random generator.
        dimension: Max ARMA order.
        integrate: If True, apply cumsum (ARIMA d=1).

    Returns:
        Array of shape (T,).
    """
    components = []
    generators = [
        lambda: generate_piecewise_linear(T, rng),
        lambda: sample_arma_from_roots(T, rng, dimension=dimension),
        lambda: generate_sinusoid(T, rng),
        lambda: generate_sinusoid(T, rng),
    ]

    # Randomly enable/disable each component
    # Trend (index 0) is always enabled; at least one other is enabled
    enabled = np.zeros(4, dtype=bool)
    enabled[0] = True
    enabled[rng.integers(1, 4)] = True  # ensure at least one non-trend
    for i in range(4):
        if not enabled[i]:
            enabled[i] = rng.random() < 0.5

    # Generate enabled components
    trend = None
    non_trend = []
    for i, (gen, on) in enumerate(zip(generators, enabled)):
        if not on:
            continue
        y = gen()
        # Normalize to unit std (skip if constant)
        std = y.std()
        if std > 1e-12:
            y = y / std
        if i == 0:
            trend = y
        else:
            non_trend.append(y)

    # Combine non-trend components with random weights
    weights = rng.uniform(0, 1, size=len(non_trend))
    total_weight = weights.sum()
    sample = np.zeros(T)
    for comp, w in zip(non_trend, weights):
        sample += comp * w

    # Apply trend
    if trend is not None:
        if rng.random() < 0.5 and not np.allclose(sample, 0):
            # Multiplicative trend (creates heteroskedasticity)
            sample = sample * trend
        else:
            trend_weight = rng.uniform(0, 1)
            total_weight += trend_weight
            sample += trend * trend_weight

    # Normalize by total weight
    if total_weight > 0:
        sample /= total_weight

    # Random scale
    std = np.exp(rng.uniform(np.log(0.1), np.log(1000)))
    sample = sample * std

    # Integrate for ARIMA(1, p, q)
    if integrate:
        sample = np.cumsum(sample)

    # Safety: retry on NaN (rare edge case)
    if np.isnan(sample).any():
        return _generate_single_series(T, rng, dimension, integrate)

    return sample


def generate_synthetic_batch(batch_size: int = 16, T_raw: int = 4096,
                             C: int = 4, seed: int | None = None,
                             dimension: int = 8,
                             integrate: bool = True,
                             ) -> tuple[torch.Tensor]:
    """Generate a batch of composite synthetic time series.

    Each batch-channel combination is an independently sampled series
    following the TimesFM recipe: piecewise trend + ARMA (root-sampled)
    + sinusoids, with optional integration (ARIMA d=1).

    Args:
        batch_size: Number of samples in the batch.
        T_raw: Length of each time series.
        C: Number of channels per sample.
        seed: Random seed for reproducibility.
        dimension: Max ARMA order (p, q in [1, dimension]).
        integrate: If True, apply cumsum for ARIMA(1, p, q).

    Returns:
        Tuple of (X,) where X has shape [batch_size, T_raw, C].
    """
    rng = np.random.default_rng(seed)

    series_list = []
    for _ in range(batch_size * C):
        series = _generate_single_series(T_raw, rng, dimension, integrate)
        series_list.append(series)

    X = np.array(series_list).reshape(batch_size, C, T_raw).transpose(0, 2, 1)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    return (X,)
