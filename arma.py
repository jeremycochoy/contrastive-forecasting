from typing import Literal
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from numpy.random import Generator
import torch

# TODO: Improve type annotations

def _sample_params(n: int, rng: Generator, method: Literal["uniform", "gaussian"]) -> np.ndarray:
    """Sample n AR/MA parameters, then shrink L1 to keep processes well-behaved."""
    if method == "uniform":
        p = rng.uniform(-1, 1, size=n)
    else:  # "gaussian"
        p = rng.standard_normal(n)
    s = np.abs(p).sum()
    if s > 0.95:  # shrink only if needed
        p *= 0.95 / s
    return p

def _sample_ar_ma(rng: Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random (p,q) in {1..8}, random method, stable-ish AR/MA params."""
    p = rng.integers(1, 9); q = rng.integers(1, 9)
    method = "gaussian" if rng.random() < 0.5 else "uniform"
    arparams = _sample_params(p, rng, method)     # φ_1..φ_p
    maparams = _sample_params(q, rng, method)     # θ_1..θ_q
    # statsmodels expects poly form: AR: 1 - φ1 L - ...;  MA: 1 + θ1 L + ...
    ar_poly = np.r_[1.0, -arparams]
    ma_poly = np.r_[1.0,  maparams]
    return ar_poly, ma_poly

def generate_arma_batch(batch_size: int = 16, T_raw: int = 4096, C: int = 4, mean: float = 0.0, std: float = 1.0, seed: int | None = None) -> tuple[torch.Tensor, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Returns X with shape [batch_size, T_raw, C].
    Each batch item uses independently sampled ARMA(p,q) with p,q∈{1..8}.
    Returns X with shape [batch_size, T_raw, C] and parameters with shape [batch_size, 2].
    """
    rng = np.random.default_rng(seed)
    X = np.empty((batch_size, T_raw, C), dtype=float)

    parameters: list[tuple[np.ndarray, np.ndarray]] = []
    for b in range(batch_size):
        ar_poly, ma_poly = _sample_ar_ma(rng)
        arma = ArmaProcess(ar=ar_poly, ma=ma_poly)
        # draw T_raw*C values (innovation std = `std`), reshape into channels, add mean
        series = arma.generate_sample(nsample=T_raw * C, scale=std, distrvs=rng.standard_normal)
        X[b] = series.reshape(T_raw, C) + mean
        parameters.append((ar_poly, ma_poly))

    X = torch.from_numpy(X).to(dtype=torch.float32)
    return X, parameters
