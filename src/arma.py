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

def _sample_ar_ma(rng: Generator, dimension: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Random (p,q) in {1..8}, random method, stable-ish AR/MA params."""
    p = rng.integers(1, dimension + 1); q = rng.integers(1, dimension + 1)
    method = "gaussian" if rng.random() < 0.5 else "uniform"
    arparams = _sample_params(p, rng, method)     # φ_1..φ_p
    maparams = _sample_params(q, rng, method)     # θ_1..θ_q
    # statsmodels expects poly form: AR: 1 - φ1 L - ...;  MA: 1 + θ1 L + ...
    ar_poly = np.r_[1.0, -arparams]
    ma_poly = np.r_[1.0,  maparams]
    return ar_poly, ma_poly

def generate_arma_batch(batch_size: int = 16, T_raw: int = 4096, C: int = 4, mean: float = 0.0, std: float = 1.0, seed: int | None = None, dimension: int = 8) -> tuple[torch.Tensor, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Returns X with shape [batch_size, T_raw, C].
    Each batch-channel combination uses independently sampled ARMA(p,q) with p,q∈{1..8}.
    Returns X with shape [batch_size, T_raw, C] and parameters with shape [batch_size * C].
    """
    rng = np.random.default_rng(seed)
    
    # Generate batch_size * C ARMA processes (one per batch-channel combination)
    parameters: list[tuple[np.ndarray, np.ndarray]] = []
    series_list = []
    for _ in range(batch_size * C):
        ar_poly, ma_poly = _sample_ar_ma(rng, dimension=dimension)
        arma = ArmaProcess(ar=ar_poly, ma=ma_poly)
        # Generate T_raw values for this ARMA process
        series = arma.generate_sample(nsample=T_raw, scale=std, distrvs=rng.standard_normal)
        series_list.append(series)
        parameters.append((ar_poly, ma_poly))
    
    # Reshape: [batch_size * C, T_raw] -> [batch_size, T_raw, C]
    X = np.array(series_list).reshape(batch_size, C, T_raw).transpose(0, 2, 1)  # [batch_size, T_raw, C]
    X = X + mean

    X = torch.from_numpy(X).to(dtype=torch.float32)
    return X, parameters
