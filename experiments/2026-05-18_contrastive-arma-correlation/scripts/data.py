"""
Joint ARMA + correlation data generator.

Per batch sample:
  - Sample 4x4 correlation matrix C with the uniform-per-pair sampler.
  - Sample 4 ARMA(p,q) processes (p,q ~ {1..4}) per channel.
  - Drive each channel's ARMA filter with Cholesky-correlated innovations
    so the channels' innovations have covariance C.
  - z-score per (b, k) so each channel has zero mean / unit variance.

Returns
-------
y : torch.Tensor  [B, T, K]   the standardized observed series
C : torch.Tensor  [B, K, K]   the correlation matrix used to mix innovations
ar_padded, ma_padded : torch.Tensor [B, K, dim]
                        AR/MA coefficients per (b, k), zero-padded to `dim`
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import lfilter

from src.correlation import sample_correlation_matrices_uniform
from src.arma import _sample_ar_ma


def generate_arma_correlated_batch(
    batch_size: int = 16,
    T_raw: int = 4096,
    K: int = 4,
    dimension: int = 4,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    standardize: bool = True,
):
    """Returns
    -------
    y   : [B, T_raw, K] z-scored observed series
    C   : [B, K, K]     correlation matrix
    ar  : [B, K, dimension]  AR coefficients (zero-padded to `dimension`)
    ma  : [B, K, dimension]  MA coefficients
    """
    rng = np.random.default_rng(seed)
    B, T = batch_size, T_raw

    # 1. Sample correlation matrix per batch element (CPU is fine; we move to
    # device at the end).
    cpu_gen = torch.Generator(device="cpu")
    if seed is not None:
        cpu_gen.manual_seed(int(seed))
    C = sample_correlation_matrices_uniform(B, K=K, device="cpu", generator=cpu_gen)
    C_np = C.numpy()
    L = np.linalg.cholesky(C_np)  # [B, K, K]

    # 2. Sample ARMA polynomials per (b, k); collect padded coefficient arrays.
    ar_pad = np.zeros((B, K, dimension), dtype=np.float32)
    ma_pad = np.zeros((B, K, dimension), dtype=np.float32)
    ar_polys: list[list[np.ndarray]] = []  # for lfilter (with leading 1)
    ma_polys: list[list[np.ndarray]] = []
    for b in range(B):
        ar_b, ma_b = [], []
        for k in range(K):
            ar_poly, ma_poly = _sample_ar_ma(rng, dimension=dimension)
            # ar_poly is [1, -φ_1, ...] (length up to dim+1); ma_poly is
            # [1, θ_1, ...].
            ar_coeffs = -ar_poly[1:]    # [φ_1, ..., φ_p]
            ma_coeffs = ma_poly[1:]     # [θ_1, ..., θ_q]
            ar_pad[b, k, :len(ar_coeffs)] = ar_coeffs
            ma_pad[b, k, :len(ma_coeffs)] = ma_coeffs
            ar_b.append(ar_poly)
            ma_b.append(ma_poly)
        ar_polys.append(ar_b)
        ma_polys.append(ma_b)

    # 3. Sample iid innovations and apply Cholesky to inject correlation.
    z = rng.standard_normal((B, T, K)).astype(np.float32)
    eps = np.einsum("btk,bjk->btj", z, L)  # [B, T, K], cov(eps_t) = C

    # 4. Apply per-(b, k) ARMA filter.
    y = np.zeros_like(eps)
    for b in range(B):
        for k in range(K):
            y[b, :, k] = lfilter(ma_polys[b][k], ar_polys[b][k], eps[b, :, k])

    # 5. Standardise per (b, k).
    if standardize:
        y_mean = y.mean(axis=1, keepdims=True)
        y_std = y.std(axis=1, keepdims=True).clip(min=1e-6)
        y = (y - y_mean) / y_std

    y_t = torch.from_numpy(y).float().to(device)
    C_t = C.to(device)
    ar_t = torch.from_numpy(ar_pad).to(device)
    ma_t = torch.from_numpy(ma_pad).to(device)
    return y_t, C_t, ar_t, ma_t


if __name__ == "__main__":
    torch.manual_seed(0)
    y, C, ar, ma = generate_arma_correlated_batch(batch_size=4, T_raw=4096, K=4, seed=0)
    print(f"y: {tuple(y.shape)}, mean per (b,k)={y.mean(dim=1)[0].tolist()}")
    print(f"C[0]:\n{C[0]}")
    print(f"AR[0,0]: {ar[0,0].tolist()}")
    print(f"MA[0,0]: {ma[0,0].tolist()}")
    # Sanity: empirical innovation correlation should approximate C if we
    # could invert the ARMA filters. As a cheap proxy use diff(y).
    diff_y = y[0].diff(dim=0)
    emp_diff = torch.corrcoef(diff_y.T)
    print(f"corrcoef(diff(y))[0]:\n{emp_diff}")
    print(f"|C - corrcoef(diff(y))| MAE: {(C[0] - emp_diff).abs().mean().item():.3f}")
