"""Forked-continuation ARIMA generator (#318 follow-up, user request).

Each *pair* of samples shares an exact prefix then diverges:

    x^A_{1:l} == x^B_{1:l}   (identical past, same ARMA state at the fork l)
    x^A_{>l}  != x^B_{>l}     (two continuations from that state under
                              slightly-perturbed coefficients + fresh noise)

So the encoder sees the *same* past up to l but the future latent targets
h_{l+1} diverge — the same past now maps to multiple plausible futures, which
denies the forecaster a content-free positional / minimal predictive-state
code. Integrated (ARIMA, d=1) by default. Paired samples occupy ADJACENT batch
rows (2k, 2k+1) so a pair-aware loss can avoid treating the two plausible
futures as negatives of each other.

The fork is exact: we generate the shared prefix with `scipy.signal.lfilter`,
then continue each branch with `lfiltic` to rebuild the filter state from the
shared past (y_pre, e_pre) under the *perturbed* coefficients — coefficient
changes are absorbed by lfiltic, so the state is the true past, not a
filter-specific delay vector.
"""
from __future__ import annotations

import numpy as np
import torch
from numpy.random import Generator
from scipy.signal import lfilter, lfiltic


def _sample_params(n: int, rng: Generator, method: str = "gaussian") -> np.ndarray:
    p = rng.standard_normal(n) if method == "gaussian" else rng.uniform(-1, 1, n)
    s = np.abs(p).sum()
    if s > 0.95:                       # shrink L1 to keep the process stable
        p = p * (0.95 / s)
    return p


def _perturb(params: np.ndarray, rng: Generator, sigma: float) -> np.ndarray:
    p = params + rng.normal(0.0, sigma, size=params.shape)
    s = np.abs(p).sum()
    if s > 0.95:
        p = p * (0.95 / s)
    return p


def _gen_pair(rng: Generator, T_raw: int, l: int, sigma: float, std: float,
              burn: int, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """One forked pair (pre-integration ARMA), each length T_raw."""
    p = int(rng.integers(1, dimension + 1))
    q = int(rng.integers(1, dimension + 1))
    arp = _sample_params(p, rng)
    maq = _sample_params(q, rng)
    a = np.r_[1.0, -arp]               # AR poly: 1 - φ1 L - …
    b = np.r_[1.0,  maq]               # MA poly: 1 + θ1 L + …

    # Shared prefix (burn-in then keep l samples). e_pre are the prefix innovations.
    e_full = rng.standard_normal(burn + l) * std
    y_full = lfilter(b, a, e_full)
    y_pre = y_full[burn:]              # length l  (the shared prefix, ARMA scale)
    e_pre = e_full[burn:]
    K = max(len(a), len(b)) - 1        # filter order
    n_cont = T_raw - l

    conts = []
    for _ in range(2):
        a2 = np.r_[1.0, -_perturb(arp, rng, sigma)]
        b2 = np.r_[1.0,  _perturb(maq, rng, sigma)]
        e_cont = rng.standard_normal(n_cont) * std
        if K > 0:
            # Rebuild the perturbed filter's state from the shared past
            # (most-recent-first), so the branch continues the SAME history.
            zi = lfiltic(b2, a2, y_pre[::-1][:K], e_pre[::-1][:K])
            y_cont, _ = lfilter(b2, a2, e_cont, zi=zi)
        else:
            y_cont = lfilter(b2, a2, e_cont)
        conts.append(np.concatenate([y_pre, y_cont]))
    return conts[0].astype(np.float32), conts[1].astype(np.float32)


def generate_forked_arma_batch(
    batch_size: int,
    T_raw: int = 4096,
    C: int = 1,
    seed: int | None = None,
    *,
    rng: Generator | None = None,
    integrate: bool = True,
    perturb_sigma: float = 0.05,
    std: float = 1.0,
    fork_frac_range: tuple[float, float] = (0.25, 0.75),
    burn: int = 200,
    dimension: int = 8,
    return_labels: bool = False,
):
    """Return X [batch_size, T_raw, C] float32. Adjacent rows (2k, 2k+1) are a
    forked pair sharing the prefix up to a per-pair random fork point.

    With ``return_labels`` also returns (freq_ids, seasonality_ids) as
    [batch_size] int64, both the sentinel 0 ("unknown") — a forked-ARMA series
    has no canonical frequency/seasonality.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    lo = int(round(T_raw * fork_frac_range[0]))
    hi = int(round(T_raw * fork_frac_range[1]))
    n_pairs = (batch_size + 1) // 2
    X = np.zeros((batch_size, T_raw, C), dtype=np.float32)
    for c in range(C):
        for k in range(n_pairs):
            l = int(rng.integers(lo, hi + 1))
            xa, xb = _gen_pair(rng, T_raw, l, perturb_sigma, std, burn, dimension)
            X[2 * k, :, c] = xa
            if 2 * k + 1 < batch_size:
                X[2 * k + 1, :, c] = xb
    Xt = torch.from_numpy(X)
    if integrate:                       # ARIMA(d=1): cumsum preserves the shared
        Xt = torch.cumsum(Xt, dim=1)    # prefix (depends only on x_{1:l}) and the fork
    if return_labels:
        freq = torch.zeros(batch_size, dtype=torch.int64)
        seas = torch.zeros(batch_size, dtype=torch.int64)
        return Xt, freq, seas
    return Xt
