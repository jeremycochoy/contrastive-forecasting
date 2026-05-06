"""
Synthetic data: correlated Brownian motions for the correlation-recovery experiment.

Each batch element has its own random 4x4 correlation matrix (positive off-diagonals,
1 on the diagonal, PSD by construction). The 4 channels are integrated correlated
Gaussian increments (random walks).

Returns x of shape [B, T, K] and C of shape [B, K, K].
"""

from __future__ import annotations

import numpy as np
import torch


def sample_correlation_matrices(
    B: int,
    K: int = 4,
    n_factors: int = 2,
    norm_low: float = 0.4,
    norm_high: float = 0.95,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample B random K×K correlation matrices (positive off-diagonals, PSD).

    Construction:
      L_ki ~ U[0, 1] for k=1..K, i=1..n_factors
      target_norm_k ~ U[norm_low, norm_high]
      Scale each row L_k to have ||L_k|| = target_norm_k
      diag_eps_k = 1 - ||L_k||² (in [1-norm_high², 1-norm_low²])
      C = L Lᵀ + diag(diag_eps)

    Off-diagonals are dot products of non-negative vectors — non-negative.
    Diagonals are exactly 1 by construction. Returns float32 tensor [B, K, K].
    """
    L = torch.rand(B, K, n_factors, device=device, generator=generator)
    target = norm_low + (norm_high - norm_low) * torch.rand(
        B, K, device=device, generator=generator
    )
    norms = L.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    L = L * (target.unsqueeze(-1) / norms)
    diag_eps = 1.0 - (L * L).sum(dim=-1)
    C = L @ L.transpose(-2, -1) + torch.diag_embed(diag_eps)
    return C.float()


def sample_correlation_matrices_uniform(
    B: int,
    K: int = 4,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    eig_floor: float = 1e-6,
    oversample: float = 3.0,
) -> torch.Tensor:
    """Sample B random K×K correlation matrices with **off-diagonals iid U[0, 1]**.

    Each pair entry is drawn independently from U[0, 1]; matrices that fail the
    PSD check (smallest eigenvalue ≤ `eig_floor`) are rejected and resampled.

    For K=4 the acceptance rate is ≈ 0.43; we over-sample by `oversample`× and
    refill until B accepted candidates are collected. Returns float32 [B, K, K].
    """
    iu = torch.triu_indices(K, K, offset=1, device=device)
    rows, cols = iu[0], iu[1]
    n_pairs = rows.numel()
    eye = torch.eye(K, device=device, dtype=torch.float32)
    accepted = []
    needed = B
    while needed > 0:
        n = max(int(needed * oversample), 16)
        tris = torch.rand(n, n_pairs, device=device, generator=generator)
        cand = eye.unsqueeze(0).repeat(n, 1, 1)
        cand[:, rows, cols] = tris
        cand[:, cols, rows] = tris
        eigmin = torch.linalg.eigvalsh(cand)[:, 0]
        ok = cand[eigmin > eig_floor]
        if ok.shape[0] == 0:
            continue
        take = ok[:needed]
        accepted.append(take)
        needed -= take.shape[0]
    return torch.cat(accepted, dim=0).float()


def generate_correlated_batch(
    batch_size: int = 16,
    T_raw: int = 4096,
    K: int = 4,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    standardize: bool = True,
    sampler: str = "factor",
    n_factors: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of correlated random walks.

    `sampler` selects the C-distribution:
    - "factor"  : 2-factor non-negative loadings (default; off-diagonals
                  concentrated around 0.3–0.4, capped near 0.9).
    - "uniform" : each off-diagonal iid U[0, 1] with PSD rejection (covers the
                  full [0, 1] range much more evenly).

    Returns
    -------
    x : torch.Tensor [B, T_raw, K]   # standardized random walks
    C : torch.Tensor [B, K, K]       # correlation matrices used to generate them
    """
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))
    else:
        gen = None

    if sampler == "uniform":
        C = sample_correlation_matrices_uniform(
            batch_size, K=K, device=device, generator=gen
        )
    elif sampler == "factor":
        C = sample_correlation_matrices(
            batch_size, K=K, n_factors=n_factors, device=device, generator=gen
        )
    else:
        raise ValueError(f"Unknown sampler {sampler!r}; expected 'factor' or 'uniform'")

    L = torch.linalg.cholesky(C)  # [B, K, K]
    z = torch.randn(batch_size, T_raw, K, device=device, generator=gen)
    eps = z @ L.transpose(-2, -1)  # eps_t = L @ z_t, applied row-wise
    x = torch.cumsum(eps, dim=1)

    if standardize:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        x = (x - mean) / std

    return x.float(), C.float()


def pairwise_indices(K: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return (rows, cols) of upper-triangular off-diagonal entries of a KxK matrix.

    For K=4 → 6 pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
    """
    iu = np.triu_indices(K, k=1)
    return iu[0], iu[1]


def correlation_to_pairs(C: torch.Tensor, K: int | None = None) -> torch.Tensor:
    """Extract upper-triangular off-diagonal entries [B, num_pairs] from [B, K, K]."""
    if K is None:
        K = C.shape[-1]
    rows, cols = pairwise_indices(K)
    return C[..., rows, cols]


def pairs_to_correlation(p: torch.Tensor, K: int = 4) -> torch.Tensor:
    """Inverse of correlation_to_pairs: rebuild full symmetric KxK with 1s on diagonal."""
    B = p.shape[0]
    out = torch.eye(K, device=p.device, dtype=p.dtype).unsqueeze(0).repeat(B, 1, 1)
    rows, cols = pairwise_indices(K)
    out[:, rows, cols] = p
    out[:, cols, rows] = p
    return out


if __name__ == "__main__":
    # Quick sanity check
    torch.manual_seed(0)
    x, C = generate_correlated_batch(batch_size=8, T_raw=4096, K=4, seed=0)
    print(f"x: {tuple(x.shape)}, dtype={x.dtype}, mean per (b,k)={x.mean(dim=1)[0].tolist()}")
    print(f"C: {tuple(C.shape)}")
    print("First correlation matrix:")
    print(C[0].numpy())
    print("Empirical pairwise correlations (from x):")
    x0 = x[0]  # [T, K]
    emp = torch.corrcoef(x0.T)
    print(emp.numpy())
    print("Diff (true - empirical):")
    print((C[0] - emp).abs().mean().item())
    pairs = correlation_to_pairs(C)
    print(f"pairs: {tuple(pairs.shape)} -> {pairs[0].tolist()}")
    rebuilt = pairs_to_correlation(pairs)
    print("Roundtrip OK:", torch.allclose(rebuilt, C))
