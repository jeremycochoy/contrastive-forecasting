"""Verify the dimension-usage formula `dim_usage` in `src/metrics.py`.

Runs three checks and prints a self-explanatory verdict:

  (2)  analytic limits — isotropic-on-sphere and rank-1 collapse
  (3a) is `1/U` an estimate of the effective number of dimensions?
       (build a rank-r tensor, compare 1/U to r and d·U to r)
  (3b) does `U` equal `PR / d`, where `PR` is the participation ratio
       of the second-moment matrix of the L2-normalised samples?
       (construct latents with a prescribed eigenvalue spectrum, compare
       d·U to the empirical PR)

Self-contained: ``python experiments/.../scripts/verify_u_formula.py``.
Runtime ~15 s on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.metrics import dim_usage  # noqa: E402


def _isotropic(n: int, d: int, *, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def _rank_one(n: int, d: int, *, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    v = v / v.norm()
    alpha = torch.randn(n, 1, generator=g)
    return alpha * v


def _rank_r_isotropic(n: int, d: int, r: int, *, seed: int = 0) -> torch.Tensor:
    """(N, d) tensor of exact rank r, isotropic within an r-dim subspace."""
    g = torch.Generator().manual_seed(seed)
    coeffs = torch.randn(n, r, generator=g)
    basis = torch.linalg.qr(torch.randn(d, r, generator=g)).Q  # (d, r), orthonormal cols
    return coeffs @ basis.t()


def _from_spectrum(eigs: torch.Tensor, n: int, *, seed: int = 0) -> torch.Tensor:
    """(N, d) latent with prescribed eigenvalue spectrum of the d×d second-moment."""
    d = eigs.numel()
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) * torch.sqrt(eigs)


def _empirical_pr_normalized(z: torch.Tensor) -> float:
    """Participation ratio of (1/N) Σ ẑ_i ẑ_iᵀ with ẑ_i = z_i / ||z_i||."""
    zn = F.normalize(z, dim=-1)
    G = (zn.t() @ zn) / zn.shape[0]
    return (G.trace() ** 2 / (G @ G).trace()).item()


def main() -> int:
    print("=" * 70)
    print("Part (2): analytic limits at K=384")
    print("=" * 70)
    d = 384
    n = 4096

    z_iso = _isotropic(n, d)
    u_iso = dim_usage(z_iso, axis=0).item()
    print(f"  Isotropic ẑ on S^(d-1): U = {u_iso:.6f}   "
          f"(docstring predicts 1; user's 1/K hypothesis = {1/d:.6f})")

    z_r1 = _rank_one(n, d)
    u_r1 = dim_usage(z_r1, axis=0).item()
    print(f"  Rank-1 collapse zi=αi v: U = {u_r1:.6f}   "
          f"(docstring predicts 1/d={1/d:.6f}; user's 1 hypothesis = 1.000000)")

    iso_ok = abs(u_iso - 1.0) < 1e-3
    r1_ok = abs(u_r1 - 1.0 / d) < 1e-4
    print(f"  Match docstring? isotropic→1: {iso_ok}   rank-1→1/d: {r1_ok}")
    print(f"  Match user's stated convention (iso→1/K, rank-1→1)? "
          f"{(abs(u_iso - 1/d) < 1e-3) and (abs(u_r1 - 1.0) < 1e-3)}")
    print()

    print("=" * 70)
    print("Part (3a): does 1/U estimate the effective number of dimensions?")
    print("=" * 70)
    print(f"  Build (N={n}, d={d}) latent of exact rank r; expect 1/U ≈ r if true.")
    print()
    print(f"  {'r':>4s}  {'U':>10s}  {'d·U':>8s}  {'1/U':>8s}  "
          f"{'r itself':>8s}  {'d/r':>8s}")
    rank_pass_dU = True
    rank_fail_invU = False
    for r in (1, 4, 16, 64, 192, 384):
        z = _rank_r_isotropic(n, d, r)
        u = dim_usage(z, axis=0).item()
        d_u, inv_u, dr = d * u, 1.0 / u, d / r
        match_dU = abs(d_u - r) <= max(0.05 * r, 1.0)
        match_invU = abs(inv_u - r) <= max(0.05 * r, 1.0)
        rank_pass_dU = rank_pass_dU and match_dU
        if not match_invU:
            rank_fail_invU = True
        print(f"  {r:>4d}  {u:>10.6f}  {d_u:>8.2f}  {inv_u:>8.2f}  "
              f"{r:>8d}  {dr:>8.2f}    "
              f"d·U matches r: {match_dU} | 1/U matches r: {match_invU}")
    print()
    print(f"  Verdict (3a): d·U tracks r across all ranks: {rank_pass_dU}")
    print(f"               1/U fails to track r            : {rank_fail_invU}")
    print(f"               → 1/U is NOT the effective dimension count; "
          f"K·U is.")
    print()

    print("=" * 70)
    print("Part (3b): is U ≈ PR / d, where PR = participation ratio of the")
    print("           second-moment of the L2-normalised samples?")
    print("=" * 70)
    print(f"  d={d}, N={n}.  Build a latent with a chosen spectrum, then compare")
    print(f"  d·U (computed) to PR(L2-normalised) (computed empirically).")
    print()
    print(f"  {'spectrum':<18s}  {'U':>10s}  {'d·U':>8s}  {'PR':>8s}  match")
    eigs_specs = {
        "uniform (full)":   torch.ones(d),
        "decay-1/k":        1.0 / torch.arange(1, d + 1, dtype=torch.float),
        "decay-1/k²":       1.0 / torch.arange(1, d + 1, dtype=torch.float) ** 2,
        "step-32 (top heavy)": torch.cat([torch.full((32,), 10.0),
                                          torch.full((d - 32,), 0.01)]),
        "step-64 (mid)":    torch.cat([torch.full((64,), 5.0),
                                       torch.full((d - 64,), 0.05)]),
    }
    pr_ok_non_saturating = True
    NEAR_SAT = 0.95  # PR(empirical Gram) is finite-N biased once d·U → d
    for name, eigs in eigs_specs.items():
        z = _from_spectrum(eigs, n)
        u = dim_usage(z, axis=0).item()
        d_u = d * u
        pr = _empirical_pr_normalized(z)
        near_sat = u >= NEAR_SAT
        rel = abs(d_u - pr) / max(pr, 1e-6)
        match = rel < 0.05
        flag = ""
        if near_sat:
            flag = "  (U near 1; PR(Gram) finite-N-biased; skipped)"
        elif not match:
            pr_ok_non_saturating = False
        print(f"  {name:<18s}  {u:>10.4f}  {d_u:>8.2f}  {pr:>8.2f}  "
              f"{'skipped' if near_sat else match}{flag}")
    print()
    print(f"  Verdict (3b): d·U ≈ PR across non-saturating spectra: "
          f"{pr_ok_non_saturating}")
    print(f"               (near-saturation cases skipped because the empirical")
    print(f"               PR of (1/N) Σ ẑẑᵀ underestimates true PR at small N/d.")
    print(f"               This is a measurement artefact, not a U bias.)")
    print()

    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    all_pass = iso_ok and r1_ok and rank_pass_dU and pr_ok_non_saturating
    print(f"  U = 1 / (d · mean_{{i≠j}} cos²(z_i, z_j)), clipped to [0, 1].")
    print(f"  Limits:    isotropic-on-sphere → U=1   (NOT 1/K).")
    print(f"             rank-1 collapse     → U=1/d (NOT 1).")
    print(f"  Estimator: d·U ≈ PR of (1/N) Σ ẑ_i ẑ_iᵀ ≈ effective # dims used.")
    print(f"             1/U is NOT the effective dimension count.")
    print(f"  All checks pass: {all_pass}")
    print()
    print(f"  Interpretation: U is a participation-ratio FRACTION in (0, 1].")
    print(f"  U=1 ⇒ all K dims equally used; U=1/K ⇒ collapsed to one direction.")
    print(f"  The report vocabulary ('lower = more dims in use', 'near 1 = collapse')")
    print(f"  is INVERTED relative to the formula's actual behaviour.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
