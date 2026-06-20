"""Tests for :func:`src.loss.sigreg_loss` — LeJEPA spherical SIGReg term (#355).

SIGReg averages the Epps–Pulley statistic over ``M`` random unit-direction
1-D projections of the pooled latent. Target: the projected marginal is
``N(0, 1/K)`` (the K-d-sphere marginal). Lower is better; 0 at perfect
uniformity. Tested without dependence on a real backbone.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.loss import sigreg_loss


# ---------------------------------------------------------------------------
# Smoke / shape contract
# ---------------------------------------------------------------------------


def test_sigreg_loss_is_finite_scalar():
    torch.manual_seed(0)
    z = torch.randn(64, 8, 1, 32)
    out = sigreg_loss(z, M=32, T_knots=17)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_sigreg_accepts_2d_input():
    """Pooled [N, K] input is the canonical shape; flattening happens inside."""
    torch.manual_seed(0)
    z = torch.randn(1024, 64)
    out = sigreg_loss(z, M=32, T_knots=17)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_sigreg_is_nonnegative():
    torch.manual_seed(0)
    z = torch.randn(256, 4, 1, 64)
    out = sigreg_loss(z, M=64, T_knots=17)
    assert out.item() >= 0.0


# ---------------------------------------------------------------------------
# Behaviour: uniform on the sphere should give a small statistic; collapsed
# inputs should give a large one.
# ---------------------------------------------------------------------------


def test_sigreg_uniform_on_sphere_is_small():
    """A uniform-on-S^{K-1} pool should drive the statistic toward 0."""
    torch.manual_seed(0)
    K = 64
    N = 8192
    z = F.normalize(torch.randn(N, K), p=2, dim=-1)
    # post_normalize=False because z is already on the sphere; either path
    # should now return the same value modulo the random projections.
    out = sigreg_loss(z, M=256, T_knots=17, post_normalize=False)
    assert out.item() < 1e-3, f"expected ~0 for uniform sphere; got {out.item()}"


def test_sigreg_collapsed_is_large():
    """All samples on a single direction → projected marginal is a δ."""
    torch.manual_seed(0)
    K = 64
    N = 4096
    e1 = torch.zeros(K)
    e1[0] = 1.0
    z = e1.expand(N, K).clone()
    uniform = F.normalize(torch.randn(N, K), p=2, dim=-1)
    out_collapsed = sigreg_loss(z, M=256, T_knots=17, post_normalize=False)
    out_uniform = sigreg_loss(uniform, M=256, T_knots=17, post_normalize=False)
    assert out_collapsed.item() > 10.0 * out_uniform.item(), (
        f"collapsed={out_collapsed.item()} should dominate "
        f"uniform={out_uniform.item()}")


# ---------------------------------------------------------------------------
# Auto-scaling: target variance σ² = 1/K and integration bound W = 6/√K
# ---------------------------------------------------------------------------


def test_default_sigma2_is_one_over_K():
    """sigma2 default is 1/K (the K-d-sphere projected variance)."""
    torch.manual_seed(0)
    K = 128
    N = 8192
    z = F.normalize(torch.randn(N, K), p=2, dim=-1)
    out_default = sigreg_loss(z, M=128, T_knots=17, post_normalize=False)
    out_explicit = sigreg_loss(
        z, sigma2=1.0 / K, W=6.0 / math.sqrt(K),
        M=128, T_knots=17, post_normalize=False)
    # Same random draws → exact equality; we use fresh draws each call here.
    # Both should be small for the same reason.
    assert out_default.item() < 1e-3
    assert out_explicit.item() < 1e-3


# ---------------------------------------------------------------------------
# post_normalize flag
# ---------------------------------------------------------------------------


def test_post_normalize_unit_sphere_invariance():
    """When z is already on the sphere, post_normalize=True is a no-op."""
    torch.manual_seed(123)
    K = 32
    N = 1024
    z_unit = F.normalize(torch.randn(N, K), p=2, dim=-1)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a_pre = F.normalize(torch.randn(128, K, generator=g1), p=2, dim=-1)
    a_post = F.normalize(torch.randn(128, K, generator=g2), p=2, dim=-1)
    # Use the same `a` to make the two calls byte-identical.
    out_a = sigreg_loss(z_unit, projections=a_pre, T_knots=17, post_normalize=False)
    out_b = sigreg_loss(z_unit, projections=a_post, T_knots=17, post_normalize=True)
    assert torch.isclose(out_a, out_b, atol=1e-6)


def test_post_normalize_changes_value_for_scaled_input():
    """If z is NOT unit-norm, post_normalize=True vs False give different values."""
    torch.manual_seed(0)
    K = 32
    z = F.normalize(torch.randn(2048, K), p=2, dim=-1) * 10.0  # norm = 10
    g1 = torch.Generator().manual_seed(7)
    a = F.normalize(torch.randn(128, K, generator=g1), p=2, dim=-1)
    out_raw = sigreg_loss(z, projections=a, T_knots=17, post_normalize=False)
    out_norm = sigreg_loss(z, projections=a, T_knots=17, post_normalize=True)
    assert not torch.isclose(out_raw, out_norm, rtol=0.1), (
        f"raw={out_raw.item()} normed={out_norm.item()} should differ")
    # post_normalize=True puts vectors back on the sphere → much smaller.
    assert out_norm.item() < out_raw.item()


# ---------------------------------------------------------------------------
# Gradient: SIGReg is supposed to push the encoder toward a uniform pool —
# the term must be differentiable end-to-end.
# ---------------------------------------------------------------------------


def test_sigreg_loss_admits_backward():
    torch.manual_seed(0)
    z = torch.randn(128, 16, requires_grad=True)
    out = sigreg_loss(z, M=64, T_knots=17)
    out.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum().item() > 0.0


def test_sigreg_loss_post_normalize_admits_backward():
    """post_normalize path uses F.normalize — backward must still work."""
    torch.manual_seed(0)
    z = torch.randn(128, 16, requires_grad=True)
    out = sigreg_loss(z, M=64, T_knots=17, post_normalize=True)
    out.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


# ---------------------------------------------------------------------------
# State / determinism: no buffers; the projections matter.
# ---------------------------------------------------------------------------


def test_resampled_projections_differ_per_call():
    """Without a seeded `projections` arg, two calls should differ — fresh
    randomness every forward is the contract."""
    torch.manual_seed(0)
    z = torch.randn(256, 32)
    out1 = sigreg_loss(z, M=64, T_knots=17)
    out2 = sigreg_loss(z, M=64, T_knots=17)
    # Different `a_m` sets → different EP averages (modulo astronomically
    # unlikely collision); equality would imply the call has hidden state.
    assert not torch.isclose(out1, out2, rtol=1e-4)


def test_explicit_projections_make_call_deterministic():
    torch.manual_seed(0)
    z = torch.randn(256, 32)
    a = F.normalize(torch.randn(64, 32), p=2, dim=-1)
    out1 = sigreg_loss(z, projections=a, T_knots=17)
    out2 = sigreg_loss(z, projections=a, T_knots=17)
    assert torch.isclose(out1, out2, atol=1e-6)


# ---------------------------------------------------------------------------
# Chunking over the sample axis: a chunked vs unchunked compute must match.
# ---------------------------------------------------------------------------


def test_n_chunk_matches_unchunked():
    """The N-axis chunking is an exact sum split; result must be invariant."""
    torch.manual_seed(0)
    K = 24
    N = 1000
    z = torch.randn(N, K)
    a = F.normalize(torch.randn(32, K, generator=torch.Generator().manual_seed(9)),
                    p=2, dim=-1)
    out_full = sigreg_loss(z, projections=a, T_knots=17, n_chunk=N)
    out_small = sigreg_loss(z, projections=a, T_knots=17, n_chunk=37)
    assert torch.isclose(out_full, out_small, atol=1e-5), (
        f"chunked={out_small.item()} unchunked={out_full.item()}")


# ---------------------------------------------------------------------------
# TransformerBlock.return_embed contract — SIGReg attaches to the patch-embed
# output e_t, which the block must expose without changing pre-existing
# return arities.
# ---------------------------------------------------------------------------


def test_transformerblock_return_embed_exposes_patch_embed():
    from src.models import ConfigurableModel

    torch.manual_seed(0)
    C, H, W = 1, 64, 16
    model = ConfigurableModel(
        C=C, H=H, W=W, encoder_type='gru', num_layers=1, nhead=4,
        ffn_mult=2, rev_norm_span=32, num_encoder_layers=1)
    model.eval()  # deterministic forward — no dropkey RNG advance.
    x = torch.randn(2, 128, C)
    if model.rev_norm is not None:
        x = model.rev_norm(x, mode='norm')
    B, Tr, Cc = x.shape
    T = Tr // W
    xr = x.view(B, T, W, Cc).permute(0, 1, 3, 2)
    # Without return_embed → byte-for-byte the old 2-tuple contract.
    out_default = model.transformer(xr)
    assert len(out_default) == 2
    # With return_embed → 3-tuple, the patch-embed in [B, T, C, H] layout.
    f_flat, o_flat, e_in = model.transformer(xr, return_embed=True)
    assert e_in.shape == (B, T, Cc, H), e_in.shape
    # The first two outputs must be byte-identical to the default contract.
    assert torch.allclose(f_flat, out_default[0], atol=0, rtol=0)
    assert torch.allclose(o_flat, out_default[1], atol=0, rtol=0)


def test_sigreg_on_real_patch_embed_admits_backward():
    """End-to-end: SIGReg on the GRU patch-embed output produces a gradient
    that reaches the encoder's parameters — proves the wiring."""
    from src.models import ConfigurableModel

    torch.manual_seed(0)
    C, H, W = 1, 32, 8
    model = ConfigurableModel(
        C=C, H=H, W=W, encoder_type='gru', num_layers=1, nhead=4,
        ffn_mult=2, rev_norm_span=16, num_encoder_layers=1)
    x = torch.randn(2, 64, C)
    if model.rev_norm is not None:
        x = model.rev_norm(x, mode='norm')
    B, Tr, Cc = x.shape
    T = Tr // W
    xr = x.view(B, T, W, Cc).permute(0, 1, 3, 2)
    _, _, e_in = model.transformer(xr, return_embed=True)
    out = sigreg_loss(e_in, M=64, T_knots=17)
    out.backward()
    grads_seen = [p.grad is not None and p.grad.abs().sum().item() > 0
                  for p in model.transformer.input_to_latent.parameters()]
    assert any(grads_seen), "SIGReg gradient did not reach the patch-embed."
