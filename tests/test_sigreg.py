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


# ---------------------------------------------------------------------------
# return_embed contract for the CPC forecaster variants. The block has
# three return sites (default transformer / 'cpc' / 'linear_cpc'); without
# coverage on the cpc variants those paths are silently unverified
# (#356-P2).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forecaster_kind", ["cpc", "linear_cpc"])
def test_transformerblock_return_embed_cpc_variants(forecaster_kind):
    from src.models import ConfigurableModel

    torch.manual_seed(0)
    C, H, W = 1, 32, 8
    model = ConfigurableModel(
        C=C, H=H, W=W, encoder_type='gru', num_layers=1, nhead=4,
        ffn_mult=2, rev_norm_span=16, num_encoder_layers=1,
        forecaster_kind=forecaster_kind, cpc_k_steps=2)
    model.eval()
    xr = torch.randn(2, 8, C, W).permute(0, 1, 3, 2).reshape(2, 8, C, W)
    out_default = model.transformer(xr)
    assert len(out_default) == 2
    f_out, o_out, e_in = model.transformer(xr, return_embed=True)
    assert e_in.shape == (2, 8, C, H)
    assert torch.allclose(f_out, out_default[0], atol=0, rtol=0)
    assert torch.allclose(o_out, out_default[1], atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Memory smoke (GPU): the per-chunk body is gradient-checkpointed so n_chunk
# is a real memory knob at production K (#356-P0/P1). Without checkpointing,
# autograd retained every chunk's `ws / cos / sin` for the cos/sin backward
# and the M=1024 + K=384 path peaked at ~11.6 GB per call at B=512 — two
# calls (e_t + h_t) ≈ 23 GB on a 24 GB card.
#
# The test shape MUST drive the chunk loop through several iterations so
# the with-/without-checkpoint memory paths diverge. With (B,T,C)=(64,256,
# 1) and n_chunk=2048, N=16384 → 8 chunks. Measured on an RTX 4090:
# checkpoint=0.82 GB, no-checkpoint=1.70 GB. Threshold 1.2 GB sits between
# the two with ~380 MB margin on both sides, so removing the checkpoint
# wrapper trips the assert (#356-round-2 P1).
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="requires a CUDA device for max_memory_allocated")
def test_sigreg_loss_memory_under_threshold_at_production_K():
    """At (B,T,C,K)=(64,256,1,384), M=1024, T_knots=17, n_chunk=2048 the
    chunk loop iterates 8 times; the forward+backward peak must sit under
    the 1.2 GB threshold. Without the per-chunk checkpoint the same call
    peaks at ~1.7 GB on this shape (and ~11 GB at B=512) — so removing
    the checkpoint wrapper trips this test."""
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    B, T, C, K = 64, 256, 1, 384
    M, T_knots, n_chunk = 1024, 17, 2048
    n_chunks_expected = (B * T * C + n_chunk - 1) // n_chunk
    assert n_chunks_expected >= 4, (
        f"shape must drive the chunk loop through multiple iterations "
        f"to discriminate the checkpoint contract; got {n_chunks_expected} "
        f"chunks (need >=4)")
    z = torch.randn(B, T, C, K, device=device, requires_grad=True)
    out = sigreg_loss(z, M=M, T_knots=T_knots, n_chunk=n_chunk)
    out.backward()
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_gb = peak_bytes / (1024 ** 3)
    assert peak_gb < 1.2, (
        f"sigreg_loss peak {peak_gb:.2f} GB exceeds 1.2 GB ceiling at "
        f"(B,T,C,K)=({B},{T},{C},{K}), M={M}, n_chunk={n_chunk} "
        f"({n_chunks_expected} chunks) — gradient checkpointing may "
        f"have regressed (no-checkpoint baseline on this shape ~1.7 GB)")
