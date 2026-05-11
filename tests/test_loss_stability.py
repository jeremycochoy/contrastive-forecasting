"""Numerical-stability + equivalence tests for the contrastive loss.

The three active variants of `contrastive_latent_loss` previously used
the manual `exp(sims/τ).sum() → log(pos/neg)` pattern, which overflows
in fp16 for τ ≲ 0.05 (and in fp32 for τ ≲ 0.012). The refactor replaces
those with `torch.logsumexp` for stability.

These tests:
  (1) freeze the pre-refactor implementations as `_legacy_*` helpers,
  (2) assert the refactored loss matches the legacy loss at "safe" τ
      where both are well-defined (fp32),
  (3) assert gradients also match,
  (4) assert the refactor is finite in fp16 at τ where the legacy NaNs
      (the bug the refactor fixes).

If you change a refactored variant, update the legacy reference here too
ONLY if the math contract changed — never to "make tests pass" by
masking a bug.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.loss import contrastive_latent_loss, cosine_similarity_from_normalized


# ---------------------------------------------------------------------------
# Frozen pre-refactor reference implementations.
# Copy of src/loss.py at commit 0705c71 (head of `experiments` before the
# logsumexp refactor). Do NOT edit to chase test passes — these are the
# contract the refactor must preserve.
# ---------------------------------------------------------------------------


def _legacy_prepare(forecasted_latent: torch.Tensor, original_latent: torch.Tensor):
    orig_norm = F.normalize(original_latent, p=2, dim=-1)
    fore_norm = F.normalize(forecasted_latent, p=2, dim=-1)
    hy_hat_norm = fore_norm[:, :-1, :, :]
    hz_hat_norm = fore_norm[:, 1:, :, :]
    hx_norm = orig_norm[:, :-1, :, :]
    hy_norm = orig_norm[:, 1:, :, :]
    return hx_norm, hy_norm, hy_hat_norm, hz_hat_norm


def _legacy_no_time_neg(fl, ol, tau):
    B, T, C, H = fl.shape
    hx_norm, hy_norm, hy_hat_norm, _ = _legacy_prepare(fl, ol)
    positives = torch.exp(
        cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau
    )
    sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
    mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
    mask_mat = mask_mat.view(1, 1, C, C)
    neg_xx = torch.exp(sims_xx / tau).masked_fill(~mask_mat, 0).sum(dim=2)

    hy_norm_exp = hy_norm.unsqueeze(0)
    hy_hat_norm_exp = hy_hat_norm.unsqueeze(1)
    sims_cross_batch = cosine_similarity_from_normalized(hy_norm_exp, hy_hat_norm_exp)
    mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
    mask_batch = mask_batch.view(B, B, 1, 1)
    neg_cross_batch_exp = torch.exp(sims_cross_batch / tau).masked_fill(~mask_batch, 0)
    neg_cross_batch = neg_cross_batch_exp.sum(dim=1)

    negatives = neg_xx + neg_cross_batch
    return -torch.log(positives / negatives.sum(dim=0, keepdim=True)).mean()


def _legacy_batch(fl, ol, tau):
    B, T, C, H = fl.shape
    hx_norm, hy_norm, hy_hat_norm, hz_hat_norm = _legacy_prepare(fl, ol)
    positives = torch.exp(
        cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau
    )

    sims_xy = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_norm.unsqueeze(2))
    neg_xy = torch.exp(sims_xy / tau).sum(dim=2)

    sims_xy_hat = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
    neg_xy_hat = torch.exp(sims_xy_hat / tau).sum(dim=2)

    sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
    mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
    mask_mat = mask_mat.view(1, 1, C, C)
    neg_xx = torch.exp(sims_xx / tau).masked_fill(~mask_mat, 0).sum(dim=2)

    sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
    neg_zy = torch.exp(sims_zy / tau).sum(dim=2)

    hy_norm_exp = hy_norm.unsqueeze(0)
    hy_hat_norm_exp = hy_hat_norm.unsqueeze(1)
    sims_cross_batch = cosine_similarity_from_normalized(hy_norm_exp, hy_hat_norm_exp)
    mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
    mask_batch = mask_batch.view(B, B, 1, 1)
    neg_cross_batch_exp = torch.exp(sims_cross_batch / tau).masked_fill(~mask_batch, 0)
    neg_cross_batch = neg_cross_batch_exp.sum(dim=1)

    negatives = neg_xy + neg_xx + neg_zy + neg_xy_hat + neg_cross_batch
    return -torch.log(positives / negatives.sum(dim=0, keepdim=True)).mean()


def _legacy_batch_square(fl, ol, tau):
    B, T, C, H = fl.shape
    hx_norm, hy_norm, hy_hat_norm, hz_hat_norm = _legacy_prepare(fl, ol)
    positives = torch.exp(
        cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau
    )

    sims_xy = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_norm.unsqueeze(2))
    neg_xy = torch.exp(sims_xy / tau).sum(dim=2)

    sims_xy_hat = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
    neg_xy_hat = torch.exp(sims_xy_hat / tau).sum(dim=2)

    sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
    mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
    mask_mat = mask_mat.view(1, 1, C, C)
    neg_xx = torch.exp(sims_xx / tau).masked_fill(~mask_mat, 0).sum(dim=2)

    sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
    neg_zy = torch.exp(sims_zy / tau).sum(dim=2)

    hy_norm_exp = hy_norm.unsqueeze(0)
    hy_hat_norm_exp = hy_hat_norm.unsqueeze(1)
    sims_cross = cosine_similarity_from_normalized(hy_norm_exp, hy_hat_norm_exp)
    mask_b = ~torch.eye(B, dtype=torch.bool, device=sims_cross.device).view(B, B, 1, 1)
    neg_cross_batch_forecast_embedding = (
        torch.exp(sims_cross / tau).masked_fill(~mask_b, 0).sum(dim=1)
    )

    f_anchor = hy_hat_norm.unsqueeze(0)
    f_other = hy_hat_norm.unsqueeze(1)
    sims_ff = cosine_similarity_from_normalized(f_anchor, f_other)
    neg_cross_batch_forecast = torch.exp(sims_ff / tau).masked_fill(~mask_b, 0).sum(dim=1)

    h_anchor = hy_norm.unsqueeze(0)
    h_other = hy_norm.unsqueeze(1)
    sims_hh = cosine_similarity_from_normalized(h_anchor, h_other)
    neg_cross_batch_embedding = torch.exp(sims_hh / tau).masked_fill(~mask_b, 0).sum(dim=1)

    negatives = (neg_xy + neg_xx + neg_zy + neg_xy_hat
                 + neg_cross_batch_forecast_embedding
                 + neg_cross_batch_forecast
                 + neg_cross_batch_embedding)
    return -torch.log(positives / negatives.sum(dim=0, keepdim=True)).mean()


LEGACY = {
    "cosine_similarity_batch_no_time_neg": _legacy_no_time_neg,
    "cosine_similarity_batch": _legacy_batch,
    "cosine_similarity_batch_square": _legacy_batch_square,
}


def _new_loss(fl, ol, variant, tau):
    spec = SimpleNamespace(train_configuration={
        "loss_shape": variant,
        "contrastive_divergence_temperature": tau,
    })
    return contrastive_latent_loss((fl, ol), validation=False, spec=spec)


# ---------------------------------------------------------------------------
# Equivalence: refactored fn == legacy fn at safe τ (fp32).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", list(LEGACY.keys()))
@pytest.mark.parametrize("tau", [0.5, 0.1, 0.07])
def test_value_matches_legacy_fp32(variant, tau):
    torch.manual_seed(0)
    B, T, C, H = 6, 12, 1, 24
    fl = torch.randn(B, T, C, H, dtype=torch.float32)
    ol = torch.randn(B, T, C, H, dtype=torch.float32)
    loss_new = _new_loss(fl, ol, variant, tau)
    loss_legacy = LEGACY[variant](fl, ol, tau)
    assert torch.allclose(loss_new, loss_legacy, atol=1e-5, rtol=1e-5), (
        f"{variant} τ={tau}: new={loss_new.item()}, legacy={loss_legacy.item()}"
    )


@pytest.mark.parametrize("variant", list(LEGACY.keys()))
@pytest.mark.parametrize("tau", [0.5, 0.07])
def test_value_matches_legacy_multichannel(variant, tau):
    # C=4 exercises the cross-channel terms (non-trivial mask_mat).
    torch.manual_seed(1)
    B, T, C, H = 4, 8, 4, 16
    fl = torch.randn(B, T, C, H, dtype=torch.float32)
    ol = torch.randn(B, T, C, H, dtype=torch.float32)
    loss_new = _new_loss(fl, ol, variant, tau)
    loss_legacy = LEGACY[variant](fl, ol, tau)
    assert torch.allclose(loss_new, loss_legacy, atol=1e-5, rtol=1e-5), (
        f"{variant} τ={tau} C={C}: new={loss_new.item()}, legacy={loss_legacy.item()}"
    )


# ---------------------------------------------------------------------------
# Equivalence: gradients match.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", list(LEGACY.keys()))
def test_grad_matches_legacy_fp32(variant):
    torch.manual_seed(0)
    B, T, C, H = 4, 8, 1, 16
    tau = 0.1
    fl_a = torch.randn(B, T, C, H, dtype=torch.float32, requires_grad=True)
    ol_a = torch.randn(B, T, C, H, dtype=torch.float32, requires_grad=True)
    fl_b = fl_a.detach().clone().requires_grad_()
    ol_b = ol_a.detach().clone().requires_grad_()

    loss_new = _new_loss(fl_a, ol_a, variant, tau)
    g_new = torch.autograd.grad(loss_new, [fl_a, ol_a])

    loss_legacy = LEGACY[variant](fl_b, ol_b, tau)
    g_legacy = torch.autograd.grad(loss_legacy, [fl_b, ol_b])

    for gn, gl, name in zip(g_new, g_legacy, ("fl", "ol")):
        assert torch.allclose(gn, gl, atol=1e-5, rtol=1e-4), (
            f"{variant}: {name} grad mismatch — max abs diff "
            f"{(gn - gl).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Stability: refactor is finite where legacy NaNs.
# Pins the bug the refactor exists to fix.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", list(LEGACY.keys()))
def test_fp16_small_tau_legacy_nans_refactor_finite(variant):
    torch.manual_seed(0)
    B, T, C, H = 8, 16, 1, 32
    fl = torch.randn(B, T, C, H, dtype=torch.float16)
    ol = torch.randn(B, T, C, H, dtype=torch.float16)
    tau = 0.03

    loss_legacy = LEGACY[variant](fl, ol, tau)
    assert not torch.isfinite(loss_legacy), (
        f"{variant}: legacy was expected to overflow at τ=0.03 fp16, "
        f"got finite {loss_legacy.item()}. Bug premise no longer holds — "
        "the refactor may not be needed for this variant."
    )

    loss_new = _new_loss(fl, ol, variant, tau)
    assert torch.isfinite(loss_new), (
        f"{variant}: refactor still produced non-finite loss "
        f"{loss_new.item()} at τ=0.03 fp16"
    )


@pytest.mark.parametrize("variant", list(LEGACY.keys()))
def test_fp32_tiny_tau_legacy_nans_refactor_finite(variant):
    # τ = 0.005 → sims/τ up to 200 → exp overflows fp32 max (~88.7).
    torch.manual_seed(0)
    B, T, C, H = 6, 12, 1, 24
    fl = torch.randn(B, T, C, H, dtype=torch.float32)
    ol = torch.randn(B, T, C, H, dtype=torch.float32)
    tau = 0.005

    loss_legacy = LEGACY[variant](fl, ol, tau)
    assert not torch.isfinite(loss_legacy), (
        f"{variant}: legacy was expected to overflow at τ=0.005 fp32, "
        f"got finite {loss_legacy.item()}."
    )

    loss_new = _new_loss(fl, ol, variant, tau)
    assert torch.isfinite(loss_new), (
        f"{variant}: refactor not finite at τ=0.005 fp32"
    )
