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
#
# Logically equivalent to src/loss.py at commit 0705c71 (head of
# `experiments` before the logsumexp refactor). The `cosine_similarity_batch`
# helper uses the broadcast form for `sims_cross_batch` for readability;
# the pre-refactor code uses a memory-optimised matmul that produces
# the same values for unit-normalised vectors. Equivalence between
# broadcast and matmul is a property of the cosine: cos(u, v) = u·v.
#
# Do NOT edit these helpers to chase test passes — they are the
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


# ---------------------------------------------------------------------------
# `include_positive_in_denominator` — the normalized InfoNCE used ONLY by the
# diagnostic `loss_tau_ref` column. Two contracts:
#   (a) when True, the loss is a proper normalized InfoNCE → always ≥ 0;
#   (b) when False (default), the loss is byte-for-byte the legacy training
#       objective (negatives-only) — this guards against any accidental
#       drift in the training loss, which must NOT change for any
#       past/running experiment.
# ---------------------------------------------------------------------------

# Variants for which `include_positive_in_denominator=True` is implemented
# (the logsumexp-form variants that compute log_pos / log_neg_total).
NORMALIZED_VARIANTS = [
    "cosine_similarity_batch",
    "cosine_similarity_batch_no_time_neg",
    "cosine_similarity_batch_square",
]

# `cosine_similarity_batch_full_fh_negs` is a NEW logsumexp-only variant: it
# has no frozen pre-refactor (`exp/sum/log`) reference, so the
# legacy-equivalence guard below cannot apply to it. It IS exercised by the
# generic normalized-form / default-path contracts (non-negativity, ≥-default,
# default==no-kwarg) at small τ — the meaningful logsumexp-stability pin for
# a variant with no legacy form.
NORMALIZED_VARIANTS_ALL = NORMALIZED_VARIANTS + [
    "cosine_similarity_batch_full_fh_negs",
]


def _new_loss_flag(fl, ol, variant, tau, include_positive_in_denominator):
    spec = SimpleNamespace(train_configuration={
        "loss_shape": variant,
        "contrastive_divergence_temperature": tau,
    })
    return contrastive_latent_loss(
        (fl, ol), validation=False, spec=spec,
        include_positive_in_denominator=include_positive_in_denominator,
    )


@pytest.mark.parametrize("variant", NORMALIZED_VARIANTS_ALL)
@pytest.mark.parametrize("tau", [0.5, 0.07])
def test_normalized_form_is_nonnegative(variant, tau):
    # Normalized InfoNCE = -log(e^pos / (e^pos + Σ_neg e^neg)). The argument
    # of the log is in (0, 1], so the loss is ≥ 0 by construction — even
    # when positives strongly separate from negatives (where the default
    # negatives-only form goes negative).
    torch.manual_seed(0)
    B, T, C, H = 6, 12, 1, 24
    fl = torch.randn(B, T, C, H, dtype=torch.float32)
    ol = torch.randn(B, T, C, H, dtype=torch.float32)
    loss_norm = _new_loss_flag(fl, ol, variant, tau, True)
    assert torch.isfinite(loss_norm), f"{variant} τ={tau}: non-finite"
    assert loss_norm.item() >= -1e-6, (
        f"{variant} τ={tau}: normalized loss must be ≥ 0, "
        f"got {loss_norm.item()}"
    )


@pytest.mark.parametrize("variant", NORMALIZED_VARIANTS_ALL)
@pytest.mark.parametrize("tau", [0.5, 0.1, 0.07])
def test_normalized_ge_default_always(variant, tau):
    # Data-independent identity: adding the positive to the denominator can
    # only enlarge it, so the normalized loss is ALWAYS ≥ the default
    # negatives-only loss:
    #   loss_norm - loss_default
    #     = mean(logsumexp([pos, negtot]) - negtot)
    #     = mean(softplus(pos - negtot)) ≥ 0.
    # This holds for every input and pins the relationship between the
    # diagnostic column and the (unchanged) training objective.
    torch.manual_seed(7)
    B, T, C, H = 5, 10, 1, 16
    fl = torch.randn(B, T, C, H, dtype=torch.float32)
    ol = torch.randn(B, T, C, H, dtype=torch.float32)
    loss_default = _new_loss_flag(fl, ol, variant, tau, False)
    loss_norm = _new_loss_flag(fl, ol, variant, tau, True)
    assert loss_norm.item() + 1e-6 >= loss_default.item(), (
        f"{variant} τ={tau}: normalized ({loss_norm.item()}) must be ≥ "
        f"default ({loss_default.item()})"
    )


def test_default_goes_negative_normalized_stays_nonneg_constructed():
    # Deterministic construction proving the change's purpose: per (b,t)
    # use orthonormal basis vectors so the (h_{t+1}, f_t) positive has
    # cos = 1 while EVERY negative pair has cos = 0. With small τ the
    # positive term 1/τ dominates log(#negatives), so the default
    # negatives-only loss is clearly NEGATIVE while the normalized form
    # stays ≥ 0. Uses cosine_similarity_batch (full negative set).
    B, T, C, H = 4, 6, 1, 64
    variant, tau = "cosine_similarity_batch", 0.05
    # Orthonormal one-hot vectors → all distinct (b,t) latents orthogonal.
    ol = torch.zeros(B, T, C, H, dtype=torch.float32)
    idx = 0
    for b in range(B):
        for t in range(T):
            ol[b, t, 0, idx % H] = 1.0
            idx += 1
    # Forecast f_t := h_{t+1} so the positive pair (hy=ol[:,1:],
    # hy_hat=fl[:,:-1]) is identical (cos=1); all other (cross-time,
    # cross-channel, cross-batch) pairs are orthogonal one-hots (cos=0).
    fl = torch.zeros_like(ol)
    fl[:, :-1, :, :] = ol[:, 1:, :, :]
    fl[:, -1, :, :] = ol[:, -1, :, :]  # unused last slot, keep finite/normable
    loss_default = _new_loss_flag(fl, ol, variant, tau, False)
    loss_norm = _new_loss_flag(fl, ol, variant, tau, True)
    assert loss_default.item() < 0.0, (
        f"expected default (negatives-only) loss < 0 in the fully "
        f"separated regime, got {loss_default.item()}"
    )
    assert loss_norm.item() >= -1e-6, (
        f"normalized loss must stay ≥ 0, got {loss_norm.item()}"
    )


@pytest.mark.parametrize("variant", NORMALIZED_VARIANTS)
@pytest.mark.parametrize("tau", [0.5, 0.1, 0.07])
def test_default_flag_matches_legacy_training_loss(variant, tau):
    # The TRAINING path must be unchanged: with the flag at its default
    # (False), the loss must equal the frozen legacy training objective
    # bit-for-bit (within fp32 tolerance). This is the regression guard
    # against accidental training-loss drift from this change.
    torch.manual_seed(0)
    B, T, C, H = 6, 12, 1, 24
    fl = torch.randn(B, T, C, H, dtype=torch.float32)
    ol = torch.randn(B, T, C, H, dtype=torch.float32)
    loss_default = _new_loss_flag(fl, ol, variant, tau, False)
    loss_legacy = LEGACY[variant](fl, ol, tau)
    assert torch.allclose(loss_default, loss_legacy, atol=1e-5, rtol=1e-5), (
        f"{variant} τ={tau}: default-flag loss drifted from legacy "
        f"training objective — new={loss_default.item()}, "
        f"legacy={loss_legacy.item()}"
    )


@pytest.mark.parametrize("variant", NORMALIZED_VARIANTS_ALL)
def test_default_flag_matches_no_flag_call(variant):
    # Passing include_positive_in_denominator=False must be identical to
    # not passing the kwarg at all (the existing training call site).
    torch.manual_seed(1)
    B, T, C, H = 4, 8, 4, 16
    tau = 0.07
    fl = torch.randn(B, T, C, H, dtype=torch.float32)
    ol = torch.randn(B, T, C, H, dtype=torch.float32)
    loss_no_kwarg = _new_loss(fl, ol, variant, tau)
    loss_false = _new_loss_flag(fl, ol, variant, tau, False)
    assert torch.equal(loss_no_kwarg, loss_false), (
        f"{variant}: default kwarg path differs from no-kwarg path"
    )


def test_normalized_flag_unsupported_variant_raises():
    # The flag is only implemented for the logsumexp-form variants. For any
    # other loss_shape it must fail loud (NotImplementedError), never
    # silently log an unintended reference metric.
    torch.manual_seed(0)
    B, T, C, H = 4, 8, 1, 16
    fl = torch.randn(B, T, C, H, dtype=torch.float32)
    ol = torch.randn(B, T, C, H, dtype=torch.float32)
    with pytest.raises(NotImplementedError):
        _new_loss_flag(fl, ol, "cosine_similarity", 0.07, True)


def test_unsupported_variant_default_flag_still_works():
    # The guard must NOT trip when the flag is False — the non-logsumexp
    # variants stay fully usable for training (unchanged behaviour).
    torch.manual_seed(0)
    B, T, C, H = 4, 8, 1, 16
    fl = torch.randn(B, T, C, H, dtype=torch.float32)
    ol = torch.randn(B, T, C, H, dtype=torch.float32)
    loss = _new_loss_flag(fl, ol, "cosine_similarity", 0.07, False)
    assert torch.isfinite(loss)
