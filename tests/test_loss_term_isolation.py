"""Tests for #382 loss-term isolation flags.

Two new config keys on the ``cosine_similarity_batch_split_pred_rep`` shape
(default 1.0 both ⇒ historical objective byte-for-byte):

* ``pred_loss_weight``: scalar multiplier on L_pred.
* ``rep_loss_weight``:  scalar multiplier on L_rep.

Setting one to 0.0 must make that term contribute exactly zero to the total
(both in value AND in the gradient w.r.t. every input tensor).
"""

from types import SimpleNamespace

import pytest
import torch

from src.loss import contrastive_latent_loss

SHAPE = "cosine_similarity_batch_split_pred_rep"


def _spec(tau=0.1, w_pred=1.0, w_rep=1.0):
    return SimpleNamespace(train_configuration={
        "contrastive_divergence_temperature": tau,
        "contrastive_latent_noise": None,
        "loss_shape": SHAPE,
        "contrastive_latent_delay": 0,
        "pred_loss_weight": w_pred,
        "rep_loss_weight": w_rep,
    })


def _latents(B=4, T=6, C=1, H=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    o = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    return f, o


class TestWeightDefaultsMatchLegacy:

    def test_both_ones_matches_no_key(self):
        """Defaults 1.0/1.0 (whether via config keys or absence of keys) must
        produce byte-for-byte the same loss as the legacy call."""
        f, o = _latents(seed=17)
        legacy_spec = SimpleNamespace(train_configuration={
            "contrastive_divergence_temperature": 0.1,
            "contrastive_latent_noise": None,
            "loss_shape": SHAPE,
            "contrastive_latent_delay": 0,
        })
        weighted = contrastive_latent_loss((f, o), False, _spec(w_pred=1.0, w_rep=1.0))
        legacy = contrastive_latent_loss((f, o), False, legacy_spec)
        assert torch.allclose(weighted, legacy, atol=0.0, rtol=0.0)


class TestPredRepDecomposition:
    """L(w_pred, w_rep) = w_pred·L_pred + w_rep·L_rep, so the loss is linear
    in each weight. Verify by testing three points on the (w_pred, w_rep)
    plane."""

    def test_additive_composition(self):
        """L(1,1) == L(1,0) + L(0,1) exactly (fp64)."""
        f, o = _latents(seed=23)
        l11 = contrastive_latent_loss((f, o), False, _spec(w_pred=1.0, w_rep=1.0))
        l10 = contrastive_latent_loss((f, o), False, _spec(w_pred=1.0, w_rep=0.0))
        l01 = contrastive_latent_loss((f, o), False, _spec(w_pred=0.0, w_rep=1.0))
        assert torch.allclose(l11, l10 + l01, atol=1e-12, rtol=0.0), (
            f"L(1,1)={l11.item()} != L(1,0)+L(0,1) = "
            f"{l10.item()} + {l01.item()} = {(l10+l01).item()}")

    def test_zero_zero_gives_zero(self):
        """Both weights 0 ⇒ loss is exactly 0.0, gradient is zero everywhere."""
        f, o = _latents(seed=29)
        f = f.requires_grad_(True)
        o = o.requires_grad_(True)
        loss = contrastive_latent_loss((f, o), False, _spec(w_pred=0.0, w_rep=0.0))
        assert loss.item() == 0.0
        loss.backward()
        assert f.grad is not None and torch.all(f.grad == 0)
        assert o.grad is not None and torch.all(o.grad == 0)

    def test_rep_arm_matches_pure_l_rep(self):
        """The `rep` arm (w_pred=0, w_rep=1) must equal the standalone L_rep
        as computed by the reference formula."""
        f, o = _latents(seed=37, C=2)
        got = contrastive_latent_loss((f, o), False, _spec(w_pred=0.0, w_rep=1.0))
        want = _reference_l_rep(f, o, tau=0.1)
        assert torch.allclose(got, want, atol=1e-9, rtol=1e-9), (
            f"got {got.item():.10f} want {want.item():.10f}")

    def test_pred_arm_matches_pure_l_pred(self):
        """The `pred` arm (w_pred=1, w_rep=0) must equal the standalone L_pred
        as computed by the reference formula."""
        f, o = _latents(seed=41, C=2)
        got = contrastive_latent_loss((f, o), False, _spec(w_pred=1.0, w_rep=0.0))
        want = _reference_l_pred(f, o, tau=0.1)
        assert torch.allclose(got, want, atol=1e-9, rtol=1e-9), (
            f"got {got.item():.10f} want {want.item():.10f}")


class TestZeroWeightMeansNoGradient:
    """A term with weight 0 must contribute zero to the gradient in every
    direction. Not just numerically zero — the multiplication by 0 must not
    leak any signal from the other term either."""

    def test_pred_weight_zero_matches_rep_only_gradient(self):
        """∇L when (w_pred=0, w_rep=1) == ∇L_rep alone: L_rep never touches f,
        so ∂L/∂f is exactly zero; ∂L/∂o matches the reference."""
        f, o = _latents(seed=53)
        f1 = f.clone().requires_grad_(True)
        o1 = o.clone().requires_grad_(True)
        loss_iso = contrastive_latent_loss(
            (f1, o1), False, _spec(w_pred=0.0, w_rep=1.0))
        loss_iso.backward()
        assert torch.all(f1.grad == 0), "L_rep does not depend on f — grad must be 0"

        o2 = o.clone().requires_grad_(True)
        _reference_l_rep(f.clone(), o2, tau=0.1).backward()
        assert torch.allclose(o1.grad, o2.grad, atol=1e-10)

    def test_rep_weight_zero_matches_pred_only_gradient(self):
        """∇L when (w_pred=1, w_rep=0) == ∇L_pred alone."""
        f, o = _latents(seed=59)
        f1 = f.clone().requires_grad_(True)
        o1 = o.clone().requires_grad_(True)
        loss_iso = contrastive_latent_loss(
            (f1, o1), False, _spec(w_pred=1.0, w_rep=0.0))
        loss_iso.backward()
        grad_f_iso, grad_o_iso = f1.grad.clone(), o1.grad.clone()

        f2 = f.clone().requires_grad_(True)
        o2 = o.clone().requires_grad_(True)
        _reference_l_pred(f2, o2, tau=0.1).backward()
        grad_f_ref, grad_o_ref = f2.grad, o2.grad
        assert torch.allclose(grad_f_iso, grad_f_ref, atol=1e-10)
        assert torch.allclose(grad_o_iso, grad_o_ref, atol=1e-10)


class TestScalarLinearity:

    @pytest.mark.parametrize("w_pred,w_rep", [(0.0, 1.0), (1.0, 0.0), (0.5, 2.0), (3.0, 0.7)])
    def test_linear_in_both_weights(self, w_pred, w_rep):
        """L(w_p, w_r) == w_p·L(1,0) + w_r·L(0,1) exactly."""
        f, o = _latents(seed=67)
        l = contrastive_latent_loss((f, o), False, _spec(w_pred=w_pred, w_rep=w_rep))
        l10 = contrastive_latent_loss((f, o), False, _spec(w_pred=1.0, w_rep=0.0))
        l01 = contrastive_latent_loss((f, o), False, _spec(w_pred=0.0, w_rep=1.0))
        assert torch.allclose(l, w_pred * l10 + w_rep * l01, atol=1e-11)


class TestOtherShapesIgnorePredRepWeights:
    """The two new weights must be a strict no-op for every loss shape other
    than ``..._split_pred_rep``. Confirms we only edited the split branch."""

    @pytest.mark.parametrize("shape", [
        "cosine_similarity_batch",
        "cosine_similarity_batch_no_time_neg",
        "cosine_similarity_batch_full_hh_negs_xshh_allt",
    ])
    def test_other_shape_weights_are_noop(self, shape):
        f, o = _latents(seed=71)
        tc_base = {
            "contrastive_divergence_temperature": 0.1,
            "contrastive_latent_noise": None,
            "loss_shape": shape,
            "contrastive_latent_delay": 0,
        }
        with_weights = SimpleNamespace(train_configuration={
            **tc_base, "pred_loss_weight": 0.0, "rep_loss_weight": 0.0})
        no_weights = SimpleNamespace(train_configuration=tc_base)
        assert torch.allclose(
            contrastive_latent_loss((f, o), False, with_weights),
            contrastive_latent_loss((f, o), False, no_weights),
            atol=0.0, rtol=0.0,
        )


# --- Reference implementations (independent brute-force) --------------------
import torch.nn.functional as F


def _reference_l_pred(f, o, tau):
    """Standalone reference for L_pred (normalized InfoNCE with f-anchored
    families in the denominator). Independent of the production impl."""
    neg_inf = float('-inf')
    orig_norm = F.normalize(o, p=2, dim=-1)
    fore_norm = F.normalize(f, p=2, dim=-1)
    hy_hat_norm = fore_norm[:, :-1]
    hz_hat_norm = fore_norm[:, 1:]
    hy_norm = orig_norm[:, 1:]
    B = hy_norm.shape[0]

    log_pos = (hy_norm * hy_hat_norm).sum(-1) / tau
    sims_zy = torch.einsum('btih,btjh->btij', hz_hat_norm, hy_hat_norm) / tau
    log_neg_zy = torch.logsumexp(sims_zy, dim=2)
    sims_cb = torch.einsum('atch,btch->abtc', hy_hat_norm, hy_norm) / tau
    sims_cb = sims_cb.masked_fill(
        torch.eye(B, dtype=torch.bool).view(B, B, 1, 1), neg_inf)
    log_neg_cross_batch = torch.logsumexp(sims_cb, dim=1)
    negs_pred = torch.stack([log_neg_zy, log_neg_cross_batch], dim=0)
    log_neg_total_pred = torch.logsumexp(
        torch.logsumexp(negs_pred, dim=0), dim=0, keepdim=True)
    log_denom_pred = torch.logsumexp(
        torch.stack([log_pos, log_neg_total_pred.expand_as(log_pos)], dim=0), dim=0)
    return (log_denom_pred - log_pos).mean()


def _reference_l_rep(f, o, tau):
    """Standalone reference for L_rep (pooled LSE of h-anchored families,
    no positive)."""
    neg_inf = float('-inf')
    orig_norm = F.normalize(o, p=2, dim=-1)
    hx_norm = orig_norm[:, :-1]
    B, Tm1, C, _ = hx_norm.shape
    T = orig_norm.shape[1]

    sims_xx = torch.einsum('btih,btjh->btij', hx_norm, hx_norm) / tau
    sims_xx = sims_xx.masked_fill(
        torch.eye(C, dtype=torch.bool).view(1, 1, C, C), neg_inf)
    log_neg_xx = torch.logsumexp(sims_xx, dim=2)

    sims_hh = torch.einsum('btch,blch->btcl', hx_norm, orig_norm) / tau
    t_idx = torch.arange(Tm1).view(Tm1, 1)
    l_idx = torch.arange(T).view(1, T)
    sims_hh = sims_hh.masked_fill(
        (l_idx == t_idx).view(1, Tm1, 1, T), neg_inf)
    log_neg_hh_all = torch.logsumexp(sims_hh, dim=3)

    sims_xs = torch.einsum('atch,blch->abtcl', hx_norm, orig_norm) / tau
    sims_xs = sims_xs.masked_fill(
        torch.eye(B, dtype=torch.bool).view(B, B, 1, 1, 1), neg_inf)
    log_neg_xs_allt = torch.logsumexp(sims_xs, dim=(1, 4))

    negs_rep = torch.stack([log_neg_xx, log_neg_hh_all, log_neg_xs_allt], dim=0)
    return torch.logsumexp(
        torch.logsumexp(negs_rep, dim=0), dim=0, keepdim=True).mean()
