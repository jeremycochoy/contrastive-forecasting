"""Tests for `src.loss.contrastive_latent_loss` variants.

Focus: the `cosine_similarity_batch_add_pos_htft` variant added in PR
"loss: add cosine_similarity_batch_add_pos_htft (extra (h_t, f_t)
positive)". The new variant is identical to `cosine_similarity_batch`
except it adds (h_t, f_t) — same-channel, same-time encoder-vs-forecaster
— as an additional positive pair on top of (h_{t+1}, f_t).
"""

from types import SimpleNamespace

import pytest
import torch

from src.loss import contrastive_latent_loss


def _make_spec(loss_shape: str, tau: float = 0.07) -> SimpleNamespace:
    return SimpleNamespace(train_configuration={
        'contrastive_divergence_temperature': tau,
        'contrastive_latent_noise': None,
        'loss_shape': loss_shape,
        'contrastive_latent_delay': 0,
    })


def _random_inputs(B=2, T=3, C=2, H=8, seed=0):
    """Return (forecasted_latent, original_latent) of shape (B, T, C, H)."""
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(B, T, C, H, generator=g)
    h = torch.randn(B, T, C, H, generator=g)
    return f, h


class TestAddPosHTFT:
    def test_loss_is_finite_scalar(self):
        f, h = _random_inputs(seed=42)
        loss = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_pos_htft'))
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_baseline_finite_scalar(self):
        f, h = _random_inputs(seed=42)
        loss = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch'))
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_new_variant_differs_from_baseline(self):
        """Adding (h_t, f_t) to the numerator must change the loss value."""
        f, h = _random_inputs(seed=42)
        loss_old = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch'))
        loss_new = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_pos_htft'))
        assert not torch.allclose(loss_old, loss_new), (
            "new variant should have a measurable effect vs baseline")

    def test_perfect_h_t_match_lowers_loss(self):
        """When cos(h_t, f_t) = 1 exactly, the new positive boosts the
        numerator, so the new loss must be < the baseline loss on the
        same inputs."""
        # Build h freely, then set f := h shifted so f_t == h_t at every
        # (b, t<T-1, c). Concretely, set f[:, :-1] = h[:, :-1]; that gives
        # cos(h_t, f_t) = 1 on the (T-1) shared slice the loss uses, while
        # leaving f[:, -1] independent (it's only used through f[:, :-1]
        # in the loss, but assigning the full tensor is simplest).
        B, T, C, H = 2, 3, 2, 8
        g = torch.Generator().manual_seed(123)
        h = torch.randn(B, T, C, H, generator=g)
        f = torch.randn(B, T, C, H, generator=g)
        # f_t := h_t for t in [0, T-2] — i.e. f[:, :-1] = h[:, :-1].
        f[:, :-1, :, :] = h[:, :-1, :, :]

        loss_old = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch'))
        loss_new = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_pos_htft'))
        assert torch.isfinite(loss_old) and torch.isfinite(loss_new)
        assert loss_new.item() < loss_old.item(), (
            f"new variant should reduce loss when cos(h_t, f_t)=1; "
            f"got new={loss_new.item():.6f} old={loss_old.item():.6f}")
