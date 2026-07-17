"""Tests for `src.loss.contrastive_latent_loss` variants.

Focus: the `cosine_similarity_batch_add_pos_htft` variant added in PR
"loss: add cosine_similarity_batch_add_pos_htft (extra (h_t, f_t)
positive)". The new variant is identical to `cosine_similarity_batch`
except it adds (h_t, f_t) — same-channel, same-time encoder-vs-forecaster
— as an additional positive pair on top of (h_{t+1}, f_t).
"""

import math
import os
from types import SimpleNamespace

import pytest
import torch

from src.loss import (
    contrastive_latent_loss,
    infonce_floor,
    _effective_negative_count,
)


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


class TestAddPosHTFTAddFCrossNegs:
    """Tests for `cosine_similarity_batch_add_pos_htft_add_f_cross_negs` (Exp 4).

    This variant is *cumulative* on top of `cosine_similarity_batch_add_pos_htft`
    (Exp 3): same multi-positive numerator, same h-side negatives, plus a new
    f-side cross-(b,c) negative term at fixed t.
    """

    def test_loss_is_finite_scalar(self):
        f, h = _random_inputs(seed=42)
        loss = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec(
                'cosine_similarity_batch_add_pos_htft_add_f_cross_negs'))
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_differs_from_predecessor(self):
        """The new f-cross-bc term must change the loss vs Exp 3 predecessor."""
        f, h = _random_inputs(seed=42)
        loss_pred = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_pos_htft'))
        loss_new = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec(
                'cosine_similarity_batch_add_pos_htft_add_f_cross_negs'))
        assert torch.isfinite(loss_pred) and torch.isfinite(loss_new)
        assert not torch.allclose(loss_pred, loss_new), (
            "new f-cross-bc term should have a measurable effect vs Exp 3 "
            f"predecessor; got pred={loss_pred.item():.6f} "
            f"new={loss_new.item():.6f}")

    def test_collinear_f_raises_loss(self):
        """When all f's are collinear (cos(f_{b,c,t}, f_{b',c',t})=1 for every
        off-diagonal (b,c) pair at every t), the new neg term shoots up, so the
        new variant's loss must be strictly higher than the Exp 3 predecessor's
        on the same inputs."""
        B, T, C, H = 2, 3, 2, 8
        g = torch.Generator().manual_seed(7)
        h = torch.randn(B, T, C, H, generator=g)
        # Build f such that every f[b,t,c,:] points in the SAME direction for
        # all (b,c) at every t — concretely, copy a single per-t direction
        # across all (b,c). This guarantees normalized cos = 1 between every
        # off-diagonal (b,c) pair at every fixed t.
        per_t_dir = torch.randn(T, H, generator=g)
        f = per_t_dir.view(1, T, 1, H).expand(B, T, C, H).contiguous()

        loss_pred = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_pos_htft'))
        loss_new = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec(
                'cosine_similarity_batch_add_pos_htft_add_f_cross_negs'))
        assert torch.isfinite(loss_pred) and torch.isfinite(loss_new)
        assert loss_new.item() > loss_pred.item(), (
            f"collinear f's should raise loss via new f-cross-bc neg term; "
            f"got new={loss_new.item():.6f} pred={loss_pred.item():.6f}")

    def test_smoke_C1(self):
        """Smoke test with C=1 (actual training config). Masked-diagonal logic
        on B*C=B should not blow up — all (b,c) cross-pairs reduce to
        cross-batch only."""
        B, T, C, H = 4, 4, 1, 8
        g = torch.Generator().manual_seed(99)
        f = torch.randn(B, T, C, H, generator=g)
        h = torch.randn(B, T, C, H, generator=g)
        loss = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec(
                'cosine_similarity_batch_add_pos_htft_add_f_cross_negs'))
        assert loss.dim() == 0
        assert torch.isfinite(loss), f"got non-finite loss: {loss}"


class TestAddFCrossNegsOnly:
    """Tests for `cosine_similarity_batch_add_f_cross_negs` (Exp 4 NON-cumulative).

    This variant is identical to `cosine_similarity_batch` except the
    `negatives` term includes an extra f-side cross-(b,c) term at fixed t.
    The numerator (positives) is unchanged — does NOT include the (h_t, f_t)
    positive used by Exp 3 / the cumulative variant.
    """

    def test_loss_is_finite_scalar(self):
        f, h = _random_inputs(seed=42)
        loss = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_f_cross_negs'))
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_differs_from_baseline(self):
        """The new f-cross-bc term must change the loss vs the
        `cosine_similarity_batch` baseline."""
        f, h = _random_inputs(seed=42)
        loss_base = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch'))
        loss_new = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_f_cross_negs'))
        assert torch.isfinite(loss_base) and torch.isfinite(loss_new)
        assert not torch.allclose(loss_base, loss_new), (
            "new f-cross-bc term should have a measurable effect vs baseline; "
            f"got base={loss_base.item():.6f} new={loss_new.item():.6f}")

    def test_collinear_f_raises_loss(self):
        """When all f's are collinear (cos(f_{b,c,t}, f_{b',c',t})=1 for every
        off-diagonal (b,c) pair at every t), the new neg term shoots up, so the
        new variant's loss must be strictly higher than the baseline's on the
        same inputs."""
        B, T, C, H = 2, 3, 2, 8
        g = torch.Generator().manual_seed(7)
        h = torch.randn(B, T, C, H, generator=g)
        # Build f such that every f[b,t,c,:] points in the SAME direction for
        # all (b,c) at every t — concretely, copy a single per-t direction
        # across all (b,c). This guarantees normalized cos = 1 between every
        # off-diagonal (b,c) pair at every fixed t.
        per_t_dir = torch.randn(T, H, generator=g)
        f = per_t_dir.view(1, T, 1, H).expand(B, T, C, H).contiguous()

        loss_base = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch'))
        loss_new = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_f_cross_negs'))
        assert torch.isfinite(loss_base) and torch.isfinite(loss_new)
        assert loss_new.item() > loss_base.item(), (
            f"collinear f's should raise loss via new f-cross-bc neg term; "
            f"got new={loss_new.item():.6f} base={loss_base.item():.6f}")


class TestAddNegHTFT:
    """Tests for `cosine_similarity_batch_add_neg_htft` (corrected Exp 3).

    Identical to `cosine_similarity_batch` except `negatives` includes an
    explicit per-(b,t,c) same-channel (h_t, f_t) NEGATIVE — pushing the
    forecaster output `f_t` AWAY from the encoder output `h_t` (which f_t
    already has access to via its causal context). Numerator unchanged.

    The original Exp 3 (PR #179, `cosine_similarity_batch_add_pos_htft`)
    used the OPPOSITE sign — pulling (h_t, f_t) together — which created a
    degenerate f_t ≈ h_t shortcut. This corrected variant flips the sign.
    """

    def test_loss_is_finite_scalar(self):
        f, h = _random_inputs(seed=42)
        loss = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_neg_htft'))
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_differs_from_baseline(self):
        """The new neg term must change the loss vs the
        `cosine_similarity_batch` baseline on the same inputs."""
        f, h = _random_inputs(seed=42)
        loss_base = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch'))
        loss_new = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_neg_htft'))
        assert torch.isfinite(loss_base) and torch.isfinite(loss_new)
        assert not torch.allclose(loss_base, loss_new), (
            "new (h_t, f_t) neg term should have a measurable effect vs "
            f"baseline; got base={loss_base.item():.6f} "
            f"new={loss_new.item():.6f}")

    def test_collinear_h_t_f_t_raises_loss(self):
        """When cos(h_t, f_t) = 1 exactly (collinear), the new neg term
        inflates the negatives sum, so the new variant's loss must be
        strictly higher than the baseline's on the same inputs.

        Mirror of the `test_perfect_h_t_match_lowers_loss` test for the
        positive-Exp-3 variant — same setup (f[:, :-1] = h[:, :-1] forces
        cos(h_t, f_t) = 1) but with the OPPOSITE sign expectation: now the
        loss should go UP (penalty), not down (reward).
        """
        B, T, C, H = 2, 3, 2, 8
        g = torch.Generator().manual_seed(123)
        h = torch.randn(B, T, C, H, generator=g)
        f = torch.randn(B, T, C, H, generator=g)
        # f_t := h_t for t in [0, T-2] — i.e. f[:, :-1] = h[:, :-1].
        f[:, :-1, :, :] = h[:, :-1, :, :]

        loss_base = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch'))
        loss_new = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_neg_htft'))
        assert torch.isfinite(loss_base) and torch.isfinite(loss_new)
        assert loss_new.item() > loss_base.item(), (
            f"collinear (h_t, f_t) should raise loss via new neg term; "
            f"got new={loss_new.item():.6f} base={loss_base.item():.6f}")

    def test_smoke_C1(self):
        """Smoke with the actual training config shape (C=1)."""
        B, T, C, H = 4, 4, 1, 8
        g = torch.Generator().manual_seed(99)
        f = torch.randn(B, T, C, H, generator=g)
        h = torch.randn(B, T, C, H, generator=g)
        loss = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_neg_htft'))
        assert loss.dim() == 0
        assert torch.isfinite(loss), f"got non-finite loss: {loss}"


class TestAddSkipFNegs:
    """Tests for `cosine_similarity_batch_add_skip_f_negs` (Exp 5).

    NON-cumulative variant: identical to `cosine_similarity_batch` except the
    `negatives` term includes an extra `f_t` vs `f_{t+2}` skip-step
    forecaster term, same-(b, c). For C=1 the existing `neg_zy` already
    covers `f_t` vs `f_{t+1}` same-channel; `f_t` vs `f_{t+2}` is genuinely
    novel — not in any other negative term.
    """

    def test_loss_is_finite_scalar(self):
        # Use T=4 so T>=3 holds (skip-pair t=0..T-3 is non-empty).
        f, h = _random_inputs(B=2, T=4, C=1, H=8, seed=42)
        loss = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_skip_f_negs'))
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_differs_from_baseline(self):
        """The new skip-f neg term must change the loss vs the
        `cosine_similarity_batch` baseline on the same inputs."""
        f, h = _random_inputs(B=2, T=4, C=1, H=8, seed=42)
        loss_base = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch'))
        loss_new = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_skip_f_negs'))
        assert torch.isfinite(loss_base) and torch.isfinite(loss_new)
        assert not torch.allclose(loss_base, loss_new), (
            "new skip-f neg term should have a measurable effect vs "
            f"baseline; got base={loss_base.item():.6f} "
            f"new={loss_new.item():.6f}")

    def test_collinear_skip_pairs_raise_loss(self):
        """When `f_{t+2} == f_t` for every (b, c, t in 0..T-3), the new neg
        term cos(f_t, f_{t+2}) = 1 inflates the negatives sum on the first
        T-2 positions, so the new variant's loss must be strictly higher
        than the baseline's on the same inputs."""
        B, T, C, H = 2, 4, 1, 8
        g = torch.Generator().manual_seed(11)
        h = torch.randn(B, T, C, H, generator=g)
        f = torch.randn(B, T, C, H, generator=g)
        # Force f_{t+2} == f_t for t=0..T-3 — i.e. f[:, 2:T] = f[:, :T-2].
        f[:, 2:T, :, :] = f[:, :T - 2, :, :].clone()

        loss_base = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch'))
        loss_new = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_skip_f_negs'))
        assert torch.isfinite(loss_base) and torch.isfinite(loss_new)
        assert loss_new.item() > loss_base.item(), (
            f"collinear skip pairs should raise loss via new skip-f neg "
            f"term; got new={loss_new.item():.6f} "
            f"base={loss_base.item():.6f}")

    def test_smoke_C1_T_large(self):
        """Smoke with the actual training config shape (C=1, T patches > 3)."""
        B, T, C, H = 4, 8, 1, 16
        g = torch.Generator().manual_seed(99)
        f = torch.randn(B, T, C, H, generator=g)
        h = torch.randn(B, T, C, H, generator=g)
        loss = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_add_skip_f_negs'))
        assert loss.dim() == 0
        assert torch.isfinite(loss), f"got non-finite loss: {loss}"


class TestFullFHNegs:
    """Tests for `cosine_similarity_batch_full_fh_negs`.

    Identical to `cosine_similarity_batch` except the single l = t
    forecaster–encoder negative (`log_neg_xy_hat` = cos(h_t, f_t)) is
    REPLACED by the full set of (f_t, h_l) negatives over every time
    position l, masking out only the positive target l = t+1. At C = 1
    (the training config) the kept l = t slice equals the old
    same-channel xy_hat term, and l ∈ {0..t-1, t+2..T-1} are the
    genuinely new negatives — so the variant's denominator is a strict
    superset of the baseline's at C = 1.
    """

    def test_loss_is_finite_scalar(self):
        f, h = _random_inputs(seed=42)
        loss = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_full_fh_negs'))
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_differs_from_baseline(self):
        """Replacing the l = t-only f–h negative with the full all-l set
        must change the loss vs the `cosine_similarity_batch` baseline."""
        f, h = _random_inputs(seed=42)
        loss_base = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch'))
        loss_new = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_full_fh_negs'))
        assert torch.isfinite(loss_base) and torch.isfinite(loss_new)
        assert not torch.allclose(loss_base, loss_new), (
            "full-fh-negs should have a measurable effect vs baseline; "
            f"got base={loss_base.item():.6f} new={loss_new.item():.6f}")

    def test_strictly_above_baseline_C1(self):
        """At C = 1 the variant's f–h negative is a logsumexp over a
        strict superset of the baseline's single l = t term (every other
        loss term is byte-for-byte identical), so the variant's loss must
        be strictly greater than `cosine_similarity_batch`'s on the same
        inputs whenever T ≥ 3 (each anchor then has ≥1 extra l)."""
        f, h = _random_inputs(B=3, T=4, C=1, H=8, seed=7)
        loss_base = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch'))
        loss_new = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_full_fh_negs'))
        assert torch.isfinite(loss_base) and torch.isfinite(loss_new)
        assert loss_new.item() > loss_base.item(), (
            "superset of negatives must raise the loss at C=1; got "
            f"new={loss_new.item():.6f} base={loss_base.item():.6f}")

    def test_positive_target_excluded(self):
        """The positive target l = t+1 must be masked OUT of the (f_t, h_l)
        negative set. With B=1, C=1 the cross-(b,c) negatives degenerate to
        -inf and drop out, so the only active negatives depend purely on h
        (log_neg_xy) or are zero by orthogonal construction (log_neg_zy,
        log_neg_fh_all). Aligning f_t exactly with the positive h_{t+1}
        (and nothing else) then leaves every negative term unchanged and
        only raises log_pos by 1/τ, so the loss must drop by exactly 1/τ.
        If the mask were broken, that alignment would also inject a cos=1
        spike at l = t+1 into the negatives and this exact relation fails.
        """
        B, T, C, H = 1, 4, 1, 8
        tau = 0.07
        eye = torch.eye(H)
        # h[b, l, 0, :] = e_l  → mutually orthogonal across l.
        h = eye[:T].view(1, T, 1, H).expand(B, T, C, H).contiguous()
        # f orthogonal to every h_l (disjoint basis vectors e_T..e_{2T-1}).
        f_orth = eye[T:2 * T].view(1, T, 1, H).expand(B, T, C, H).contiguous()
        loss_orth = contrastive_latent_loss(
            (f_orth, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_full_fh_negs', tau))
        # f_t := h_{t+1} = e_{t+1} for t=0..T-2 (aligned ONLY with the
        # masked positive); f[:, T-1] left orthogonal.
        f_pos = f_orth.clone()
        f_pos[:, :T - 1, 0, :] = eye[1:T]
        loss_pos = contrastive_latent_loss(
            (f_pos, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_full_fh_negs', tau))
        assert torch.isfinite(loss_orth) and torch.isfinite(loss_pos)
        delta = (loss_orth - loss_pos).item()
        assert abs(delta - 1.0 / tau) < 1e-3, (
            "aligning f_t with the masked positive h_{t+1} must lower the "
            f"loss by exactly 1/τ ({1.0 / tau:.4f}); got Δ={delta:.4f} "
            f"(orth={loss_orth.item():.4f} pos={loss_pos.item():.4f})")

    def test_smoke_C1_with_grad(self):
        """Smoke at the real training config shape (C=1, several patches):
        forward + backward, gradients finite."""
        B, T, C, H = 4, 8, 1, 16
        g = torch.Generator().manual_seed(99)
        f = torch.randn(B, T, C, H, generator=g, requires_grad=True)
        h = torch.randn(B, T, C, H, generator=g)
        loss = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_full_fh_negs'))
        assert loss.dim() == 0
        assert torch.isfinite(loss), f"got non-finite loss: {loss}"
        loss.backward()
        assert f.grad is not None and torch.isfinite(f.grad).all()


class TestPosInDenominatorFlag:
    """`include_positive_in_denominator` as a *training* knob.

    The positive can be put in BOTH numerator and denominator (proper
    normalized InfoNCE, always ≥ 0) via either the function arg
    (diagnostic loss_tau_ref) OR the `train_configuration` key
    (`--pos-in-denominator` CLI flag → run-level objective). Tested on
    `cosine_similarity_batch_full_fh_negs`.
    """

    NAME = 'cosine_similarity_batch_full_fh_negs'

    def _spec_with_flag(self, value):
        spec = _make_spec(self.NAME)
        spec.train_configuration['include_positive_in_denominator'] = value
        return spec

    def test_config_key_matches_function_arg(self):
        """The config key and the function arg are OR-ed and must produce
        the identical loss value (same normalized-InfoNCE path)."""
        f, h = _random_inputs(B=3, T=4, C=1, H=8, seed=1)
        loss_arg = contrastive_latent_loss(
            (f, h), validation=False, spec=_make_spec(self.NAME),
            include_positive_in_denominator=True)
        loss_cfg = contrastive_latent_loss(
            (f, h), validation=False, spec=self._spec_with_flag(True))
        assert torch.isfinite(loss_arg) and torch.isfinite(loss_cfg)
        assert torch.allclose(loss_arg, loss_cfg), (
            f"config key != function arg: cfg={loss_cfg.item():.6f} "
            f"arg={loss_arg.item():.6f}")

    def test_normalized_form_is_nonnegative_and_differs(self):
        """With a near-perfect forecast the default negatives-only loss
        goes NEGATIVE (unbounded below), while the positive-in-denominator
        normalized form stays ≥ 0 — the whole point of the flag."""
        B, T, C, H = 2, 4, 1, 8
        g = torch.Generator().manual_seed(123)
        h = torch.randn(B, T, C, H, generator=g)
        f = torch.randn(B, T, C, H, generator=g)
        f[:, :-1, :, :] = h[:, 1:, :, :]          # f_t := h_{t+1} (perfect)
        loss_negonly = contrastive_latent_loss(
            (f, h), validation=False, spec=_make_spec(self.NAME))
        loss_norm = contrastive_latent_loss(
            (f, h), validation=False, spec=self._spec_with_flag(True))
        assert torch.isfinite(loss_negonly) and torch.isfinite(loss_norm)
        assert loss_negonly.item() < 0.0, (
            "sanity: near-perfect forecast should drive the negatives-only "
            f"loss negative; got {loss_negonly.item():.4f}")
        assert loss_norm.item() >= -1e-6, (
            f"normalized form must be ≥ 0; got {loss_norm.item():.6f}")
        assert loss_norm.item() > loss_negonly.item()

    def test_config_key_false_is_negatives_only(self):
        """An explicit False config key must be a no-op (identical to the
        default negatives-only objective) — historical runs unchanged."""
        f, h = _random_inputs(B=3, T=4, C=1, H=8, seed=2)
        loss_default = contrastive_latent_loss(
            (f, h), validation=False, spec=_make_spec(self.NAME))
        loss_false = contrastive_latent_loss(
            (f, h), validation=False, spec=self._spec_with_flag(False))
        assert torch.allclose(loss_default, loss_false)

    def test_config_key_unsupported_shape_raises(self):
        """Requesting positive-in-denominator via the config key with a
        non-logsumexp loss_shape must fail loud, not silently no-op."""
        f, h = _random_inputs(B=2, T=3, C=1, H=8, seed=3)
        spec = _make_spec('cosine_similarity')
        spec.train_configuration['include_positive_in_denominator'] = True
        with pytest.raises(NotImplementedError):
            contrastive_latent_loss((f, h), validation=False, spec=spec)


class TestFullFHNegsBackwardPerf:
    """Quantifies the backward-pass cost of replacing the single l = t
    f–h negative with the full all-l set, vs the `cosine_similarity_batch`
    baseline. Reports fwd/bwd medians + ratios; the assertion is a loose
    regression guard, not a tight benchmark (CPU timing is noisy)."""

    @staticmethod
    def _bench(loss_shape, B, T, C, H, iters):
        from time import perf_counter
        from statistics import median
        g = torch.Generator().manual_seed(0)
        f0 = torch.randn(B, T, C, H, generator=g)
        h0 = torch.randn(B, T, C, H, generator=g)
        spec = _make_spec(loss_shape)
        for _ in range(3):                                # warmup
            f = f0.clone().requires_grad_(True)
            contrastive_latent_loss((f, h0), False, spec).backward()
        fwd, bwd = [], []
        for _ in range(iters):
            f = f0.clone().requires_grad_(True)
            t0 = perf_counter()
            loss = contrastive_latent_loss((f, h0), False, spec)
            t1 = perf_counter()
            loss.backward()
            t2 = perf_counter()
            fwd.append(t1 - t0)
            bwd.append(t2 - t1)
        return median(fwd), median(bwd)

    def test_backward_slowdown_vs_baseline(self, capsys):
        # Real training PROPORTIONS: the loss sees latents of shape
        # [B, T, C, H] where T is the PATCH count = t_raw // W. The
        # production run uses --t-raw 4096 with W=16 (MODEL_CONFIG) ⇒
        # T = 4096/16 = 256 (NOT /6 — the 6 in the run script is
        # --n-heads/--num-layers), H = d_model = 384, C = 1. Batch is
        # cut to 32 (the B² cross-batch term dominates wall time) so the
        # test stays ~1s; the bwd ratio is shape-stable (~1.1× measured
        # at B=32→256, T=64→256), so this faithfully answers "how much
        # slower is the backward pass at training shape".
        B, T, C, H, iters = 32, 256, 1, 384, 8
        base_fwd, base_bwd = self._bench(
            'cosine_similarity_batch', B, T, C, H, iters)
        new_fwd, new_bwd = self._bench(
            'cosine_similarity_batch_full_fh_negs', B, T, C, H, iters)
        bwd_ratio = new_bwd / base_bwd
        fwd_ratio = new_fwd / base_fwd
        msg = (
            f"\n[full_fh_negs perf @ B={B},T={T},C={C},H={H},iters={iters}]\n"
            f"  baseline  fwd={base_fwd*1e3:7.2f}ms  bwd={base_bwd*1e3:7.2f}ms\n"
            f"  full_fh   fwd={new_fwd*1e3:7.2f}ms  bwd={new_bwd*1e3:7.2f}ms\n"
            f"  ratio     fwd×{fwd_ratio:.2f}        bwd×{bwd_ratio:.2f}")
        with capsys.disabled():
            print(msg)
        assert base_bwd > 0 and new_bwd > 0
        # Loose guard: the extra term is one [B,C,T-1,T] matmul + logsumexp,
        # comparable to the existing cross-batch term — well under 4× bwd.
        assert bwd_ratio < 4.0, f"backward {bwd_ratio:.2f}× slower:{msg}"


class TestCrossedLossSiblings:
    """Tests for the #303 sibling crossed-negative variants:

      (B) ``cosine_similarity_batch_full_hh_negs``     (h_t, h_l) ∀ l≠t
      (C) ``cosine_similarity_batch_full_ff_negs``     (f_t, f_l) ∀ l≠t
      (A)+(B) ``cosine_similarity_batch_full_fh_hh_negs``  full_fh_negs's
              (f_t, h_l) ∀ l≠t+1 term PLUS (B)'s (h_t, h_l) ∀ l≠t term.

    Each is ``cosine_similarity_batch`` with the single l=t f–h negative
    (``log_neg_xy_hat``) REPLACED by an all-time crossed term — the same
    structural transform that produces ``cosine_similarity_batch_full_fh_negs``.
    """

    B_NAME = 'cosine_similarity_batch_full_hh_negs'
    C_NAME = 'cosine_similarity_batch_full_ff_negs'
    AB_NAME = 'cosine_similarity_batch_full_fh_hh_negs'
    ALL = (B_NAME, C_NAME, AB_NAME)

    def test_finite_scalar_and_grad(self):
        """Forward + backward at the real training shape (C=1); finite."""
        for name in self.ALL:
            B, T, C, H = 4, 8, 1, 16
            g = torch.Generator().manual_seed(99)
            f = torch.randn(B, T, C, H, generator=g, requires_grad=True)
            h = torch.randn(B, T, C, H, generator=g)
            loss = contrastive_latent_loss(
                (f, h), validation=False, spec=_make_spec(name))
            assert loss.dim() == 0 and torch.isfinite(loss), name
            loss.backward()
            assert f.grad is not None and torch.isfinite(f.grad).all(), name

    def test_each_differs_from_baseline_and_from_each_other(self):
        """All three must differ from the cosine_similarity_batch baseline
        and from cosine_similarity_batch_full_fh_negs and from one another
        — they are distinct crossed-negative families, not aliases."""
        f, h = _random_inputs(B=3, T=5, C=1, H=8, seed=42)
        vals = {n: contrastive_latent_loss(
                    (f, h), validation=False, spec=_make_spec(n)).item()
                for n in ('cosine_similarity_batch',
                          'cosine_similarity_batch_full_fh_negs',
                          *self.ALL)}
        keys = list(vals)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                assert abs(vals[keys[i]] - vals[keys[j]]) > 1e-6, (
                    f"{keys[i]} == {keys[j]} ({vals[keys[i]]:.6f}); "
                    "variants must be distinct")

    def test_exact_value_orthonormal_C1(self):
        """Closed form pins the full negative composition AND the masks.

        B=1, C=1; h_l = e_l (orthonormal across l); f_t = e_{T+t}
        (orthonormal, disjoint basis from h). Then every cross pair has
        cos=0 except the masked self/positive ones, C=1 nulls log_neg_xx,
        and B=1 nulls the cross-batch term, so the negatives-only loss is
        a pure count of active negatives:

          full_hh / full_ff →  log(T+1)
          full_fh_hh        →  log(2T)

        If a mask (self l=t for h–h/f–f, positive l=t+1 for the f–h part
        of A+B) were broken, an unmasked cos=1 would inject a ~1/τ spike
        and these exact equalities fail.
        """
        B, T, C, H = 1, 5, 1, 16
        tau = 0.07
        eye = torch.eye(H)
        h = eye[:T].view(1, T, 1, H).contiguous()              # h_l = e_l
        f = eye[T:2 * T].view(1, T, 1, H).contiguous()          # f_t = e_{T+t}
        expect = {
            self.B_NAME: math.log(T + 1),
            self.C_NAME: math.log(T + 1),
            self.AB_NAME: math.log(2 * T),
        }
        for name, exp in expect.items():
            loss = contrastive_latent_loss(
                (f, h), validation=False, spec=_make_spec(name, tau))
            assert torch.isfinite(loss), name
            assert abs(loss.item() - exp) < 1e-4, (
                f"{name}: expected {exp:.6f} (orthonormal C1 closed form), "
                f"got {loss.item():.6f} — negative composition or a mask "
                "is wrong")

    def test_positive_target_excluded_in_AB_fh_term(self):
        """The (A)+(B) variant must still mask l=t+1 in its (f_t, h_l)
        sub-term. Construction (B=1,C=1): h_l = e_l, f orthogonal to h
        ⇒ all negatives cos=0; then set f_t := h_{t+1} (= the masked
        positive only). h-only terms (xy, hh_all) and zy are unchanged,
        the f–h term's sole nonzero entry sits exactly at the masked
        l=t+1, so only log_pos rises by 1/τ ⇒ loss drops by exactly 1/τ.
        A broken mask would also spike the negatives and break this.
        """
        B, T, C, H = 1, 4, 1, 8
        tau = 0.07
        eye = torch.eye(H)
        h = eye[:T].view(1, T, 1, H).expand(B, T, C, H).contiguous()
        f_orth = eye[T:2 * T].view(1, T, 1, H).expand(B, T, C, H).contiguous()
        loss_orth = contrastive_latent_loss(
            (f_orth, h), validation=False,
            spec=_make_spec(self.AB_NAME, tau))
        f_pos = f_orth.clone()
        f_pos[:, :T - 1, 0, :] = eye[1:T]                       # f_t := h_{t+1}
        loss_pos = contrastive_latent_loss(
            (f_pos, h), validation=False,
            spec=_make_spec(self.AB_NAME, tau))
        assert torch.isfinite(loss_orth) and torch.isfinite(loss_pos)
        delta = (loss_orth - loss_pos).item()
        assert abs(delta - 1.0 / tau) < 1e-3, (
            f"aligning f_t with the masked positive must drop the loss by "
            f"exactly 1/τ ({1.0 / tau:.4f}); got Δ={delta:.4f}")

    def test_AB_is_superset_of_A_and_B(self):
        """A+B's negative set = full_fh_negs's ∪ full_hh_negs's extra term.
        Adding negatives to the logsumexp denominator can only raise the
        negatives-only loss, so on identical inputs (T≥3, generic random)
        full_fh_hh_negs is strictly greater than BOTH full_fh_negs and
        full_hh_negs."""
        f, h = _random_inputs(B=3, T=4, C=1, H=8, seed=7)
        v_ab = contrastive_latent_loss(
            (f, h), validation=False, spec=_make_spec(self.AB_NAME)).item()
        v_a = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_full_fh_negs')).item()
        v_b = contrastive_latent_loss(
            (f, h), validation=False, spec=_make_spec(self.B_NAME)).item()
        assert v_ab > v_a > 0 and v_ab > v_b, (
            f"expected full_fh_hh > full_fh and > full_hh; got "
            f"ab={v_ab:.6f} a={v_a:.6f} b={v_b:.6f}")

    def test_pos_in_denominator_config_key_matches_arg_and_nonneg(self):
        """--pos-in-denominator (config key) must equal the function arg
        and yield a proper normalized InfoNCE (≥ 0) for all three."""
        for name in self.ALL:
            f, h = _random_inputs(B=3, T=4, C=1, H=8, seed=1)
            loss_arg = contrastive_latent_loss(
                (f, h), validation=False, spec=_make_spec(name),
                include_positive_in_denominator=True)
            spec_cfg = _make_spec(name)
            spec_cfg.train_configuration['include_positive_in_denominator'] = True
            loss_cfg = contrastive_latent_loss(
                (f, h), validation=False, spec=spec_cfg)
            assert torch.allclose(loss_arg, loss_cfg), name
            assert loss_cfg.item() >= -1e-6, f"{name}: normalized must be ≥0"


class TestCrossBranchAblationExtended:
    """#307 — the two extra all-time crossed-negative combos that extend
    the #303 combined branch (cross-batch axis UNCHANGED — still the
    standard f↔h `log_neg_cross_batch`):

      (B)+(C)     ``cosine_similarity_batch_full_hh_ff_negs``
                  (h_t,h_l) ∀ l≠t  AND  (f_t,f_l) ∀ l≠t  (no all-time f–h)
      (A)+(B)+(C) ``cosine_similarity_batch_full_fh_hh_ff_negs``
                  the (A) (f_t,h_l) ∀ l≠t+1 term PLUS (B) PLUS (C).

    Same structural transform as ``cosine_similarity_batch_full_fh_negs``.
    """

    BC_NAME = 'cosine_similarity_batch_full_hh_ff_negs'
    ABC_NAME = 'cosine_similarity_batch_full_fh_hh_ff_negs'
    ALL = (BC_NAME, ABC_NAME)

    def test_finite_scalar_and_grad(self):
        """Forward + backward at the real training shape (C=1); finite."""
        for name in self.ALL:
            B, T, C, H = 4, 8, 1, 16
            g = torch.Generator().manual_seed(99)
            f = torch.randn(B, T, C, H, generator=g, requires_grad=True)
            h = torch.randn(B, T, C, H, generator=g)
            loss = contrastive_latent_loss(
                (f, h), validation=False, spec=_make_spec(name))
            assert loss.dim() == 0 and torch.isfinite(loss), name
            loss.backward()
            assert f.grad is not None and torch.isfinite(f.grad).all(), name

    def test_all_seven_arms_distinct(self):
        """The full #303+#307 combined family + baseline + A must all be
        distinct functions on generic random inputs (no aliasing)."""
        f, h = _random_inputs(B=3, T=5, C=1, H=8, seed=42)
        names = ('cosine_similarity_batch',
                 'cosine_similarity_batch_full_fh_negs',
                 'cosine_similarity_batch_full_hh_negs',
                 'cosine_similarity_batch_full_ff_negs',
                 'cosine_similarity_batch_full_fh_hh_negs',
                 *self.ALL)
        vals = {n: contrastive_latent_loss(
                    (f, h), validation=False, spec=_make_spec(n)).item()
                for n in names}
        keys = list(vals)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                assert abs(vals[keys[i]] - vals[keys[j]]) > 1e-6, (
                    f"{keys[i]} == {keys[j]} ({vals[keys[i]]:.6f})")

    def test_exact_value_orthonormal_C1(self):
        """Closed form pins the full negative composition AND the masks.

        B=1, C=1; h_l = e_l (orthonormal across l), f_t = e_{T+t}
        (orthonormal, disjoint from h). Every cross pair has cos=0 except
        the masked self/positive ones; C=1 nulls log_neg_xx; B=1 nulls the
        cross-batch term. Negatives-only loss = log(count of active negs):
        each of xy, zy contributes 1; each all-time term contributes its
        (T-1) unmasked cos=0 entries:

          full_hh_ff    (B)+(C):      1+1 + (T-1)+(T-1)          = log(2T)
          full_fh_hh_ff (A)+(B)+(C):  1+1 + (T-1)+(T-1)+(T-1)    = log(3T-1)

        A broken self-mask (l=t for h–h/f–f) or positive-mask (l=t+1 for
        the f–h sub-term of A+B+C) injects an unmasked cos=1 ≈ 1/τ spike
        and these exact equalities fail.
        """
        B, T, C, H = 1, 5, 1, 16
        tau = 0.07
        eye = torch.eye(H)
        h = eye[:T].view(1, T, 1, H).contiguous()              # h_l = e_l
        f = eye[T:2 * T].view(1, T, 1, H).contiguous()          # f_t = e_{T+t}
        expect = {
            self.BC_NAME: math.log(2 * T),
            self.ABC_NAME: math.log(3 * T - 1),
        }
        for name, exp in expect.items():
            loss = contrastive_latent_loss(
                (f, h), validation=False, spec=_make_spec(name, tau))
            assert torch.isfinite(loss), name
            assert abs(loss.item() - exp) < 1e-4, (
                f"{name}: expected {exp:.6f} (orthonormal C1 closed form), "
                f"got {loss.item():.6f} — negative composition or a mask "
                "is wrong")

    def test_superset_ordering(self):
        """A+B+C ⊃ B+C ⊃ {B, C}, and A+B+C ⊃ A+B — adding negatives to a
        logsumexp denominator can only raise the negatives-only loss, so
        on identical generic inputs (T≥3) the strict-superset orderings
        hold. Pins that each combo really carries the union of its parts."""
        f, h = _random_inputs(B=3, T=4, C=1, H=8, seed=7)

        def v(n):
            return contrastive_latent_loss(
                (f, h), validation=False, spec=_make_spec(n)).item()
        bc = v(self.BC_NAME)
        abc = v(self.ABC_NAME)
        b = v('cosine_similarity_batch_full_hh_negs')
        c = v('cosine_similarity_batch_full_ff_negs')
        ab = v('cosine_similarity_batch_full_fh_hh_negs')
        assert bc > b > 0 and bc > c, f"bc={bc} b={b} c={c}"
        assert abc > bc and abc > ab, f"abc={abc} bc={bc} ab={ab}"

    def test_pos_in_denominator_config_key_matches_arg_and_nonneg(self):
        for name in self.ALL:
            f, h = _random_inputs(B=3, T=4, C=1, H=8, seed=1)
            loss_arg = contrastive_latent_loss(
                (f, h), validation=False, spec=_make_spec(name),
                include_positive_in_denominator=True)
            spec_cfg = _make_spec(name)
            spec_cfg.train_configuration['include_positive_in_denominator'] = True
            loss_cfg = contrastive_latent_loss(
                (f, h), validation=False, spec=spec_cfg)
            assert torch.allclose(loss_arg, loss_cfg), name
            assert loss_cfg.item() >= -1e-6, f"{name}: normalized must be ≥0"


class TestCrossBranchNegativeFree:
    """#307 — ``cosine_similarity_batch_full_hh_negs_xbfree``.

    Arm (B)'s all-time (h_t,h_l) ∀ l≠t transform, AND the cross-batch axis
    rebuilt the ``cosine_similarity_batch_square`` way: the f↔h cross-batch
    term ``log_neg_cross_fe`` is DROPPED; the two within-branch square
    edges (f_b↔f_b', h_{b,t+1}↔h_{b',t+1}, b≠b') are kept. Net: NO f↔h
    NEGATIVE anywhere (all-time or cross-batch); the f↔h *positive* is
    retained.
    """

    NAME = 'cosine_similarity_batch_full_hh_negs_xbfree'

    def test_finite_scalar_and_grad(self):
        B, T, C, H = 4, 8, 1, 16
        g = torch.Generator().manual_seed(99)
        f = torch.randn(B, T, C, H, generator=g, requires_grad=True)
        h = torch.randn(B, T, C, H, generator=g)
        loss = contrastive_latent_loss(
            (f, h), validation=False, spec=_make_spec(self.NAME))
        assert loss.dim() == 0 and torch.isfinite(loss)
        loss.backward()
        assert f.grad is not None and torch.isfinite(f.grad).all()

    def test_distinct_from_relatives(self):
        """Must differ from baseline, A, B, square, and the #307 combos —
        it is a distinct cross-batch composition, not an alias of (B)."""
        f, h = _random_inputs(B=3, T=5, C=1, H=8, seed=42)

        def v(n):
            return contrastive_latent_loss(
                (f, h), validation=False, spec=_make_spec(n)).item()
        mine = v(self.NAME)
        for other in ('cosine_similarity_batch',
                      'cosine_similarity_batch_square',
                      'cosine_similarity_batch_full_fh_negs',
                      'cosine_similarity_batch_full_hh_negs',
                      'cosine_similarity_batch_full_hh_ff_negs',
                      'cosine_similarity_batch_full_fh_hh_ff_negs'):
            assert abs(mine - v(other)) > 1e-6, (
                f"{self.NAME} == {other} ({mine:.6f})")

    def test_exact_value_orthonormal_B1_C1(self):
        """B=1 kills BOTH within-branch cross-batch edges (eye-mask on the
        single batch element), so this reduces to (B)'s B=1 closed form
        log(T+1): xy(1)+zy(1)+hh_all(T-1). Pins the standard terms, the
        all-time h–h builder, and its self-mask l=t."""
        B, T, C, H = 1, 5, 1, 16
        tau = 0.07
        eye = torch.eye(H)
        h = eye[:T].view(1, T, 1, H).contiguous()
        f = eye[T:2 * T].view(1, T, 1, H).contiguous()
        loss = contrastive_latent_loss(
            (f, h), validation=False, spec=_make_spec(self.NAME, tau))
        assert abs(loss.item() - math.log(T + 1)) < 1e-4, loss.item()

    def test_exact_value_orthonormal_B2_C1(self):
        """The discriminating closed form. B=2, C=1, four disjoint
        orthonormal blocks: h[0,l]=e_l, h[1,l]=e_{T+l}, f[0,t]=e_{2T+t},
        f[1,t]=e_{3T+t}. Every distinct pair has cos=0, every self-pair
        cos=1. Per anchor the active negatives are xy(1), zy(1),
        hh_all(T-1 zeros ⇒ T-1), cross_ff(1 off-diag zero ⇒ 1),
        cross_hh(1 off-diag zero ⇒ 1) ⇒ Σ = T+3; both batch rows equal so
        log_neg_total = log(2(T+3)); log_pos = 0 ⇒ loss = log(2(T+3)).

        Contrast: (B) full_hh_negs on the SAME inputs keeps the f↔h
        cross-batch term instead (1 off-diag zero ⇒ 1) and has NO
        cross_ff/cross_hh ⇒ Σ = T+2 ⇒ loss = log(2(T+2)). The exact +1
        inside the sum pins that xbfree DROPPED cross_fe and ADDED both
        cross_ff and cross_hh (and that no f↔h negative survived)."""
        T = 5
        B, C, H = 2, 1, 4 * T
        tau = 0.07
        eye = torch.eye(H)
        h = torch.empty(B, T, C, H)
        f = torch.empty(B, T, C, H)
        h[0, :, 0, :] = eye[0:T]
        h[1, :, 0, :] = eye[T:2 * T]
        f[0, :, 0, :] = eye[2 * T:3 * T]
        f[1, :, 0, :] = eye[3 * T:4 * T]
        loss = contrastive_latent_loss(
            (f, h), validation=False, spec=_make_spec(self.NAME, tau))
        assert abs(loss.item() - math.log(2 * (T + 3))) < 1e-4, (
            f"xbfree B2 closed form: expected {math.log(2*(T+3)):.6f}, got "
            f"{loss.item():.6f} — cross-batch composition (drop cross_fe, "
            "add cross_ff+cross_hh) or the h–h all-time term is wrong")
        loss_b = contrastive_latent_loss(
            (f, h), validation=False,
            spec=_make_spec('cosine_similarity_batch_full_hh_negs', tau))
        assert abs(loss_b.item() - math.log(2 * (T + 2))) < 1e-4, (
            "sanity: (B) on the same inputs must give log(2(T+2)) "
            f"(its single f↔h cross-batch term); got {loss_b.item():.6f}")

    def test_fh_positive_retained(self):
        """The f↔h *positive* (cos(h_{t+1}, f_t)) must still be the
        numerator. B=1 (cross-batch edges nulled); h_l=e_l, f orthogonal.
        Setting f_t := h_{t+1} aligns ONLY the positive (h-only xy/hh_all
        and f–f zy stay cos=0), so log_pos rises by exactly 1/τ and every
        negative is unchanged ⇒ loss drops by exactly 1/τ. If the f↔h
        positive had been removed/altered this exact relation breaks."""
        B, T, C, H = 1, 4, 1, 8
        tau = 0.07
        eye = torch.eye(H)
        h = eye[:T].view(1, T, 1, H).expand(B, T, C, H).contiguous()
        f_orth = eye[T:2 * T].view(1, T, 1, H).expand(B, T, C, H).contiguous()
        loss_orth = contrastive_latent_loss(
            (f_orth, h), validation=False, spec=_make_spec(self.NAME, tau))
        f_pos = f_orth.clone()
        f_pos[:, :T - 1, 0, :] = eye[1:T]                       # f_t := h_{t+1}
        loss_pos = contrastive_latent_loss(
            (f_pos, h), validation=False, spec=_make_spec(self.NAME, tau))
        delta = (loss_orth - loss_pos).item()
        assert abs(delta - 1.0 / tau) < 1e-3, (
            f"f↔h positive must lower loss by exactly 1/τ ({1.0/tau:.4f}); "
            f"got Δ={delta:.4f}")

    def test_pos_in_denominator_config_key_matches_arg_and_nonneg(self):
        f, h = _random_inputs(B=3, T=4, C=1, H=8, seed=1)
        loss_arg = contrastive_latent_loss(
            (f, h), validation=False, spec=_make_spec(self.NAME),
            include_positive_in_denominator=True)
        spec_cfg = _make_spec(self.NAME)
        spec_cfg.train_configuration['include_positive_in_denominator'] = True
        loss_cfg = contrastive_latent_loss(
            (f, h), validation=False, spec=spec_cfg)
        assert torch.allclose(loss_arg, loss_cfg)
        assert loss_cfg.item() >= -1e-6


class TestCrossSeriesSameStepHH:
    """#318 — ``cosine_similarity_batch_full_hh_negs_xshh``.

    β (``cosine_similarity_batch_full_hh_negs``) with exactly two edits:
      (1) ADD cross-series, same-step h↔h negatives cos(h_{b,t}, h_{b',t})
          ∀ b' ≠ b, anchored at h_t (every step l = t).
      (2) REMOVE the adjacent ``log_neg_xy`` = cos(h_t, h_{t+1}); at C = 1 it
          is byte-for-byte the l = t+1 slice already in (B)'s all-time term.
    Everything else (log_pos, xx, zy, all-time hh, cross-batch f↔h) = (B).
    """

    NAME = 'cosine_similarity_batch_full_hh_negs_xshh'

    @staticmethod
    def _ref_loss(f, h, tau, pos_in_denom=False):
        """Transparent fp64 reference for the C = 1 objective. Builds every
        negative family from its math definition (independent of the
        production branch's internal structure) and applies the SAME
        reduction (per-anchor logsumexp over families → logsumexp over the
        batch → minus log_pos, mean). Pins both edits: zy/hh_all/cross_fe are
        (B)'s, log_xy is ABSENT, log_xshh is the new cross-series same-step
        edge. C = 1 (the training config), so the cross-channel xx family is
        −inf and omitted."""
        f = f.double()
        h = h.double()
        B, T, C, H = f.shape
        assert C == 1, "reference is for the C=1 training config"
        fn = torch.nn.functional.normalize(f, dim=-1)[:, :, 0, :]   # [B,T,H]
        hn = torch.nn.functional.normalize(h, dim=-1)[:, :, 0, :]   # [B,T,H]
        f_t, f_tp1 = fn[:, :-1], fn[:, 1:]                          # [B,T-1,H]
        h_t, h_tp1 = hn[:, :-1], hn[:, 1:]                          # [B,T-1,H]
        cos = lambda a, b: (a * b).sum(-1)
        ninf = float('-inf')
        log_pos = cos(h_tp1, f_t) / tau                            # [B,T-1]
        log_zy = cos(f_tp1, f_t) / tau                             # [B,T-1]
        # all-time hh: logsumexp_{l≠t} cos(h_t, h_l)/τ
        sims_hh = torch.einsum('bth,blh->btl', h_t, hn) / tau      # [B,T-1,T]
        it = torch.arange(T - 1).view(T - 1, 1)
        il = torch.arange(T).view(1, T)
        sims_hh = sims_hh.masked_fill((il == it).view(1, T - 1, T), ninf)
        log_hh = torch.logsumexp(sims_hh, dim=2)                   # [B,T-1]
        eyeB = torch.eye(B, dtype=torch.bool).view(B, 1, B)
        # cross-batch f↔h: logsumexp_{b'≠b} cos(f_{b,t}, h_{b',t+1})/τ
        sims_cb = torch.einsum('bth,cth->btc', f_t, h_tp1) / tau   # [B,T-1,B']
        log_cb = torch.logsumexp(sims_cb.masked_fill(eyeB, ninf), dim=2)
        # NEW cross-series same-step h↔h: logsumexp_{b'≠b} cos(h_{b,t}, h_{b',t})/τ
        sims_xs = torch.einsum('bth,cth->btc', h_t, h_t) / tau     # [B,T-1,B']
        log_xs = torch.logsumexp(sims_xs.masked_fill(eyeB, ninf), dim=2)
        fams = torch.stack([log_zy, log_hh, log_cb, log_xs], dim=0)
        log_neg_per_anchor = torch.logsumexp(fams, dim=0)          # [B,T-1]
        log_neg_total = torch.logsumexp(log_neg_per_anchor, dim=0, keepdim=True)
        if pos_in_denom:
            log_denom = torch.logsumexp(
                torch.stack([log_pos, log_neg_total.expand_as(log_pos)], 0), 0)
            return (log_denom - log_pos).mean()
        return (log_neg_total - log_pos).mean()

    def test_finite_scalar_and_grad(self):
        B, T, C, H = 4, 8, 1, 16
        g = torch.Generator().manual_seed(318)
        f = torch.randn(B, T, C, H, generator=g, requires_grad=True)
        h = torch.randn(B, T, C, H, generator=g)
        loss = contrastive_latent_loss(
            (f, h), validation=False, spec=_make_spec(self.NAME))
        assert loss.dim() == 0 and torch.isfinite(loss)
        loss.backward()
        assert f.grad is not None and torch.isfinite(f.grad).all()

    def test_matches_transparent_reference(self):
        """The defining test: production loss == independent fp64 reference,
        for BOTH the negatives-only and the --pos-in-denominator objective.
        A wrong term set (kept xy, missing/incorrect xshh, altered cross-fe)
        would break this."""
        for seed in (0, 7, 42):
            f, h = _random_inputs(B=4, T=6, C=1, H=12, seed=seed)
            f, h = f.double(), h.double()
            for pid in (False, True):
                spec = _make_spec(self.NAME, tau=0.1)
                if pid:
                    spec.train_configuration['include_positive_in_denominator'] = True
                got = contrastive_latent_loss((f, h), False, spec)
                want = self._ref_loss(f, h, 0.1, pos_in_denom=pid)
                assert torch.allclose(got, want, atol=1e-9, rtol=1e-9), (
                    f"seed={seed} pid={pid}: got {got.item():.10f} "
                    f"want {want.item():.10f}")

    def test_distinct_from_relatives(self):
        """Must differ from (B), square, xbfree, the base, and A+B+C — a
        distinct composition, not an alias."""
        f, h = _random_inputs(B=3, T=5, C=1, H=8, seed=11)
        v = lambda n: contrastive_latent_loss(
            (f, h), False, _make_spec(n)).item()
        mine = v(self.NAME)
        for other in ('cosine_similarity_batch',
                      'cosine_similarity_batch_square',
                      'cosine_similarity_batch_full_hh_negs',
                      'cosine_similarity_batch_full_hh_negs_xbfree',
                      'cosine_similarity_batch_full_fh_hh_ff_negs'):
            assert abs(mine - v(other)) > 1e-6, f"{self.NAME} == {other}"

    def test_exact_value_orthonormal_B2_C1(self):
        """Disjoint orthonormal blocks (every distinct pair cos=0, self
        cos=1). Active negatives per anchor: zy(1), hh_all(T-1), cross_fe(1),
        xshh(1) ⇒ Σ = T+2; both batch rows equal ⇒ log_neg_total =
        log(2(T+2)); log_pos = 0 ⇒ loss = log(2(T+2)). xx is −inf (C=1), xy
        is removed. Contrast: xbfree on the same inputs keeps cross_ff +
        cross_hh (no f↔h) ⇒ T+3 ⇒ log(2(T+3)); the +1 difference pins that
        xshh adds exactly ONE cross-series edge and xy was dropped."""
        T = 5
        B, C, H = 2, 1, 4 * T
        tau = 0.07
        eye = torch.eye(H)
        h = torch.empty(B, T, C, H)
        f = torch.empty(B, T, C, H)
        h[0, :, 0, :] = eye[0:T]
        h[1, :, 0, :] = eye[T:2 * T]
        f[0, :, 0, :] = eye[2 * T:3 * T]
        f[1, :, 0, :] = eye[3 * T:4 * T]
        loss = contrastive_latent_loss(
            (f, h), False, _make_spec(self.NAME, tau))
        assert abs(loss.item() - math.log(2 * (T + 2))) < 1e-4, loss.item()
        loss_xb = contrastive_latent_loss(
            (f, h), False,
            _make_spec('cosine_similarity_batch_full_hh_negs_xbfree', tau))
        assert abs(loss_xb.item() - math.log(2 * (T + 3))) < 1e-4, (
            "sanity: xbfree must give log(2(T+3)) on these inputs")

    def test_drops_adjacent_xy_vs_beta(self):
        """The discriminating, human-readable case for edit (2). On disjoint
        orthonormal blocks set h_{0,1} := h_{0,0} (so the ADJACENT pair
        cos(h_{0,0}, h_{0,1}) = 1) while every cross-series and f↔h pair stays
        orthogonal. (B) carries this as the standalone log_neg_xy edge AND
        inside hh_all; this variant carries it ONLY inside hh_all (xy dropped).
        So (B)'s anchor (0,0) denominator gets the e^{1/τ} spike twice, this
        variant once ⇒ loss_xshh < loss_beta strictly."""
        T = 5
        B, C, H = 2, 1, 4 * T
        tau = 0.1
        eye = torch.eye(H)
        h = torch.empty(B, T, C, H)
        f = torch.empty(B, T, C, H)
        h[0, :, 0, :] = eye[0:T]
        h[1, :, 0, :] = eye[T:2 * T]
        f[0, :, 0, :] = eye[2 * T:3 * T]
        f[1, :, 0, :] = eye[3 * T:4 * T]
        h[0, 1, 0, :] = eye[0]                       # h_{0,1} := h_{0,0}
        mine = contrastive_latent_loss(
            (f, h), False, _make_spec(self.NAME, tau)).item()
        beta = contrastive_latent_loss(
            (f, h), False,
            _make_spec('cosine_similarity_batch_full_hh_negs', tau)).item()
        assert mine < beta - 1e-4, (
            f"dropping the duplicated adjacent xy must lower the denominator "
            f"vs (B): mine={mine:.5f} beta={beta:.5f}")

    def test_pos_in_denominator_config_key_matches_arg_and_nonneg(self):
        f, h = _random_inputs(B=3, T=4, C=1, H=8, seed=3)
        loss_arg = contrastive_latent_loss(
            (f, h), False, _make_spec(self.NAME),
            include_positive_in_denominator=True)
        spec_cfg = _make_spec(self.NAME)
        spec_cfg.train_configuration['include_positive_in_denominator'] = True
        loss_cfg = contrastive_latent_loss((f, h), False, spec_cfg)
        assert torch.allclose(loss_arg, loss_cfg)
        assert loss_cfg.item() >= -1e-6


class TestCrossSeriesAllTimeHH:
    """#318 ablation — ``cosine_similarity_batch_full_hh_negs_xshh_allt``.

    The all-time sibling of ``_xshh``: identical (β, drop adjacent xy, keep the
    rest) except the cross-series h↔h edge ranges over EVERY source step l, not
    only the same step — LSE_{b'≠b, ∀l} cos(h_{b,t}, h_{b',l})/τ, the
    cross-series analog of (B)'s within-series all-time term. ``_xshh`` is its
    l = t slice; this is the strict superset.
    """

    NAME = 'cosine_similarity_batch_full_hh_negs_xshh_allt'

    @staticmethod
    def _ref_loss(f, h, tau, pos_in_denom=False):
        """Transparent fp64 reference (C = 1). Same families/reduction as the
        _xshh reference, but the cross-series edge is all-time."""
        f = f.double()
        h = h.double()
        B, T, C, H = f.shape
        assert C == 1
        fn = torch.nn.functional.normalize(f, dim=-1)[:, :, 0, :]
        hn = torch.nn.functional.normalize(h, dim=-1)[:, :, 0, :]
        f_t, f_tp1 = fn[:, :-1], fn[:, 1:]
        h_t, h_tp1 = hn[:, :-1], hn[:, 1:]
        cos = lambda a, b: (a * b).sum(-1)
        ninf = float('-inf')
        log_pos = cos(h_tp1, f_t) / tau
        log_zy = cos(f_tp1, f_t) / tau
        sims_hh = torch.einsum('bth,blh->btl', h_t, hn) / tau
        it = torch.arange(T - 1).view(T - 1, 1)
        il = torch.arange(T).view(1, T)
        sims_hh = sims_hh.masked_fill((il == it).view(1, T - 1, T), ninf)
        log_hh = torch.logsumexp(sims_hh, dim=2)
        sims_cb = torch.einsum('bth,cth->btc', f_t, h_tp1) / tau
        eyeB = torch.eye(B, dtype=torch.bool).view(B, 1, B)
        log_cb = torch.logsumexp(sims_cb.masked_fill(eyeB, ninf), dim=2)
        # all-time cross-series: cos(h_{b,t}, h_{b',l}) ∀ l, mask whole series b'=b
        sims_xs = torch.einsum('bth,clh->btcl', h_t, hn) / tau          # [B,T-1,B',T]
        bmask = (torch.arange(B).view(B, 1, 1, 1) == torch.arange(B).view(1, 1, B, 1))
        sims_xs = sims_xs.masked_fill(bmask, ninf)
        log_xs = torch.logsumexp(sims_xs.flatten(2), dim=2)             # over (b',l)
        fams = torch.stack([log_zy, log_hh, log_cb, log_xs], dim=0)
        log_neg_per_anchor = torch.logsumexp(fams, dim=0)
        log_neg_total = torch.logsumexp(log_neg_per_anchor, dim=0, keepdim=True)
        if pos_in_denom:
            log_denom = torch.logsumexp(
                torch.stack([log_pos, log_neg_total.expand_as(log_pos)], 0), 0)
            return (log_denom - log_pos).mean()
        return (log_neg_total - log_pos).mean()

    def test_finite_scalar_and_grad(self):
        # h requires grad too: the cross-series term is gradient-checkpointed
        # and depends on h, so this exercises the checkpoint backward path.
        B, T, C, H = 4, 8, 1, 16
        g = torch.Generator().manual_seed(3180)
        f = torch.randn(B, T, C, H, generator=g, requires_grad=True)
        h = torch.randn(B, T, C, H, generator=g, requires_grad=True)
        loss = contrastive_latent_loss(
            (f, h), validation=False, spec=_make_spec(self.NAME))
        assert loss.dim() == 0 and torch.isfinite(loss)
        loss.backward()
        assert f.grad is not None and torch.isfinite(f.grad).all()
        assert h.grad is not None and torch.isfinite(h.grad).all()

    def test_chunk_size_invariant(self):
        """The checkpointed chunked LSE must be exact: XSHH_ALLT_CHUNK only
        trades memory for kernel launches, never the value."""
        f, h = _random_inputs(B=7, T=6, C=1, H=10, seed=5)
        f, h = f.double(), h.double()
        vals = []
        for ch in ("1", "3", "16"):
            os.environ['XSHH_ALLT_CHUNK'] = ch
            vals.append(contrastive_latent_loss(
                (f, h), False, _make_spec(self.NAME, 0.1)).item())
        os.environ.pop('XSHH_ALLT_CHUNK', None)
        assert max(vals) - min(vals) < 1e-9, f"chunk-size changed value: {vals}"

    def test_matches_transparent_reference(self):
        """Production == fp64 reference for both objectives, and the chunked
        logsumexp is chunk-invariant (B=5 spans multiple CH-independent
        sizes; here a single chunk, but the seeds exercise the masking)."""
        for seed in (0, 7, 42):
            f, h = _random_inputs(B=5, T=6, C=1, H=12, seed=seed)
            f, h = f.double(), h.double()
            for pid in (False, True):
                spec = _make_spec(self.NAME, tau=0.1)
                if pid:
                    spec.train_configuration['include_positive_in_denominator'] = True
                got = contrastive_latent_loss((f, h), False, spec)
                want = self._ref_loss(f, h, 0.1, pos_in_denom=pid)
                assert torch.allclose(got, want, atol=1e-9, rtol=1e-9), (
                    f"seed={seed} pid={pid}: got {got.item():.10f} want {want.item():.10f}")

    def test_distinct_from_samestep_and_relatives(self):
        """Must differ from the same-step _xshh, from (B), and from the base —
        the all-time edge is a strict superset of same-step, not an alias."""
        f, h = _random_inputs(B=3, T=5, C=1, H=8, seed=13)
        v = lambda n: contrastive_latent_loss((f, h), False, _make_spec(n)).item()
        mine = v(self.NAME)
        for other in ('cosine_similarity_batch_full_hh_negs_xshh',
                      'cosine_similarity_batch_full_hh_negs',
                      'cosine_similarity_batch'):
            assert abs(mine - v(other)) > 1e-6, f"{self.NAME} == {other}"

    def test_exact_value_orthonormal_B2_C1(self):
        """Disjoint orthonormal blocks, B=2. Active negs per anchor: zy(1),
        hh_all(T-1), cross_fe(1), xs_allt(the other series' T states ⇒ T) ⇒
        Σ = 2T+1 ⇒ loss = log(2(2T+1)). Contrast same-step _xshh = log(2(T+2)):
        the all-time edge adds T−1 more cross-series negatives (the b'≠b states
        at l ≠ t), pinning that l ranges over ALL steps, not just l=t."""
        T = 5
        B, C, H = 2, 1, 4 * T
        tau = 0.07
        eye = torch.eye(H)
        h = torch.empty(B, T, C, H); f = torch.empty(B, T, C, H)
        h[0, :, 0, :] = eye[0:T]; h[1, :, 0, :] = eye[T:2 * T]
        f[0, :, 0, :] = eye[2 * T:3 * T]; f[1, :, 0, :] = eye[3 * T:4 * T]
        loss = contrastive_latent_loss((f, h), False, _make_spec(self.NAME, tau))
        assert abs(loss.item() - math.log(2 * (2 * T + 1))) < 1e-4, loss.item()
        loss_ss = contrastive_latent_loss(
            (f, h), False,
            _make_spec('cosine_similarity_batch_full_hh_negs_xshh', tau))
        assert abs(loss_ss.item() - math.log(2 * (T + 2))) < 1e-4, (
            "sanity: same-step _xshh must give log(2(T+2)) on these inputs")

    def test_pos_in_denominator_config_key_matches_arg_and_nonneg(self):
        f, h = _random_inputs(B=3, T=4, C=1, H=8, seed=4)
        loss_arg = contrastive_latent_loss(
            (f, h), False, _make_spec(self.NAME),
            include_positive_in_denominator=True)
        spec_cfg = _make_spec(self.NAME)
        spec_cfg.train_configuration['include_positive_in_denominator'] = True
        loss_cfg = contrastive_latent_loss((f, h), False, spec_cfg)
        assert torch.allclose(loss_arg, loss_cfg)
        assert loss_cfg.item() >= -1e-6


class TestAlignLoss:
    """The BYOL/SimSiam alignment add-on (#309): L_align = (2 − 2·cos(f_t,
    sg(h_{t+1}))).mean(), weight λ = `align_loss_weight` (default 0 = off).
    `2 − 2·cos = ‖f̂ − ĥ‖²` ∈ [0, 4], minimum 0 at cos = 1 — already ≥ 0 /
    min-0 (the `2` is the built-in constant); stop-grad on the target."""

    NAME = 'cosine_similarity_batch_full_fh_negs'

    def _spec(self, align=None, tau=0.1):
        spec = _make_spec(self.NAME, tau)
        spec.train_configuration['include_positive_in_denominator'] = True
        if align is not None:
            spec.train_configuration['align_loss_weight'] = align
        return spec

    def test_default_off_is_noop(self):
        f, h = _random_inputs(B=3, T=4, C=1, H=8, seed=1)
        base = contrastive_latent_loss((f, h), False, self._spec())
        zero = contrastive_latent_loss((f, h), False, self._spec(align=0.0))
        assert torch.allclose(base, zero)

    def test_perfect_forecast_zero_contribution(self):
        # f_t := h_{t+1} ⇒ cos = 1 ⇒ L_align = 0 ⇒ loss unchanged.
        B, T, C, H = 2, 5, 1, 8
        g = torch.Generator().manual_seed(7)
        h = torch.randn(B, T, C, H, generator=g)
        f = torch.randn(B, T, C, H, generator=g)
        f[:, :-1] = h[:, 1:]
        base = contrastive_latent_loss((f, h), False, self._spec())
        withal = contrastive_latent_loss((f, h), False, self._spec(align=1.0))
        assert abs((withal - base).item()) < 1e-5

    def test_orthogonal_adds_two_lambda(self):
        # f_t ⊥ h_{t+1} ⇒ cos = 0 ⇒ L_align = 2 ⇒ loss += λ·2 exactly.
        B, T, C, H = 2, 4, 1, 8
        eye = torch.eye(H)
        h = eye[1].view(1, 1, 1, H).expand(B, T, C, H).contiguous()   # e1
        f = eye[0].view(1, 1, 1, H).expand(B, T, C, H).contiguous()   # e0 ⊥ e1
        for lam in (0.5, 1.0, 2.0):
            base = contrastive_latent_loss((f, h), False, self._spec())
            withal = contrastive_latent_loss((f, h), False, self._spec(align=lam))
            assert abs((withal - base).item() - 2.0 * lam) < 1e-5

    def test_lambda_scales_linearly(self):
        f, h = _random_inputs(B=3, T=5, C=1, H=8, seed=3)
        base = contrastive_latent_loss((f, h), False, self._spec())
        d1 = (contrastive_latent_loss((f, h), False, self._spec(align=1.0)) - base).item()
        d2 = (contrastive_latent_loss((f, h), False, self._spec(align=2.0)) - base).item()
        assert d1 > 0 and abs(d2 - 2 * d1) < 1e-5

    def test_stopgrad_blocks_encoder_target_grad(self):
        fv, hv = _random_inputs(B=2, T=4, C=1, H=8, seed=11)
        f1, h1 = fv.clone().requires_grad_(True), hv.clone().requires_grad_(True)
        f2, h2 = fv.clone().requires_grad_(True), hv.clone().requires_grad_(True)
        contrastive_latent_loss((f1, h1), False, self._spec(align=1.0)).backward()
        contrastive_latent_loss((f2, h2), False, self._spec()).backward()
        # align adds NO gradient to the stop-grad'd encoder target h …
        assert (h1.grad - h2.grad).abs().max().item() < 1e-6
        # … but it DOES add gradient to the forecaster f.
        assert (f1.grad - f2.grad).abs().max().item() > 1e-4

    def test_applies_to_any_loss_shape(self):
        # L_align needs only the positive pair, so it works on non-logsumexp
        # variants too (here the legacy 'cosine_similarity' form). The exact
        # added value is λ·(2 − 2·cos(f_t, h_{t+1})).mean().
        f, h = _random_inputs(B=2, T=4, C=1, H=8, seed=9)
        base = _make_spec('cosine_similarity', tau=0.1)
        al = _make_spec('cosine_similarity', tau=0.1)
        al.train_configuration['align_loss_weight'] = 1.0
        d = (contrastive_latent_loss((f, h), False, al)
             - contrastive_latent_loss((f, h), False, base)).item()
        import torch.nn.functional as _F
        fn, hn = _F.normalize(f, dim=-1), _F.normalize(h, dim=-1)
        cos = (fn[:, :-1] * hn[:, 1:]).sum(-1)
        expected = (2.0 - 2.0 * cos).mean().item()
        assert abs(d - expected) < 1e-5


class TestContrastiveFloor:
    """`subtract_contrastive_floor` (#309) re-bases the loss by the constant
    floor `log(1 + N·e^(−1/τ))`. Gradient-neutral; needs the normalized form."""

    NAME = 'cosine_similarity_batch_full_fh_negs'

    def _spec(self, floor=False, tau=0.1, pos_denom=True):
        spec = _make_spec(self.NAME, tau)
        spec.train_configuration['include_positive_in_denominator'] = pos_denom
        spec.train_configuration['subtract_contrastive_floor'] = floor
        return spec

    def test_infonce_floor_formula(self):
        # Strict best lower bound: cos_pos = +1, cos_neg = −1 ⇒
        # floor = log(exp(+1/τ) + N·exp(−1/τ)) − 1/τ = log(1 + N·e^(−2/τ)).
        for tau, n in [(0.1, 100), (0.07, 5000), (0.2, 256 * 511)]:
            assert abs(infonce_floor(tau, n)
                       - math.log1p(n * math.exp(-2.0 / tau))) < 1e-12

    def test_subtract_floor_keeps_loss_nonnegative(self):
        # With the strict-min floor, `loss − floor ≥ 0` on any batch.
        for seed in (3, 11, 42, 101):
            f, h = _random_inputs(B=3, T=5, C=1, H=8, seed=seed)
            L = contrastive_latent_loss((f, h), False, self._spec(floor=True))
            assert L.item() >= -1e-6, (
                f"rebased loss must be ≥ 0 with strict-min floor; got {L.item()}")

    def test_effective_negative_count_matches_structure(self):
        # full_fh_negs, C=1: per-anchor = (xy+xx+zy)=2 + fh_all(T-1) + (B-1); ×B.
        assert _effective_negative_count(self.NAME, B=4, T=5, C=1) == 4 * (2 + 4 + 3)
        # base variant: 2 + xy_hat(1) + (B-1); ×B.
        assert _effective_negative_count('cosine_similarity_batch', 4, 5, 1) == 4 * (2 + 1 + 3)
        # two all-time terms add another (T-1).
        assert _effective_negative_count(
            'cosine_similarity_batch_full_fh_hh_negs', 4, 5, 1) == 4 * (2 + 4 + 4 + 3)

    def test_subtract_floor_shifts_by_constant(self):
        f, h = _random_inputs(B=3, T=5, C=1, H=8, seed=5)
        no_floor = contrastive_latent_loss((f, h), False, self._spec(floor=False))
        with_floor = contrastive_latent_loss((f, h), False, self._spec(floor=True))
        n = _effective_negative_count(self.NAME, B=3, T=5, C=1)
        expected = no_floor.item() - infonce_floor(0.1, n)
        assert abs(with_floor.item() - expected) < 1e-5

    def test_subtract_floor_is_gradient_neutral(self):
        fv, hv = _random_inputs(B=2, T=4, C=1, H=8, seed=6)
        f1, h1 = fv.clone().requires_grad_(True), hv.clone().requires_grad_(True)
        f2, h2 = fv.clone().requires_grad_(True), hv.clone().requires_grad_(True)
        contrastive_latent_loss((f1, h1), False, self._spec(floor=True)).backward()
        contrastive_latent_loss((f2, h2), False, self._spec(floor=False)).backward()
        assert (f1.grad - f2.grad).abs().max().item() < 1e-6
        assert (h1.grad - h2.grad).abs().max().item() < 1e-6

    def test_requires_pos_in_denominator(self):
        f, h = _random_inputs(B=2, T=4, C=1, H=8, seed=7)
        with pytest.raises(NotImplementedError):
            contrastive_latent_loss(
                (f, h), False, self._spec(floor=True, pos_denom=False))


class TestCPCInfonceAuxLoss:
    """`cpc_infonce_aux_loss` — the #344 CPC InfoNCE auxiliary term.

    Pins the paper-faithful properties: normalized (≥ 0), joint-trained
    (gradient reaches W_1 AND both latents — no stop-grad), exact under
    source-batch chunking (logsumexp associativity), and → 0 at perfect
    next-step alignment with spread negatives.
    """

    @staticmethod
    def _w1(H, seed=0, scale=None):
        g = torch.Generator().manual_seed(seed)
        w = torch.nn.Linear(H, H, bias=False)
        with torch.no_grad():
            if scale is None:
                w.weight.copy_(torch.randn(H, H, generator=g) / math.sqrt(H))
            else:
                w.weight.copy_(scale * torch.eye(H))
        return w

    def test_returns_nonneg_finite_scalar(self):
        from src.loss import cpc_infonce_aux_loss
        f, h = _random_inputs(B=4, T=5, C=2, H=16, seed=1)
        loss = cpc_infonce_aux_loss(f, h, self._w1(16))
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0  # positive in the denominator ⇒ ≥ 0

    def test_grad_flows_to_w1_and_both_latents(self):
        # Paper-exact: NO stop-grad — gradient must reach W_1, the AR context
        # h_t (forecasted), AND the encoder targets e (original).
        from src.loss import cpc_infonce_aux_loss
        f, h = _random_inputs(B=4, T=5, C=2, H=16, seed=2)
        f.requires_grad_(True)
        h.requires_grad_(True)
        w1 = self._w1(16)
        loss = cpc_infonce_aux_loss(f, h, w1)
        loss.backward()
        for name, t in (("W_1", w1.weight), ("forecasted", f), ("original", h)):
            assert t.grad is not None, f"no grad on {name}"
            assert t.grad.abs().sum() > 0, f"zero grad on {name}"

    def test_chunk_size_invariance(self):
        # logsumexp is associative ⇒ the checkpointed cross-batch chunking is
        # exact and chunk-size independent.
        from src.loss import cpc_infonce_aux_loss
        f, h = _random_inputs(B=8, T=4, C=1, H=16, seed=3)
        w1 = self._w1(16)
        vals = []
        for ch in ("1", "3", "8", "100"):
            os.environ["CPC_CB_CHUNK"] = ch
            try:
                vals.append(cpc_infonce_aux_loss(f, h, w1).item())
            finally:
                os.environ.pop("CPC_CB_CHUNK", None)
        for v in vals[1:]:
            assert abs(v - vals[0]) < 1e-5, (vals)

    def test_perfect_alignment_near_zero(self):
        # If the forecaster perfectly predicts the next embedding direction
        # (h_t ∝ e_{t+1}) and W_1 = big·I, the positive score dominates the
        # (near-orthogonal, high-dim) negatives ⇒ loss → 0.
        from src.loss import cpc_infonce_aux_loss
        g = torch.Generator().manual_seed(4)
        B, T, C, H = 4, 6, 1, 64
        h = torch.randn(B, T, C, H, generator=g)
        # forecasted[:, t] := original[:, t+1] (perfect next-step direction).
        f = torch.empty_like(h)
        f[:, :-1] = h[:, 1:]
        f[:, -1] = h[:, -1]
        w1 = self._w1(H, scale=20.0)
        loss = cpc_infonce_aux_loss(f, h, w1)
        assert loss.item() < 0.05, loss.item()

    def test_short_sequence_returns_zero(self):
        from src.loss import cpc_infonce_aux_loss
        f, h = _random_inputs(B=2, T=1, C=1, H=8, seed=5)
        loss = cpc_infonce_aux_loss(f, h, self._w1(8))
        assert loss.item() == 0.0

    def test_rejects_5d_forecaster(self):
        # The cpc_multistep forecaster stack is 5-D [B,T,C,K,H]; the single-step
        # aux term must reject it loudly (not a cryptic unpack error).
        from src.loss import cpc_infonce_aux_loss
        f5 = torch.randn(2, 4, 1, 3, 8)  # 5-D
        h = torch.randn(2, 4, 1, 8)
        with pytest.raises(ValueError):
            cpc_infonce_aux_loss(f5, h, self._w1(8))


class TestCPCInfonceAllLoss:
    """`cpc_infonce_all_loss` — the #348 full marginal-proposal CPC variant.

    Same log-bilinear next-step objective as the aux term, but with the FULL
    batch×time grid as negatives (mask only the positive). Pins: ≥ 0, joint
    gradient (W_1 + both latents), chunk-size invariance, → 0 at perfect
    alignment, T<2 → 0, rejects 5-D, and the defining superset property —
    `all` >= `aux` on identical inputs (its negatives are a strict superset).
    """

    @staticmethod
    def _w1(H, seed=0, scale=None):
        g = torch.Generator().manual_seed(seed)
        w = torch.nn.Linear(H, H, bias=False)
        with torch.no_grad():
            if scale is None:
                w.weight.copy_(torch.randn(H, H, generator=g) / math.sqrt(H))
            else:
                w.weight.copy_(scale * torch.eye(H))
        return w

    def test_returns_nonneg_finite_scalar(self):
        from src.loss import cpc_infonce_all_loss
        f, h = _random_inputs(B=4, T=5, C=2, H=16, seed=1)
        loss = cpc_infonce_all_loss(f, h, self._w1(16))
        assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() >= 0.0

    def test_grad_flows_to_w1_and_both_latents(self):
        from src.loss import cpc_infonce_all_loss
        f, h = _random_inputs(B=4, T=5, C=2, H=16, seed=2)
        f.requires_grad_(True); h.requires_grad_(True)
        w1 = self._w1(16)
        cpc_infonce_all_loss(f, h, w1).backward()
        for name, t in (("W_1", w1.weight), ("forecasted", f), ("original", h)):
            assert t.grad is not None and t.grad.abs().sum() > 0, f"no grad on {name}"

    def test_chunk_size_invariance(self):
        # logsumexp is associative ⇒ source-batch chunking is exact.
        from src.loss import cpc_infonce_all_loss
        f, h = _random_inputs(B=8, T=4, C=1, H=16, seed=3)
        w1 = self._w1(16)
        vals = []
        for ch in ("1", "3", "8", "100"):
            os.environ["CPC_ALL_CHUNK"] = ch
            try:
                vals.append(cpc_infonce_all_loss(f, h, w1).item())
            finally:
                os.environ.pop("CPC_ALL_CHUNK", None)
        for v in vals[1:]:
            assert abs(v - vals[0]) < 1e-5, vals

    def test_full_grid_superset_ordering(self):
        # The full grid (marginal_only=False) is a strict superset of BOTH the
        # cross-sequence-only (marginal_only=True) set AND the narrow aux set, so
        # its denominator — hence loss — is >= both. (cross-only is NOT a superset
        # of aux: it drops aux's same-sequence negatives, so they are unordered.)
        from src.loss import cpc_infonce_all_loss, cpc_infonce_aux_loss
        f, h = _random_inputs(B=6, T=5, C=1, H=16, seed=7)
        w1 = self._w1(16)
        full = cpc_infonce_all_loss(f, h, w1, marginal_only=False).item()
        cross = cpc_infonce_all_loss(f, h, w1, marginal_only=True).item()
        aux = cpc_infonce_aux_loss(f, h, w1).item()
        assert full >= cross - 1e-6, (full, cross)
        assert full >= aux - 1e-6, (full, aux)
        assert cross >= 0.0

    def test_perfect_alignment_near_zero(self):
        from src.loss import cpc_infonce_all_loss
        g = torch.Generator().manual_seed(4)
        B, T, C, H = 4, 6, 1, 64
        h = torch.randn(B, T, C, H, generator=g)
        f = torch.empty_like(h); f[:, :-1] = h[:, 1:]; f[:, -1] = h[:, -1]
        loss = cpc_infonce_all_loss(f, h, self._w1(H, scale=20.0))
        assert loss.item() < 0.05, loss.item()

    def test_short_sequence_returns_zero(self):
        from src.loss import cpc_infonce_all_loss
        f, h = _random_inputs(B=2, T=1, C=1, H=8, seed=5)
        assert cpc_infonce_all_loss(f, h, self._w1(8)).item() == 0.0

    def test_rejects_5d_forecaster(self):
        from src.loss import cpc_infonce_all_loss
        with pytest.raises(ValueError):
            cpc_infonce_all_loss(torch.randn(2, 4, 1, 3, 8), torch.randn(2, 4, 1, 8), self._w1(8))


class TestAlignLoss:
    """`align_loss` — the standalone BYOL alignment term (#344 follow-up arm)."""

    def test_nonneg_scalar(self):
        from src.loss import align_loss
        f, h = _random_inputs(B=4, T=5, C=2, H=16, seed=7)
        loss = align_loss(f, h, 1.0)
        assert loss.ndim == 0 and 0.0 <= loss.item() <= 4.0

    def test_grad_only_through_forecaster(self):
        # encoder target is stop-gradded ⇒ grad reaches f, NOT h.
        from src.loss import align_loss
        f, h = _random_inputs(B=4, T=5, C=2, H=16, seed=8)
        f.requires_grad_(True)
        h.requires_grad_(True)
        align_loss(f, h, 1.0).backward()
        assert f.grad is not None and f.grad.abs().sum() > 0
        assert h.grad is None or h.grad.abs().sum() == 0

    def test_matches_inline_align_in_contrastive(self):
        # Standalone align == the align add-on inside contrastive_latent_loss
        # (contrastive weight isolated by differencing align_w=1 vs 0).
        from src.loss import align_loss, contrastive_latent_loss
        f, h = _random_inputs(B=3, T=4, C=2, H=8, seed=9)
        spec = _make_spec("cosine_similarity_batch", tau=0.1)
        base = contrastive_latent_loss((f, h), False, spec, align_loss_weight=0.0)
        withal = contrastive_latent_loss((f, h), False, spec, align_loss_weight=1.0)
        standalone = align_loss(f, h, 1.0)
        assert abs((withal - base).item() - standalone.item()) < 1e-5

    def test_perfect_alignment_zero(self):
        from src.loss import align_loss
        g = torch.Generator().manual_seed(10)
        h = torch.randn(2, 6, 1, 16, generator=g)
        f = torch.empty_like(h)
        f[:, :-1] = h[:, 1:]   # f_t := h_{t+1} (perfect next-step alignment)
        f[:, -1] = h[:, -1]
        assert align_loss(f, h, 1.0).item() < 1e-5
