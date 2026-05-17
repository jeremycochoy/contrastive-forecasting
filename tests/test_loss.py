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
