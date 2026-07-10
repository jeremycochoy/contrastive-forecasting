"""Tests for #374 — `cosine_similarity_batch_split_pred_rep`.

Same negative families as `cosine_similarity_batch_full_hh_negs_xshh_allt`,
but split into two independent terms with a single positive:

  L_pred = normalized InfoNCE with the f-anchored families in the denominator
           (cross-batch f_t ↔ h'_{t+1} + adjacent f_{t+1} ↔ f_t).
  L_rep  = pooled logsumexp of the h-anchored families, no positive
           (cross-channel h_t ↔ h_t + within-series all-time h_t ↔ h_l
            + cross-series all-time h_t ↔ h_{b',l}).
  L      = L_pred + L_rep.

Same τ in both, same teacher-side / stopgrad positive, same batch pooling
inside each term. `include_positive_in_denominator` and
`subtract_contrastive_floor` are rejected: L_pred is normalized by
construction; the floor formula is derived for the combined shape.
"""

import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.loss import contrastive_latent_loss

NAME = "cosine_similarity_batch_split_pred_rep"


def _spec(tau=0.1, teacher=False, stopgrad=False, pos_in_denom=False,
          sub_floor=False, align=None, moco_negatives=False):
    tc = {
        "contrastive_divergence_temperature": tau,
        "contrastive_latent_noise": None,
        "loss_shape": NAME,
        "contrastive_latent_delay": 0,
        "include_positive_in_denominator": pos_in_denom,
        "stopgrad_positive_h": stopgrad,
        "subtract_contrastive_floor": sub_floor,
        "moco_negatives": moco_negatives,
    }
    if align is not None:
        tc["align_loss_weight"] = align
    return SimpleNamespace(train_configuration=tc)


def _latents(B, T, C, H, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(B, T, C, H, generator=g, dtype=dtype)
    o = torch.randn(B, T, C, H, generator=g, dtype=dtype)
    return f, o


def _split_reference(f, o, tau, teacher_o=None, stopgrad=False, moco_negatives=False):
    """Independent brute-force reference: build every negative family as a
    full dense Gram (no chunking / checkpointing), then combine into
    L_pred + L_rep exactly as specified in the issue."""
    neg_inf = float('-inf')
    orig_norm = F.normalize(o, p=2, dim=-1)
    fore_norm = F.normalize(f, p=2, dim=-1)
    hy_hat_norm = fore_norm[:, :-1]           # f_t         [B, T-1, C, H]
    hz_hat_norm = fore_norm[:, 1:]            # f_{t+1}
    hx_norm = orig_norm[:, :-1]               # h_t
    hy_norm = orig_norm[:, 1:]                # h_{t+1}
    B, Tm1, C, H = hx_norm.shape
    T = orig_norm.shape[1]

    if teacher_o is not None:
        hy_pos = F.normalize(teacher_o, p=2, dim=-1)[:, 1:]
    elif stopgrad:
        hy_pos = hy_norm.detach()
    else:
        hy_pos = hy_norm
    log_pos = (hy_pos * hy_hat_norm).sum(-1) / tau         # [B, T-1, C]

    # h-anchored (repulsion) families.
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

    # f-anchored (prediction) families.
    sims_zy = torch.einsum('btih,btjh->btij', hz_hat_norm, hy_hat_norm) / tau
    log_neg_zy = torch.logsumexp(sims_zy, dim=2)

    # Arm 3 moco_negatives (#374): the cross-batch f↔h keys are the teacher's
    # h^T_{t+1} when the flag is on AND a teacher was provided (otherwise
    # student h, matching the base split).
    if moco_negatives and teacher_o is not None:
        hy_cb = F.normalize(teacher_o, p=2, dim=-1)[:, 1:]
    else:
        hy_cb = hy_norm
    sims_cb = torch.einsum('atch,btch->abtc', hy_hat_norm, hy_cb) / tau
    sims_cb = sims_cb.masked_fill(
        torch.eye(B, dtype=torch.bool).view(B, B, 1, 1), neg_inf)
    log_neg_cross_batch = torch.logsumexp(sims_cb, dim=1)

    # L_pred: normalized InfoNCE with the f-anchored families.
    negs_pred = torch.stack([log_neg_zy, log_neg_cross_batch], dim=0)
    log_neg_per_anchor_pred = torch.logsumexp(negs_pred, dim=0)
    log_neg_total_pred = torch.logsumexp(
        log_neg_per_anchor_pred, dim=0, keepdim=True)
    log_denom_pred = torch.logsumexp(
        torch.stack([log_pos, log_neg_total_pred.expand_as(log_pos)], dim=0),
        dim=0,
    )
    loss_pred = (log_denom_pred - log_pos).mean()

    # L_rep: pooled LSE of the h-anchored families, no positive.
    negs_rep = torch.stack([log_neg_xx, log_neg_hh_all, log_neg_xs_allt], dim=0)
    log_neg_per_anchor_rep = torch.logsumexp(negs_rep, dim=0)
    log_neg_total_rep = torch.logsumexp(
        log_neg_per_anchor_rep, dim=0, keepdim=True)
    loss_rep = log_neg_total_rep.mean()

    return loss_pred + loss_rep, loss_pred, loss_rep


class TestSplitPredRep:

    def test_finite_scalar_and_grad(self):
        B, T, C, H = 4, 6, 2, 12
        g = torch.Generator().manual_seed(3740)
        f = torch.randn(B, T, C, H, generator=g, requires_grad=True)
        h = torch.randn(B, T, C, H, generator=g, requires_grad=True)
        loss = contrastive_latent_loss((f, h), False, _spec())
        assert loss.dim() == 0 and torch.isfinite(loss)
        loss.backward()
        assert f.grad is not None and torch.isfinite(f.grad).all()
        assert h.grad is not None and torch.isfinite(h.grad).all()

    def test_matches_transparent_reference_c1(self):
        """Production impl == fp64 brute-force reference for a range of
        seeds and small (B, T, C=1) shapes."""
        for seed in (0, 7, 42):
            f, o = _latents(B=5, T=6, C=1, H=12, seed=seed)
            got = contrastive_latent_loss((f, o), False, _spec(tau=0.1))
            want, _, _ = _split_reference(f, o, 0.1)
            assert torch.allclose(got, want, atol=1e-9, rtol=1e-9), (
                f"seed={seed}: got {got.item():.10f} want {want.item():.10f}")

    def test_matches_transparent_reference_multichannel(self):
        """Multi-channel: `log_neg_xx` becomes active (masked C×C)."""
        f, o = _latents(B=4, T=5, C=3, H=10, seed=137)
        got = contrastive_latent_loss((f, o), False, _spec(tau=0.1))
        want, _, _ = _split_reference(f, o, 0.1)
        assert torch.allclose(got, want, atol=1e-9, rtol=1e-9), (
            f"got {got.item():.10f} want {want.item():.10f}")

    def test_decomposes_into_pred_plus_rep(self):
        """Sanity: the reference `L_pred + L_rep` sum matches the production
        impl (proves the two terms are exactly what the spec calls for and
        not smuggled through a single joint LSE)."""
        f, o = _latents(B=4, T=5, C=1, H=8, seed=11)
        got = contrastive_latent_loss((f, o), False, _spec(tau=0.1))
        total, l_pred, l_rep = _split_reference(f, o, 0.1)
        assert torch.allclose(got, l_pred + l_rep, atol=1e-9)
        assert torch.allclose(got, total, atol=1e-9)
        # L_pred is a normalized InfoNCE (positive in denom) ⇒ ≥ 0.
        assert l_pred.item() >= -1e-9

    def test_distinct_from_arm_c(self):
        """Splitting the shared denominator must move the value: `..._xshh_allt`
        (a single pooled denominator over all five families) is a different
        objective from the split (two independent terms)."""
        f, o = _latents(B=3, T=5, C=1, H=8, seed=13, dtype=torch.float32)
        v_split = contrastive_latent_loss((f, o), False, _spec(tau=0.1)).item()
        arm_c_spec = _spec(tau=0.1)
        arm_c_spec.train_configuration['loss_shape'] = (
            'cosine_similarity_batch_full_hh_negs_xshh_allt')
        v_c = contrastive_latent_loss((f, o), False, arm_c_spec).item()
        assert abs(v_split - v_c) > 1e-6, (
            f"split loss must differ from arm C; got split={v_split} c={v_c}")

    def test_chunk_size_invariant(self):
        """The xs_allt chunked LSE must be exact: XSHH_ALLT_CHUNK is a
        memory/kernel-launches knob, never a value knob."""
        f, o = _latents(B=7, T=6, C=1, H=10, seed=5)
        vals = []
        for ch in ("1", "3", "16"):
            os.environ['XSHH_ALLT_CHUNK'] = ch
            vals.append(contrastive_latent_loss(
                (f, o), False, _spec(tau=0.1)).item())
        os.environ.pop('XSHH_ALLT_CHUNK', None)
        assert max(vals) - min(vals) < 1e-9, (
            f"chunk-size changed value: {vals}")

    def test_teacher_positive_replaces_student_in_log_pos(self):
        """`teacher_original_latent` must replace the student `h_{t+1}` in
        L_pred's positive only. Reference passes the same teacher in."""
        B, T, C, H = 4, 5, 1, 10
        g = torch.Generator().manual_seed(917)
        f = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        o = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        teacher = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        got = contrastive_latent_loss(
            (f, o), False, _spec(tau=0.1),
            teacher_original_latent=teacher,
        )
        want, _, _ = _split_reference(f, o, 0.1, teacher_o=teacher)
        assert torch.allclose(got, want, atol=1e-9, rtol=1e-9)

    def test_stopgrad_positive_h_forward_invariant(self):
        """`stopgrad_positive_h` detaches only the backward edge — the
        forward loss value is identical to the no-flag run."""
        f, o = _latents(B=4, T=5, C=1, H=10, seed=71)
        base = contrastive_latent_loss((f, o), False, _spec(tau=0.1))
        stopgr = contrastive_latent_loss(
            (f, o), False, _spec(tau=0.1, stopgrad=True))
        assert torch.allclose(base, stopgr, atol=1e-12, rtol=1e-12)

    def test_stopgrad_positive_h_cuts_pos_edge_of_h_grad(self):
        """With `stopgrad_positive_h`, the ENCODER gradient dh differs from
        the no-flag run (the positive's h-edge is dropped); f-grad is
        identical (no f-edge is touched by the flag)."""
        B, T, C, H = 3, 5, 1, 8
        g = torch.Generator().manual_seed(83)

        def _grads(sg):
            f = torch.randn(B, T, C, H, generator=torch.Generator().manual_seed(83),
                            dtype=torch.float64, requires_grad=True)
            o = torch.randn(B, T, C, H, generator=torch.Generator().manual_seed(84),
                            dtype=torch.float64, requires_grad=True)
            loss = contrastive_latent_loss(
                (f, o), False, _spec(tau=0.1, stopgrad=sg))
            loss.backward()
            return f.grad.clone(), o.grad.clone()

        df0, dh0 = _grads(False)
        df1, dh1 = _grads(True)
        assert torch.allclose(df0, df1, atol=1e-12, rtol=1e-12)
        assert not torch.allclose(dh0, dh1, atol=1e-6)

    def test_rejects_pos_in_denominator(self):
        """The split's L_pred is ALREADY normalized-InfoNCE — the flag is
        not a knob here. Requesting it must fail loud."""
        f, o = _latents(B=3, T=4, C=1, H=8, seed=2)
        with pytest.raises(NotImplementedError):
            contrastive_latent_loss(
                (f, o), False, _spec(tau=0.1, pos_in_denom=True))
        with pytest.raises(NotImplementedError):
            contrastive_latent_loss(
                (f, o), False, _spec(tau=0.1),
                include_positive_in_denominator=True)

    def test_rejects_subtract_contrastive_floor(self):
        """The floor formula is derived for the COMBINED shape's single
        denominator; it does not apply to the split (issue #374 spec)."""
        f, o = _latents(B=3, T=4, C=1, H=8, seed=2)
        with pytest.raises(NotImplementedError):
            contrastive_latent_loss(
                (f, o), False, _spec(tau=0.1, sub_floor=True))

    def test_align_loss_still_additive(self):
        """`align_loss_weight` applies to any shape; it adds a BYOL term on
        top of the split total."""
        f, o = _latents(B=3, T=4, C=1, H=8, seed=6, dtype=torch.float32)
        base = contrastive_latent_loss((f, o), False, _spec(tau=0.1)).item()
        withal = contrastive_latent_loss(
            (f, o), False, _spec(tau=0.1, align=0.5)).item()
        # cos ∈ [-1, 1] ⇒ 2 - 2·cos ∈ [0, 4] ⇒ increment ∈ [0, 2] (λ=0.5)
        assert withal > base - 1e-9
        assert withal - base <= 2.0 + 1e-6

    def test_moco_negatives_off_is_base_split(self):
        """With `moco_negatives=False` the cross-batch f↔h keys are the
        student's h — byte-for-byte the base split-shape loss."""
        B, T, C, H = 4, 5, 2, 12
        g = torch.Generator().manual_seed(7373)
        f = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        o = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        teacher = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        base = contrastive_latent_loss(
            (f, o), False, _spec(tau=0.1),
            teacher_original_latent=teacher)
        with_flag = contrastive_latent_loss(
            (f, o), False, _spec(tau=0.1, moco_negatives=False),
            teacher_original_latent=teacher)
        assert torch.allclose(base, with_flag, atol=1e-12, rtol=1e-12)

    def test_moco_negatives_on_uses_teacher_in_cross_batch(self):
        """With `moco_negatives=True` the cross-batch f↔h keys are the
        teacher's h^T; reference reproduces the swap exactly."""
        B, T, C, H = 4, 5, 2, 12
        g = torch.Generator().manual_seed(7374)
        f = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        o = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        teacher = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        got = contrastive_latent_loss(
            (f, o), False, _spec(tau=0.1, moco_negatives=True),
            teacher_original_latent=teacher)
        want, _, _ = _split_reference(
            f, o, 0.1, teacher_o=teacher, moco_negatives=True)
        assert torch.allclose(got, want, atol=1e-9, rtol=1e-9)

    def test_moco_negatives_shifts_loss_vs_base_split(self):
        """The moco flag SHOULD change the loss (the negatives are drawn
        from a different distribution). Sanity check that the flag is
        actually plumbed and not silently dropped."""
        B, T, C, H = 4, 5, 2, 12
        g = torch.Generator().manual_seed(7375)
        f = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        o = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        teacher = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        base = contrastive_latent_loss(
            (f, o), False, _spec(tau=0.1),
            teacher_original_latent=teacher)
        moco = contrastive_latent_loss(
            (f, o), False, _spec(tau=0.1, moco_negatives=True),
            teacher_original_latent=teacher)
        assert not torch.allclose(base, moco, atol=1e-6)

    def test_moco_negatives_requires_teacher(self):
        """Passing `moco_negatives=True` without an EMA teacher must fail
        loud (the flag has nothing to route the negatives through)."""
        f, o = _latents(B=3, T=4, C=1, H=8, seed=2)
        with pytest.raises(ValueError):
            contrastive_latent_loss(
                (f, o), False, _spec(tau=0.1, moco_negatives=True))

    def test_moco_negatives_rejects_other_loss_shapes(self):
        """The flag is only wired into the split branch; any other
        loss_shape reaching contrastive_latent_loss with it set must
        raise (avoids silently training WITHOUT teacher-in-negs)."""
        f, o = _latents(B=3, T=4, C=1, H=8, seed=2)
        teacher = torch.randn(3, 4, 1, 8, generator=torch.Generator().manual_seed(1),
                              dtype=torch.float64)
        tc = {
            "contrastive_divergence_temperature": 0.1,
            "contrastive_latent_noise": None,
            "loss_shape": "cosine_similarity_batch_full_hh_negs_xshh_allt",
            "contrastive_latent_delay": 0,
            "include_positive_in_denominator": False,
            "stopgrad_positive_h": False,
            "subtract_contrastive_floor": False,
            "moco_negatives": True,
        }
        with pytest.raises(NotImplementedError):
            contrastive_latent_loss(
                (f, o), False, SimpleNamespace(train_configuration=tc),
                teacher_original_latent=teacher)

    def test_zero_when_teacher_positive_and_negs_are_far(self):
        """If f_t := teacher's h^T_{t+1} (positive cos = 1) AND every h_t is
        pushed toward the same axis (h-anchored negatives are large, but the
        rep term is a constant of the h layout, independent of f). The
        L_pred term collapses to log(1 + N·e^{-1/τ}) — its infonce floor —
        because the f-anchored negatives are randomised and their pooled
        LSE stays finite. Sanity: L_pred ≥ 0, L_rep is a real scalar,
        and the sum is finite."""
        f, o = _latents(B=3, T=4, C=1, H=8, seed=9)
        teacher = o.clone()                                          # same as student
        f[:, :-1] = F.normalize(teacher, p=2, dim=-1)[:, 1:]        # cos(f_t, h^T_{t+1}) = 1
        loss = contrastive_latent_loss(
            (f, o), False, _spec(tau=0.1),
            teacher_original_latent=teacher,
        )
        _, l_pred, _ = _split_reference(f, o, 0.1, teacher_o=teacher)
        assert torch.isfinite(loss)
        assert l_pred.item() >= -1e-9
