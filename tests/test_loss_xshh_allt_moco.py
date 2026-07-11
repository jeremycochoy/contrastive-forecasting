"""Tests for #374 arm 4 — `moco_negatives` on the xshh_allt loss shape.

`cosine_similarity_batch_full_hh_negs_xshh_allt` (arm C's baseline) gains the
same MoCo-negatives swap as the split shape: with `moco_negatives` on AND an
EMA teacher available, ONLY the cross-batch f↔h family draws its keys h'_{t+1}
from the teacher (`hy_teacher_norm`). The three h↔h families (xx, hh_all,
xs_allt) stay pure student on BOTH sides, and keep delivering gradient to the
encoder. Mirrors the split-shape moco tests in test_loss_split_pred_rep.py.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.loss import contrastive_latent_loss

NAME = "cosine_similarity_batch_full_hh_negs_xshh_allt"


def _spec(tau=0.1, pos_in_denom=False, moco_negatives=False):
    tc = {
        "contrastive_divergence_temperature": tau,
        "contrastive_latent_noise": None,
        "loss_shape": NAME,
        "contrastive_latent_delay": 0,
        "include_positive_in_denominator": pos_in_denom,
        "stopgrad_positive_h": False,
        "subtract_contrastive_floor": False,
        "moco_negatives": moco_negatives,
    }
    return SimpleNamespace(train_configuration=tc)


def _latents(B, T, C, H, seed, n=2, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(B, T, C, H, generator=g, dtype=dtype) for _ in range(n))


def _xshh_allt_reference(f, o, tau, teacher_o=None, moco_negatives=False,
                         pos_in_denom=False):
    """Independent brute-force reference: every negative family as a full
    dense Gram (no chunking / checkpointing), pooled into the single
    xshh_allt denominator. The moco swap routes ONLY the cross-batch f↔h
    keys through the teacher; the teacher also replaces the student in
    log_pos whenever provided (the #353 behaviour, moco or not)."""
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
    else:
        hy_pos = hy_norm
    log_pos = (hy_pos * hy_hat_norm).sum(-1) / tau         # [B, T-1, C]

    sims_xx = torch.einsum('btih,btjh->btij', hx_norm, hx_norm) / tau
    sims_xx = sims_xx.masked_fill(
        torch.eye(C, dtype=torch.bool).view(1, 1, C, C), neg_inf)
    log_neg_xx = torch.logsumexp(sims_xx, dim=2)

    sims_zy = torch.einsum('btih,btjh->btij', hz_hat_norm, hy_hat_norm) / tau
    log_neg_zy = torch.logsumexp(sims_zy, dim=2)

    sims_hh = torch.einsum('btch,blch->btcl', hx_norm, orig_norm) / tau
    t_idx = torch.arange(Tm1).view(Tm1, 1)
    l_idx = torch.arange(T).view(1, T)
    sims_hh = sims_hh.masked_fill(
        (l_idx == t_idx).view(1, Tm1, 1, T), neg_inf)
    log_neg_hh_all = torch.logsumexp(sims_hh, dim=3)

    # Arm 4 moco_negatives (#374): the cross-batch f↔h keys are the
    # teacher's h^T_{t+1} when the flag is on AND a teacher was provided
    # (otherwise student h, matching the base xshh_allt).
    if moco_negatives and teacher_o is not None:
        hy_cb = F.normalize(teacher_o, p=2, dim=-1)[:, 1:]
    else:
        hy_cb = hy_norm
    sims_cb = torch.einsum('atch,btch->abtc', hy_hat_norm, hy_cb) / tau
    sims_cb = sims_cb.masked_fill(
        torch.eye(B, dtype=torch.bool).view(B, B, 1, 1), neg_inf)
    log_neg_cross_batch = torch.logsumexp(sims_cb, dim=1)

    sims_xs = torch.einsum('atch,blch->abtcl', hx_norm, orig_norm) / tau
    sims_xs = sims_xs.masked_fill(
        torch.eye(B, dtype=torch.bool).view(B, B, 1, 1, 1), neg_inf)
    log_neg_xs_allt = torch.logsumexp(sims_xs, dim=(1, 4))

    negatives = torch.stack(
        [log_neg_xx, log_neg_zy, log_neg_hh_all,
         log_neg_cross_batch, log_neg_xs_allt], dim=0)
    log_neg_per_anchor = torch.logsumexp(negatives, dim=0)
    log_neg_total = torch.logsumexp(log_neg_per_anchor, dim=0, keepdim=True)
    if pos_in_denom:
        log_denom = torch.logsumexp(
            torch.stack(
                [log_pos, log_neg_total.expand_as(log_pos)], dim=0),
            dim=0,
        )
        return (log_denom - log_pos).mean()
    return (log_neg_total - log_pos).mean()


class TestXshhAlltMocoNegatives:

    def test_moco_negatives_off_is_base_xshh_allt(self):
        """With `moco_negatives=False` (or the key absent, i.e. a pre-flag
        config) the cross-batch f↔h keys are the student's h — byte-for-byte
        the base xshh_allt loss, teacher only inside log_pos."""
        f, o, teacher = _latents(B=4, T=5, C=2, H=12, seed=7473, n=3)
        legacy = _spec(tau=0.1)
        del legacy.train_configuration['moco_negatives']
        base = contrastive_latent_loss(
            (f, o), False, legacy, teacher_original_latent=teacher)
        off = contrastive_latent_loss(
            (f, o), False, _spec(tau=0.1, moco_negatives=False),
            teacher_original_latent=teacher)
        assert torch.allclose(base, off, atol=1e-12, rtol=1e-12)
        want = _xshh_allt_reference(f, o, 0.1, teacher_o=teacher)
        assert torch.allclose(off, want, atol=1e-9, rtol=1e-9)

    def test_moco_negatives_on_uses_teacher_in_cross_batch(self):
        """With `moco_negatives=True` the cross-batch f↔h keys are the
        teacher's h^T; the hand-computed reference reproduces the swap
        exactly, with and without pos-in-denominator (arm 4 trains with
        --pos-in-denominator on)."""
        f, o, teacher = _latents(B=4, T=5, C=2, H=12, seed=7474, n=3)
        for pid in (False, True):
            got = contrastive_latent_loss(
                (f, o), False,
                _spec(tau=0.1, pos_in_denom=pid, moco_negatives=True),
                teacher_original_latent=teacher)
            want = _xshh_allt_reference(
                f, o, 0.1, teacher_o=teacher, moco_negatives=True,
                pos_in_denom=pid)
            assert torch.allclose(got, want, atol=1e-9, rtol=1e-9), (
                f"pid={pid}: got {got.item():.10f} want {want.item():.10f}")

    def test_moco_negatives_shifts_loss_vs_base(self):
        """The moco flag SHOULD change the loss (the cross-batch negatives
        are drawn from a different encoder). Sanity check that the flag is
        actually plumbed into the xshh_allt branch and not silently
        dropped."""
        f, o, teacher = _latents(B=4, T=5, C=2, H=12, seed=7475, n=3)
        base = contrastive_latent_loss(
            (f, o), False, _spec(tau=0.1),
            teacher_original_latent=teacher)
        moco = contrastive_latent_loss(
            (f, o), False, _spec(tau=0.1, moco_negatives=True),
            teacher_original_latent=teacher)
        assert not torch.allclose(base, moco, atol=1e-6)

    def test_moco_negatives_hh_families_still_gradient_to_encoder(self):
        """With moco_negatives on, the three h↔h families (xx, hh_all,
        xs_allt) must still deliver gradient to the encoder. The flag only
        swaps the cross_batch f↔h key side; with the positive on the
        (no-grad) teacher and f detached, a nonzero gradient on o can come
        only from the pure-student repulsion families."""
        f, o, teacher = _latents(B=4, T=5, C=2, H=12, seed=9102, n=3)
        f = f.detach()
        teacher = teacher.detach()
        o.requires_grad_(True)
        loss = contrastive_latent_loss(
            (f, o), False, _spec(tau=0.1, moco_negatives=True),
            teacher_original_latent=teacher)
        loss.backward()
        assert o.grad is not None
        assert o.grad.abs().sum().item() > 1e-6, (
            "encoder grad from h↔h families vanished (moco_negatives "
            "leaked into the repulsion terms)")

    def test_moco_negatives_requires_teacher(self):
        """Passing `moco_negatives=True` without an EMA teacher must fail
        loud for the xshh_allt shape too (same guard as the split)."""
        f, o = _latents(B=3, T=4, C=1, H=8, seed=2)
        with pytest.raises(ValueError):
            contrastive_latent_loss(
                (f, o), False, _spec(tau=0.1, moco_negatives=True))
