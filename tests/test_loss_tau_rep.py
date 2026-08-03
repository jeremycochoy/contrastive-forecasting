"""Tests for #379 — separate `--tau-rep` for the L_rep term.

`cosine_similarity_batch_split_pred_rep` and `cosine_similarity_batch_rep_only`
now honour a `contrastive_divergence_temperature_rep` config key
(set via `--tau-rep` at train time). When unset (default), the L_rep
term shares the L_pred temperature `contrastive_divergence_temperature`
— byte-for-byte identical to the pre-#379 objective. When set, L_pred
keeps its τ and L_rep divides its h-anchored family LSE (and moco_rep
positive) by τ_rep instead.

Tests:
- unset ⇒ identical to today's loss (both split_pred_rep and rep_only)
- set with split_pred_rep ⇒ L_pred numeric matches τ=0.10, L_rep numeric
  matches τ=1.0 recomputation
- set with rep_only ⇒ whole loss matches a τ_rep recomputation
- moco_rep_keys: log_pos_h (the moco positive) also routes through τ_rep
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.loss import contrastive_latent_loss


def _spec(loss_shape, tau=0.1, tau_rep=None, moco_rep=False,
          moco_negatives=False, align=None):
    tc = {
        "contrastive_divergence_temperature": tau,
        "contrastive_latent_noise": None,
        "loss_shape": loss_shape,
        "contrastive_latent_delay": 0,
        "include_positive_in_denominator": False,
        "stopgrad_positive_h": False,
        "subtract_contrastive_floor": False,
        "moco_negatives": moco_negatives,
        "moco_rep_keys": moco_rep,
    }
    if tau_rep is not None:
        tc["contrastive_divergence_temperature_rep"] = tau_rep
    if align is not None:
        tc["align_loss_weight"] = align
    return SimpleNamespace(train_configuration=tc)


def _latents(B, T, C, H, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(B, T, C, H, generator=g, dtype=dtype)
    o = torch.randn(B, T, C, H, generator=g, dtype=dtype)
    return f, o


def _split_reference(f, o, tau, tau_rep=None, teacher_o=None,
                     moco_negatives=False, moco_rep=False):
    """Independent reference for split_pred_rep — a full dense recomputation
    with an explicit τ per family. Passing tau_rep=None ⇒ shares tau."""
    if tau_rep is None:
        tau_rep = tau
    neg_inf = float('-inf')
    orig_norm = F.normalize(o, p=2, dim=-1)
    fore_norm = F.normalize(f, p=2, dim=-1)
    hy_hat_norm = fore_norm[:, :-1]
    hz_hat_norm = fore_norm[:, 1:]
    hx_norm = orig_norm[:, :-1]
    hy_norm = orig_norm[:, 1:]
    B, Tm1, C, H = hx_norm.shape
    T = orig_norm.shape[1]

    if teacher_o is not None:
        teach_norm = F.normalize(teacher_o, p=2, dim=-1)
        hy_pos = teach_norm[:, 1:]
        if moco_rep:
            hx_key = teach_norm[:, :-1]
            orig_key = teach_norm
        else:
            hx_key = hx_norm
            orig_key = orig_norm
    else:
        hy_pos = hy_norm
        hx_key = hx_norm
        orig_key = orig_norm
    log_pos = (hy_pos * hy_hat_norm).sum(-1) / tau

    # h-anchored / L_rep — divides by tau_rep.
    sims_xx = torch.einsum('btih,btjh->btij', hx_norm, hx_key) / tau_rep
    sims_xx = sims_xx.masked_fill(
        torch.eye(C, dtype=torch.bool).view(1, 1, C, C), neg_inf)
    log_neg_xx = torch.logsumexp(sims_xx, dim=2)

    sims_hh = torch.einsum('btch,blch->btcl', hx_norm, orig_key) / tau_rep
    t_idx = torch.arange(Tm1).view(Tm1, 1)
    l_idx = torch.arange(T).view(1, T)
    sims_hh = sims_hh.masked_fill(
        (l_idx == t_idx).view(1, Tm1, 1, T), neg_inf)
    log_neg_hh_all = torch.logsumexp(sims_hh, dim=3)

    sims_xs = torch.einsum('atch,blch->abtcl', hx_norm, orig_key) / tau_rep
    sims_xs = sims_xs.masked_fill(
        torch.eye(B, dtype=torch.bool).view(B, B, 1, 1, 1), neg_inf)
    log_neg_xs_allt = torch.logsumexp(sims_xs, dim=(1, 4))

    # f-anchored / L_pred — divides by tau.
    sims_zy = torch.einsum('btih,btjh->btij', hz_hat_norm, hy_hat_norm) / tau
    log_neg_zy = torch.logsumexp(sims_zy, dim=2)
    if moco_negatives and teacher_o is not None:
        hy_cb = F.normalize(teacher_o, p=2, dim=-1)[:, 1:]
    else:
        hy_cb = hy_norm
    sims_cb = torch.einsum('atch,btch->abtc', hy_hat_norm, hy_cb) / tau
    sims_cb = sims_cb.masked_fill(
        torch.eye(B, dtype=torch.bool).view(B, B, 1, 1), neg_inf)
    log_neg_cross_batch = torch.logsumexp(sims_cb, dim=1)

    # L_pred: normalized InfoNCE.
    negs_pred = torch.stack([log_neg_zy, log_neg_cross_batch], dim=0)
    log_neg_per_anchor_pred = torch.logsumexp(negs_pred, dim=0)
    log_neg_total_pred = torch.logsumexp(
        log_neg_per_anchor_pred, dim=0, keepdim=True)
    log_denom_pred = torch.logsumexp(
        torch.stack([log_pos, log_neg_total_pred.expand_as(log_pos)], dim=0),
        dim=0,
    )
    loss_pred = (log_denom_pred - log_pos).mean()

    # L_rep: pooled LSE (or InfoNCE if moco_rep). Positive uses tau_rep.
    negs_rep = torch.stack([log_neg_xx, log_neg_hh_all, log_neg_xs_allt], dim=0)
    log_neg_per_anchor_rep = torch.logsumexp(negs_rep, dim=0)
    log_neg_total_rep = torch.logsumexp(
        log_neg_per_anchor_rep, dim=0, keepdim=True)
    if moco_rep and teacher_o is not None:
        teach_norm = F.normalize(teacher_o, p=2, dim=-1)
        hx_teacher_norm = teach_norm[:, :-1]
        log_pos_h = (hx_norm * hx_teacher_norm).sum(-1) / tau_rep
        log_denom_rep = torch.logsumexp(
            torch.stack(
                [log_pos_h, log_neg_total_rep.expand_as(log_pos_h)], dim=0),
            dim=0,
        )
        loss_rep = (log_denom_rep - log_pos_h).mean()
    else:
        loss_rep = log_neg_total_rep.mean()

    return loss_pred + loss_rep, loss_pred, loss_rep


def _rep_only_reference(f, o, tau_rep, teacher_o=None, moco_rep=False):
    """Independent reference for rep_only — every LSE term (and the moco
    positive when on) divides by tau_rep."""
    neg_inf = float('-inf')
    orig_norm = F.normalize(o, p=2, dim=-1)
    hx_norm = orig_norm[:, :-1]
    B, Tm1, C, H = hx_norm.shape
    T = orig_norm.shape[1]

    if teacher_o is not None and moco_rep:
        teach_norm = F.normalize(teacher_o, p=2, dim=-1)
        hx_key = teach_norm[:, :-1]
        orig_key = teach_norm
    else:
        hx_key = hx_norm
        orig_key = orig_norm

    sims_xx = torch.einsum('btih,btjh->btij', hx_norm, hx_key) / tau_rep
    sims_xx = sims_xx.masked_fill(
        torch.eye(C, dtype=torch.bool).view(1, 1, C, C), neg_inf)
    log_neg_xx = torch.logsumexp(sims_xx, dim=2)

    sims_hh = torch.einsum('btch,blch->btcl', hx_norm, orig_key) / tau_rep
    t_idx = torch.arange(Tm1).view(Tm1, 1)
    l_idx = torch.arange(T).view(1, T)
    sims_hh = sims_hh.masked_fill(
        (l_idx == t_idx).view(1, Tm1, 1, T), neg_inf)
    log_neg_hh_all = torch.logsumexp(sims_hh, dim=3)

    sims_xs = torch.einsum('atch,blch->abtcl', hx_norm, orig_key) / tau_rep
    sims_xs = sims_xs.masked_fill(
        torch.eye(B, dtype=torch.bool).view(B, B, 1, 1, 1), neg_inf)
    log_neg_xs_allt = torch.logsumexp(sims_xs, dim=(1, 4))

    negs_rep = torch.stack([log_neg_xx, log_neg_hh_all, log_neg_xs_allt], dim=0)
    log_neg_per_anchor_rep = torch.logsumexp(negs_rep, dim=0)
    log_neg_total_rep = torch.logsumexp(
        log_neg_per_anchor_rep, dim=0, keepdim=True)
    if moco_rep and teacher_o is not None:
        teach_norm = F.normalize(teacher_o, p=2, dim=-1)
        hx_teacher_norm = teach_norm[:, :-1]
        log_pos_h = (hx_norm * hx_teacher_norm).sum(-1) / tau_rep
        log_denom_rep = torch.logsumexp(
            torch.stack(
                [log_pos_h, log_neg_total_rep.expand_as(log_pos_h)], dim=0),
            dim=0,
        )
        return (log_denom_rep - log_pos_h).mean()
    return log_neg_total_rep.mean()


class TestTauRepUnsetPreservesHistoricalObjective:
    """When --tau-rep is unset, the split_pred_rep / rep_only losses must be
    byte-for-byte identical to the historical (single-τ) form."""

    @pytest.mark.parametrize("shape", [
        "cosine_similarity_batch_split_pred_rep",
        "cosine_similarity_batch_rep_only",
    ])
    def test_absent_key_matches_shared_tau(self, shape):
        f, o = _latents(B=4, T=5, C=2, H=12, seed=379)
        base = contrastive_latent_loss(
            (f, o), False, _spec(shape, tau=0.1))
        with_key_none = contrastive_latent_loss(
            (f, o), False, _spec(shape, tau=0.1, tau_rep=None))
        assert torch.allclose(base, with_key_none, atol=1e-12, rtol=1e-12)

    def test_split_pred_rep_tau_rep_matches_tau_gives_original(self):
        """Explicitly setting tau_rep==tau matches the base loss."""
        f, o = _latents(B=4, T=5, C=2, H=12, seed=1379)
        base = contrastive_latent_loss(
            (f, o), False,
            _spec("cosine_similarity_batch_split_pred_rep", tau=0.1))
        equal = contrastive_latent_loss(
            (f, o), False,
            _spec("cosine_similarity_batch_split_pred_rep",
                  tau=0.1, tau_rep=0.1))
        assert torch.allclose(base, equal, atol=1e-12, rtol=1e-12)

    def test_rep_only_tau_rep_matches_tau_gives_original(self):
        f, o = _latents(B=4, T=5, C=2, H=12, seed=2379)
        base = contrastive_latent_loss(
            (f, o), False,
            _spec("cosine_similarity_batch_rep_only", tau=0.1))
        equal = contrastive_latent_loss(
            (f, o), False,
            _spec("cosine_similarity_batch_rep_only", tau=0.1, tau_rep=0.1))
        assert torch.allclose(base, equal, atol=1e-12, rtol=1e-12)


class TestTauRepSplitPredRep:
    """With tau_rep=1.0, L_pred stays on tau=0.10, L_rep uses tau=1.0."""

    def test_l_pred_stays_on_tau_l_rep_uses_tau_rep(self):
        f, o = _latents(B=4, T=5, C=2, H=12, seed=3739)
        got = contrastive_latent_loss(
            (f, o), False,
            _spec("cosine_similarity_batch_split_pred_rep",
                  tau=0.10, tau_rep=1.0))
        want, want_pred, want_rep = _split_reference(
            f, o, tau=0.10, tau_rep=1.0)
        assert torch.allclose(got, want, atol=1e-9, rtol=1e-9)
        # Sanity: split into pred + rep from a reference where L_pred alone
        # uses tau=0.10 must equal a base call with tau_rep=0.10 (identical
        # objective) — a witness that L_pred is independent of tau_rep.
        base = contrastive_latent_loss(
            (f, o), False,
            _spec("cosine_similarity_batch_split_pred_rep", tau=0.10))
        _, base_pred, _ = _split_reference(f, o, tau=0.10)
        assert torch.allclose(want_pred, base_pred, atol=1e-9, rtol=1e-9)
        # And L_rep at tau_rep=1.0 differs from L_rep at tau_rep=0.10
        # (otherwise τ_rep is silently a no-op).
        _, _, base_rep_at_pt1 = _split_reference(f, o, tau=0.10, tau_rep=0.10)
        assert not torch.allclose(want_rep, base_rep_at_pt1, atol=1e-4)

    def test_shifts_loss_vs_shared_tau(self):
        """Sanity — the flag must change the loss (else silently a no-op)."""
        f, o = _latents(B=4, T=5, C=2, H=12, seed=3741, dtype=torch.float32)
        base = contrastive_latent_loss(
            (f, o), False,
            _spec("cosine_similarity_batch_split_pred_rep", tau=0.10)).item()
        with_tr = contrastive_latent_loss(
            (f, o), False,
            _spec("cosine_similarity_batch_split_pred_rep",
                  tau=0.10, tau_rep=1.0)).item()
        assert abs(base - with_tr) > 1e-4

    def test_gradient_finite(self):
        B, T, C, H = 4, 6, 2, 12
        g = torch.Generator().manual_seed(3743)
        f = torch.randn(B, T, C, H, generator=g, requires_grad=True)
        h = torch.randn(B, T, C, H, generator=g, requires_grad=True)
        loss = contrastive_latent_loss(
            (f, h), False,
            _spec("cosine_similarity_batch_split_pred_rep",
                  tau=0.10, tau_rep=1.0))
        loss.backward()
        assert torch.isfinite(f.grad).all() and torch.isfinite(h.grad).all()

    def test_moco_negatives_shape_still_wires_tau_rep_correctly(self):
        """arm3_tr1: moco_negatives on (teacher-in-cross-batch), tau_rep=1.0.
        L_pred uses teacher on the f↔h key side but stays on tau=0.10."""
        B, T, C, H = 4, 5, 2, 12
        g = torch.Generator().manual_seed(3745)
        f = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        o = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        teacher = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        got = contrastive_latent_loss(
            (f, o), False,
            _spec("cosine_similarity_batch_split_pred_rep",
                  tau=0.10, tau_rep=1.0, moco_negatives=True),
            teacher_original_latent=teacher)
        want, _, _ = _split_reference(
            f, o, tau=0.10, tau_rep=1.0,
            teacher_o=teacher, moco_negatives=True)
        assert torch.allclose(got, want, atol=1e-9, rtol=1e-9)

    def test_moco_rep_keys_positive_uses_tau_rep(self):
        """bimoco_tr1: moco_rep_keys on — the log_pos_h positive must
        divide by tau_rep (not tau)."""
        B, T, C, H = 4, 5, 2, 12
        g = torch.Generator().manual_seed(3747)
        f = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        o = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        teacher = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        got = contrastive_latent_loss(
            (f, o), False,
            _spec("cosine_similarity_batch_split_pred_rep",
                  tau=0.10, tau_rep=1.0, moco_rep=True,
                  moco_negatives=True),
            teacher_original_latent=teacher)
        want, _, _ = _split_reference(
            f, o, tau=0.10, tau_rep=1.0,
            teacher_o=teacher, moco_negatives=True, moco_rep=True)
        assert torch.allclose(got, want, atol=1e-9, rtol=1e-9)


class TestTauRepRepOnly:
    """rep_only is entirely L_rep, so every LSE term uses tau_rep."""

    def test_matches_tau_rep_recomputation(self):
        f, o = _latents(B=4, T=5, C=2, H=12, seed=3749)
        got = contrastive_latent_loss(
            (f, o), False,
            _spec("cosine_similarity_batch_rep_only",
                  tau=0.10, tau_rep=1.0))
        want = _rep_only_reference(f, o, tau_rep=1.0)
        assert torch.allclose(got, want, atol=1e-9, rtol=1e-9)

    def test_moco_rep_keys_positive_uses_tau_rep(self):
        B, T, C, H = 4, 5, 2, 12
        g = torch.Generator().manual_seed(3751)
        f = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        o = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        teacher = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
        got = contrastive_latent_loss(
            (f, o), False,
            _spec("cosine_similarity_batch_rep_only",
                  tau=0.10, tau_rep=1.0, moco_rep=True),
            teacher_original_latent=teacher)
        want = _rep_only_reference(
            f, o, tau_rep=1.0, teacher_o=teacher, moco_rep=True)
        assert torch.allclose(got, want, atol=1e-9, rtol=1e-9)

    def test_shifts_loss_vs_shared_tau(self):
        f, o = _latents(B=4, T=5, C=2, H=12, seed=3753, dtype=torch.float32)
        base = contrastive_latent_loss(
            (f, o), False,
            _spec("cosine_similarity_batch_rep_only", tau=0.10)).item()
        with_tr = contrastive_latent_loss(
            (f, o), False,
            _spec("cosine_similarity_batch_rep_only",
                  tau=0.10, tau_rep=1.0)).item()
        assert abs(base - with_tr) > 1e-4
