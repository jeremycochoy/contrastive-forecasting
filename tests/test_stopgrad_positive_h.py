"""`stopgrad_positive_h` (#336 follow-up): SimSiam/BYOL-style target
stop-grad on the InfoNCE positive of the xshh_allt loss.

The config key detaches the ENCODER side h_{t+1} of the positive cosine
sim(h_{t+1}, f_{t+1}) everywhere that term appears — numerator and, under
pos-in-denominator, denominator — while every negative keeps its gradient
on h. This file pins, on CPU in fp64:

  1. FORWARD INVARIANCE — the loss VALUE is bit-identical with the flag on
     vs off (detach cuts a backward edge only), for both the negatives-only
     and pos-in-denominator forms, and with the floor subtraction on.
  2. GRADIENT CORRECTNESS — with the flag on, (df, dh) equal an independent
     brute-force reference that applies the same detach (full dense Gram,
     no chunking/checkpoint), across shapes including C=1 and chunked runs.
  3. EDGE CUT — vs the no-flag run on the same inputs: the FORECASTER
     gradient df is unchanged (no f-edge is touched), and the ENCODER
     gradient dh differs (the positive's h-edge was carried before).
  4. GUARD — any other loss_shape with the key set raises
     NotImplementedError rather than silently training without the
     stop-grad.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.loss import contrastive_latent_loss

LOSS = "cosine_similarity_batch_full_hh_negs_xshh_allt"


def _spec(tau, pos_in_denom, stopgrad=False, sub_floor=False):
    return SimpleNamespace(train_configuration={
        "contrastive_divergence_temperature": tau,
        "contrastive_latent_noise": None,
        "loss_shape": LOSS,
        "include_positive_in_denominator": pos_in_denom,
        "stopgrad_positive_h": stopgrad,
        "subtract_contrastive_floor": sub_floor,
    })


def _latents(B, T, C, H, seed):
    """Deterministic fp64 latents (isolates math from fp32 noise)."""
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    o = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    return f, o


def _brute_reference(f, o, tau, pos_in_denom, stopgrad):
    """Independent brute-force xshh_allt loss (every negative as a full
    dense Gram, no chunk/checkpoint), with the positive's encoder side
    optionally detached — mirrors the production branch term-for-term."""
    neg_inf = float('-inf')
    orig_norm = F.normalize(o, p=2, dim=-1)
    fore_norm = F.normalize(f, p=2, dim=-1)
    hy_hat_norm = fore_norm[:, :-1]          # f_t
    hz_hat_norm = fore_norm[:, 1:]           # f_{t+1}
    hx_norm = orig_norm[:, :-1]              # h_t
    hy_norm = orig_norm[:, 1:]               # h_{t+1}
    B, Tm1, C, H = hx_norm.shape
    T = orig_norm.shape[1]

    hy_pos = hy_norm.detach() if stopgrad else hy_norm
    log_pos = (hy_pos * hy_hat_norm).sum(-1) / tau                       # [B,T-1,C]

    sims_xx = torch.einsum('btih,btjh->btij', hx_norm, hx_norm) / tau    # [B,T-1,C,C]
    sims_xx = sims_xx.masked_fill(torch.eye(C, dtype=torch.bool).view(1, 1, C, C), neg_inf)
    log_neg_xx = torch.logsumexp(sims_xx, dim=2)                         # [B,T-1,C]

    sims_zy = torch.einsum('btih,btjh->btij', hz_hat_norm, hy_hat_norm) / tau
    log_neg_zy = torch.logsumexp(sims_zy, dim=2)                         # [B,T-1,C]

    sims_hh = torch.einsum('btch,blch->btcl', hx_norm, orig_norm) / tau  # [B,T-1,C,T]
    t_idx = torch.arange(Tm1).view(Tm1, 1)
    l_idx = torch.arange(T).view(1, T)
    sims_hh = sims_hh.masked_fill((l_idx == t_idx).view(1, Tm1, 1, T), neg_inf)
    log_neg_hh_all = torch.logsumexp(sims_hh, dim=3)                     # [B,T-1,C]

    sims_cb = torch.einsum('atch,btch->abtc', hy_hat_norm, hy_norm) / tau  # [B,B,T-1,C]
    sims_cb = sims_cb.masked_fill(torch.eye(B, dtype=torch.bool).view(B, B, 1, 1), neg_inf)
    log_neg_cross_batch = torch.logsumexp(sims_cb, dim=1)                # [B,T-1,C]

    sims_xs = torch.einsum('atch,blch->abtcl', hx_norm, orig_norm) / tau  # [B,B,T-1,C,T]
    sims_xs = sims_xs.masked_fill(torch.eye(B, dtype=torch.bool).view(B, B, 1, 1, 1), neg_inf)
    log_neg_xs_allt = torch.logsumexp(sims_xs, dim=(1, 4))               # [B,T-1,C]

    negatives = torch.stack([log_neg_xx, log_neg_zy, log_neg_hh_all,
                             log_neg_cross_batch, log_neg_xs_allt], dim=0)
    log_neg_per_anchor = torch.logsumexp(negatives, dim=0)
    log_neg_total = torch.logsumexp(log_neg_per_anchor, dim=0, keepdim=True)
    if pos_in_denom:
        log_denom = torch.logsumexp(
            torch.stack([log_pos, log_neg_total.expand_as(log_pos)], dim=0), dim=0)
        return (log_denom - log_pos).mean()
    return (log_neg_total - log_pos).mean()


def _loss_and_grads(f, o, spec, chunk="2"):
    saved = os.environ.get("XSHH_ALLT_CHUNK")
    try:
        os.environ["XSHH_ALLT_CHUNK"] = chunk
        fc = f.clone().requires_grad_(True)
        oc = o.clone().requires_grad_(True)
        loss = contrastive_latent_loss((fc, oc), validation=False, spec=spec)
        loss.backward()
        return loss.detach(), fc.grad.clone(), oc.grad.clone()
    finally:
        if saved is None:
            os.environ.pop("XSHH_ALLT_CHUNK", None)
        else:
            os.environ["XSHH_ALLT_CHUNK"] = saved


def _assert_match(name, got, ref, atol=1e-6, rtol=1e-5):
    d = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, atol=atol, rtol=rtol), \
        f"{name}: max|Δ|={d:.3e} exceeds tol"


SHAPES = [(4, 8, 1, 8), (5, 12, 3, 8), (3, 6, 2, 4)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("pos_in_denom", [False, True])
def test_forward_value_unchanged(shape, pos_in_denom):
    """(1) The flag must not change the loss VALUE (it cuts a backward
    edge only)."""
    B, T, C, H = shape
    f, o = _latents(B, T, C, H, seed=20260610 + B)
    loss_off, _, _ = _loss_and_grads(f, o, _spec(0.1, pos_in_denom, stopgrad=False))
    loss_on, _, _ = _loss_and_grads(f, o, _spec(0.1, pos_in_denom, stopgrad=True))
    assert torch.equal(loss_on, loss_off)


def test_forward_value_unchanged_with_floor():
    """(1) Same forward invariance under --subtract-contrastive-floor
    (the #328 recipe trains with floor subtraction on)."""
    f, o = _latents(4, 8, 1, 8, seed=7)
    loss_off, _, _ = _loss_and_grads(
        f, o, _spec(0.1, pos_in_denom=True, stopgrad=False, sub_floor=True))
    loss_on, _, _ = _loss_and_grads(
        f, o, _spec(0.1, pos_in_denom=True, stopgrad=True, sub_floor=True))
    assert torch.equal(loss_on, loss_off)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("pos_in_denom", [False, True])
@pytest.mark.parametrize("chunk", ["1", "2", "1000"])
def test_grads_match_brute_force(shape, pos_in_denom, chunk):
    """(2) With the flag on, loss AND both gradients equal the dense
    brute-force reference with the same detach, for several chunk sizes
    (the chunked/checkpointed path must cut exactly the same edge)."""
    B, T, C, H = shape
    tau = 0.1
    f, o = _latents(B, T, C, H, seed=20260611 + B)

    fr = f.clone().requires_grad_(True)
    orr = o.clone().requires_grad_(True)
    ref = _brute_reference(fr, orr, tau, pos_in_denom, stopgrad=True)
    ref.backward()

    loss, df, do = _loss_and_grads(
        f, o, _spec(tau, pos_in_denom, stopgrad=True), chunk=chunk)
    _assert_match("loss", loss, ref.detach())
    _assert_match("df", df, fr.grad)
    _assert_match("do", do, orr.grad)


@pytest.mark.parametrize("pos_in_denom", [False, True])
def test_f_grad_unchanged_h_grad_differs(pos_in_denom):
    """(3) vs the no-flag run: df identical (no forecaster edge touched),
    dh different (the positive's encoder edge existed before the cut)."""
    f, o = _latents(4, 8, 2, 8, seed=99)
    _, df_off, do_off = _loss_and_grads(f, o, _spec(0.1, pos_in_denom, stopgrad=False))
    _, df_on, do_on = _loss_and_grads(f, o, _spec(0.1, pos_in_denom, stopgrad=True))
    _assert_match("df on==off", df_on, df_off, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(do_on, do_off, atol=1e-9), \
        "encoder gradient should change when the positive edge is cut"


def test_other_loss_shape_raises():
    """(4) The key on any other loss_shape fails loud."""
    f, o = _latents(2, 4, 1, 4, seed=3)
    spec = _spec(0.1, pos_in_denom=False, stopgrad=True)
    spec.train_configuration["loss_shape"] = \
        "cosine_similarity_batch_full_hh_negs_xshh"
    with pytest.raises(NotImplementedError, match="stopgrad_positive_h"):
        contrastive_latent_loss((f, o), validation=False, spec=spec)
