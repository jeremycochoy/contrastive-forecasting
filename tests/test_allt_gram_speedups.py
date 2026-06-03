"""Correctness of the all-time cross-series Gram speedups (#327).

`src.loss.contrastive_latent_loss` has two env-flag-gated, default-OFF
speedups for the `cosine_similarity_batch_full_hh_negs_xshh_allt` loss, whose
all-time cross-series negative is the O((B·T)²·H) term that dominates
large-batch step time:

  XSHH_ALLT_FUSED  — a hand-written fused autograd.Function replacing the
                     per-chunk gradient-checkpoint path (tested here).
  XSHH_ALLT_SHARD  — source-dim sharding across DDP ranks (tested in
                     tests/test_dist_gather.py with 2 gloo processes).

This file pins, on CPU in a single process:

  1. The DEFAULT (flags OFF) path equals an independent brute-force reference
     — the full [B, B, T-1, T] Gram, no chunking/checkpoint — in loss AND in
     the h-latent gradients. So the default path is both correct and (since
     the diff leaves it byte-identical) unchanged.

  2. XSHH_ALLT_FUSED=1 equals that reference in loss AND h-latent gradients to
     < 1e-5, across several XSHH_ALLT_CHUNK values and shapes (incl. C=1,
     T≈64, small B), for both the negatives-only and pos-in-denominator forms.

  3. With a learnable-τ tensor, XSHH_ALLT_FUSED=1 reproduces the reference
     temperature gradient too (the fused Function's `tau` backward).
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.loss import contrastive_latent_loss

LOSS = "cosine_similarity_batch_full_hh_negs_xshh_allt"


def _spec(tau, pos_in_denom):
    return SimpleNamespace(train_configuration={
        "contrastive_divergence_temperature": tau,
        "contrastive_latent_noise": None,
        "loss_shape": LOSS,
        "include_positive_in_denominator": pos_in_denom,
    })


def _latents(B, T, C, H, seed):
    """Deterministic fp64 latents (fp64 makes the <1e-5 bar a comfortable
    margin — actual deltas are ~1e-12 — and isolates math from fp32 noise)."""
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    o = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    return f, o


def _brute_reference(f, o, tau, pos_in_denom):
    """Independent brute-force xshh_allt loss: every negative materialised
    (the all-time term as a full [B, B, T-1, T] Gram), no chunk/checkpoint/
    fused/shard path. Mirrors the branch term-for-term."""
    neg_inf = float('-inf')
    orig_norm = F.normalize(o, p=2, dim=-1)
    fore_norm = F.normalize(f, p=2, dim=-1)
    hy_hat_norm = fore_norm[:, :-1]          # f_t
    hz_hat_norm = fore_norm[:, 1:]           # f_{t+1}
    hx_norm = orig_norm[:, :-1]              # h_t
    hy_norm = orig_norm[:, 1:]               # h_{t+1}
    B, Tm1, C, H = hx_norm.shape
    T = orig_norm.shape[1]

    log_pos = (hy_norm * hy_hat_norm).sum(-1) / tau                      # [B,T-1,C]

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
    log_neg_cross_batch = torch.logsumexp(sims_cb, dim=1)               # [B,T-1,C]

    sims_xs = torch.einsum('atch,blch->abtcl', hx_norm, orig_norm) / tau  # [B,B,T-1,C,T]
    sims_xs = sims_xs.masked_fill(torch.eye(B, dtype=torch.bool).view(B, B, 1, 1, 1), neg_inf)
    log_neg_xs_allt = torch.logsumexp(sims_xs, dim=(1, 4))             # [B,T-1,C]

    negatives = torch.stack([log_neg_xx, log_neg_zy, log_neg_hh_all,
                             log_neg_cross_batch, log_neg_xs_allt], dim=0)
    log_neg_per_anchor = torch.logsumexp(negatives, dim=0)
    log_neg_total = torch.logsumexp(log_neg_per_anchor, dim=0, keepdim=True)
    if pos_in_denom:
        log_denom = torch.logsumexp(
            torch.stack([log_pos, log_neg_total.expand_as(log_pos)], dim=0), dim=0)
        return (log_denom - log_pos).mean()
    return (log_neg_total - log_pos).mean()


def _loss_and_grads(f, o, spec, env, tau_override=None):
    """Run contrastive_latent_loss under a specific env-flag dict, returning
    (loss, df, do[, dtau]). Restores os.environ afterwards."""
    saved = {k: os.environ.get(k) for k in
             ("XSHH_ALLT_FUSED", "XSHH_ALLT_SHARD", "XSHH_ALLT_CHUNK")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        os.environ.update(env)
        fc = f.clone().requires_grad_(True)
        oc = o.clone().requires_grad_(True)
        loss = contrastive_latent_loss((fc, oc), validation=False, spec=spec,
                                       tau_override=tau_override)
        loss.backward()
        out = (loss.detach(), fc.grad.clone(), oc.grad.clone())
        if tau_override is not None and tau_override.requires_grad:
            out = out + (tau_override.grad.clone(),)
        return out
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _assert_match(name, got, ref, atol=1e-6, rtol=1e-5):
    d = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, atol=atol, rtol=rtol), \
        f"{name}: max|Δ|={d:.3e} exceeds tol"
    return d


SHAPES = [(4, 64, 1, 8), (5, 12, 3, 8), (3, 8, 1, 4), (6, 16, 2, 8)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("pos_in_denom", [False, True])
def test_default_path_matches_brute_force(shape, pos_in_denom):
    """(1) The default (flags OFF) path == independent brute-force reference,
    loss AND h-latent grads. Pins today's behaviour as correct + unchanged."""
    B, T, C, H = shape
    tau = 0.1
    f, o = _latents(B, T, C, H, seed=20270327 + B)

    fr = f.clone().requires_grad_(True)
    orr = o.clone().requires_grad_(True)
    ref = _brute_reference(fr, orr, tau, pos_in_denom)
    ref.backward()

    loss, df, do = _loss_and_grads(f, o, _spec(tau, pos_in_denom),
                                   env={"XSHH_ALLT_CHUNK": "2"})
    _assert_match("default loss", loss, ref.detach())
    _assert_match("default df", df, fr.grad)
    _assert_match("default do", do, orr.grad)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("chunk", [1, 2, 8, 1000])
@pytest.mark.parametrize("pos_in_denom", [False, True])
def test_fused_matches_reference(shape, chunk, pos_in_denom):
    """(2) XSHH_ALLT_FUSED=1 == brute-force reference, loss AND h-latent grads,
    across chunk sizes and shapes (incl. C=1, T=64, small B)."""
    B, T, C, H = shape
    tau = 0.1
    f, o = _latents(B, T, C, H, seed=20270327 + B)

    fr = f.clone().requires_grad_(True)
    orr = o.clone().requires_grad_(True)
    ref = _brute_reference(fr, orr, tau, pos_in_denom)
    ref.backward()

    loss, df, do = _loss_and_grads(
        f, o, _spec(tau, pos_in_denom),
        env={"XSHH_ALLT_FUSED": "1", "XSHH_ALLT_CHUNK": str(chunk)})
    _assert_match("fused loss", loss, ref.detach())
    _assert_match("fused df", df, fr.grad)
    _assert_match("fused do", do, orr.grad)


@pytest.mark.parametrize("chunk", [1, 3, 1000])
def test_fused_equals_default_path(chunk):
    """Fused ON and the default checkpoint path agree directly (loss + grads)
    — the drop-in guarantee, independent of the brute-force reference."""
    B, T, C, H = 6, 20, 2, 8
    tau = 0.1
    f, o = _latents(B, T, C, H, seed=99)
    off = _loss_and_grads(f, o, _spec(tau, False),
                          env={"XSHH_ALLT_CHUNK": str(chunk)})
    on = _loss_and_grads(f, o, _spec(tau, False),
                         env={"XSHH_ALLT_FUSED": "1", "XSHH_ALLT_CHUNK": str(chunk)})
    for name, a, b in zip(("loss", "df", "do"), on, off):
        _assert_match(f"fused-vs-default {name}", a, b)


@pytest.mark.parametrize("flag_env", [
    {"XSHH_ALLT_CHUNK": "2"},                          # default checkpoint path
    {"XSHH_ALLT_FUSED": "1", "XSHH_ALLT_CHUNK": "2"},  # fused path
])
def test_tau_gradient_matches_reference(flag_env):
    """(3) Learnable-τ: both the default and fused paths reproduce the
    reference temperature gradient (exercises the fused Function's tau
    backward and the closure-tau grad of the checkpoint path)."""
    B, T, C, H = 5, 16, 2, 8
    tau0 = 0.1
    f, o = _latents(B, T, C, H, seed=7)

    fr = f.clone().requires_grad_(True)
    orr = o.clone().requires_grad_(True)
    tau_r = torch.tensor(tau0, dtype=torch.float64, requires_grad=True)
    ref = _brute_reference(fr, orr, tau_r, pos_in_denom=False)
    ref.backward()

    tau_t = torch.tensor(tau0, dtype=torch.float64, requires_grad=True)
    loss, df, do, dtau = _loss_and_grads(
        f, o, _spec(tau0, False), env=flag_env, tau_override=tau_t)
    _assert_match("loss", loss, ref.detach())
    _assert_match("df", df, fr.grad)
    _assert_match("do", do, orr.grad)
    _assert_match("dtau", dtau, tau_r.grad)
