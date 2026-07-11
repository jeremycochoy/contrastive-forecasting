"""Tests for src/metrics.py — diagnostic metrics for the backbone.

Synthetic tensors only; no real backbone needed.
"""

import math

import pytest
import torch

import torch.nn.functional as F

from src.metrics import (
    q_random,
    q_naive_latent,
    q_hard_neg,
    m_z,
    dim_usage,
    u_batch,
    u_temporal,
    retrieval_auc_top1_legacy,
    retrieval_auc_topk,
)
from src.models import compute_metrics


# ---------------------------------------------------------------------------
# q_random
# ---------------------------------------------------------------------------

def test_q_random_perfect_forecast_is_zero():
    torch.manual_seed(0)
    h_target = torch.randn(64, 4, 3, 16)
    f = h_target.clone()
    q = q_random(f, h_target)
    assert q.item() == pytest.approx(0.0, abs=1e-6)


def test_q_random_random_forecast_near_one():
    torch.manual_seed(0)
    h_target = torch.randn(128, 6, 3, 32)
    f = torch.randn_like(h_target)
    q = q_random(f, h_target).item()
    assert 0.85 <= q <= 1.15, f"Q_random={q}, expected ≈1"


def test_q_random_anti_correlated_about_two():
    torch.manual_seed(0)
    h_target = torch.randn(128, 4, 3, 32)
    f = -h_target
    q = q_random(f, h_target).item()
    assert 1.7 <= q <= 2.3, f"Q_random={q}, expected ≈2"


# ---------------------------------------------------------------------------
# q_naive_latent
# ---------------------------------------------------------------------------

def test_q_naive_better_than_naive_is_zero():
    torch.manual_seed(0)
    h_target = torch.randn(64, 4, 3, 16)
    h_prev = torch.randn_like(h_target)
    q = q_naive_latent(h_target, h_target, h_prev).item()
    assert q == pytest.approx(0.0, abs=1e-6)


def test_q_naive_worse_than_naive_is_large():
    torch.manual_seed(0)
    h_target = torch.randn(64, 4, 3, 16)
    f = torch.randn_like(h_target)
    h_prev = h_target.clone()  # naive baseline is perfect → denom ≈ 0
    q = q_naive_latent(f, h_target, h_prev).item()
    assert q > 1e3, f"Q_naive={q}, expected very large"


# ---------------------------------------------------------------------------
# dim_usage
# ---------------------------------------------------------------------------

def test_dim_usage_orthonormal_is_one():
    z = torch.eye(8)  # (n=8, d=8) — pairwise cos² = 0
    u = dim_usage(z, axis=0).item()
    assert u == pytest.approx(1.0, abs=1e-6)


def test_dim_usage_collinear_is_one_over_d():
    d = 8
    base = torch.randn(d)
    z = base.unsqueeze(0).repeat(8, 1) * torch.randn(8, 1)  # (n=8, d=8) all collinear
    u = dim_usage(z, axis=0).item()
    assert u == pytest.approx(1.0 / d, abs=1e-3)


def test_dim_usage_chunked_equals_unchunked(monkeypatch):
    # The chunked accumulation path (#369, DIM_USAGE_CHUNK < n) must be
    # numerically identical to the one-shot matmul path.
    torch.manual_seed(0)
    z = torch.randn(64, 3, 16)  # (n=64, fixed=3, d=16)
    monkeypatch.setenv("DIM_USAGE_CHUNK", "0")   # one-shot path
    u_full = dim_usage(z, axis=0).item()
    monkeypatch.setenv("DIM_USAGE_CHUNK", "7")   # chunked path, n % chunk != 0
    u_chunked = dim_usage(z, axis=0).item()
    assert u_chunked == pytest.approx(u_full, abs=1e-6)


def test_u_batch_isotropic_near_one():
    torch.manual_seed(0)
    z = torch.randn(256, 4, 3, 32)  # (B=256, T, C, H)
    u = u_batch(z).item()
    assert 0.85 <= u <= 1.0, f"U_batch={u}"


def test_u_temporal_axis_param():
    torch.manual_seed(0)
    z = torch.randn(8, 256, 3, 32)  # time at axis=1, n=256 isotropic
    u = u_temporal(z, time_axis=1).item()
    assert 0.85 <= u <= 1.0, f"U_temporal={u}"


# ---------------------------------------------------------------------------
# retrieval_auc_top1_legacy
# ---------------------------------------------------------------------------

def test_retrieval_perfect_forecast():
    torch.manual_seed(0)
    B, T, C, H = 4, 32, 3, 16
    h_full = torch.randn(B, T + 1, C, H)
    f = h_full[:, 1:T + 1, :, :].clone()  # f[t] = h[t+1] exactly
    auc, top1 = retrieval_auc_top1_legacy(f, h_full)
    assert auc.item() == pytest.approx(1.0, abs=1e-6)
    assert top1.item() == pytest.approx(1.0, abs=1e-6)


def test_retrieval_forecast_equals_past_lag():
    torch.manual_seed(0)
    B, T, C, H = 4, 32, 3, 16
    h_full = torch.randn(B, T + 1, C, H)
    # f[t] = h[t-1] (so f exactly matches the k=1 negative; positive can't beat it).
    f = torch.empty(B, T, C, H)
    for t in range(T):
        f[:, t, :, :] = h_full[:, max(t - 1, 0), :, :]
    auc, top1 = retrieval_auc_top1_legacy(f, h_full)
    # k=1 negative IS f itself → sim=1, positive can never beat it → Top1=0.
    assert top1.item() == pytest.approx(0.0, abs=1e-6)
    # AUC: positive can still beat negatives k∈{2,4,8} at ~chance, never k=1.
    # Mean over 4 negatives ≤ 3/4; with 0 from k=1 and ~0.5 from others ≈ 0.375.
    assert auc.item() < 0.6


def test_retrieval_returns_nan_when_T_too_small():
    B, T, C, H = 2, 4, 1, 8  # T=4 < max_lag=8
    h_full = torch.randn(B, T + 1, C, H)
    f = torch.randn(B, T, C, H)
    auc, top1 = retrieval_auc_top1_legacy(f, h_full)
    assert torch.isnan(auc) and torch.isnan(top1)


# ---------------------------------------------------------------------------
# q_hard_neg
# ---------------------------------------------------------------------------

def test_q_hard_neg_perfect_forecast_is_zero():
    # sim_pos = 1 ⇒ err_p = 0 ⇒ ratio = 0 regardless of negs.
    sim_pos = torch.ones(8, 4)
    sim_neg = torch.rand(8, 4, 16) * 0.5
    q = q_hard_neg(sim_pos, sim_neg).item()
    assert q == pytest.approx(0.0, abs=1e-6)


def test_q_hard_neg_tied_with_hardest_is_one():
    # If sim_pos == max(sim_neg) per query, err_p / err_h = 1.
    sim_neg = torch.rand(8, 4, 16) * 0.5
    sim_pos = sim_neg.max(dim=-1).values
    q = q_hard_neg(sim_pos, sim_neg).item()
    assert q == pytest.approx(1.0, abs=1e-6)


def test_q_hard_neg_loses_to_hardest_is_greater_than_one():
    sim_neg = torch.full((8, 4, 16), 0.8)
    sim_pos = torch.full((8, 4), 0.3)
    # err_p = 0.7, err_h = 0.2 → ratio = 3.5
    q = q_hard_neg(sim_pos, sim_neg).item()
    assert q == pytest.approx(3.5, abs=1e-6)


def test_q_hard_neg_shape_assertion():
    with pytest.raises(AssertionError):
        q_hard_neg(torch.zeros(8, 4), torch.zeros(8, 3, 16))  # mismatched leading dims


# ---------------------------------------------------------------------------
# m_z
# ---------------------------------------------------------------------------

def test_m_z_perfect_high_d_is_large():
    # Random isotropic negatives in dim d: σ_neg ≈ 1/√d, μ ≈ 0.
    # sim_pos = 1 ⇒ m_z ≈ √d.
    torch.manual_seed(0)
    d = 256
    n_neg = 512
    pos_vec = torch.randn(d)
    pos_vec = pos_vec / pos_vec.norm()
    neg_vecs = torch.randn(n_neg, d)
    neg_vecs = neg_vecs / neg_vecs.norm(dim=-1, keepdim=True)
    sim_pos = torch.tensor([1.0])
    sim_neg = (pos_vec.unsqueeze(0) * neg_vecs).sum(-1).unsqueeze(0)  # (1, N)
    z = m_z(sim_pos, sim_neg).item()
    # In dim 256, √d ≈ 16; tolerate noise.
    assert 12 < z < 20, f"m_z={z}, expected ≈ √d ≈ 16"


def test_m_z_random_forecast_near_zero():
    # Random forecast against random negatives in high d: m_z ≈ 0.
    torch.manual_seed(0)
    d = 256
    B, n_neg = 64, 256
    f = torch.randn(B, d)
    f = f / f.norm(dim=-1, keepdim=True)
    targets = torch.randn(B, d)
    targets = targets / targets.norm(dim=-1, keepdim=True)
    negs = torch.randn(B, n_neg, d)
    negs = negs / negs.norm(dim=-1, keepdim=True)
    sim_pos = (f * targets).sum(-1)                    # (B,)
    sim_neg = (f.unsqueeze(1) * negs).sum(-1)          # (B, n_neg)
    z = m_z(sim_pos, sim_neg).item()
    assert abs(z) < 1.0, f"m_z={z}, expected ≈ 0 for random forecast"


def test_m_z_shape_assertion():
    with pytest.raises(AssertionError):
        m_z(torch.zeros(8, 4), torch.zeros(8, 3, 16))


# ---------------------------------------------------------------------------
# retrieval_auc_topk
# ---------------------------------------------------------------------------

def test_retrieval_topk_perfect_forecast():
    torch.manual_seed(0)
    B, T, C, H = 8, 32, 1, 16
    h_full = torch.randn(B, T + 1, C, H)
    f = h_full[:, 1:T + 1, :, :].clone()
    out = retrieval_auc_topk(f, h_full, n_batch_negs=4)
    assert out["auc"].item() == pytest.approx(1.0, abs=1e-6)
    assert out["top1"].item() == pytest.approx(1.0, abs=1e-6)
    assert out["top3"].item() == pytest.approx(1.0, abs=1e-6)
    assert out["mrr"].item() == pytest.approx(1.0, abs=1e-6)
    assert out["q_hard_neg"].item() == pytest.approx(0.0, abs=1e-6)
    # m_z: large positive (sim_pos = 1, neg mean ≈ 0, neg std small).
    assert out["m_z"].item() > 3.0


def test_retrieval_topk_returns_nan_when_T_too_small():
    B, T, C, H = 4, 4, 1, 8  # T=4 < max_lag=8
    h_full = torch.randn(B, T + 1, C, H)
    f = torch.randn(B, T, C, H)
    out = retrieval_auc_topk(f, h_full)
    for key in ("auc", "mrr", "top1", "top3", "q_hard_neg", "m_z"):
        assert torch.isnan(out[key]), f"{key} not NaN"


def test_retrieval_topk_returns_nan_when_B_is_one():
    B, T, C, H = 1, 32, 1, 8
    h_full = torch.randn(B, T + 1, C, H)
    f = torch.randn(B, T, C, H)
    out = retrieval_auc_topk(f, h_full)
    for key in ("auc", "mrr", "top1", "top3", "q_hard_neg", "m_z"):
        assert torch.isnan(out[key])


def test_retrieval_topk_product_structure_matches_reference():
    # Verify the negative pool is the cross-batch × temporal-offset
    # product (n_b × n_lags), and that auc/top1/mrr match a parallel
    # reference computation following the documented layout.
    torch.manual_seed(42)
    B, T, C, H = 4, 16, 1, 8
    h_full = torch.randn(B, T + 1, C, H)
    f = torch.randn(B, T, C, H)
    lookback_lags = (1, 2, 4)
    n_batch_negs = 2  # 2 b' × 3 lags = 6 negs per query

    out = retrieval_auc_topk(
        f, h_full, lookback_lags=lookback_lags,
        n_batch_negs=n_batch_negs, top_k=(1, 3),
    )

    # Reference: replicate rand_b and the per-lag gather/cosine path.
    max_lag = max(lookback_lags)
    f_v = f[:, max_lag:T, :, :]
    pos = h_full[:, max_lag + 1:T + 1, :, :]
    sim_pos_ref = torch.nn.functional.cosine_similarity(f_v, pos, dim=-1)

    n_b = min(n_batch_negs, B - 1)
    raw = torch.arange(n_b).unsqueeze(0).expand(B, n_b)
    b_idx = torch.arange(B).unsqueeze(1)
    rand_b = raw + (raw >= b_idx).long()

    sims = []
    for k in lookback_lags:
        h_slice = h_full[:, max_lag + 1 - k:T + 1 - k, :, :]
        h_gather = h_slice[rand_b]
        sim_k = torch.nn.functional.cosine_similarity(
            f_v.unsqueeze(1), h_gather, dim=-1,
        )
        sims.append(sim_k.permute(0, 2, 3, 1))
    sim_neg_ref = torch.cat(sims, dim=-1)

    # Pool size is the product (not the sum).
    expected_n_neg = n_b * len(lookback_lags)
    assert sim_neg_ref.shape[-1] == expected_n_neg

    beats_ref = (sim_pos_ref.unsqueeze(-1) > sim_neg_ref).float()
    auc_ref = beats_ref.mean(dim=-1).mean()
    n_beats_ref = beats_ref.sum(dim=-1)
    rank_ref = expected_n_neg - n_beats_ref + 1.0
    mrr_ref = (1.0 / rank_ref).mean()
    top1_ref = (n_beats_ref >= expected_n_neg).float().mean()

    assert torch.allclose(out["auc"], auc_ref, atol=1e-6), (out["auc"], auc_ref)
    assert torch.allclose(out["mrr"], mrr_ref, atol=1e-6)
    assert torch.allclose(out["top1"], top1_ref, atol=1e-6)


def test_retrieval_topk_brute_force_oracle():
    # Independent oracle: enumerate every (b, t, b', k) explicitly and
    # compute the expected negative sim from the documented layout
    # h_full[b', t+1-k, c, :]. The product-structure test above uses the
    # same vectorized slice as the impl; an off-by-one introduced in
    # both would pass that test. This loop has no shared slicing logic
    # with the impl, so it catches alignment bugs.
    torch.manual_seed(7)
    B, T, C, H = 4, 10, 1, 8
    h_full = torch.randn(B, T + 1, C, H)
    f = torch.randn(B, T, C, H)
    lookback_lags = (1, 2, 4)  # max_lag = 4, T_v = 6
    n_batch_negs = 2
    max_lag = max(lookback_lags)
    T_v = T - max_lag

    n_b = min(n_batch_negs, B - 1)
    raw = torch.arange(n_b).unsqueeze(0).expand(B, n_b)
    b_idx = torch.arange(B).unsqueeze(1)
    rand_b = raw + (raw >= b_idx).long()

    n_neg = n_b * len(lookback_lags)
    expected_sim_neg = torch.empty(B, T_v, C, n_neg)
    for b in range(B):
        for t_idx in range(T_v):
            t = max_lag + t_idx
            f_vec = f[b, t, 0, :]
            slot = 0
            for k in lookback_lags:
                for j in range(n_b):
                    bp = rand_b[b, j].item()
                    neg_vec = h_full[bp, t + 1 - k, 0, :]
                    expected_sim_neg[b, t_idx, 0, slot] = torch.nn.functional.cosine_similarity(
                        f_vec.unsqueeze(0), neg_vec.unsqueeze(0)
                    )
                    slot += 1

    sim_pos_expected = torch.nn.functional.cosine_similarity(
        f[:, max_lag:T, :, :], h_full[:, max_lag + 1:T + 1, :, :], dim=-1
    )
    # AUC and MRR are order-invariant in the neg axis; the impl's
    # internal lag/batch interleaving doesn't have to match `slot`.
    beats_expected = (sim_pos_expected.unsqueeze(-1) > expected_sim_neg).float()
    auc_expected = beats_expected.mean(dim=-1).mean()
    n_beats_expected = beats_expected.sum(dim=-1)
    rank_expected = n_neg - n_beats_expected + 1.0
    mrr_expected = (1.0 / rank_expected).mean()

    out = retrieval_auc_topk(
        f, h_full, lookback_lags=lookback_lags,
        n_batch_negs=n_batch_negs, top_k=(1,),
    )
    assert torch.allclose(out["auc"], auc_expected, atol=1e-6), (out["auc"], auc_expected)
    assert torch.allclose(out["mrr"], mrr_expected, atol=1e-6), (out["mrr"], mrr_expected)


def test_retrieval_topk_random_forecast_auc_near_half():
    torch.manual_seed(0)
    B, T, C, H = 16, 32, 1, 256
    h_full = torch.randn(B, T + 1, C, H)
    f = torch.randn(B, T, C, H)
    out = retrieval_auc_topk(f, h_full, n_batch_negs=8)
    # Random forecast: AUC should hover around 0.5.
    assert 0.4 < out["auc"].item() < 0.6
    # MRR for n_neg = 32 random: ≈ harmonic-mean-ish, well below 1.
    assert out["mrr"].item() < 0.3


# ---------------------------------------------------------------------------
# einsum memory-rewrite equivalence pins.
#
# Two per-step diagnostics used to materialise a [..., B, ..., H] broadcast
# intermediate (~12-26 GB fp32 at training scale) just to dot-product over
# H, OOM-ing every step at B=256/T=256/H=384. They were rewritten to
# contract H *inside* an einsum (→ batched matmul, ~MBs). These freeze the
# pre-rewrite formulas as references and assert the live functions are
# numerically identical (the value is a logged metric — must not drift).
# ---------------------------------------------------------------------------


def _frozen_compute_metrics_broadcast(f_lat, o_lat, cld):
    """Verbatim pre-einsum src.models.compute_metrics (broadcast form)."""
    fn = F.normalize(f_lat, p=2, dim=-1)
    on = F.normalize(o_lat, p=2, dim=-1)
    hyh = fn[:, :-cld, :, :]
    hyn = on[:, cld:, :, :]
    hxn = on[:, :-cld, :, :]
    ff = (hyh * hyn).sum(-1).mean().item()
    fp = (hyh * hxn).sum(-1).mean().item()
    tp = (hyn * hxn).sum(-1).mean().item()
    B = hyh.shape[0]
    sims = (hyh.unsqueeze(0) * hyn.unsqueeze(1)).sum(-1)   # [B,B,T',C,H]→sum H
    m = ~torch.eye(B, dtype=torch.bool, device=sims.device)
    m = m.view(B, B, 1, 1)
    cb = sims.masked_fill(~m, 0).mean().item()
    return ff, fp, tp, cb


@pytest.mark.parametrize("B,T,C,H,cld", [(5, 6, 2, 8, 1),
                                         (7, 9, 1, 16, 1),
                                         (4, 5, 3, 12, 2)])
def test_compute_metrics_einsum_equals_broadcast(B, T, C, H, cld):
    g = torch.Generator().manual_seed(0)
    f = torch.randn(B, T, C, H, generator=g)
    o = torch.randn(B, T, C, H, generator=g)
    ref = _frozen_compute_metrics_broadcast(f, o, cld)
    got = compute_metrics(f, o, cld)
    for r, v, name in zip(ref, got, ("ff", "fp", "tp", "cross_batch")):
        assert math.isclose(r, v, rel_tol=1e-5, abs_tol=1e-6), (
            f"{name}: einsum {v} != broadcast {r}")


def _frozen_retrieval_sim_neg_gather(f, h_full, lags, n_batch_negs):
    """Verbatim pre-einsum sim_neg path of retrieval_auc_topk (the only
    code that changed — everything downstream is byte-identical, so
    sim_neg equivalence ⇒ identical public dict)."""
    B, T, C, H = f.shape
    max_lag = max(lags)
    f_v = f[:, max_lag:T, :, :]
    n_b = min(n_batch_negs, B - 1)
    raw = torch.arange(n_b).unsqueeze(0).expand(B, n_b)
    b_idx = torch.arange(B).unsqueeze(1)
    rand_b = raw + (raw >= b_idx).long()
    sp = []
    for k in lags:
        hs = h_full[:, max_lag + 1 - k:T + 1 - k, :, :]
        sk = F.cosine_similarity(f_v.unsqueeze(1), hs[rand_b], dim=-1)
        sp.append(sk.permute(0, 2, 3, 1))
    return torch.cat(sp, dim=-1)


def _live_retrieval_sim_neg_einsum(f, h_full, lags, n_batch_negs):
    """Mirror of the rewritten sim_neg path in src.metrics."""
    B, T, C, H = f.shape
    max_lag = max(lags)
    f_v = f[:, max_lag:T, :, :]
    n_b = min(n_batch_negs, B - 1)
    raw = torch.arange(n_b).unsqueeze(0).expand(B, n_b)
    b_idx = torch.arange(B).unsqueeze(1)
    rand_b = raw + (raw >= b_idx).long()
    f_n = F.normalize(f_v, dim=-1, eps=1e-8)
    sp = []
    for k in lags:
        hs = h_full[:, max_lag + 1 - k:T + 1 - k, :, :]
        h_n = F.normalize(hs, dim=-1, eps=1e-8)
        full = torch.einsum('btch,Btch->bBtc', f_n, h_n)
        sp.append(full[b_idx, rand_b].permute(0, 2, 3, 1))
    return torch.cat(sp, dim=-1)


@pytest.mark.parametrize("B,T,H", [(6, 14, 8), (9, 20, 16), (5, 12, 10)])
def test_retrieval_auc_topk_sim_neg_einsum_equals_gather(B, T, H):
    lags, n_neg = (1, 2, 4, 8), 128
    g = torch.Generator().manual_seed(1)
    f = torch.randn(B, T, 1, H, generator=g)
    h_full = torch.randn(B, T + 1, 1, H, generator=g)
    ref = _frozen_retrieval_sim_neg_gather(f, h_full, lags, n_neg)
    got = _live_retrieval_sim_neg_einsum(f, h_full, lags, n_neg)
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-4), (
        f"sim_neg drift: max|Δ|={(ref - got).abs().max().item():.2e}")
    # Public fn still runs and returns finite scalars.
    out = retrieval_auc_topk(f, h_full, lookback_lags=lags, n_batch_negs=n_neg)
    assert all(torch.isfinite(v) for v in out.values())
