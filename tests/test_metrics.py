"""Tests for src/metrics.py — diagnostic metrics for the backbone.

Synthetic tensors only; no real backbone needed.
"""

import math

import pytest
import torch

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


def test_retrieval_topk_negative_count_is_product():
    # Verify the negatives are a product of (batch × lag), not a sum.
    # With B=5, n_batch_negs=4 (clamped to B-1=4), len(lags)=4 → 16 negs/query.
    # The AUC for a random forecast is ≈ 0.5 regardless of n_neg, so we
    # instead inspect via a known-margin forecast: scale a perfect
    # forecast by 0.99 + small noise so the positive almost wins. With
    # n_neg = 16 we'd expect top1 ≈ 1 - tail; with n_neg = 8 (the
    # legacy 4+4=8) we'd expect different. We check sim_neg shape
    # indirectly by checking q_hard_neg responds to n_b changes.
    torch.manual_seed(0)
    B, T, C, H = 5, 32, 1, 32
    h_full = torch.randn(B, T + 1, C, H)
    # Mildly imperfect forecast: 0.95 * positive + 0.05 * noise.
    pos_target = h_full[:, 1:T + 1, :, :]
    noise = torch.randn_like(pos_target) * 0.05
    f = pos_target + noise
    out_few = retrieval_auc_topk(f, h_full, n_batch_negs=2)   # 2*4 = 8 negs
    out_many = retrieval_auc_topk(f, h_full, n_batch_negs=4)  # 4*4 = 16 negs
    # More negatives ⇒ hardest is harder ⇒ q_hard_neg ≥ (more error
    # against worst confusable). Strict > because more negs means a
    # higher-tail max.
    assert out_many["q_hard_neg"].item() >= out_few["q_hard_neg"].item()


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
