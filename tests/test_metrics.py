"""Tests for src/metrics.py — diagnostic metrics for the backbone.

Synthetic tensors only; no real backbone needed.
"""

import math

import pytest
import torch

from src.metrics import (
    q_random,
    q_naive_latent,
    dim_usage,
    u_batch,
    u_temporal,
    retrieval_auc_top1,
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
# retrieval_auc_top1
# ---------------------------------------------------------------------------

def test_retrieval_perfect_forecast():
    torch.manual_seed(0)
    B, T, C, H = 4, 32, 3, 16
    h_full = torch.randn(B, T + 1, C, H)
    f = h_full[:, 1:T + 1, :, :].clone()  # f[t] = h[t+1] exactly
    auc, top1 = retrieval_auc_top1(f, h_full)
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
    auc, top1 = retrieval_auc_top1(f, h_full)
    # k=1 negative IS f itself → sim=1, positive can never beat it → Top1=0.
    assert top1.item() == pytest.approx(0.0, abs=1e-6)
    # AUC: positive can still beat negatives k∈{2,4,8} at ~chance, never k=1.
    # Mean over 4 negatives ≤ 3/4; with 0 from k=1 and ~0.5 from others ≈ 0.375.
    assert auc.item() < 0.6


def test_retrieval_returns_nan_when_T_too_small():
    B, T, C, H = 2, 4, 1, 8  # T=4 < max_lag=8
    h_full = torch.randn(B, T + 1, C, H)
    f = torch.randn(B, T, C, H)
    auc, top1 = retrieval_auc_top1(f, h_full)
    assert torch.isnan(auc) and torch.isnan(top1)
