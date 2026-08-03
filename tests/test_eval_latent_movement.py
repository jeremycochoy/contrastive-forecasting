"""Unit tests for :mod:`src.eval_latent_movement` (#379).

Pins the ``mean_one_minus_cos`` reduction against an independent
scipy/sklearn implementation on hand-picked toy tensors, so a bug in
the F.cosine_similarity axis handling would show up as a numeric
disagreement rather than a silently-wrong plot.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.eval_latent_movement import mean_one_minus_cos


def _mean_1_cos_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Independent reference: 1 - cos per last-dim vector pair, then mean."""
    a_n = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=-1, keepdims=True)
    cos = (a_n * b_n).sum(axis=-1)
    return float((1.0 - cos).mean())


def test_identical_tensors_movement_zero():
    """cos(x, x) = 1 → 1 - cos = 0 elementwise → mean = 0."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 3, 8)
    assert mean_one_minus_cos(x, x) == pytest.approx(0.0, abs=1e-6)


def test_opposite_tensors_movement_two():
    """cos(x, -x) = -1 → 1 - cos = 2 elementwise → mean = 2."""
    torch.manual_seed(0)
    x = torch.randn(3, 4, 2, 6)
    assert mean_one_minus_cos(x, -x) == pytest.approx(2.0, abs=1e-6)


def test_orthogonal_tensors_movement_one():
    """Hand-built orthogonal pair: every (b, t, c) sees cos = 0 → mean = 1."""
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).view(1, 2, 1, 2)
    b = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).view(1, 2, 1, 2)
    assert mean_one_minus_cos(a, b) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("shape", [(2, 3, 1, 4), (4, 8, 2, 16), (1, 5, 3, 6)])
def test_matches_numpy_reference(shape):
    """Random tensors: torch reduction bit-close to numpy reference."""
    rng = np.random.default_rng(20260722)
    a_np = rng.standard_normal(shape).astype(np.float64)
    b_np = rng.standard_normal(shape).astype(np.float64)
    got = mean_one_minus_cos(torch.from_numpy(a_np), torch.from_numpy(b_np))
    assert got == pytest.approx(_mean_1_cos_numpy(a_np, b_np), rel=1e-6, abs=1e-6)


def test_matches_scipy_when_available():
    """Cross-check against scipy.spatial.distance.cosine when installed."""
    scipy_spatial = pytest.importorskip("scipy.spatial.distance")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 3)).astype(np.float64)
    b = rng.standard_normal((4, 3)).astype(np.float64)
    scipy_mean = float(np.mean(
        [scipy_spatial.cosine(a[i], b[i]) for i in range(a.shape[0])]))
    got = mean_one_minus_cos(torch.from_numpy(a), torch.from_numpy(b))
    assert got == pytest.approx(scipy_mean, rel=1e-6, abs=1e-6)
