"""Tests for the forked-continuation ARIMA generator (#318 follow-up)."""
import numpy as np
import torch

from src.synthetic_forked_arma import generate_forked_arma_batch


def _first_divergence(a, b):
    d = np.abs(a - b)
    nz = np.nonzero(d > 0)[0]
    return int(nz[0]) if len(nz) else len(a)


def test_shape_and_dtype():
    X = generate_forked_arma_batch(8, T_raw=512, C=1, seed=0)
    assert X.shape == (8, 512, 1) and X.dtype == torch.float32
    assert torch.isfinite(X).all()


def test_exact_shared_prefix_then_diverge_no_integrate():
    # Without integration the ARMA values share the prefix EXACTLY up to the
    # fork l, then differ. Each adjacent (2k, 2k+1) is a forked pair.
    X = generate_forked_arma_batch(6, T_raw=512, C=1, seed=1, integrate=False).numpy()
    for k in range(3):
        a, b = X[2 * k, :, 0], X[2 * k + 1, :, 0]
        l = _first_divergence(a, b)
        assert 128 <= l <= 384, f"fork {l} out of [0.25,0.75]·T"          # fork range
        assert np.array_equal(a[:l], b[:l]), "prefix must be bit-identical"  # exact
        assert not np.array_equal(a[l:], b[l:]), "futures must diverge"      # divergent


def test_integrated_preserves_fork():
    # cumsum (ARIMA d=1) preserves the shared prefix and the divergence.
    X = generate_forked_arma_batch(4, T_raw=512, C=1, seed=2, integrate=True).numpy()
    for k in range(2):
        a, b = X[2 * k, :, 0], X[2 * k + 1, :, 0]
        l = _first_divergence(a, b)
        assert 128 <= l <= 384
        assert np.allclose(a[:l], b[:l], atol=0), "integrated prefix must match"
        assert not np.allclose(a[l:], b[l:]), "integrated futures must diverge"


def test_pairs_are_distinct_across_pairs():
    # Different pairs are independent (different prefixes).
    X = generate_forked_arma_batch(4, T_raw=256, C=1, seed=3, integrate=False).numpy()
    assert not np.array_equal(X[0, :, 0], X[2, :, 0]), "pairs must be independent"


def test_labels_are_unknown_sentinel():
    X, freq, seas = generate_forked_arma_batch(6, T_raw=256, C=1, seed=4, return_labels=True)
    assert X.shape == (6, 256, 1)
    assert freq.shape == (6,) and seas.shape == (6,)
    assert freq.dtype == torch.int64 and (freq == 0).all()   # no canonical frequency
    assert (seas == 0).all()


def test_reproducible():
    a = generate_forked_arma_batch(4, T_raw=256, C=1, seed=7).numpy()
    b = generate_forked_arma_batch(4, T_raw=256, C=1, seed=7).numpy()
    assert np.array_equal(a, b)


def test_multichannel():
    X = generate_forked_arma_batch(4, T_raw=256, C=3, seed=5, integrate=False).numpy()
    assert X.shape == (4, 256, 3)
    # each channel forks independently but the pair still shares its per-channel prefix
    for c in range(3):
        l = _first_divergence(X[0, :, c], X[1, :, c])
        assert np.array_equal(X[0, :l, c], X[1, :l, c])
