"""Tests for the regime-crossfade synthetic stream (#325)."""
import numpy as np
import pytest
import torch

from src.synthetic_crossfade import (
    generate_crossfade_batch,
    _sample_crossfade_weight,
    _zscore_per_series,
)


def _real(n, T, C, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, T, C, generator=g)


# --- s(t) weight ----------------------------------------------------------

def test_weight_is_monotone_unit_interval_starting_at_zero():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = _sample_crossfade_weight(256, rng)
        assert s.shape == (256,) and s.dtype == np.float32
        assert s[0] == 0.0                                  # t<=l region present at t=0
        assert (s >= 0).all() and (s <= 1).all()            # bounded
        assert (np.diff(s) >= -1e-6).all()                  # non-decreasing


def test_weight_reaches_one_when_ramp_ends_before_T():
    # A narrow fade centred mid-window saturates to 1 by the end.
    rng = np.random.default_rng(1)
    saw_one = any(_sample_crossfade_weight(256, rng)[-1] == 1.0 for _ in range(50))
    assert saw_one


# --- z-norm ---------------------------------------------------------------

def test_zscore_zero_mean_unit_std_per_channel():
    x = torch.randn(128, 3) * 5 + 2
    z = _zscore_per_series(x)
    assert torch.allclose(z.mean(0), torch.zeros(3), atol=1e-5)
    assert torch.allclose(z.std(0, unbiased=False), torch.ones(3), atol=1e-4)


# --- generator ------------------------------------------------------------

def test_shape_dtype_finite():
    X = generate_crossfade_batch(_real(16, 128, 1), 6, rng=np.random.default_rng(0))
    assert X.shape == (6, 128, 1) and X.dtype == torch.float32
    assert torch.isfinite(X).all()


def test_each_row_is_pointwise_convex_blend_of_two_z_normed_reals():
    # With N=2 the two sources are rows 0 and 1; every output point must lie on
    # the segment between their z-normed values (a per-t convex blend).
    real = _real(2, 200, 1)
    z0, z1 = _zscore_per_series(real[0]), _zscore_per_series(real[1])
    lo, hi = torch.minimum(z0, z1), torch.maximum(z0, z1)
    X = generate_crossfade_batch(real, 8, rng=np.random.default_rng(3))
    assert (X >= lo - 1e-4).all() and (X <= hi + 1e-4).all()


def test_blend_weight_is_shared_across_channels():
    # One s(t) per sample, shared across channels: the implied weight recovered
    # from each channel must agree (where the two sources differ enough).
    real = _real(2, 128, 4)
    z0, z1 = _zscore_per_series(real[0]), _zscore_per_series(real[1])
    denom = z1 - z0
    X = generate_crossfade_batch(real, 5, rng=np.random.default_rng(4))
    big = denom.abs() > 0.5                                  # avoid 0/0
    for row in X:
        s = ((row - z0) / denom)[big]                        # per (t,c) implied weight
        # reshape back to compare channels at equal t is awkward with masking;
        # instead check all recovered weights are valid and the row is monotone-
        # consistent: min/max implied weight per row stay within [0,1].
        assert (s >= -1e-3).all() and (s <= 1 + 1e-3).all()


def test_blend_weight_identical_across_channels_explicit():
    # One s(t) per sample, shared across channels: recover the implied weight
    # from each channel and require the channels to agree at every t. (Two
    # z-normed series necessarily cross, so divide only where |z1-z0| is large.)
    T, C = 96, 3
    real = _real(2, T, C, seed=11)
    z0, z1 = _zscore_per_series(real[0]), _zscore_per_series(real[1])
    denom = z1 - z0
    mask = denom.abs() > 0.3                                  # [T, C] safe positions
    X = generate_crossfade_batch(real, 4, rng=np.random.default_rng(7))
    checked = 0
    for row in X:
        s = (row - z0) / denom                               # [T, C]
        for t in range(T):
            safe = mask[t]
            if int(safe.sum()) >= 2:
                vals = s[t][safe]
                assert float(vals.max() - vals.min()) < 1e-3
                checked += 1
    assert checked > 0                                       # the test actually ran


def test_labels_are_unknown_sentinel():
    X, freq, seas = generate_crossfade_batch(
        _real(8, 64, 1), 5, rng=np.random.default_rng(0), return_labels=True)
    assert X.shape == (5, 64, 1)
    assert freq.shape == (5,) and seas.shape == (5,)
    assert freq.dtype == torch.int64 and (freq == 0).all() and (seas == 0).all()


def test_reproducible():
    a = generate_crossfade_batch(_real(8, 64, 1, seed=1), 4, rng=np.random.default_rng(9))
    b = generate_crossfade_batch(_real(8, 64, 1, seed=1), 4, rng=np.random.default_rng(9))
    assert torch.equal(a, b)


def test_empty_when_n_out_zero():
    X = generate_crossfade_batch(_real(4, 32, 1), 0, rng=np.random.default_rng(0))
    assert X.shape == (0, 32, 1)


def test_requires_two_sources():
    with pytest.raises(ValueError):
        generate_crossfade_batch(_real(1, 32, 1), 2, rng=np.random.default_rng(0))


# --- 3-way dataloader mix -------------------------------------------------

class _FakeHF:
    """Yields a fixed real batch forever (no network)."""
    def __init__(self, x):
        self.x = x

    def __iter__(self):
        while True:
            yield self.x


def test_loader_appends_crossfade_block_after_real():
    from src.dataloader import MixedForkedArmaLoader
    real = _real(10, 128, 1, seed=2)
    loader = MixedForkedArmaLoader(
        hf_loader=_FakeHF(real), synth_bs=0, cross_bs=4,
        T_raw=128, C=1, seed=0, emit_freq_ids=False)
    batch = next(iter(loader))
    assert batch.shape == (14, 128, 1)                       # 10 real + 4 crossfade
    assert torch.equal(batch[:10], real)                     # real block unchanged
    assert torch.isfinite(batch).all()


def test_loader_fork_plus_crossfade_block_sizes_and_labels():
    from src.dataloader import MixedForkedArmaLoader
    real = _real(8, 128, 1, seed=5)
    loader = MixedForkedArmaLoader(
        hf_loader=_FakeHF(real), synth_bs=3, cross_bs=2,
        T_raw=128, C=1, seed=0, emit_freq_ids=True)
    x, freq, seas = next(iter(loader))
    assert x.shape == (13, 128, 1)                           # 8 real + 3 fork + 2 cross
    assert freq.shape == (13,) and seas.shape == (13,)
    assert (freq[8:] == 0).all() and (seas[8:] == 0).all()   # synth labels = unknown
    assert torch.equal(x[:8], real)


def test_loader_cross_block_raises_when_real_subbatch_too_small():
    from src.dataloader import MixedForkedArmaLoader
    real = _real(1, 64, 1)
    loader = MixedForkedArmaLoader(
        hf_loader=_FakeHF(real), synth_bs=0, cross_bs=2,
        T_raw=64, C=1, seed=0, emit_freq_ids=False)
    with pytest.raises(ValueError):
        next(iter(loader))


def test_factory_three_way_split_and_fork_only_default():
    from src import dataloader as dl

    captured = {}
    orig = dl.HFStreamingLoader

    class _Stub:
        def __init__(self, *a, **k):
            captured["batch_size"] = k.get("batch_size")

        def __iter__(self):
            while True:
                yield torch.zeros(captured["batch_size"], 16, 1)

    dl.HFStreamingLoader = _Stub
    try:
        ldr = dl.create_mixed_forked_arma_dataloader(
            repo_id="x", batch_size=100, C=1, mix_ratio=0.10,
            crossfade_ratio=0.10, T_raw=16)
        assert captured["batch_size"] == 80                  # 100 - 10 fork - 10 cross
        assert ldr.synth_bs == 10 and ldr.cross_bs == 10
        # fork-only default leaves crossfade off
        ldr2 = dl.create_mixed_forked_arma_dataloader(
            repo_id="x", batch_size=100, C=1, mix_ratio=0.10, T_raw=16)
        assert ldr2.cross_bs == 0 and captured["batch_size"] == 90
    finally:
        dl.HFStreamingLoader = orig


def test_factory_rejects_ratios_over_one():
    from src.dataloader import create_mixed_forked_arma_dataloader
    with pytest.raises(ValueError):
        create_mixed_forked_arma_dataloader(
            repo_id="x", batch_size=100, C=1, mix_ratio=0.6, crossfade_ratio=0.6)
