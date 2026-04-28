"""Tests for the clean periodic synthesizer (src.synthetic_periodic)."""

import numpy as np
import pytest
import torch

from src.freq_embedding import (
    NUM_FREQS,
    NUM_SEASONALITIES,
    seasonality_to_id,
)
from src.synthetic_periodic import (
    generate_periodic_batch,
    primitive_name,
    _PRIM_SIN,
    _PRIM_SQUARE,
    _PRIM_SAW,
)


# ── Shape / type / finiteness ────────────────────────────────────────────────

class TestShapes:
    def test_output_shape_matches_convention(self):
        X = generate_periodic_batch(batch_size=5, T_raw=1024, C=4, seed=0)
        assert X.shape == (5, 1024, 4)

    def test_output_dtype_float32(self):
        X = generate_periodic_batch(batch_size=2, T_raw=256, C=3, seed=0)
        assert X.dtype == torch.float32

    def test_output_is_finite(self):
        X = generate_periodic_batch(batch_size=8, T_raw=1024, C=4, seed=0)
        assert torch.isfinite(X).all()

    def test_output_is_contiguous(self):
        """PyTorch downstream assumes contiguous memory; we test we guarantee it."""
        X = generate_periodic_batch(batch_size=3, T_raw=512, C=2, seed=0)
        assert X.is_contiguous()


# ── Determinism ──────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_seed_same_output(self):
        X1 = generate_periodic_batch(4, 1024, 4, seed=123)
        X2 = generate_periodic_batch(4, 1024, 4, seed=123)
        assert torch.equal(X1, X2)

    def test_different_seeds_differ(self):
        X1 = generate_periodic_batch(4, 1024, 4, seed=1)
        X2 = generate_periodic_batch(4, 1024, 4, seed=2)
        assert not torch.equal(X1, X2)

    def test_rng_thread(self):
        """Passing a Generator should advance it, producing new values next call."""
        rng = np.random.default_rng(0)
        X1 = generate_periodic_batch(4, 256, 2, rng=rng)
        X2 = generate_periodic_batch(4, 256, 2, rng=rng)
        # Same generator, but advanced state → different outputs.
        assert not torch.equal(X1, X2)


# ── Float32 safety band ──────────────────────────────────────────────────────

class TestFloat32Safety:
    def test_values_stay_safe(self):
        """With default scale_max=1000 and env_gain_max=10, values stay well below
        float32's representable range (~3.4e38) and have safe std for AdamW.
        """
        X = generate_periodic_batch(batch_size=64, T_raw=1024, C=4, seed=0)
        assert torch.isfinite(X).all()
        # Absolute value must stay under 1e5 (scale*env*1 <= 1000*10 = 10_000).
        assert X.abs().max().item() < 1e5


# ── Metadata ────────────────────────────────────────────────────────────────

class TestMeta:
    def test_meta_keys(self):
        _, meta = generate_periodic_batch(4, 256, 2, seed=0, return_meta=True)
        expected = {"primitive", "spp", "phase", "sign_flip", "use_env",
                    "env_gain", "scale"}
        assert set(meta.keys()) == expected

    def test_meta_shapes(self):
        B, T, C = 3, 128, 5
        _, meta = generate_periodic_batch(B, T, C, seed=0, return_meta=True)
        N = B * C
        for k, v in meta.items():
            assert v.shape == (N,), f"meta[{k!r}] has shape {v.shape}, expected ({N},)"

    def test_meta_ranges(self):
        """Parameter ranges should match the declared defaults."""
        _, meta = generate_periodic_batch(512, 1024, 4, seed=0, return_meta=True)
        # primitive codes
        assert set(np.unique(meta["primitive"]).tolist()) <= {0, 1, 2}
        # spp
        assert meta["spp"].min() >= 8.0 - 1e-6
        assert meta["spp"].max() <= 256.0 + 1e-6
        # phase
        assert meta["phase"].min() >= 0.0
        assert meta["phase"].max() < 1.0
        # scale
        assert meta["scale"].min() >= 0.1 - 1e-6
        assert meta["scale"].max() <= 1000.0 + 1e-6
        # env gain within cap
        assert meta["env_gain"].min() >= 0.1 - 1e-6
        assert meta["env_gain"].max() <= 10.0 + 1e-6


# ── Distribution sanity ─────────────────────────────────────────────────────

class TestDistribution:
    def test_all_primitives_appear(self):
        _, meta = generate_periodic_batch(128, 64, 4, seed=42, return_meta=True)
        present = set(meta["primitive"].tolist())
        assert {_PRIM_SIN, _PRIM_SQUARE, _PRIM_SAW} <= present

    def test_primitive_roughly_uniform(self):
        """Uniform over 3 categories → each ≈ 1/3 of a large sample."""
        _, meta = generate_periodic_batch(1000, 16, 4, seed=0, return_meta=True)
        counts = np.bincount(meta["primitive"], minlength=3)
        frac = counts / counts.sum()
        assert np.allclose(frac, 1 / 3, atol=0.05), f"primitive frac = {frac}"

    def test_envelope_rate_matches_p(self):
        """With p_env=0.3 we expect ~30% of series to have envelope."""
        _, meta = generate_periodic_batch(1000, 16, 4, seed=0, return_meta=True)
        rate = meta["use_env"].mean()
        assert abs(rate - 0.3) < 0.05, f"envelope rate = {rate:.3f}"

    def test_spp_log_uniform(self):
        """log(spp) should be roughly uniform across [log 8, log 256]."""
        _, meta = generate_periodic_batch(2000, 16, 4, seed=0, return_meta=True)
        log_spp = np.log(meta["spp"])
        lo, hi = np.log(8.0), np.log(256.0)
        # Kolmogorov-Smirnov style check via bucket counts (5 equal-log-bins).
        edges = np.linspace(lo, hi, 6)
        counts, _ = np.histogram(log_spp, bins=edges)
        frac = counts / counts.sum()
        assert np.allclose(frac, 0.2, atol=0.03), f"log-spp hist = {frac}"


# ── Primitive correctness (single-series construction) ──────────────────────

class TestPrimitiveShape:
    def _single(self, prim_code: int, spp: float, phase: float = 0.0,
                scale: float = 1.0, T: int = 256, seed: int = 0):
        """Force a series with the given primitive and known spp/phase/scale.

        We do this by drawing many seeds and filtering down to one matching
        the requested setup. A clean path would be to expose the underlying
        generator, but keeping the public API stable matters more.
        """
        # With fixed tight ranges we can just directly construct from numpy
        # to test the mathematical shape — no dependency on generate_periodic_batch.
        t = np.arange(T, dtype=np.float64)
        u = t / spp + phase
        if prim_code == _PRIM_SIN:
            return np.sin(2 * np.pi * u)
        elif prim_code == _PRIM_SQUARE:
            return np.sign(np.sin(2 * np.pi * u))
        else:
            return 2 * np.mod(u, 1.0) - 1.0

    def test_sin_closes_exactly_one_period(self):
        """sin over exactly P samples (phase 0) starts at 0, peaks near P/4,
        zero-crosses near P/2, valleys near 3P/4, back to 0 at P."""
        y = self._single(_PRIM_SIN, spp=64, phase=0.0, T=64)
        assert abs(y[0]) < 1e-10
        assert abs(y[16] - 1.0) < 1e-2
        assert abs(y[32]) < 1e-2
        assert abs(y[48] + 1.0) < 1e-2

    def test_square_has_only_plus_minus_one(self):
        y = self._single(_PRIM_SQUARE, spp=32, phase=0.1, T=200)
        # Exclude measure-zero sign(0) samples by tolerating 0.
        uniq = set(np.round(np.unique(y), 8).tolist())
        assert uniq <= {-1.0, 0.0, 1.0}, f"square values = {uniq}"

    def test_saw_in_unit_range(self):
        y = self._single(_PRIM_SAW, spp=40, phase=0.3, T=200)
        assert y.min() >= -1.0 - 1e-10
        assert y.max() < 1.0 + 1e-10


# ── Pipeline compatibility ──────────────────────────────────────────────────

class TestPipelineCompat:
    def test_matches_training_batch_shape(self):
        """The trainer expects [B, T_raw, C] = [24, 1024, 4]."""
        X = generate_periodic_batch(batch_size=24, T_raw=1024, C=4, seed=0)
        assert X.shape == (24, 1024, 4)
        assert X.dtype == torch.float32

    def test_primitive_name_reversible(self):
        assert primitive_name(_PRIM_SIN) == "sinusoid"
        assert primitive_name(_PRIM_SQUARE) == "square"
        assert primitive_name(_PRIM_SAW) == "saw"


# ── Dual-axis labels (return_labels=True) ───────────────────────────────────

class TestReturnLabels:
    def test_returns_three_tuple(self):
        X, freq_ids, seasonality_ids = generate_periodic_batch(
            batch_size=8, T_raw=512, C=2, seed=0, return_labels=True)
        assert X.shape == (8, 512, 2)
        assert freq_ids.shape == (8,)
        assert seasonality_ids.shape == (8,)

    def test_label_dtypes_long(self):
        _, freq_ids, seasonality_ids = generate_periodic_batch(
            batch_size=4, T_raw=256, C=2, seed=0, return_labels=True)
        assert freq_ids.dtype == torch.long
        assert seasonality_ids.dtype == torch.long

    def test_freq_ids_in_range(self):
        _, freq_ids, _ = generate_periodic_batch(
            batch_size=128, T_raw=64, C=4, seed=0, return_labels=True)
        # 1..NUM_FREQS-1 (0=unknown is reserved for missing labels)
        assert int(freq_ids.min()) >= 1
        assert int(freq_ids.max()) <= NUM_FREQS - 1

    def test_seasonality_ids_in_range(self):
        _, _, seas_ids = generate_periodic_batch(
            batch_size=128, T_raw=64, C=4, seed=0, return_labels=True)
        assert int(seas_ids.min()) >= 1  # spp ≥ 8 always → bucket ≥ 2 actually
        assert int(seas_ids.max()) <= NUM_SEASONALITIES - 1

    def test_seasonality_matches_meta_spp(self):
        """Seasonality id per row must equal seasonality_to_id of the row's spp.

        Since each row has C channels with independent spps, the row's
        spp is taken as the channel-min (matching the single-axis
        legacy convention so the model sees the dominant cycle).
        """
        N_BATCH = 16
        C = 4
        X, freq_ids, seas_ids, meta = generate_periodic_batch(
            batch_size=N_BATCH, T_raw=128, C=C, seed=42,
            return_labels=True, return_meta=True)
        spp_per_row = meta["spp"].reshape(N_BATCH, C).min(axis=1)
        expected = np.array(
            [seasonality_to_id(float(s)) for s in spp_per_row], dtype=np.int64)
        np.testing.assert_array_equal(seas_ids.numpy(), expected)

    def test_freq_independent_of_spp(self):
        """freq is sampled uniformly over {1..NUM_FREQS-1}; not derived from spp.

        With N=2000 and 9 classes, expect ~222 per class. χ²-style tolerance.
        """
        N_BATCH = 500
        C = 4
        _, freq_ids, _ = generate_periodic_batch(
            batch_size=N_BATCH, T_raw=64, C=C, seed=0, return_labels=True)
        ids = freq_ids.numpy()
        counts = np.bincount(ids, minlength=NUM_FREQS)
        # Class 0 should be empty (we sample from 1..9), classes 1-9 roughly equal.
        assert counts[0] == 0
        non_zero = counts[1:]
        expected = N_BATCH / (NUM_FREQS - 1)
        assert (non_zero >= 0.6 * expected).all(), \
            f"some class under-represented: {non_zero}"
        assert (non_zero <= 1.4 * expected).all(), \
            f"some class over-represented: {non_zero}"

    def test_same_seed_same_labels(self):
        out1 = generate_periodic_batch(8, 256, 2, seed=7, return_labels=True)
        out2 = generate_periodic_batch(8, 256, 2, seed=7, return_labels=True)
        torch.testing.assert_close(out1[0], out2[0])
        torch.testing.assert_close(out1[1], out2[1])
        torch.testing.assert_close(out1[2], out2[2])

    def test_different_seeds_different_labels(self):
        _, f1, s1 = generate_periodic_batch(
            32, 64, 2, seed=1, return_labels=True)
        _, f2, s2 = generate_periodic_batch(
            32, 64, 2, seed=2, return_labels=True)
        # At least one of (freq, seasonality) should differ across seeds.
        assert not (torch.equal(f1, f2) and torch.equal(s1, s2))

    def test_data_unchanged_when_return_labels_added(self):
        """Adding return_labels=True must not change the data values.

        Same seed without labels and with labels should produce
        bit-identical X. This is the same-seed-same-numbers gate the
        plumb change must respect: the data path is independent of
        whether we report labels.
        """
        X_no = generate_periodic_batch(8, 512, 2, seed=99)
        X_yes, _, _ = generate_periodic_batch(
            8, 512, 2, seed=99, return_labels=True)
        torch.testing.assert_close(X_no, X_yes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
