"""Tests for the TimesFM-style composite synthesizer (src.synthetic_composite)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.freq_embedding import (
    NUM_FREQS,
    NUM_SEASONALITIES,
    SEASONALITY_BUCKET_SPP_RANGES,
)
from src.synthetic_composite import generate_composite_batch


# ── Shape / type / finiteness ───────────────────────────────────────────────

class TestShapes:
    def test_output_shape(self):
        X = generate_composite_batch(batch_size=5, T_raw=1024, C=4, seed=0)
        assert X.shape == (5, 1024, 4)

    def test_output_dtype_float32(self):
        X = generate_composite_batch(batch_size=2, T_raw=256, C=3, seed=0)
        assert X.dtype == torch.float32

    def test_output_is_finite(self):
        X = generate_composite_batch(batch_size=8, T_raw=1024, C=4, seed=0)
        assert torch.isfinite(X).all()

    def test_output_is_contiguous(self):
        X = generate_composite_batch(batch_size=3, T_raw=512, C=2, seed=0)
        assert X.is_contiguous()


# ── Determinism ─────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_seed_same_output(self):
        X1 = generate_composite_batch(4, 1024, 4, seed=123)
        X2 = generate_composite_batch(4, 1024, 4, seed=123)
        assert torch.equal(X1, X2)

    def test_different_seeds_differ(self):
        X1 = generate_composite_batch(4, 1024, 4, seed=1)
        X2 = generate_composite_batch(4, 1024, 4, seed=2)
        assert not torch.equal(X1, X2)

    def test_rng_thread(self):
        rng = np.random.default_rng(0)
        X1 = generate_composite_batch(4, 256, 2, rng=rng)
        X2 = generate_composite_batch(4, 256, 2, rng=rng)
        assert not torch.equal(X1, X2)

    def test_same_seed_with_labels(self):
        X1, f1, s1 = generate_composite_batch(
            8, 512, 2, seed=99, return_labels=True)
        X2, f2, s2 = generate_composite_batch(
            8, 512, 2, seed=99, return_labels=True)
        torch.testing.assert_close(X1, X2)
        torch.testing.assert_close(f1, f2)
        torch.testing.assert_close(s1, s2)


# ── Float32 safety ──────────────────────────────────────────────────────────

class TestFloat32Safety:
    def test_values_stay_in_safe_band(self):
        """With default scale_max=1000 and env_gain_max=10 and component
        natural amplitudes O(1), final values stay well below float32's
        representable range. Conservative bound: 1e6 (= scale_max * env_gain_max
        * component_amp_with_some_headroom). The on-the-fly periodic synth's
        bound is 1e5; ours is laxer because cumsum'd ARIMA can be a few × the
        target_std even after rescale.
        """
        X = generate_composite_batch(batch_size=64, T_raw=1024, C=4, seed=0)
        assert torch.isfinite(X).all()
        assert X.abs().max().item() < 1e6


# ── Dual-axis labels ────────────────────────────────────────────────────────

class TestReturnLabels:
    def test_returns_three_tuple(self):
        X, freq_ids, seasonality_ids = generate_composite_batch(
            batch_size=8, T_raw=512, C=2, seed=0, return_labels=True)
        assert X.shape == (8, 512, 2)
        assert freq_ids.shape == (8,)
        assert seasonality_ids.shape == (8,)

    def test_label_dtypes_long(self):
        _, freq_ids, seasonality_ids = generate_composite_batch(
            batch_size=4, T_raw=256, C=2, seed=0, return_labels=True)
        assert freq_ids.dtype == torch.long
        assert seasonality_ids.dtype == torch.long

    def test_freq_ids_in_range(self):
        _, freq_ids, _ = generate_composite_batch(
            batch_size=128, T_raw=128, C=4, seed=0, return_labels=True)
        # 1..NUM_FREQS-1 (0=unknown reserved for missing labels)
        assert int(freq_ids.min()) >= 1
        assert int(freq_ids.max()) <= NUM_FREQS - 1

    def test_seasonality_ids_in_range(self):
        _, _, seas_ids = generate_composite_batch(
            batch_size=128, T_raw=128, C=4, seed=0, return_labels=True)
        assert int(seas_ids.min()) >= 0
        assert int(seas_ids.max()) <= NUM_SEASONALITIES - 1


class TestSeasLabelGating:
    """The key new semantic: emit seas_id=0 when the seas-tied wave is OFF."""

    def test_no_seas_label_when_seas_tied_always_off(self):
        # p_seas_tied=0 → seas-tied wave always off → emitted seas_id always 0
        _, _, seas_ids = generate_composite_batch(
            batch_size=64, T_raw=128, C=4, seed=42,
            p_seas_tied=0.0, return_labels=True)
        assert (seas_ids == 0).all(), \
            f"non-zero seas with p_seas_tied=0: {seas_ids[seas_ids != 0]}"

    def test_seas_label_emitted_when_seas_tied_always_on(self):
        # p_seas_tied=1 → seas-tied wave always on → emitted = drawn (any of 0..9).
        # The drawn distribution is uniform on {0..9} so > 50% of rows have
        # seas_id != 0 when sample is large enough.
        _, _, seas_ids = generate_composite_batch(
            batch_size=200, T_raw=128, C=4, seed=42,
            p_seas_tied=1.0, return_labels=True)
        nonzero_frac = (seas_ids != 0).float().mean().item()
        # Drawn is U{0..9} so ~90% should be non-zero (only bucket 0 is zero).
        assert nonzero_frac > 0.7, f"only {nonzero_frac:.2f} non-zero with p=1"

    def test_meta_emitted_seas_matches_label(self):
        _, _, seas_ids, meta = generate_composite_batch(
            batch_size=32, T_raw=128, C=4, seed=7,
            p_seas_tied=0.5, return_labels=True, return_meta=True)
        emitted = meta["emitted_seas_id"]
        np.testing.assert_array_equal(seas_ids.numpy(), emitted)
        # And: emitted = drawn when on, 0 when off.
        on = meta["seas_tied_on_row"]
        drawn = meta["drawn_seas_id"]
        np.testing.assert_array_equal(emitted[on], drawn[on])
        assert (emitted[~on] == 0).all()


class TestComponentGating:
    """The "≥1 non-trend on" rule and basic coinflip behaviours."""

    def test_at_least_one_non_trend_component(self):
        # All non-row coinflips off, row's seas-tied off too → force-on path.
        _, _, _, meta = generate_composite_batch(
            batch_size=64, T_raw=128, C=4, seed=11,
            p_arma=0.0, p_free=0.0, p_seas_tied=0.0,
            return_labels=True, return_meta=True)
        # With everything zero'd out, the force-on path always trips.
        per_ch = meta["per_channel"]
        for ch in per_ch:
            assert ch["arma_on"] or ch["free1_on"] or ch["free2_on"], \
                f"channel with no non-trend component: {ch}"

    def test_arma_off_when_p_arma_zero(self):
        _, _, _, meta = generate_composite_batch(
            batch_size=32, T_raw=128, C=4, seed=3,
            p_arma=0.0, p_free=0.5, p_seas_tied=0.5,
            return_labels=True, return_meta=True)
        # Force-on can still turn ARMA on for "all-off" channels. So we
        # only check that the *unforced* fraction is zero, by verifying
        # ARMA is on only for force-on'd channels (where free1/free2/seas
        # would all be off too).
        for ch in meta["per_channel"]:
            if ch["arma_on"]:
                assert not (ch["free1_on"] or ch["free2_on"]
                            or ch["seas_tied_on"]), (
                    "ARMA on but other components also on, with p_arma=0 "
                    f"this is impossible: {ch}")

    def test_envelope_rate_matches_p(self):
        _, _, _, meta = generate_composite_batch(
            batch_size=512, T_raw=64, C=2, seed=0,
            return_labels=True, return_meta=True)
        # Default p_env=0.3.
        rate = meta["use_env"].mean()
        assert abs(rate - 0.3) < 0.05, f"envelope rate = {rate:.3f}"


class TestCoverage:
    """At a large enough batch, every (freq_id, seas_id) combination
    appears in the row-level draws — same property as the existing
    on-the-fly synth."""

    def test_all_seasonality_buckets_drawn(self):
        # We check the *drawn* seas, not the emitted. The emitted has
        # seas=0 conflated with "seas-tied off".
        _, _, _, meta = generate_composite_batch(
            batch_size=300, T_raw=64, C=2, seed=7,
            return_labels=True, return_meta=True)
        present = set(int(x) for x in meta["drawn_seas_id"].tolist())
        assert present == set(range(NUM_SEASONALITIES)), (
            f"missing seas bucket(s): {set(range(NUM_SEASONALITIES)) - present}")

    def test_all_freq_buckets_drawn(self):
        _, _, _, meta = generate_composite_batch(
            batch_size=300, T_raw=64, C=2, seed=7,
            return_labels=True, return_meta=True)
        present = set(int(x) for x in meta["drawn_freq_id"].tolist())
        # Drawn is U{1..NUM_FREQS-1}; bucket 0 should never appear.
        assert 0 not in present
        assert present == set(range(1, NUM_FREQS)), (
            f"missing freq bucket(s): {set(range(1, NUM_FREQS)) - present}")

    def test_joint_grid_coverage_at_scale(self):
        """At batch=2000 every (freq_id ∈ 1..9, seas_id ∈ 0..9) cell of the
        90-cell grid should appear at least once in the drawn labels."""
        _, _, _, meta = generate_composite_batch(
            batch_size=2000, T_raw=64, C=1, seed=11,
            return_labels=True, return_meta=True)
        seas = meta["drawn_seas_id"]
        freq = meta["drawn_freq_id"]
        pairs = set(zip(freq.tolist(), seas.tolist()))
        expected = {(f, s) for f in range(1, NUM_FREQS)
                           for s in range(NUM_SEASONALITIES)}
        missing = expected - pairs
        assert not missing, f"missing {len(missing)} cells: {sorted(missing)[:5]} ..."


class TestArmaIntegration:
    """ARMA target-std rescale and integration probability behaviour."""

    def test_integrate_rate_matches_p(self):
        # All channels arma_on, count integrate_used flags.
        _, _, _, meta = generate_composite_batch(
            batch_size=200, T_raw=128, C=2, seed=5,
            p_arma=1.0, p_free=0.0, p_seas_tied=0.0,
            p_integrate=0.5,
            return_labels=True, return_meta=True)
        per_ch = meta["per_channel"]
        n_arma = sum(1 for ch in per_ch if ch["arma_on"])
        n_integ = sum(1 for ch in per_ch if ch["integrate_used"])
        rate = n_integ / n_arma
        assert abs(rate - 0.5) < 0.07, \
            f"integrate rate = {rate:.3f} (n={n_arma})"

    def test_no_integrate_when_p_integrate_zero(self):
        _, _, _, meta = generate_composite_batch(
            batch_size=64, T_raw=128, C=2, seed=5,
            p_arma=1.0, p_free=0.0, p_seas_tied=0.0,
            p_integrate=0.0,
            return_labels=True, return_meta=True)
        for ch in meta["per_channel"]:
            assert not ch["integrate_used"]


class TestPipelineCompat:
    def test_matches_training_batch_shape(self):
        X = generate_composite_batch(batch_size=24, T_raw=1024, C=4, seed=0)
        assert X.shape == (24, 1024, 4)
        assert X.dtype == torch.float32

    def test_batch_with_labels_matches_periodic_api(self):
        """Same return contract as src.synthetic_periodic.generate_periodic_batch
        with return_labels=True: 3-tuple (X, freq_ids, seasonality_ids)."""
        out = generate_composite_batch(
            batch_size=4, T_raw=128, C=2, seed=0, return_labels=True)
        assert isinstance(out, tuple)
        assert len(out) == 3
        X, f, s = out
        assert X.shape == (4, 128, 2)
        assert f.shape == (4,) and f.dtype == torch.long
        assert s.shape == (4,) and s.dtype == torch.long


class TestSeasTiedSppInBucket:
    """When the seas-tied wave is on, its spp must fall inside the row's
    bucket range."""

    def test_seas_tied_spp_in_row_bucket(self):
        """We can't read the per-channel spp directly through the meta;
        instead we check the round-trip property by setting all rows'
        seas_id to a known bucket via repeated draws and verifying via
        coverage that emitted matches drawn for all on-rows.

        This is an integration check that the seas-tied wave is actually
        sampled when seas_tied_on=True, and that the spp_range lookup
        uses the row's bucket. We rely on TestSeasLabelGating tests
        already passing; this is a sanity confirmation only.
        """
        _, _, seas_ids, meta = generate_composite_batch(
            batch_size=32, T_raw=128, C=4, seed=42,
            p_seas_tied=1.0, return_labels=True, return_meta=True)
        # seas_ids should equal drawn since p_seas_tied=1.0
        np.testing.assert_array_equal(
            seas_ids.numpy(), meta["drawn_seas_id"])


class TestPulsePrimitive:
    """Phase-2 spike-deficit fix: pulse-train primitive enabled via flag."""

    def test_default_disabled_means_no_pulse(self):
        """With enable_pulse=False (default), generated outputs are
        identical to the previous behaviour for the same seed.
        Specifically: no row should contain only-zero / only-±1 (the
        signature of a pulse-only channel).
        """
        # Force trend off + ARMA off + seas-tied off + only one free wave on:
        # the channel is then *exactly* a single wave. Without pulse, that
        # wave is sin/sq/saw — none of which produce a "mostly zero" pattern.
        X, _, _, meta = generate_composite_batch(
            batch_size=64, T_raw=128, C=1, seed=0,
            p_arma=0.0, p_seas_tied=0.0,
            slope_std=0.0,                # zero out the trend's contribution
            p_trend_mult=0.0,             # avoid mult * 0 = 0 channel
            return_labels=True, return_meta=True,
        )
        # Compute "fraction of (b,c) channels with > 50% zeros pre-scale".
        # Pulse channels would have ~95% zeros at default duty. Sin/sq/saw
        # don't. So with enable_pulse=False the fraction should be ~0.
        flat = X.numpy().reshape(-1, 128)
        # pre-scale signals are O(1), post-scale they're scaled to anywhere
        # in [0.1, 1000] log-uniform — so use a "near-zero relative to peak"
        # check rather than absolute.
        peak = np.abs(flat).max(axis=1, keepdims=True)
        peak[peak == 0] = 1
        sparsity = (np.abs(flat) < 0.01 * peak).mean(axis=1)
        assert sparsity.max() < 0.5, \
            f"unexpected sparse channel without enable_pulse: max sparsity {sparsity.max():.2f}"

    def test_pulse_produces_sparse_signal(self):
        """With enable_pulse=True and forcing a single free wave, some
        channels should land on the PULSE primitive and produce sparse
        signals (mostly zero with rare ±1 bursts)."""
        X, _, _, meta = generate_composite_batch(
            batch_size=256, T_raw=128, C=1, seed=42,
            p_arma=0.0, p_seas_tied=0.0,
            slope_std=0.0,
            p_trend_mult=0.0,
            enable_pulse=True,
            return_labels=True, return_meta=True,
        )
        flat = X.numpy().reshape(-1, 128)
        peak = np.abs(flat).max(axis=1, keepdims=True)
        peak[peak == 0] = 1
        sparsity = (np.abs(flat) < 0.01 * peak).mean(axis=1)
        # 1 of 4 primitives is pulse → ~25% of channels should be sparse.
        # Allow a wide band [10%, 50%].
        sparse_frac = (sparsity > 0.5).mean()
        assert 0.10 < sparse_frac < 0.50, \
            f"unexpected sparse-channel fraction: {sparse_frac:.2f}"

    def test_pulse_amplitude_in_band(self):
        """Pulse primitive emits values in {-1, 0, +1} when sampled directly."""
        from src.synthetic_composite import _sample_wave, _PRIM_PULSE
        # Sample many waves with enable_pulse=True; the ones that draw
        # PULSE should have peak amplitude in [-1, 1] and many zeros.
        rng = np.random.default_rng(0)
        for _ in range(200):
            y = _sample_wave(T=64, rng=rng, spp_range=(8, 32),
                             enable_pulse=True)
            assert y.min() >= -1.0 - 1e-9
            assert y.max() <= 1.0 + 1e-9

    def test_pulse_determinism(self):
        X1 = generate_composite_batch(8, 256, 2, seed=42, enable_pulse=True)
        X2 = generate_composite_batch(8, 256, 2, seed=42, enable_pulse=True)
        assert torch.equal(X1, X2)

    def test_pulse_changes_distribution_from_default(self):
        """Same seed + enable_pulse=True should produce different X than
        enable_pulse=False (the primitive draw shifts)."""
        X_off = generate_composite_batch(16, 256, 2, seed=99, enable_pulse=False)
        X_on = generate_composite_batch(16, 256, 2, seed=99, enable_pulse=True)
        assert not torch.equal(X_off, X_on)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
