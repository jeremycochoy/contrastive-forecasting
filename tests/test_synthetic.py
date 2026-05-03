"""TDD tests for synthetic data generation (TimesFM-style composite)."""

import pytest
import numpy as np
import torch

from src.synthetic import (
    generate_piecewise_linear,
    generate_sinusoid,
    sample_arma_from_roots,
    generate_synthetic_batch,
)


# ── Piecewise linear trend ──

class TestPiecewiseLinear:
    def test_output_length(self):
        rng = np.random.default_rng(0)
        y = generate_piecewise_linear(1024, rng)
        assert len(y) == 1024

    def test_is_piecewise_constant_derivative(self):
        """Derivative of piecewise linear should be piecewise constant."""
        rng = np.random.default_rng(42)
        y = generate_piecewise_linear(512, rng)
        dy = np.diff(y)
        # Count unique values (within tolerance) — should be small (2-8 segments)
        unique_slopes = len(np.unique(np.round(dy, 8)))
        assert unique_slopes <= 10

    def test_reproducible_with_seed(self):
        y1 = generate_piecewise_linear(256, np.random.default_rng(0))
        y2 = generate_piecewise_linear(256, np.random.default_rng(0))
        np.testing.assert_array_equal(y1, y2)


# ── Sinusoid ──

class TestSinusoid:
    def test_output_length(self):
        rng = np.random.default_rng(0)
        y = generate_sinusoid(1024, rng)
        assert len(y) == 1024

    def test_bounded(self):
        """Sinusoid amplitude should be <= 1 (unit amplitude)."""
        rng = np.random.default_rng(42)
        y = generate_sinusoid(2048, rng)
        assert np.abs(y).max() <= 1.0 + 1e-6

    def test_period_in_valid_range(self):
        """Over many samples, periods should span a wide range (log-uniform)."""
        periods = []
        for seed in range(100):
            rng = np.random.default_rng(seed)
            y = generate_sinusoid(1024, rng)
            # Estimate period via FFT
            fft = np.abs(np.fft.rfft(y))
            peak_freq = np.argmax(fft[1:]) + 1  # skip DC
            period = 1024 / peak_freq
            periods.append(period)
        # Log-uniform should give some short AND long periods
        assert min(periods) < 20, f"No short periods found, min={min(periods)}"
        assert max(periods) > 100, f"No long periods found, max={max(periods)}"

    def test_reproducible_with_seed(self):
        y1 = generate_sinusoid(256, np.random.default_rng(0))
        y2 = generate_sinusoid(256, np.random.default_rng(0))
        np.testing.assert_array_equal(y1, y2)


# ── Root-based ARMA sampling ──

class TestSampleArmaFromRoots:
    def test_output_length(self):
        rng = np.random.default_rng(0)
        y = sample_arma_from_roots(512, rng, dimension=8)
        assert len(y) == 512

    def test_stable_process(self):
        """Generated ARMA should be stationary (bounded variance)."""
        rng = np.random.default_rng(42)
        y = sample_arma_from_roots(4096, rng, dimension=8)
        # Stationary process: variance in first half ≈ variance in second half
        var_first = np.var(y[:2048])
        var_second = np.var(y[2048:])
        ratio = max(var_first, var_second) / (min(var_first, var_second) + 1e-10)
        assert ratio < 10, f"Variance ratio {ratio:.1f} too large for stationary process"

    def test_no_nan(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            y = sample_arma_from_roots(1024, rng, dimension=8)
            assert not np.isnan(y).any(), f"NaN at seed {seed}"

    def test_reproducible(self):
        y1 = sample_arma_from_roots(256, np.random.default_rng(0), dimension=4)
        y2 = sample_arma_from_roots(256, np.random.default_rng(0), dimension=4)
        np.testing.assert_array_equal(y1, y2)


# ── Composite synthetic batch ──

class TestGenerateSyntheticBatch:
    def test_output_shape(self):
        X, = generate_synthetic_batch(batch_size=4, T_raw=256, C=2, seed=0)
        assert X.shape == (4, 256, 2)

    def test_dtype_float32(self):
        X, = generate_synthetic_batch(batch_size=2, T_raw=128, C=1, seed=0)
        assert X.dtype == torch.float32

    def test_no_nan(self):
        X, = generate_synthetic_batch(batch_size=8, T_raw=512, C=2, seed=0)
        assert not torch.isnan(X).any()
        assert not torch.isinf(X).any()

    def test_seed_reproducibility(self):
        X1, = generate_synthetic_batch(batch_size=2, T_raw=128, C=2, seed=42)
        X2, = generate_synthetic_batch(batch_size=2, T_raw=128, C=2, seed=42)
        assert torch.allclose(X1, X2)

    def test_different_seeds_differ(self):
        X1, = generate_synthetic_batch(batch_size=2, T_raw=128, C=2, seed=0)
        X2, = generate_synthetic_batch(batch_size=2, T_raw=128, C=2, seed=1)
        assert not torch.allclose(X1, X2)

    def test_channels_independent(self):
        """Each channel should be a different series."""
        X, = generate_synthetic_batch(batch_size=1, T_raw=256, C=4, seed=0)
        # Check that channels are not identical
        for i in range(1, 4):
            assert not torch.allclose(X[0, :, 0], X[0, :, i]), \
                f"Channel 0 and {i} are identical"

    def test_non_constant(self):
        """Output should not be constant (at least some variance)."""
        X, = generate_synthetic_batch(batch_size=4, T_raw=512, C=1, seed=0)
        for b in range(4):
            assert X[b, :, 0].std() > 1e-6, f"Batch {b} is constant"


# ── End-to-end training step ──

class TestSyntheticEndToEnd:
    def test_one_train_step_with_synthetic_data(self):
        """Full pipeline: synthetic data + RevEWMNorm + model + contrastive loss + backward."""
        from types import SimpleNamespace
        from src.models import ConfigurableModel
        from src.loss import contrastive_latent_loss

        C, H, W = 2, 64, 16
        model = ConfigurableModel(
            C=C, H=H, W=W, encoder_type='gru', num_layers=1, nhead=2,
            ffn_mult=2, rev_norm_span=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        X, = generate_synthetic_batch(batch_size=2, T_raw=64, C=C, seed=0)

        # Apply RevEWMNorm + transformer (skip channel mixing)
        if model.rev_norm is not None:
            X = model.rev_norm(X, mode='norm')
        B, Tr, Cc = X.shape
        T = Tr // W
        xr = X.view(B, T, W, Cc).permute(0, 1, 3, 2)
        f_flat, o_flat = model.transformer(xr)
        f_lat = f_flat.reshape(B, Cc, T, H).permute(0, 2, 1, 3)
        o_lat = o_flat.reshape(B, Cc, T, H).permute(0, 2, 1, 3)

        spec = SimpleNamespace(train_configuration={
            'contrastive_divergence_temperature': 0.07,
            'contrastive_latent_noise': None,
            'loss_shape': 'cosine_similarity_batch_no_time_neg',
            'contrastive_latent_delay': 0,
        })
        loss = contrastive_latent_loss((f_lat, o_lat), validation=False, spec=spec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)
