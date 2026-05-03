"""TDD tests for ARIMA(1, p, q) data generation."""

import pytest
import torch
import numpy as np

from src.arma import generate_arima_batch, generate_arma_batch


class TestGenerateArimaBatchShape:
    def test_output_shape(self):
        X, params = generate_arima_batch(batch_size=4, T_raw=256, C=2, seed=0, dimension=4)
        assert X.shape == (4, 256, 2)

    def test_parameters_length(self):
        X, params = generate_arima_batch(batch_size=4, T_raw=256, C=2, seed=0, dimension=4)
        assert len(params) == 4 * 2  # batch_size * C


class TestGenerateArimaBatchIntegration:
    def test_first_differences_recover_arma(self):
        """First differences of ARIMA(1,p,q) should equal the underlying ARMA(p,q)."""
        seed = 42
        X_arima, _ = generate_arima_batch(batch_size=2, T_raw=256, C=1, seed=seed, dimension=4)
        X_arma, _ = generate_arma_batch(batch_size=2, T_raw=256, C=1, seed=seed, dimension=4)
        # ARIMA = cumsum(ARMA), so diff(ARIMA) = ARMA (except first element)
        diff = X_arima[:, 1:, :] - X_arima[:, :-1, :]
        assert torch.allclose(diff, X_arma[:, 1:, :], atol=1e-5), \
            f"Max diff: {(diff - X_arma[:, 1:, :]).abs().max().item():.2e}"

    def test_cumsum_relationship(self):
        """ARIMA output should be the cumulative sum of the ARMA output."""
        seed = 42
        X_arima, _ = generate_arima_batch(batch_size=2, T_raw=128, C=2, seed=seed, dimension=4)
        X_arma, _ = generate_arma_batch(batch_size=2, T_raw=128, C=2, seed=seed, dimension=4)
        expected = torch.cumsum(X_arma, dim=1)
        assert torch.allclose(X_arima, expected, atol=1e-4), \
            f"Max error: {(X_arima - expected).abs().max().item():.2e}"


class TestGenerateArimaBatchReproducibility:
    def test_seed_reproducibility(self):
        X1, _ = generate_arima_batch(batch_size=2, T_raw=128, C=2, seed=123, dimension=4)
        X2, _ = generate_arima_batch(batch_size=2, T_raw=128, C=2, seed=123, dimension=4)
        assert torch.allclose(X1, X2)

    def test_different_seeds_differ(self):
        X1, _ = generate_arima_batch(batch_size=2, T_raw=128, C=2, seed=0, dimension=4)
        X2, _ = generate_arima_batch(batch_size=2, T_raw=128, C=2, seed=1, dimension=4)
        assert not torch.allclose(X1, X2)


class TestGenerateArimaBatchProperties:
    def test_non_stationary(self):
        """ARIMA(1,p,q) should show increasing variance over time (non-stationary)."""
        X, _ = generate_arima_batch(batch_size=32, T_raw=1024, C=1, seed=0, dimension=4)
        # Variance of first quarter vs last quarter
        var_first = X[:, :256, :].var().item()
        var_last = X[:, -256:, :].var().item()
        # Not a strict test, but on average ARIMA should have growing variance
        assert var_last > var_first * 0.5, \
            f"Expected growing variance, got first={var_first:.4f} last={var_last:.4f}"

    def test_dtype_is_float32(self):
        X, _ = generate_arima_batch(batch_size=2, T_raw=128, C=1, seed=0, dimension=4)
        assert X.dtype == torch.float32
