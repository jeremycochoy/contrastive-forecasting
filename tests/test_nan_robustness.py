"""Tests for NaN robustness in the data loading and model forward pipeline.

Root cause: the HF dataset can contain rows that are entirely NaN (no valid
observations). The _forward_fill_nan function silently passes these through,
causing NaN to propagate through RevEWMNorm, the transformer, and the loss,
crashing training (observed at step 24970 of forecasting head training).

These tests ensure:
1. _forward_fill_nan handles all-NaN sequences by replacing with zeros
2. The data loader produces NaN-free batches even when source data has all-NaN rows
3. RevEWMNorm produces finite output even for edge-case inputs
4. The full forward pipeline (backbone + head) produces finite output for valid data
"""

import numpy as np
import pytest
import torch

from src.dataloader import _forward_fill_nan


class TestForwardFillNaN:
    """Tests for _forward_fill_nan handling of NaN values."""

    def test_no_nan_passthrough(self):
        """Normal data should be unchanged."""
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        original = arr.copy()
        _forward_fill_nan(arr)
        np.testing.assert_array_equal(arr, original)

    def test_middle_nan_forward_filled(self):
        """NaN in the middle should be forward-filled from previous value."""
        arr = np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float32)
        _forward_fill_nan(arr)
        np.testing.assert_array_equal(arr, [1.0, 1.0, 3.0, 3.0])

    def test_leading_nan_backfilled(self):
        """Leading NaN should be backfilled from first valid value."""
        arr = np.array([np.nan, np.nan, 3.0, 4.0], dtype=np.float32)
        _forward_fill_nan(arr)
        np.testing.assert_array_equal(arr, [3.0, 3.0, 3.0, 4.0])

    def test_all_nan_replaced_with_zeros(self):
        """All-NaN sequences must be replaced with zeros (not left as NaN).

        This is the root cause of the step-24970 crash: an all-NaN row from the
        HF dataset was silently passed through, causing NaN to propagate through
        the entire pipeline.
        """
        arr = np.full(1024, np.nan, dtype=np.float32)
        _forward_fill_nan(arr)
        assert not np.isnan(arr).any(), (
            "All-NaN sequence must not contain NaN after forward_fill_nan"
        )
        np.testing.assert_array_equal(arr, np.zeros(1024, dtype=np.float32))

    def test_all_nan_short_sequence(self):
        """All-NaN short sequence should also be replaced with zeros."""
        arr = np.full(5, np.nan, dtype=np.float32)
        _forward_fill_nan(arr)
        assert not np.isnan(arr).any()
        np.testing.assert_array_equal(arr, np.zeros(5, dtype=np.float32))

    def test_single_nan(self):
        """Single-element all-NaN array should become zero."""
        arr = np.array([np.nan], dtype=np.float32)
        _forward_fill_nan(arr)
        assert not np.isnan(arr).any()
        assert arr[0] == 0.0

    def test_single_valid_value(self):
        """One valid value surrounded by NaN should fill everything."""
        arr = np.array([np.nan, np.nan, 5.0, np.nan, np.nan], dtype=np.float32)
        _forward_fill_nan(arr)
        np.testing.assert_array_equal(arr, [5.0, 5.0, 5.0, 5.0, 5.0])

    def test_output_never_nan(self):
        """No matter the input pattern, output must never contain NaN."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            arr = rng.standard_normal(1024).astype(np.float32)
            # Randomly set some values to NaN (0-100% of values)
            nan_frac = rng.random()
            mask = rng.random(1024) < nan_frac
            arr[mask] = np.nan
            _forward_fill_nan(arr)
            assert not np.isnan(arr).any(), (
                f"Output contains NaN with {mask.sum()} NaN inputs "
                f"({nan_frac:.1%} NaN fraction)"
            )


class TestRevEWMNormRobustness:
    """Tests that RevEWMNorm handles edge cases without producing NaN/Inf."""

    @pytest.fixture
    def rev_norm(self):
        from src.norm import RevEWMNorm
        return RevEWMNorm(num_features=4, span=32, patch_size=16)

    def test_constant_input(self, rev_norm):
        """Constant input (zero variance) should not produce NaN."""
        x = torch.ones(2, 1024, 4)
        out = rev_norm(x, mode='norm')
        assert torch.isfinite(out).all(), "Constant input produced non-finite output"

    def test_zero_input(self, rev_norm):
        """All-zero input should not produce NaN."""
        x = torch.zeros(2, 1024, 4)
        out = rev_norm(x, mode='norm')
        assert torch.isfinite(out).all(), "Zero input produced non-finite output"

    def test_large_magnitude_input(self, rev_norm):
        """Very large values (like the 920k seen in data) should be handled."""
        x = torch.randn(2, 1024, 4) * 1e6
        out = rev_norm(x, mode='norm')
        assert torch.isfinite(out).all(), "Large-magnitude input produced non-finite output"

    def test_mixed_scale_channels(self, rev_norm):
        """Different channels with vastly different scales."""
        x = torch.randn(2, 1024, 4)
        x[:, :, 0] *= 1e-6  # tiny
        x[:, :, 1] *= 1e6   # huge
        x[:, :, 2] = 0.0    # constant zero
        x[:, :, 3] = 42.0   # constant nonzero
        out = rev_norm(x, mode='norm')
        assert torch.isfinite(out).all(), "Mixed-scale input produced non-finite output"


class TestPipelineNaNFree:
    """End-to-end test: valid data through backbone should never produce NaN."""

    @pytest.fixture
    def backbone(self):
        from src.models import ConfigurableModel
        model = ConfigurableModel(
            C=4, H=64, W=16,  # Smaller H for faster testing
            encoder_type="gru", num_layers=2, nhead=4,
            ffn_mult=2.0, activation="gelu", depthwise_conv=3, dropout=0.0,
            rev_norm_span=32,
        )
        model.eval()
        return model

    def test_normal_input_finite(self, backbone):
        """Normal random input should produce finite latents."""
        x = torch.randn(4, 1024, 4)
        with torch.no_grad():
            if backbone.rev_norm is not None:
                x_norm = backbone.rev_norm(x, mode='norm')
            else:
                x_norm = x
            B, T_raw, C = x_norm.shape
            T = T_raw // backbone.W
            xr = x_norm.view(B, T, backbone.W, C).permute(0, 1, 3, 2)
            f_flat, o_flat = backbone.transformer(xr)
        assert torch.isfinite(f_flat).all(), "Normal input produced NaN in f_flat"
        assert torch.isfinite(o_flat).all(), "Normal input produced NaN in o_flat"

    def test_zero_channel_finite(self, backbone):
        """Input with a zero channel (after all-NaN replacement) should be finite."""
        x = torch.randn(4, 1024, 4)
        x[:, :, 2] = 0.0  # Simulate all-NaN -> zero replacement
        with torch.no_grad():
            if backbone.rev_norm is not None:
                x_norm = backbone.rev_norm(x, mode='norm')
            else:
                x_norm = x
            B, T_raw, C = x_norm.shape
            T = T_raw // backbone.W
            xr = x_norm.view(B, T, backbone.W, C).permute(0, 1, 3, 2)
            f_flat, o_flat = backbone.transformer(xr)
        assert torch.isfinite(f_flat).all(), "Zero-channel input produced NaN"
        assert torch.isfinite(o_flat).all(), "Zero-channel input produced NaN"
