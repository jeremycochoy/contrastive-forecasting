"""TDD tests for ForecastingHead -- GRU-based head that decodes backbone
forecaster latents into future normalized values.

All tests are runnable offline using small random tensors and mocked backbones.
"""

import math

import pytest
import torch
import torch.nn as nn

from src.forecasting_head import (
    ForecastingHead,
    W,
    FORECAST_LEN,
    extract_forecaster_latents,
    compute_valid_targets,
    forecast_autoregressive,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# Backbone dimensions used throughout the tests (small for speed)
B, C, T_RAW, H = 2, 4, 1024, 512
T = T_RAW // W  # 64 patches


class FakeRevNorm(nn.Module):
    """Minimal stand-in for RevEWMNorm that records stats but is an identity."""

    def __init__(self, C, T_raw):
        super().__init__()
        self.mean = None
        self.stdev = None
        self._C = C
        self._T_raw = T_raw

    def __call__(self, x, mode='norm'):
        B = x.shape[0]
        self.mean = torch.zeros(B, self._T_raw, self._C)
        self.stdev = torch.ones(B, self._T_raw, self._C)
        return x


class FakeTransformer(nn.Module):
    """Returns random (f_flat, o_flat) of the correct shape."""

    def __init__(self, H):
        super().__init__()
        self._H = H
        # Need a parameter so that .to(device) works on the parent
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, xr):
        # xr: (B, T, C, W) -> internally reshaped to (B*C, T, H)
        B_orig, T, C_orig, _W = xr.shape
        BC = B_orig * C_orig
        f_flat = torch.randn(BC, T, self._H)
        o_flat = torch.randn(BC, T, self._H)
        return f_flat, o_flat


class FakeBackbone(nn.Module):
    """Mimics ConfigurableModel with the minimum interface needed by
    extract_forecaster_latents and forecast_autoregressive."""

    def __init__(self, C=4, H=512, W_=16, T_raw=1024):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W_
        self.rev_norm = FakeRevNorm(C, T_raw)
        self.transformer = FakeTransformer(H)


# ---------------------------------------------------------------------------
# 1. ForecastingHead output shape
# ---------------------------------------------------------------------------

class TestForecastingHeadShape:
    def test_output_shape(self):
        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        x = torch.randn(B * C, T, H)
        out = head(x)
        assert out.shape == (B * C, T, 128)

    def test_output_shape_different_forecast_len(self):
        head = ForecastingHead(H=H, hidden_dim=64, num_gru_layers=1, forecast_len=64)
        x = torch.randn(B * C, T, H)
        out = head(x)
        assert out.shape == (B * C, T, 64)

    def test_output_shape_single_batch(self):
        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        x = torch.randn(1, T, H)
        out = head(x)
        assert out.shape == (1, T, 128)


# ---------------------------------------------------------------------------
# 2. Parameter count is reasonable (< 1M)
# ---------------------------------------------------------------------------

class TestParameterCount:
    def test_param_count_under_1m(self):
        head = ForecastingHead(H=512, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        n_params = sum(p.numel() for p in head.parameters())
        assert n_params < 1_000_000, f"Too many params: {n_params:,}"

    def test_param_count_positive(self):
        head = ForecastingHead(H=512, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        n_params = sum(p.numel() for p in head.parameters())
        assert n_params > 0


# ---------------------------------------------------------------------------
# 3. Forward pass produces finite values (no NaN)
# ---------------------------------------------------------------------------

class TestFiniteValues:
    def test_no_nan_output(self):
        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        head.eval()
        x = torch.randn(B * C, T, H)
        out = head(x)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_no_nan_with_large_input(self):
        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        head.eval()
        x = torch.randn(B * C, T, H) * 100
        out = head(x)
        assert not torch.isnan(out).any(), "Output contains NaN with large input"

    def test_no_nan_with_zero_input(self):
        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        head.eval()
        x = torch.zeros(B * C, T, H)
        out = head(x)
        assert not torch.isnan(out).any(), "Output contains NaN with zero input"


# ---------------------------------------------------------------------------
# 4. Valid patch count = 56 for T=64, W=16, forecast_len=128
# ---------------------------------------------------------------------------

class TestValidPatchCount:
    def test_valid_patch_count_default(self):
        """For T=64, W=16, forecast_len=128:
        target for patch t starts at (t+1)*16 and needs 128 values.
        (t+1)*16 + 128 <= 1024 -> t+1 <= 56 -> t <= 55
        So valid patches are t=0..55, i.e. T_valid=56.
        """
        x_norm = torch.randn(B, T_RAW, C)
        targets, T_valid = compute_valid_targets(x_norm, W=16, forecast_len=128)
        assert T_valid == 56

    def test_valid_patch_count_small_forecast(self):
        """forecast_len=16: (t+1)*16 + 16 <= 1024 -> t <= 62, T_valid=63."""
        x_norm = torch.randn(B, T_RAW, C)
        targets, T_valid = compute_valid_targets(x_norm, W=16, forecast_len=16)
        assert T_valid == 63

    def test_valid_patch_count_large_forecast(self):
        """forecast_len=512: (t+1)*16 + 512 <= 1024 -> t <= 31, T_valid=32."""
        x_norm = torch.randn(B, T_RAW, C)
        targets, T_valid = compute_valid_targets(x_norm, W=16, forecast_len=512)
        assert T_valid == 32


# ---------------------------------------------------------------------------
# 5. Target extraction shape and values correct
# ---------------------------------------------------------------------------

class TestTargetExtraction:
    def test_target_shape(self):
        x_norm = torch.randn(B, T_RAW, C)
        targets, T_valid = compute_valid_targets(x_norm, W=16, forecast_len=128)
        assert targets.shape == (B * C, T_valid, 128)

    def test_target_values_match_input(self):
        """Verify the extracted target for patch t equals
        x_norm[:, (t+1)*W : (t+1)*W + forecast_len, :]."""
        torch.manual_seed(42)
        x_norm = torch.randn(B, T_RAW, C)
        targets, T_valid = compute_valid_targets(x_norm, W=16, forecast_len=128)

        # Check a few specific patches
        for t in [0, 10, 55]:
            start = (t + 1) * 16
            end = start + 128
            # targets is (B*C, T_valid, 128)
            # x_norm is (B, T_raw, C)
            expected = x_norm[:, start:end, :]  # (B, 128, C)
            expected = expected.permute(0, 2, 1).reshape(B * C, 128)  # (B*C, 128)
            actual = targets[:, t, :]  # (B*C, 128)
            assert torch.allclose(actual, expected), \
                f"Target mismatch at patch t={t}"

    def test_target_single_channel(self):
        """Target extraction with B=1, C=1 padded to C=4 should work."""
        x_norm = torch.randn(1, T_RAW, 4)
        targets, T_valid = compute_valid_targets(x_norm, W=16, forecast_len=128)
        assert targets.shape == (4, T_valid, 128)


# ---------------------------------------------------------------------------
# 6. MSE loss backward works (gradient flows to head params only)
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_backward_updates_head_params(self):
        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        x = torch.randn(B * C, T, H)
        out = head(x)  # (B*C, T, 128)
        # Use first 56 patches as valid
        pred = out[:, :56, :]
        target = torch.randn_like(pred)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()

        # All head parameters should have gradients
        for name, param in head.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_backbone_grads_frozen(self):
        """When backbone is frozen, only head params get gradients."""
        backbone = FakeBackbone(C=C, H=H, W_=W, T_raw=T_RAW)
        for p in backbone.parameters():
            p.requires_grad = False

        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)

        x = torch.randn(B, T_RAW, C)
        f_bc, x_norm = extract_forecaster_latents(backbone, x)
        out = head(f_bc)
        pred = out[:, :56, :]
        target = torch.randn_like(pred)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()

        # Head params should have gradients
        head_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in head.parameters())
        assert head_has_grad, "Head parameters should have gradients"

        # Backbone params should not
        for name, p in backbone.named_parameters():
            if p.requires_grad:
                assert p.grad is None or p.grad.abs().sum() == 0, \
                    f"Backbone param {name} should not have gradient"


# ---------------------------------------------------------------------------
# 7. Autoregressive rollout: correct shapes
# ---------------------------------------------------------------------------

class TestAutoregressiveRollout:
    def test_single_step_shape(self):
        """Rolling forward by exactly 128 steps (one head prediction)."""
        backbone = FakeBackbone(C=C, H=H, W_=W, T_raw=T_RAW)
        backbone.eval()
        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        head.eval()
        x_context = torch.randn(1, T_RAW, C)
        forecast = forecast_autoregressive(backbone, head, x_context, horizon=128, device='cpu')
        assert forecast.shape == (128, C)

    def test_multi_step_shape(self):
        """Rolling forward 256 steps requires 2 iterations."""
        backbone = FakeBackbone(C=C, H=H, W_=W, T_raw=T_RAW)
        backbone.eval()
        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        head.eval()
        x_context = torch.randn(1, T_RAW, C)
        forecast = forecast_autoregressive(backbone, head, x_context, horizon=256, device='cpu')
        assert forecast.shape == (256, C)

    def test_non_multiple_horizon_shape(self):
        """Horizon that isn't a multiple of 128: should still return exact shape."""
        backbone = FakeBackbone(C=C, H=H, W_=W, T_raw=T_RAW)
        backbone.eval()
        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        head.eval()
        x_context = torch.randn(1, T_RAW, C)
        forecast = forecast_autoregressive(backbone, head, x_context, horizon=200, device='cpu')
        assert forecast.shape == (200, C)

    def test_2d_context_input(self):
        """x_context as (T_context, C) without batch dim."""
        backbone = FakeBackbone(C=C, H=H, W_=W, T_raw=T_RAW)
        backbone.eval()
        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        head.eval()
        x_context = torch.randn(T_RAW, C)
        forecast = forecast_autoregressive(backbone, head, x_context, horizon=128, device='cpu')
        assert forecast.shape == (128, C)

    def test_output_is_numpy(self):
        """Return type should be a numpy array."""
        import numpy as np
        backbone = FakeBackbone(C=C, H=H, W_=W, T_raw=T_RAW)
        backbone.eval()
        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        head.eval()
        x_context = torch.randn(1, T_RAW, C)
        forecast = forecast_autoregressive(backbone, head, x_context, horizon=128, device='cpu')
        assert isinstance(forecast, np.ndarray)

    def test_output_finite(self):
        """Autoregressive output should contain no NaN or Inf."""
        import numpy as np
        backbone = FakeBackbone(C=C, H=H, W_=W, T_raw=T_RAW)
        backbone.eval()
        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        head.eval()
        x_context = torch.randn(1, T_RAW, C)
        forecast = forecast_autoregressive(backbone, head, x_context, horizon=128, device='cpu')
        assert np.all(np.isfinite(forecast)), "Forecast contains NaN or Inf"


# ---------------------------------------------------------------------------
# 8. Channel padding: C=1 input padded to C=4 for backbone inference
# ---------------------------------------------------------------------------

class TestChannelPadding:
    def test_c1_expand_to_c4(self):
        """A C=1 input should be expandable to C=4 for backbone inference."""
        backbone = FakeBackbone(C=4, H=H, W_=W, T_raw=T_RAW)
        backbone.eval()
        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        head.eval()

        x_c1 = torch.randn(1, T_RAW, 1)
        # Pad to C=4 by repeating
        x_c4 = x_c1.expand(-1, -1, 4)
        assert x_c4.shape == (1, T_RAW, 4)

        # Should be able to extract latents from padded input
        f_bc, x_norm = extract_forecaster_latents(backbone, x_c4)
        assert f_bc.shape == (4, T, H)

        # And run the head
        out = head(f_bc)
        assert out.shape == (4, T, 128)

    def test_c2_repeat_to_c4(self):
        """A C=2 input repeated to C=4."""
        x_c2 = torch.randn(1, T_RAW, 2)
        x_c4 = x_c2.repeat(1, 1, 2)
        assert x_c4.shape == (1, T_RAW, 4)

    def test_autoregressive_with_padded_channel(self):
        """Full autoregressive rollout with a C=1 series padded to C=4."""
        backbone = FakeBackbone(C=4, H=H, W_=W, T_raw=T_RAW)
        backbone.eval()
        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        head.eval()

        x_c1 = torch.randn(1, T_RAW, 1)
        x_c4 = x_c1.expand(-1, -1, 4).contiguous()
        forecast = forecast_autoregressive(backbone, head, x_c4, horizon=128, device='cpu')
        # Returns (horizon, C=4) -- caller picks channel 0 for the actual forecast
        assert forecast.shape == (128, 4)


# ---------------------------------------------------------------------------
# 9. extract_forecaster_latents shape and interface
# ---------------------------------------------------------------------------

class TestExtractForecasterLatents:
    def test_output_shapes(self):
        backbone = FakeBackbone(C=C, H=H, W_=W, T_raw=T_RAW)
        x = torch.randn(B, T_RAW, C)
        f_bc, x_norm = extract_forecaster_latents(backbone, x)
        assert f_bc.shape == (B * C, T, H)
        assert x_norm.shape == (B, T_RAW, C)

    def test_no_gradient_on_latents(self):
        """Latents should be detached (backbone is frozen)."""
        backbone = FakeBackbone(C=C, H=H, W_=W, T_raw=T_RAW)
        x = torch.randn(B, T_RAW, C)
        f_bc, x_norm = extract_forecaster_latents(backbone, x)
        assert not f_bc.requires_grad


# ---------------------------------------------------------------------------
# 10. Integration: full training-like forward pass
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_training_step(self):
        """End-to-end: fake backbone -> extract latents -> head -> loss -> backward."""
        backbone = FakeBackbone(C=C, H=H, W_=W, T_raw=T_RAW)
        for p in backbone.parameters():
            p.requires_grad = False

        head = ForecastingHead(H=H, hidden_dim=128, num_gru_layers=2, forecast_len=128)
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        optimizer.zero_grad()

        x = torch.randn(B, T_RAW, C)
        f_bc, x_norm = extract_forecaster_latents(backbone, x)
        targets, T_valid = compute_valid_targets(x_norm, W=W, forecast_len=128)

        preds = head(f_bc)[:, :T_valid, :]
        loss = torch.nn.functional.mse_loss(preds, targets)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)
