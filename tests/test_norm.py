"""TDD tests for RevEWMNorm — reversible exponential weighted moving normalization."""

import pytest
import torch

from src.norm import RevEWMNorm


class TestRevEWMNormShape:
    def test_output_shape_preserved(self):
        norm = RevEWMNorm(num_features=4, span=300, patch_size=32)
        x = torch.randn(2, 128, 4)
        out = norm(x, mode='norm')
        assert out.shape == x.shape

    def test_denorm_shape_preserved(self):
        norm = RevEWMNorm(num_features=4, span=300, patch_size=32)
        x = torch.randn(2, 128, 4)
        normed = norm(x, mode='norm')
        out = norm(normed, mode='denorm')
        assert out.shape == x.shape


class TestRevEWMNormRoundtrip:
    def test_normalize_denormalize_recovers_input(self):
        """norm then denorm should recover the original input."""
        norm = RevEWMNorm(num_features=4, span=300, patch_size=32)
        x = torch.randn(2, 256, 4)
        normed = norm(x, mode='norm')
        recovered = norm(normed, mode='denorm')
        assert torch.allclose(x, recovered, atol=1e-5), \
            f"Max error: {(x - recovered).abs().max().item():.2e}"

    def test_roundtrip_with_affine(self):
        norm = RevEWMNorm(num_features=4, span=300, patch_size=32, affine=True)
        x = torch.randn(2, 256, 4)
        normed = norm(x, mode='norm')
        recovered = norm(normed, mode='denorm')
        assert torch.allclose(x, recovered, atol=1e-5), \
            f"Max error: {(x - recovered).abs().max().item():.2e}"


class TestRevEWMNormFirstPatch:
    def test_first_patch_near_zero_mean(self):
        """After normalization, the first patch should have near-zero mean
        because EMA is initialized from it."""
        norm = RevEWMNorm(num_features=1, span=300, patch_size=32)
        x = torch.randn(1, 256, 1) + 10.0  # shifted mean
        normed = norm(x, mode='norm')
        first_patch_mean = normed[:, :32, :].mean().item()
        assert abs(first_patch_mean) < 1.0, \
            f"First patch mean after norm should be near zero, got {first_patch_mean:.4f}"

    def test_no_initial_spike(self):
        """The first timestep shouldn't have an extreme value (no cold-start spike)."""
        norm = RevEWMNorm(num_features=1, span=300, patch_size=32)
        x = torch.randn(1, 256, 1) + 5.0
        normed = norm(x, mode='norm')
        # First value should be reasonable, not an outlier
        first_val = normed[0, 0, 0].abs().item()
        rest_std = normed[0, 1:, 0].std().item()
        assert first_val < 5 * rest_std + 1, \
            f"First value {first_val:.2f} is too large compared to rest std {rest_std:.2f}"


class TestRevEWMNormEMA:
    def test_constant_input_ema_converges(self):
        """For constant input, EMA mean should converge to that constant."""
        norm = RevEWMNorm(num_features=1, span=50, patch_size=32)
        x = torch.full((1, 512, 1), 3.0)
        normed = norm(x, mode='norm')
        # After normalization of a constant, output should be near zero
        assert normed.abs().max().item() < 0.1, \
            f"Constant input should normalize to ~0, got max {normed.abs().max().item():.4f}"

    def test_per_channel_independence(self):
        """Each channel should be normalized independently."""
        norm = RevEWMNorm(num_features=2, span=300, patch_size=32)
        x = torch.randn(1, 128, 2)
        x[:, :, 0] += 100  # shift channel 0 heavily
        normed = norm(x, mode='norm')
        # Channel 1 should not be affected by channel 0's offset
        norm_c1_only = RevEWMNorm(num_features=1, span=300, patch_size=32)
        normed_c1 = norm_c1_only(x[:, :, 1:2], mode='norm')
        assert torch.allclose(normed[:, :, 1:2], normed_c1, atol=1e-5)


class TestRevEWMNormSafety:
    def test_no_nan_or_inf(self):
        norm = RevEWMNorm(num_features=4, span=300, patch_size=32)
        x = torch.randn(2, 256, 4)
        normed = norm(x, mode='norm')
        assert not torch.isnan(normed).any()
        assert not torch.isinf(normed).any()

    def test_near_zero_input_no_nan(self):
        """Very small input values shouldn't cause NaN from division by ~0 std."""
        norm = RevEWMNorm(num_features=1, span=300, patch_size=32)
        x = torch.full((1, 128, 1), 1e-8)
        normed = norm(x, mode='norm')
        assert not torch.isnan(normed).any()
        assert not torch.isinf(normed).any()

    def test_denorm_before_norm_raises(self):
        """Denormalize without a prior normalize should raise."""
        norm = RevEWMNorm(num_features=4, span=300, patch_size=32)
        x = torch.randn(2, 128, 4)
        with pytest.raises(RuntimeError):
            norm(x, mode='denorm')

    def test_invalid_mode_raises(self):
        norm = RevEWMNorm(num_features=4, span=300, patch_size=32)
        x = torch.randn(2, 128, 4)
        with pytest.raises(ValueError):
            norm(x, mode='invalid')


class TestRevEWMNormAffine:
    def test_affine_has_parameters(self):
        norm = RevEWMNorm(num_features=4, span=300, patch_size=32, affine=True)
        param_names = [n for n, _ in norm.named_parameters()]
        assert 'affine_weight' in param_names
        assert 'affine_bias' in param_names

    def test_no_affine_has_no_parameters(self):
        norm = RevEWMNorm(num_features=4, span=300, patch_size=32, affine=False)
        assert len(list(norm.parameters())) == 0


class TestRevEWMNormGrad:
    def test_statistics_are_detached(self):
        """Mean and stdev should not carry gradient history."""
        norm = RevEWMNorm(num_features=2, span=300, patch_size=32)
        x = torch.randn(1, 128, 2, requires_grad=True)
        normed = norm(x, mode='norm')
        assert not norm.mean.requires_grad
        assert not norm.stdev.requires_grad


class TestRevEWMNormIntegration:
    def test_with_configurable_model(self):
        """RevEWMNorm should work end-to-end inside ConfigurableModel."""
        from src.models import ConfigurableModel
        model = ConfigurableModel(
            C=2, H=64, W=16, encoder_type='mlp', num_layers=1, nhead=2,
            ffn_mult=2, rev_norm_span=300)
        x = torch.randn(1, 64, 2)
        out, out_orig = model(x)
        assert out.shape == (1, 4, 2, 64)
        assert out_orig.shape == (1, 4, 2, 64)

    def test_full_train_step_arima_with_revnorm(self):
        """End-to-end: ARIMA data + RevEWMNorm model + one training step."""
        from types import SimpleNamespace
        from src.models import ConfigurableModel
        from src.arma import generate_arima_batch
        from src.loss import contrastive_latent_loss

        C, H, W = 2, 64, 16
        model = ConfigurableModel(
            C=C, H=H, W=W, encoder_type='gru', num_layers=1, nhead=2,
            ffn_mult=2, rev_norm_span=300)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Generate ARIMA(1, p, q) data
        x, _ = generate_arima_batch(batch_size=2, T_raw=64, C=C, seed=0, dimension=4)

        # Forward
        f_lat, o_lat = model(x)
        B, T, Cm, Hm = f_lat.shape
        f_lat = f_lat.permute(0, 1, 2, 3)  # already [B, T, C, H]
        o_lat = o_lat.permute(0, 1, 2, 3)

        spec = SimpleNamespace(train_configuration={
            'contrastive_divergence_temperature': 0.07,
            'contrastive_latent_noise': None,
            'loss_shape': 'cosine_similarity_batch_no_time_neg',
            'contrastive_latent_delay': 0,
        })
        loss = contrastive_latent_loss((f_lat, o_lat), validation=False, spec=spec)

        # Backward + step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)
