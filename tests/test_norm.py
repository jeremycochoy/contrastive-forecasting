"""TDD tests for RevEWMNorm — reversible exponential weighted moving normalization."""

import pytest
import torch

from src.norm import RevEWMNorm, compute_patch_stats, PATCH_STATS_DIM


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


class TestRevEWMNormFloat32Accuracy:
    """The fast path runs in float32 (vs the previous float64). Verify
    the EMA mean and stdev still match a float64 reference implementation
    within a tolerance that's well under any plausible model-noise floor.
    """

    def _f64_reference(self, x, span, patch_size):
        """Independent float64 implementation of the cumsum-EMA recipe.
        Used purely as a numerical reference; no caching tricks.
        """
        B, T, C = x.shape
        W = min(patch_size, T)
        alpha = 2.0 / (span + 1.0)
        x64 = x.to(torch.float64)
        first = x64[:, :W, :]
        m_init = first.mean(dim=1, keepdim=True)
        v_init = first.var(dim=1, keepdim=True, unbiased=False)
        arange = torch.arange(T, dtype=torch.float64)
        decay = (1.0 - alpha) ** arange
        decay_shift = decay * (1.0 - alpha)
        inv_decay = 1.0 / decay
        d, dsh, idc = decay.view(1, T, 1), decay_shift.view(1, T, 1), inv_decay.view(1, T, 1)
        cs = torch.cumsum(x64 * idc, dim=1)
        m = alpha * cs * d + dsh * m_init
        rsq = (x64 - m) ** 2
        cs2 = torch.cumsum(rsq * idc, dim=1)
        v = alpha * cs2 * d + dsh * v_init
        return m.float(), torch.sqrt(v.clamp(min=1e-12)).float()

    def test_matches_float64_reference_span_128(self):
        torch.manual_seed(0)
        x = torch.randn(8, 1024, 4) * 100  # realistic scale
        norm = RevEWMNorm(num_features=4, span=128, patch_size=16)
        norm(x, mode='norm')
        m_ref, s_ref = self._f64_reference(x, span=128, patch_size=16)
        assert (norm.mean - m_ref).abs().max().item() < 1e-3, \
            f"mean disagreement {(norm.mean - m_ref).abs().max():.2e}"
        assert (norm.stdev - s_ref).abs().max().item() < 1e-3, \
            f"stdev disagreement {(norm.stdev - s_ref).abs().max():.2e}"

    def test_matches_float64_reference_span_512(self):
        torch.manual_seed(1)
        x = torch.randn(8, 1024, 4) * 100
        norm = RevEWMNorm(num_features=4, span=512, patch_size=16)
        norm(x, mode='norm')
        m_ref, s_ref = self._f64_reference(x, span=512, patch_size=16)
        # span=512 has slightly larger float32 cumsum precision loss but
        # still well within model-noise floor.
        assert (norm.mean - m_ref).abs().max().item() < 5e-3
        assert (norm.stdev - s_ref).abs().max().item() < 5e-3

    def test_buffer_cache_consistent_across_calls(self):
        """Repeated calls with same shape should reuse the cache and
        give identical results to non-cached recomputation."""
        torch.manual_seed(2)
        x1 = torch.randn(4, 512, 2) * 10
        x2 = torch.randn(4, 512, 2) * 10
        norm = RevEWMNorm(num_features=2, span=128, patch_size=16)
        norm(x1, mode='norm')
        m1 = norm.mean.clone()
        norm(x2, mode='norm')         # second call hits the cache
        norm(x1, mode='norm')         # third call back on x1 — cache reuse
        m1_again = norm.mean
        assert torch.equal(m1, m1_again), "cached path produced different result"

    def test_dtype_preserved(self):
        """Output mean/stdev should match the input dtype (float32)."""
        x = torch.randn(2, 64, 2).float()
        norm = RevEWMNorm(num_features=2, span=64, patch_size=8)
        norm(x, mode='norm')
        assert norm.mean.dtype == torch.float32
        assert norm.stdev.dtype == torch.float32


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


class TestComputePatchStats:
    def test_returns_none_for_kind_none(self):
        mean = torch.randn(2, 64, 3)
        stdev = torch.ones_like(mean)
        out = compute_patch_stats(mean, stdev, W=16, kind='none')
        assert out is None

    def test_diff_shape(self):
        B, T, C, W = 2, 64, 3, 16
        mean = torch.randn(B, T, C)
        stdev = torch.ones(B, T, C) * 2.0
        out = compute_patch_stats(mean, stdev, W=W, kind='diff')
        assert out.shape == (B, T // W, C, PATCH_STATS_DIM)

    def test_raw_shape(self):
        B, T, C, W = 2, 64, 3, 16
        mean = torch.randn(B, T, C)
        stdev = torch.ones(B, T, C) * 2.0
        out = compute_patch_stats(mean, stdev, W=W, kind='raw')
        assert out.shape == (B, T // W, C, PATCH_STATS_DIM)

    def test_diff_first_patch_is_zero(self):
        """No previous patch ⇒ diffs at t=0 must be exactly zero (left-pad)."""
        mean = torch.randn(2, 64, 3)
        stdev = torch.rand(2, 64, 3) + 0.1
        out = compute_patch_stats(mean, stdev, W=16, kind='diff')
        assert torch.allclose(out[:, 0, :, :], torch.zeros_like(out[:, 0, :, :]))

    def test_diff_constant_input_is_zero(self):
        """If mean and std are constant in time, dmean and dlogstd are 0 everywhere."""
        B, T, C = 1, 64, 2
        mean = torch.full((B, T, C), 5.0)
        stdev = torch.full((B, T, C), 2.0)
        out = compute_patch_stats(mean, stdev, W=16, kind='diff')
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_diff_dmean_in_std_units(self):
        """dmean[t] = (mean_p[t] - mean_p[t-1]) / std_p[t-1]. Hand-checked case."""
        B, T, C, W = 1, 32, 1, 16
        # Patch 0 mean=0, std=2; Patch 1 mean=4, std=2.
        x_mean = torch.cat(
            [torch.zeros(B, W, C), torch.full((B, W, C), 4.0)], dim=1
        )
        x_std = torch.full((B, T, C), 2.0)
        out = compute_patch_stats(x_mean, x_std, W=W, kind='diff')
        # Expected dmean[1] = (4 - 0) / 2 = 2.0
        assert torch.allclose(out[0, 1, 0, 0], torch.tensor(2.0), atol=1e-5)
        # dlogstd[1] = log(2) - log(2) = 0
        assert torch.allclose(out[0, 1, 0, 1], torch.tensor(0.0), atol=1e-5)
        # First patch is zero
        assert torch.allclose(out[0, 0, :, :], torch.zeros(C, 2), atol=1e-6)

    def test_diff_dlogstd_log_ratio(self):
        """dlogstd[t] = log(std[t]) - log(std[t-1]) = log(std[t] / std[t-1])."""
        B, T, C, W = 1, 32, 1, 16
        x_mean = torch.zeros(B, T, C)
        # Patch 0 std=1; Patch 1 std=4 ⇒ log ratio = log 4 = 2*log 2.
        x_std = torch.cat(
            [torch.ones(B, W, C), torch.full((B, W, C), 4.0)], dim=1
        )
        out = compute_patch_stats(x_mean, x_std, W=W, kind='diff')
        expected = torch.log(torch.tensor(4.0))
        assert torch.allclose(out[0, 1, 0, 1], expected, atol=1e-5)

    def test_diff_no_nan_with_zero_std(self):
        """Division must use the eps clamp so std=0 doesn't blow up."""
        B, T, C = 1, 32, 1
        mean = torch.randn(B, T, C)
        stdev = torch.zeros(B, T, C)
        out = compute_patch_stats(mean, stdev, W=16, kind='diff')
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_invalid_kind_raises(self):
        mean = torch.randn(1, 32, 1)
        stdev = torch.ones_like(mean)
        with pytest.raises(ValueError):
            compute_patch_stats(mean, stdev, W=16, kind='bogus')

    def test_t_not_multiple_of_w_raises(self):
        mean = torch.randn(1, 33, 1)
        stdev = torch.ones_like(mean)
        with pytest.raises(ValueError):
            compute_patch_stats(mean, stdev, W=16, kind='diff')


class TestConfigurableModelWithPatchStats:
    """End-to-end: model with patch_stats_kind='diff' trains without crashing."""

    def test_forward_shape_with_patch_stats_diff(self):
        from src.models import ConfigurableModel
        model = ConfigurableModel(
            C=2, H=64, W=16, encoder_type='mlp', num_layers=1, nhead=2,
            ffn_mult=2, rev_norm_span=32, patch_stats_kind='diff')
        x = torch.randn(1, 64, 2)
        out, out_orig = model(x)
        # Output shape should be unchanged from baseline.
        assert out.shape == (1, 4, 2, 64)
        assert out_orig.shape == (1, 4, 2, 64)

    def test_encoder_input_dim_widens(self):
        """Encoder's first projection must accept W+2 inputs when stats are on."""
        from src.models import ConfigurableModel
        baseline = ConfigurableModel(
            C=2, H=64, W=16, encoder_type='mlp', num_layers=1, nhead=2,
            ffn_mult=2, rev_norm_span=32, patch_stats_kind='none')
        with_stats = ConfigurableModel(
            C=2, H=64, W=16, encoder_type='mlp', num_layers=1, nhead=2,
            ffn_mult=2, rev_norm_span=32, patch_stats_kind='diff')
        # MLPEncoder.linear1 takes W as in_features.
        assert baseline.encoder.linear1.in_features == 16
        assert with_stats.encoder.linear1.in_features == 16 + PATCH_STATS_DIM

    def test_encoder_input_dim_with_freq_emb_and_stats(self):
        """W + freq_emb_dim + 2 when both are on."""
        from src.models import ConfigurableModel
        m = ConfigurableModel(
            C=2, H=64, W=16, encoder_type='mlp', num_layers=1, nhead=2,
            ffn_mult=2, rev_norm_span=32,
            freq_emb_dim=3, patch_stats_kind='diff')
        assert m.encoder.linear1.in_features == 16 + 3 + 2

    def test_revin_with_patch_stats_rejected(self):
        """RevIN's stats are time-invariant ⇒ patch_stats is meaningless."""
        from src.models import ConfigurableModel
        with pytest.raises(ValueError):
            ConfigurableModel(
                C=2, H=64, W=16, encoder_type='mlp', num_layers=1,
                nhead=2, ffn_mult=2, rev_norm_kind='revin',
                rev_norm_span=32, patch_stats_kind='diff')

    def test_no_rev_norm_with_patch_stats_rejected(self):
        from src.models import ConfigurableModel
        with pytest.raises(ValueError):
            ConfigurableModel(
                C=2, H=64, W=16, encoder_type='mlp', num_layers=1,
                nhead=2, ffn_mult=2, rev_norm_span=None,
                patch_stats_kind='diff')

    def test_unknown_kind_rejected(self):
        from src.models import ConfigurableModel
        with pytest.raises(ValueError):
            ConfigurableModel(
                C=2, H=64, W=16, encoder_type='mlp', num_layers=1,
                nhead=2, ffn_mult=2, rev_norm_span=32,
                patch_stats_kind='bogus')

    def test_prepare_encoder_input_used_by_train_path(self):
        """Regression: train.py's forward_step bypassed prepare_encoder_input
        and reimplemented patching, which silently dropped the patch-stats
        concat. This test exercises a forward_step-equivalent call to make
        sure ANY future train-script pattern that takes (model, x) and
        calls model.transformer() picks up the stats."""
        from src.models import ConfigurableModel
        m = ConfigurableModel(
            C=2, H=64, W=16, encoder_type='gru', num_layers=1, nhead=2,
            ffn_mult=2, rev_norm_span=32, freq_emb_dim=3,
            patch_stats_kind='diff')
        x = torch.randn(2, 64, 2)
        # train.py-style: rev_norm then prepare_encoder_input.
        x_norm = m.rev_norm(x, mode='norm')
        freq_ids = torch.zeros(2, dtype=torch.long)
        xr = m.prepare_encoder_input(x_norm, freq_ids=freq_ids)
        # The encoder's input width MUST be W + freq + stats = 16 + 3 + 2 = 21.
        assert xr.shape[-1] == 21, (
            f"prepare_encoder_input should output width 21 (W=16 + freq_emb=3 "
            f"+ patch_stats=2), got {xr.shape[-1]}. If this fails, train.py's "
            f"forward_step is probably skipping the stats concat.")
        # Calling the transformer must succeed (encoder.skip is wired for 21).
        f_flat, o_flat = m.transformer(xr)
        assert f_flat.shape[0] == 2 * 2  # B*C
        assert f_flat.shape[2] == 64     # H

    def test_state_dict_roundtrip_with_patch_stats(self):
        """Save a patch_stats=diff backbone, reload it via state_dict — the
        encoder input dim is recoverable from the skip layer's in_features."""
        from src.models import ConfigurableModel
        m1 = ConfigurableModel(
            C=2, H=64, W=16, encoder_type='gru', num_layers=1, nhead=2,
            ffn_mult=2, rev_norm_span=32, freq_emb_dim=3,
            patch_stats_kind='diff')
        sd = m1.state_dict()
        # Auto-detect via the same arithmetic the train scripts use.
        skip_in = sd['encoder.skip.weight'].shape[1]
        freq_dim = sd['freq_embedding.embedding.weight'].shape[1]
        extra = skip_in - 16 - freq_dim
        assert extra == PATCH_STATS_DIM
        # Load into a fresh model with the detected config.
        m2 = ConfigurableModel(
            C=2, H=64, W=16, encoder_type='gru', num_layers=1, nhead=2,
            ffn_mult=2, rev_norm_span=32, freq_emb_dim=3,
            patch_stats_kind='diff')
        m2.load_state_dict(sd)

    def test_one_train_step_with_patch_stats(self):
        from types import SimpleNamespace
        from src.models import ConfigurableModel
        from src.arma import generate_arima_batch
        from src.loss import contrastive_latent_loss

        C, H, W = 2, 64, 16
        model = ConfigurableModel(
            C=C, H=H, W=W, encoder_type='gru', num_layers=1, nhead=2,
            ffn_mult=2, rev_norm_span=32, patch_stats_kind='diff')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        x, _ = generate_arima_batch(batch_size=2, T_raw=64, C=C, seed=0, dimension=4)
        f_lat, o_lat = model(x)
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
