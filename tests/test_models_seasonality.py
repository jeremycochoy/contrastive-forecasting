"""Smoke tests for the seasonality embedding integration in ConfigurableModel."""

import pytest
import torch

from src.models import ConfigurableModel
from src.freq_embedding import (
    FREQ_NAME_TO_ID,
    NUM_FREQS,
    NUM_SEASONALITIES,
    seasonality_to_id,
)


B, T_RAW, C, H, W = 2, 64, 2, 64, 16


def _model(freq_emb_dim=0, seasonality_emb_dim=0, **kw):
    return ConfigurableModel(
        C=C, H=H, W=W,
        encoder_type='mlp',
        num_layers=2, nhead=2, ffn_mult=1, dropout=0.0,
        rev_norm_kind='none', rev_norm_span=None,
        patch_stats_kind='none',
        freq_emb_dim=freq_emb_dim, num_freqs=NUM_FREQS,
        seasonality_emb_dim=seasonality_emb_dim,
        num_seasonalities=NUM_SEASONALITIES,
        **kw,
    )


class TestArchitecturalToggle:
    def test_no_seasonality_emb_means_no_module(self):
        m = _model(freq_emb_dim=3, seasonality_emb_dim=0)
        assert m.seasonality_embedding is None
        # freq embedding is still configured
        assert m.freq_embedding is not None

    def test_with_seasonality_emb_module_present(self):
        m = _model(freq_emb_dim=3, seasonality_emb_dim=3)
        assert m.seasonality_embedding is not None

    def test_encoder_width_includes_both(self):
        m = _model(freq_emb_dim=2, seasonality_emb_dim=4)
        # Encoder linear in_features = W + freq_emb_dim + seasonality_emb_dim
        # Encoder is an MLP — peek at its first linear layer.
        first = next(p for p in m.encoder.parameters() if p.dim() == 2)
        assert first.shape[1] == W + 2 + 4

    def test_backward_compat_no_emb(self):
        m = _model(freq_emb_dim=0, seasonality_emb_dim=0)
        assert m.freq_embedding is None
        assert m.seasonality_embedding is None
        first = next(p for p in m.encoder.parameters() if p.dim() == 2)
        assert first.shape[1] == W


class TestForwardPass:
    def test_forward_no_emb_runs(self):
        m = _model(freq_emb_dim=0, seasonality_emb_dim=0)
        x = torch.randn(B, T_RAW, C)
        y, y_orig = m(x)
        assert torch.isfinite(y).all()
        assert torch.isfinite(y_orig).all()

    def test_forward_with_freq_only(self):
        m = _model(freq_emb_dim=3, seasonality_emb_dim=0)
        x = torch.randn(B, T_RAW, C)
        freq_ids = torch.tensor([1, 5], dtype=torch.long)
        y, _ = m(x, freq_ids=freq_ids)
        assert torch.isfinite(y).all()

    def test_forward_with_both_labels(self):
        m = _model(freq_emb_dim=3, seasonality_emb_dim=3)
        x = torch.randn(B, T_RAW, C)
        freq_ids = torch.tensor([FREQ_NAME_TO_ID["1h"], FREQ_NAME_TO_ID["1d"]],
                                dtype=torch.long)
        seas_ids = torch.tensor(
            [seasonality_to_id(24), seasonality_to_id(7)], dtype=torch.long)
        y, _ = m(x, freq_ids=freq_ids, seasonality_ids=seas_ids)
        assert torch.isfinite(y).all()
        assert y.shape == (B, T_RAW // W, C, H)

    def test_seasonality_changes_output(self):
        """Different seasonality_ids should produce different outputs (the
        embedding is non-zero and concatenated to patches)."""
        torch.manual_seed(0)
        m = _model(freq_emb_dim=0, seasonality_emb_dim=3)
        x = torch.randn(B, T_RAW, C)
        s_a = torch.tensor([1, 1], dtype=torch.long)
        s_b = torch.tensor([7, 7], dtype=torch.long)
        y_a, _ = m(x, seasonality_ids=s_a)
        y_b, _ = m(x, seasonality_ids=s_b)
        assert not torch.allclose(y_a, y_b)

    def test_missing_seasonality_when_required_raises(self):
        m = _model(freq_emb_dim=0, seasonality_emb_dim=3)
        x = torch.randn(B, T_RAW, C)
        with pytest.raises(ValueError, match="seasonality"):
            m(x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
