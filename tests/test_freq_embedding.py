"""Tests for src.freq_embedding."""

import torch
import pytest

from src.freq_embedding import (
    FrequencyEmbedding,
    NUM_FREQS,
    FREQ_NAMES,
    FREQ_NAME_TO_ID,
    SAMPLES_PER_DAY,
    spp_to_freq_id,
)


# ── Module-level constants ───────────────────────────────────────────────────

class TestConstants:
    def test_num_freqs_matches_names(self):
        assert len(FREQ_NAMES) == NUM_FREQS
        assert NUM_FREQS == 10

    def test_name_to_id_roundtrip(self):
        for i, name in enumerate(FREQ_NAMES):
            assert FREQ_NAME_TO_ID[name] == i

    def test_unknown_is_zero(self):
        """Class 0 must be 'unknown' so default-init rows have a clean fallback."""
        assert FREQ_NAMES[0] == "unknown"

    def test_samples_per_day_covers_real_freqs(self):
        """Every non-unknown class has an entry (except the w-weekly edge case)."""
        for i in range(1, NUM_FREQS):
            assert i in SAMPLES_PER_DAY


# ── FrequencyEmbedding module ────────────────────────────────────────────────

class TestForward:
    def test_output_shape(self):
        emb = FrequencyEmbedding(emb_dim=3)
        ids = torch.tensor([0, 1, 5, 9], dtype=torch.long)
        out = emb(ids)
        assert out.shape == (4, 3)

    def test_dtype(self):
        emb = FrequencyEmbedding(emb_dim=4)
        ids = torch.tensor([3], dtype=torch.long)
        assert emb(ids).dtype == torch.float32

    def test_finite(self):
        emb = FrequencyEmbedding(emb_dim=3)
        ids = torch.arange(NUM_FREQS, dtype=torch.long)
        assert torch.isfinite(emb(ids)).all()

    def test_different_classes_different_embeddings(self):
        emb = FrequencyEmbedding(emb_dim=4)
        ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        out = emb(ids)
        # All embeddings must be distinct (probability of collision on 5×4 floats
        # with std=0.02 is vanishingly small).
        for i in range(5):
            for j in range(i + 1, 5):
                assert not torch.allclose(out[i], out[j])

    def test_same_ids_give_same_embeddings(self):
        emb = FrequencyEmbedding(emb_dim=3)
        a = emb(torch.tensor([4], dtype=torch.long))
        b = emb(torch.tensor([4], dtype=torch.long))
        assert torch.allclose(a, b)

    def test_param_count(self):
        """10 classes × 3 dims = 30 parameters. Small as advertised."""
        emb = FrequencyEmbedding(emb_dim=3)
        n = sum(p.numel() for p in emb.parameters())
        assert n == 30

    def test_backprop(self):
        emb = FrequencyEmbedding(emb_dim=3)
        ids = torch.tensor([0, 5], dtype=torch.long)
        out = emb(ids)
        loss = out.pow(2).sum()
        loss.backward()
        assert emb.embedding.weight.grad is not None
        # Only the two rows referenced should have non-zero grad.
        grad = emb.embedding.weight.grad
        assert (grad[0].abs().sum() > 0).item()
        assert (grad[5].abs().sum() > 0).item()
        assert (grad[1].abs().sum() == 0).item()
        assert (grad[3].abs().sum() == 0).item()


# ── Mixup ────────────────────────────────────────────────────────────────────

class TestMix:
    def test_alpha_1_recovers_a(self):
        emb = FrequencyEmbedding(emb_dim=4)
        ids_a = torch.tensor([3, 5, 7], dtype=torch.long)
        ids_b = torch.tensor([0, 9, 2], dtype=torch.long)
        alpha = torch.tensor([1.0, 1.0, 1.0])
        out = emb.mix(ids_a, ids_b, alpha)
        assert torch.allclose(out, emb(ids_a))

    def test_alpha_0_recovers_b(self):
        emb = FrequencyEmbedding(emb_dim=4)
        ids_a = torch.tensor([3, 5, 7], dtype=torch.long)
        ids_b = torch.tensor([0, 9, 2], dtype=torch.long)
        alpha = torch.tensor([0.0, 0.0, 0.0])
        out = emb.mix(ids_a, ids_b, alpha)
        assert torch.allclose(out, emb(ids_b))

    def test_linear_midpoint(self):
        emb = FrequencyEmbedding(emb_dim=3)
        ids_a = torch.tensor([1, 2], dtype=torch.long)
        ids_b = torch.tensor([3, 4], dtype=torch.long)
        alpha = torch.tensor([0.5, 0.5])
        out = emb.mix(ids_a, ids_b, alpha)
        expected = 0.5 * emb(ids_a) + 0.5 * emb(ids_b)
        assert torch.allclose(out, expected)

    def test_per_sample_alpha(self):
        """Different alpha per sample should be honoured independently."""
        emb = FrequencyEmbedding(emb_dim=3)
        ids_a = torch.tensor([1, 1, 1], dtype=torch.long)
        ids_b = torch.tensor([5, 5, 5], dtype=torch.long)
        alpha = torch.tensor([0.2, 0.5, 0.8])
        out = emb.mix(ids_a, ids_b, alpha)
        a_emb = emb(ids_a[:1])[0]
        b_emb = emb(ids_b[:1])[0]
        for i, a_val in enumerate([0.2, 0.5, 0.8]):
            expected = a_val * a_emb + (1 - a_val) * b_emb
            assert torch.allclose(out[i], expected), f"mismatch at {i}"


# ── spp_to_freq_id heuristic ─────────────────────────────────────────────────

class TestSppBucketing:
    def test_returns_in_range(self):
        for spp in [8, 12, 16, 24, 48, 72, 128, 200, 256]:
            f = spp_to_freq_id(spp)
            assert 1 <= f <= 9

    def test_short_spp_gets_sub_daily_class(self):
        # spp=8 is the shortest period — should map to class 1 (10s).
        assert spp_to_freq_id(8) == 1

    def test_long_spp_gets_weekly_class(self):
        assert spp_to_freq_id(250) == 9

    def test_monotonicity_rough(self):
        """Not strict monotonicity, but the trend should hold across buckets."""
        buckets = [(8, 1), (16, 4), (30, 6), (72, 7), (128, 8), (250, 9)]
        for spp, expected in buckets:
            assert spp_to_freq_id(spp) == expected, \
                f"spp={spp} mapped to {spp_to_freq_id(spp)}, expected {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
