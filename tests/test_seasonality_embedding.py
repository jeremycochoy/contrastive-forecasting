"""Tests for the seasonality embedding and source_id label mapping.

The "freq" axis encodes the wall-clock sample rate (10s, 1min, ..., 1w).
The "seasonality" axis encodes the dominant period in samples
(1, 7, 24, 168, 288, ...) bucketed by powers of two.

For training-time data we look up `(freq_id, seasonality_id)` from a
static `SOURCE_ID_TO_LABELS` table built off the bundle's `source_id`
column. Synth rows carry their own (freq, seasonality) sampled jointly.
For gift_eval, the freq comes from the task's pandas freq string and
the seasonality from `gluonts.time_feature.get_seasonality(freq)`.
"""

import torch
import pytest

from src.freq_embedding import (
    FREQ_NAME_TO_ID,
    NUM_FREQS,
    NUM_SEASONALITIES,
    SEASONALITY_NAMES,
    SeasonalityEmbedding,
    SOURCE_ID_TO_LABELS,
    gluonts_freq_to_id,
    seasonality_to_id,
)


# ── Vocab constants ─────────────────────────────────────────────────────────

class TestSeasonalityVocab:
    def test_num_seasonalities_is_ten(self):
        assert NUM_SEASONALITIES == 10
        assert len(SEASONALITY_NAMES) == NUM_SEASONALITIES

    def test_unknown_is_zero(self):
        assert SEASONALITY_NAMES[0] == "unknown"


# ── seasonality_to_id ───────────────────────────────────────────────────────

class TestSeasonalityToId:
    def test_zero_is_unknown(self):
        assert seasonality_to_id(0) == 0
        assert seasonality_to_id(-5) == 0

    def test_one_is_no_info(self):
        # gluonts's default seasonality for daily/weekly freqs is 1; we treat
        # it as the "no information" bucket so the embedding gets the same
        # row as truly-unknown samples (gift train + bundle synth).
        assert seasonality_to_id(1) == 0

    def test_common_gift_eval_values(self):
        # 7→2 (≤8), 12→3 (≤16), 24→4 (≤32), and so on.
        assert seasonality_to_id(2) == 1
        assert seasonality_to_id(7) == 2
        assert seasonality_to_id(12) == 3
        assert seasonality_to_id(24) == 4
        assert seasonality_to_id(48) == 5
        assert seasonality_to_id(96) == 6
        assert seasonality_to_id(168) == 7
        assert seasonality_to_id(288) == 8
        assert seasonality_to_id(1024) == 9

    def test_monotonic(self):
        prev = seasonality_to_id(1)
        for spp in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 8640]:
            cur = seasonality_to_id(spp)
            assert cur >= prev, f"seasonality_to_id not monotonic at spp={spp}"
            prev = cur

    def test_in_range(self):
        for spp in [1, 7, 24, 168, 288, 8640, 100000]:
            assert 0 <= seasonality_to_id(spp) < NUM_SEASONALITIES


# ── gluonts_freq_to_id ──────────────────────────────────────────────────────

class TestGluontsFreqToId:
    def test_canonical_strings(self):
        assert gluonts_freq_to_id("10s") == FREQ_NAME_TO_ID["10s"]
        assert gluonts_freq_to_id("1min") == FREQ_NAME_TO_ID["1min"]
        assert gluonts_freq_to_id("5min") == FREQ_NAME_TO_ID["5min"]
        assert gluonts_freq_to_id("10min") == FREQ_NAME_TO_ID["10min"]
        assert gluonts_freq_to_id("15min") == FREQ_NAME_TO_ID["15min"]
        assert gluonts_freq_to_id("30min") == FREQ_NAME_TO_ID["30min"]
        assert gluonts_freq_to_id("1h") == FREQ_NAME_TO_ID["1h"]
        assert gluonts_freq_to_id("1d") == FREQ_NAME_TO_ID["1d"]
        assert gluonts_freq_to_id("1w") == FREQ_NAME_TO_ID["1w"]

    def test_pandas_aliases(self):
        # gluonts/pandas use single-char shortcuts (T=min, H=hour, D=day, W=week).
        assert gluonts_freq_to_id("H") == FREQ_NAME_TO_ID["1h"]
        assert gluonts_freq_to_id("1H") == FREQ_NAME_TO_ID["1h"]
        assert gluonts_freq_to_id("D") == FREQ_NAME_TO_ID["1d"]
        assert gluonts_freq_to_id("1D") == FREQ_NAME_TO_ID["1d"]
        assert gluonts_freq_to_id("W") == FREQ_NAME_TO_ID["1w"]
        assert gluonts_freq_to_id("T") == FREQ_NAME_TO_ID["1min"]
        assert gluonts_freq_to_id("min") == FREQ_NAME_TO_ID["1min"]
        assert gluonts_freq_to_id("5T") == FREQ_NAME_TO_ID["5min"]
        assert gluonts_freq_to_id("15T") == FREQ_NAME_TO_ID["15min"]
        assert gluonts_freq_to_id("30T") == FREQ_NAME_TO_ID["30min"]

    def test_case_insensitive(self):
        assert gluonts_freq_to_id("h") == FREQ_NAME_TO_ID["1h"]
        assert gluonts_freq_to_id("d") == FREQ_NAME_TO_ID["1d"]

    def test_unknown_returns_zero(self):
        assert gluonts_freq_to_id("Y") == 0
        assert gluonts_freq_to_id("M") == 0
        assert gluonts_freq_to_id("nonsense") == 0
        assert gluonts_freq_to_id("") == 0
        assert gluonts_freq_to_id(None) == 0


# ── SOURCE_ID_TO_LABELS ─────────────────────────────────────────────────────

class TestSourceIdToLabels:
    def test_all_seven_source_ids_covered(self):
        # Bundle has source_ids 0..6 per rnd:training_data_prep/config.py.
        for sid in range(7):
            assert sid in SOURCE_ID_TO_LABELS

    def test_gift_train_is_unknown_unknown(self):
        # source_id 0 = gift; meta lost in pipeline → can't recover.
        assert SOURCE_ID_TO_LABELS[0] == (0, 0)

    def test_wiki_hourly(self):
        # source_id 1 = wiki_hourly: freq=1h, seasonality=24 (daily).
        freq_id, seas_id = SOURCE_ID_TO_LABELS[1]
        assert freq_id == FREQ_NAME_TO_ID["1h"]
        assert seas_id == seasonality_to_id(24)

    def test_wiki_daily(self):
        # source_id 2 = wiki_daily: freq=1d, seasonality=7 (weekly).
        freq_id, seas_id = SOURCE_ID_TO_LABELS[2]
        assert freq_id == FREQ_NAME_TO_ID["1d"]
        assert seas_id == seasonality_to_id(7)

    def test_wiki_stl_components(self):
        # source_ids 3,4,5: hourly freq, unknown seasonality (per user spec).
        for sid in (3, 4, 5):
            freq_id, seas_id = SOURCE_ID_TO_LABELS[sid]
            assert freq_id == FREQ_NAME_TO_ID["1h"], f"sid={sid}"
            assert seas_id == 0, f"sid={sid}"

    def test_synthetic_bundle_is_unknown(self):
        # source_id 6 = synthetic in the bundle (spp not preserved at write time).
        assert SOURCE_ID_TO_LABELS[6] == (0, 0)


# ── SeasonalityEmbedding module ─────────────────────────────────────────────

class TestSeasonalityEmbedding:
    def test_output_shape(self):
        emb = SeasonalityEmbedding(emb_dim=3)
        ids = torch.tensor([0, 1, 5, 9], dtype=torch.long)
        assert emb(ids).shape == (4, 3)

    def test_dtype(self):
        emb = SeasonalityEmbedding(emb_dim=4)
        assert emb(torch.tensor([3], dtype=torch.long)).dtype == torch.float32

    def test_finite(self):
        emb = SeasonalityEmbedding(emb_dim=3)
        ids = torch.arange(NUM_SEASONALITIES, dtype=torch.long)
        assert torch.isfinite(emb(ids)).all()

    def test_param_count_matches_freq(self):
        # Same vocab size and dim ⇒ same param count as FrequencyEmbedding.
        emb = SeasonalityEmbedding(emb_dim=3)
        n = sum(p.numel() for p in emb.parameters())
        assert n == NUM_SEASONALITIES * 3

    def test_mix_alpha_endpoints(self):
        emb = SeasonalityEmbedding(emb_dim=3)
        a = torch.tensor([2, 5], dtype=torch.long)
        b = torch.tensor([7, 1], dtype=torch.long)
        ones = torch.tensor([1.0, 1.0])
        zeros = torch.tensor([0.0, 0.0])
        assert torch.allclose(emb.mix(a, b, ones), emb(a))
        assert torch.allclose(emb.mix(a, b, zeros), emb(b))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
