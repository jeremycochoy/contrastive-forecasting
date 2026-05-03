"""TDD tests for parquet shard data loader."""

import os
import tempfile
import shutil
from unittest.mock import patch

import pytest
import numpy as np
import torch

from src.dataloader import (
    HFStreamingLoader,
    MixedPeriodicLoader,
    ShardDataset,
    create_dataloader,
)
from src.freq_embedding import (
    FREQ_NAME_TO_ID,
    NUM_FREQS,
    NUM_SEASONALITIES,
    SOURCE_ID_TO_LABELS,
    seasonality_to_id,
)


@pytest.fixture
def shard_dir():
    """Create a temp directory with two small parquet shards."""
    d = tempfile.mkdtemp()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pytest.skip("pyarrow not installed")

    schema = pa.schema([
        pa.field("series", pa.list_(pa.float32(), 1025)),
        pa.field("source_id", pa.uint8()),
        pa.field("meta", pa.string()),
    ])

    rng = np.random.default_rng(0)
    for shard_idx in range(2):
        rows = []
        for i in range(50):
            series = rng.standard_normal(1025).astype(np.float32)
            rows.append({"series": series.tolist(), "source_id": shard_idx,
                         "meta": f"test_{shard_idx}_{i}"})
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, os.path.join(d, f"shard_{shard_idx:05d}.parquet"),
                        compression="zstd")
    yield d
    shutil.rmtree(d)


class TestShardDataset:
    def test_len(self, shard_dir):
        ds = ShardDataset(shard_dir)
        assert len(ds) == 100  # 2 shards × 50 rows

    def test_getitem_shape(self, shard_dir):
        ds = ShardDataset(shard_dir)
        series = ds[0]
        assert isinstance(series, np.ndarray)
        assert series.shape == (1024,)
        assert series.dtype == np.float32

    def test_all_rows_accessible(self, shard_dir):
        ds = ShardDataset(shard_dir)
        for i in range(len(ds)):
            s = ds[i]
            assert s.shape == (1024,)

    def test_reproducible(self, shard_dir):
        ds = ShardDataset(shard_dir)
        a = ds[42]
        b = ds[42]
        np.testing.assert_array_equal(a, b)


class TestCreateDataloader:
    def test_batch_shape(self, shard_dir):
        dl = create_dataloader(shard_dir, batch_size=4, C=2)
        batch = next(iter(dl))
        # batch: [B, T_raw, C] where T_raw = 1024 (trimmed from 1025) and C=2
        assert batch.shape == (4, 1024, 2)
        assert batch.dtype == torch.float32

    def test_channels_independent(self, shard_dir):
        dl = create_dataloader(shard_dir, batch_size=2, C=4)
        batch = next(iter(dl))
        # Each channel should be a different series
        assert not torch.allclose(batch[0, :, 0], batch[0, :, 1])

    def test_in_order_deterministic(self, shard_dir):
        # Two identical loader instantiations must yield bytewise-identical
        # batches: the sampler is in-order (HF/parquet bundles are
        # pre-shuffled at upload, so re-shuffling on top is redundant and
        # breaks resume — see May 3 2026 #10-resume incident).
        dl1 = create_dataloader(shard_dir, batch_size=4, C=1)
        dl2 = create_dataloader(shard_dir, batch_size=4, C=1)
        b1 = next(iter(dl1))
        b2 = next(iter(dl2))
        assert torch.equal(b1, b2)

    def test_no_nan(self, shard_dir):
        dl = create_dataloader(shard_dir, batch_size=8, C=4)
        batch = next(iter(dl))
        assert not torch.isnan(batch).any()
        assert not torch.isinf(batch).any()

    def test_epoch_iteration(self, shard_dir):
        """Should be able to iterate through the full dataset."""
        dl = create_dataloader(shard_dir, batch_size=8, C=2)
        total_batches = 0
        for batch in dl:
            assert batch.shape[0] <= 8
            assert batch.shape[1] == 1024
            assert batch.shape[2] == 2
            total_batches += 1
        # 100 rows, C=2 channels per sample → 50 samples, bs=8 → ~6 batches
        assert total_batches > 0


# ── HFStreamingLoader: emit_source_ids ─────────────────────────────────────

def _fake_hf_rows(n_rows: int, T: int = 1024, source_id: int = 1, seed: int = 0):
    """Yield n_rows dicts mimicking the bundle's parquet schema."""
    rng = np.random.default_rng(seed)
    for i in range(n_rows):
        series = rng.standard_normal(T + 1).astype(np.float32).tolist()
        yield {"series": series, "source_id": source_id, "meta": f"row_{i}"}


class TestHFStreamingLoaderEmitSourceIds:
    def test_default_yields_tensor_only(self):
        """Existing emit_source_ids=False path must keep yielding bare tensors."""
        loader = HFStreamingLoader(
            repo_id="ignored", batch_size=2, C=2, prefetch=0)
        with patch.object(loader, "_open_stream",
                          return_value=list(_fake_hf_rows(8))):
            it = iter(loader)
            batch = next(it)
            assert isinstance(batch, torch.Tensor)
            assert batch.shape == (2, 1024, 2)

    def test_emit_source_ids_yields_pair(self):
        loader = HFStreamingLoader(
            repo_id="ignored", batch_size=2, C=2,
            emit_source_ids=True, prefetch=0)
        with patch.object(loader, "_open_stream",
                          return_value=list(_fake_hf_rows(8, source_id=1))):
            it = iter(loader)
            batch = next(it)
            assert isinstance(batch, tuple) and len(batch) == 2
            x, source_ids = batch
            assert x.shape == (2, 1024, 2)
            assert source_ids.shape == (2,)
            assert source_ids.dtype == torch.long
            assert (source_ids == 1).all()

    def test_emit_source_ids_carries_distinct_values(self):
        loader = HFStreamingLoader(
            repo_id="ignored", batch_size=2, C=1,
            emit_source_ids=True, prefetch=0)
        # 4 rows: source_ids 1, 2, 3, 4. With C=1, batch_size=2 ⇒ rows 0,1 then 2,3.
        rows = []
        for sid in (1, 2, 3, 4):
            rows.extend(_fake_hf_rows(1, source_id=sid))
        with patch.object(loader, "_open_stream", return_value=rows):
            it = iter(loader)
            x1, s1 = next(it)
            x2, s2 = next(it)
            assert s1.tolist() == [1, 2]
            assert s2.tolist() == [3, 4]


# ── MixedPeriodicLoader: dual-axis labels ──────────────────────────────────


class _FakeHFLoader:
    """Minimal stand-in for HFStreamingLoader that yields prepared tuples.

    Used to test MixedPeriodicLoader without depending on HF Hub.
    """
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)


class TestMixedPeriodicLoaderLabels:
    def test_yields_three_tuple_when_emit_freq_ids(self):
        # Two batches, each [2, 64, 1] tensor + source_ids [2]
        x = torch.zeros(2, 64, 1)
        sids = torch.tensor([1, 2], dtype=torch.long)
        hf_loader = _FakeHFLoader([(x, sids)])
        ml = MixedPeriodicLoader(
            hf_loader=hf_loader, synth_bs=0, T_raw=64, C=1,
            seed=0, emit_freq_ids=True)
        out = next(iter(ml))
        assert isinstance(out, tuple) and len(out) == 3
        xb, freq_ids, seas_ids = out
        assert xb.shape == (2, 64, 1)
        assert freq_ids.shape == (2,)
        assert seas_ids.shape == (2,)

    def test_hf_only_labels_from_source_id_table(self):
        # source_ids 1 (wiki_hourly) and 2 (wiki_daily) map to known labels.
        x = torch.zeros(2, 64, 1)
        sids = torch.tensor([1, 2], dtype=torch.long)
        hf_loader = _FakeHFLoader([(x, sids)])
        ml = MixedPeriodicLoader(
            hf_loader=hf_loader, synth_bs=0, T_raw=64, C=1,
            seed=0, emit_freq_ids=True)
        _, freq_ids, seas_ids = next(iter(ml))
        # wiki_hourly: freq=1h, seasonality=24
        assert freq_ids[0].item() == FREQ_NAME_TO_ID["1h"]
        assert seas_ids[0].item() == seasonality_to_id(24)
        # wiki_daily: freq=1d, seasonality=7
        assert freq_ids[1].item() == FREQ_NAME_TO_ID["1d"]
        assert seas_ids[1].item() == seasonality_to_id(7)

    def test_unknown_source_id_falls_back_to_zero(self):
        # source_id 0 (gift) and 6 (synth-bundle) are both (0, 0).
        x = torch.zeros(2, 64, 1)
        sids = torch.tensor([0, 6], dtype=torch.long)
        hf_loader = _FakeHFLoader([(x, sids)])
        ml = MixedPeriodicLoader(
            hf_loader=hf_loader, synth_bs=0, T_raw=64, C=1,
            seed=0, emit_freq_ids=True)
        _, freq_ids, seas_ids = next(iter(ml))
        assert freq_ids.tolist() == [0, 0]
        assert seas_ids.tolist() == [0, 0]

    def test_synth_only_labels_from_synth(self):
        """mix_ratio=1 → no HF rows; freq/seasonality come from synth labels."""
        # _EmptyHFLoader yields zero-row tensors.
        from src.dataloader import create_mixed_periodic_dataloader
        ml = create_mixed_periodic_dataloader(
            repo_id="ignored", batch_size=4, C=1, mix_ratio=1.0,
            T_raw=64, seed=42, emit_freq_ids=True)
        out = next(iter(ml))
        assert isinstance(out, tuple) and len(out) == 3
        x, freq_ids, seas_ids = out
        assert x.shape == (4, 64, 1)
        assert freq_ids.shape == (4,)
        assert seas_ids.shape == (4,)
        # Synth samples freq from {1..NUM_FREQS-1}; seasonality from spp.
        assert (freq_ids >= 1).all() and (freq_ids < NUM_FREQS).all()
        assert (seas_ids >= 0).all() and (seas_ids < NUM_SEASONALITIES).all()

    def test_mixed_concatenates_hf_then_synth(self):
        """With both HF and synth, labels are HF labels followed by synth labels."""
        hf_x = torch.ones(1, 64, 1)
        hf_sids = torch.tensor([1], dtype=torch.long)  # wiki_hourly
        hf_loader = _FakeHFLoader([(hf_x, hf_sids)] * 10)  # plenty of batches
        ml = MixedPeriodicLoader(
            hf_loader=hf_loader, synth_bs=2, T_raw=64, C=1,
            seed=42, emit_freq_ids=True)
        x, freq_ids, seas_ids = next(iter(ml))
        # HF row 0: wiki_hourly. Then 2 synth rows.
        assert freq_ids.shape == (3,)
        assert freq_ids[0].item() == FREQ_NAME_TO_ID["1h"]
        assert seas_ids[0].item() == seasonality_to_id(24)
        # Synth labels are not required to be specific — just non-zero.
        assert (freq_ids[1:] >= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
