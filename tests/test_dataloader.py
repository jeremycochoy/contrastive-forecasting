"""TDD tests for parquet shard data loader."""

import os
import tempfile
import shutil

import pytest
import numpy as np
import torch

from src.dataloader import ShardDataset, create_dataloader


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

    def test_shuffle(self, shard_dir):
        dl1 = create_dataloader(shard_dir, batch_size=4, C=1, shuffle=True)
        dl2 = create_dataloader(shard_dir, batch_size=4, C=1, shuffle=True)
        b1 = next(iter(dl1))
        b2 = next(iter(dl2))
        # With shuffling, two dataloaders should give different batches
        # (extremely unlikely to be equal by chance with 100 rows)
        assert not torch.allclose(b1, b2)

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
