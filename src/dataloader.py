"""
Parquet shard data loader for contrastive forecasting training.

Reads pre-shuffled parquet shards produced by the training data
preparation pipeline. Each shard row contains a fixed-length time
series window (1025 points; we use the first 1024).

Multiple rows are stacked to form the C independent channels per
training sample, matching the [B, T_raw, C] convention.
"""

import os
import math

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


T_RAW = 1024  # We use the first 1024 of the 1025-point windows


class ShardDataset(Dataset):
    """Memory-mapped dataset over a directory of parquet shards.

    Each __getitem__ returns a single 1-D float32 array of length T_RAW.
    The DataLoader collate function (via ``create_dataloader``) stacks
    C consecutive rows into multi-channel samples.
    """

    def __init__(self, shard_dir: str):
        import pyarrow.parquet as pq

        self.shard_dir = shard_dir
        shard_files = sorted(
            f for f in os.listdir(shard_dir) if f.endswith(".parquet")
        )
        if not shard_files:
            raise FileNotFoundError(
                f"No .parquet files found in {shard_dir}")

        # Build index: (file_path, row_offset_in_file, num_rows_in_file)
        self._index = []  # list of (path, num_rows)
        self._cumulative = [0]
        for fname in shard_files:
            path = os.path.join(shard_dir, fname)
            meta = pq.read_metadata(path)
            n = meta.num_rows
            self._index.append((path, n))
            self._cumulative.append(self._cumulative[-1] + n)

        self._total_rows = self._cumulative[-1]

        # Cache: keep the last loaded shard in memory
        self._cached_shard_idx = -1
        self._cached_data = None

    def __len__(self) -> int:
        return self._total_rows

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self._total_rows:
            raise IndexError(f"Index {idx} out of range [0, {self._total_rows})")

        # Binary search for the shard containing this index
        shard_idx = self._find_shard(idx)
        local_idx = idx - self._cumulative[shard_idx]

        # Load shard if not cached
        if shard_idx != self._cached_shard_idx:
            self._load_shard(shard_idx)

        return self._cached_data[local_idx]

    def _find_shard(self, idx: int) -> int:
        lo, hi = 0, len(self._index) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cumulative[mid + 1] <= idx:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _load_shard(self, shard_idx: int):
        import pyarrow.parquet as pq

        path, _ = self._index[shard_idx]
        table = pq.read_table(path, columns=["series"])
        # Each row is a list<float32>[1025]; take first T_RAW points
        series_col = table.column("series")
        data = []
        for row in series_col:
            arr = row.as_py()
            data.append(np.array(arr[:T_RAW], dtype=np.float32))
        self._cached_data = data
        self._cached_shard_idx = shard_idx


class _MultiChannelBatchSampler:
    """Yields index batches where each sample consists of C consecutive rows.

    This groups C rows into one multi-channel sample, then batches
    B such samples together.
    """

    def __init__(self, total_rows: int, C: int, batch_size: int,
                 shuffle: bool = True, drop_last: bool = False):
        self.C = C
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Number of complete C-channel samples
        self.num_samples = total_rows // C
        self.indices = np.arange(self.num_samples)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        batch = []
        for sample_idx in self.indices:
            # Each sample is C consecutive rows starting at sample_idx * C
            start = sample_idx * self.C
            batch.append(list(range(start, start + self.C)))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = self.num_samples
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)


def _collate_multichannel(batch_of_groups: list[list[np.ndarray]]) -> torch.Tensor:
    """Collate groups of C arrays into [B, T_raw, C] tensor."""
    B = len(batch_of_groups)
    C = len(batch_of_groups[0])
    T = len(batch_of_groups[0][0])
    out = torch.empty(B, T, C, dtype=torch.float32)
    for b, group in enumerate(batch_of_groups):
        for c, arr in enumerate(group):
            out[b, :, c] = torch.from_numpy(arr)
    return out


def create_dataloader(shard_dir: str, batch_size: int = 16, C: int = 4,
                      shuffle: bool = True, num_workers: int = 0,
                      drop_last: bool = False) -> DataLoader:
    """Create a DataLoader that yields [B, T_raw, C] tensors from parquet shards.

    Args:
        shard_dir: Directory containing parquet shard files.
        batch_size: Number of C-channel samples per batch.
        C: Number of channels (independent series) per sample.
        shuffle: Whether to shuffle sample order each epoch.
        num_workers: DataLoader workers (0 = main process).
        drop_last: Drop the last incomplete batch.

    Returns:
        DataLoader yielding tensors of shape [B, T_raw, C].
    """
    dataset = ShardDataset(shard_dir)
    sampler = _MultiChannelBatchSampler(
        total_rows=len(dataset), C=C, batch_size=batch_size,
        shuffle=shuffle, drop_last=drop_last,
    )

    # Wrap dataset + sampler into a simple iterable that yields [B, T, C]
    return _ShardDataLoader(dataset, sampler)


class _ShardDataLoader:
    """Minimal DataLoader that groups C rows into multi-channel samples."""

    def __init__(self, dataset: ShardDataset,
                 sampler: _MultiChannelBatchSampler):
        self.dataset = dataset
        self.sampler = sampler

    def __iter__(self):
        for batch_of_groups in self.sampler:
            groups = []
            for idx_group in batch_of_groups:
                groups.append([self.dataset[i] for i in idx_group])
            yield _collate_multichannel(groups)

    def __len__(self):
        return len(self.sampler)
