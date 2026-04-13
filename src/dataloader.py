"""
Data loader for contrastive forecasting training.

Supports two modes:
  1. Local parquet shards (from the data prep pipeline)
  2. HuggingFace streaming (streams shards without downloading the full dataset)

Each row contains a fixed-length time series window (1025 points; we use
the first 1024). Multiple rows are stacked to form the C independent
channels per training sample, matching the [B, T_raw, C] convention.
"""

import os
import math

import numpy as np
import torch
from torch.utils.data import Dataset


T_RAW = 1024  # We use the first 1024 of the 1025-point windows


# ── Local parquet shard loader ────────────────────────────────────────────────

class ShardDataset(Dataset):
    """Memory-mapped dataset over a directory of parquet shards.

    Each __getitem__ returns a single 1-D float32 array of length T_RAW.
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

        self._index = []  # list of (path, num_rows)
        self._cumulative = [0]
        for fname in shard_files:
            path = os.path.join(shard_dir, fname)
            meta = pq.read_metadata(path)
            n = meta.num_rows
            self._index.append((path, n))
            self._cumulative.append(self._cumulative[-1] + n)

        self._total_rows = self._cumulative[-1]
        self._cached_shard_idx = -1
        self._cached_data = None

    def __len__(self) -> int:
        return self._total_rows

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self._total_rows:
            raise IndexError(f"Index {idx} out of range [0, {self._total_rows})")

        shard_idx = self._find_shard(idx)
        local_idx = idx - self._cumulative[shard_idx]

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
        series_col = table.column("series")
        data = []
        for row in series_col:
            arr = row.as_py()
            a = np.array(arr[:T_RAW], dtype=np.float32)
            np.nan_to_num(a, copy=False, nan=0.0)
            data.append(a)
        self._cached_data = data
        self._cached_shard_idx = shard_idx


# ── HuggingFace streaming loader ─────────────────────────────────────────────

class HFStreamingLoader:
    """Streams data from a HuggingFace dataset repo, yielding [B, T_raw, C] batches.

    Data is already pre-shuffled in the parquet shards, so we stream
    sequentially. Each epoch re-starts the stream from the beginning.

    Args:
        repo_id: HuggingFace dataset repo (e.g. "user/contrastive-training-tiny-bundles").
        path_in_repo: Subdirectory within the repo (e.g. "tiny_mixed_v1").
        batch_size: Number of C-channel samples per batch.
        C: Number of channels (independent series) per sample.
        split: Dataset split to use.
    """

    def __init__(self, repo_id: str, batch_size: int = 16, C: int = 4,
                 path_in_repo: str = None, split: str = "train"):
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.C = C
        self.path_in_repo = path_in_repo
        self.split = split

    def _open_stream(self):
        from datasets import load_dataset

        kwargs = dict(streaming=True, split=self.split)
        if self.path_in_repo:
            kwargs["data_dir"] = self.path_in_repo
        return load_dataset(self.repo_id, **kwargs)

    def __iter__(self):
        stream = self._open_stream()
        buf = []

        for row in stream:
            series = row["series"]
            arr = np.array(series[:T_RAW], dtype=np.float32)
            # Replace NaN with 0 (some real-world series have missing values)
            np.nan_to_num(arr, copy=False, nan=0.0)
            buf.append(arr)

            # Once we have B*C rows, yield a batch
            if len(buf) == self.batch_size * self.C:
                yield self._flush(buf)
                buf = []

        # Yield remainder if enough for at least one sample
        if len(buf) >= self.C:
            # Trim to a multiple of C
            usable = (len(buf) // self.C) * self.C
            yield self._flush(buf[:usable])

    def _flush(self, buf: list[np.ndarray]) -> torch.Tensor:
        C = self.C
        B = len(buf) // C
        T = len(buf[0])
        out = torch.empty(B, T, C, dtype=torch.float32)
        for b in range(B):
            for c in range(C):
                out[b, :, c] = torch.from_numpy(buf[b * C + c])
        return out


# ── Batch collation for local shards ─────────────────────────────────────────

class _MultiChannelBatchSampler:
    """Yields index batches where each sample consists of C consecutive rows."""

    def __init__(self, total_rows: int, C: int, batch_size: int,
                 shuffle: bool = True, drop_last: bool = False):
        self.C = C
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_samples = total_rows // C
        self.indices = np.arange(self.num_samples)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        batch = []
        for sample_idx in self.indices:
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


# ── Public API ────────────────────────────────────────────────────────────────

def create_dataloader(shard_dir: str, batch_size: int = 16, C: int = 4,
                      shuffle: bool = True, num_workers: int = 0,
                      drop_last: bool = False) -> _ShardDataLoader:
    """Create a DataLoader from local parquet shards.

    Returns an iterable yielding tensors of shape [B, T_raw, C].
    """
    dataset = ShardDataset(shard_dir)
    sampler = _MultiChannelBatchSampler(
        total_rows=len(dataset), C=C, batch_size=batch_size,
        shuffle=shuffle, drop_last=drop_last,
    )
    return _ShardDataLoader(dataset, sampler)


def create_hf_dataloader(repo_id: str, batch_size: int = 16, C: int = 4,
                          path_in_repo: str = None,
                          split: str = "train") -> HFStreamingLoader:
    """Create a streaming DataLoader from a HuggingFace dataset repo.

    Streams parquet shards on the fly — no full download required.
    Data is already pre-shuffled by the pipeline.

    Returns an iterable yielding tensors of shape [B, T_raw, C].
    """
    return HFStreamingLoader(
        repo_id=repo_id, batch_size=batch_size, C=C,
        path_in_repo=path_in_repo, split=split,
    )
