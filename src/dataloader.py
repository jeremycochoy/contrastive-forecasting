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
import threading
import queue

import numpy as np
import torch
from torch.utils.data import Dataset


T_RAW = 1024  # We use the first 1024 of the 1025-point windows


def _forward_fill_nan(arr: np.ndarray) -> bool:
    """Replace NaN values in-place: forward-fill, then backfill any leading NaN.

    Equivalent to a naive forecast for missing observations.

    Returns True if the array was successfully cleaned, False if the
    array is all-NaN (no valid values to fill from) and should be skipped.
    """
    mask = np.isnan(arr)
    if not mask.any():
        return True
    # Forward-fill: each NaN gets the previous valid value
    idx = np.arange(len(arr))
    valid = ~mask
    if not valid.any():
        return False  # all-NaN: caller should skip this window
    # Set NaN positions to 0 so maximum.accumulate propagates the last valid index
    idx[mask] = 0
    np.maximum.accumulate(idx, out=idx)
    arr[mask] = arr[idx[mask]]
    # Backfill any remaining leading NaN (before the first valid value)
    still_nan = np.isnan(arr)
    if still_nan.any():
        first_valid = np.argmax(~still_nan)
        arr[:first_valid] = arr[first_valid]
    # If NaN still remains after both passes, signal to skip
    return not np.isnan(arr).any()


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
            if not _forward_fill_nan(a):
                continue  # skip all-NaN rows
            data.append(a)
        self._cached_data = data
        self._cached_shard_idx = shard_idx


# ── Prefetch iterator (background thread) ────────────────────────────────────

class PrefetchIterator:
    """Wraps an iterable and prefetches items in a background thread.

    This hides the latency of data loading (network I/O for HF streaming,
    disk I/O for local shards) by preparing the next batch while the GPU
    is busy with forward/backward passes.

    Args:
        iterable: Any iterable (e.g. HFStreamingLoader).
        prefetch: Number of items to buffer ahead. Default 2 is enough
            to hide one batch of I/O latency.
    """

    def __init__(self, iterable, prefetch: int = 2):
        self._iterable = iterable
        self._prefetch = prefetch

    def __iter__(self):
        q = queue.Queue(maxsize=self._prefetch)
        _sentinel = object()

        def _producer():
            try:
                for item in self._iterable:
                    q.put(item)
            except Exception as e:
                q.put(e)
            finally:
                q.put(_sentinel)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is _sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        t.join()


# ── HuggingFace streaming loader ─────────────────────────────────────────────

class HFStreamingLoader:
    """Streams data from a HuggingFace dataset repo, yielding [B, T_raw, C] batches.

    Data is already pre-shuffled in the parquet shards, so we stream
    sequentially. Each epoch re-starts the stream from the beginning.

    Uses a background prefetch thread to overlap data loading with training.

    Args:
        repo_id: HuggingFace dataset repo (e.g. "user/contrastive-training-tiny-bundles").
        path_in_repo: Subdirectory within the repo (e.g. "tiny_mixed_v1").
        batch_size: Number of C-channel samples per batch.
        C: Number of channels (independent series) per sample.
        split: Dataset split to use.
        prefetch: Number of batches to buffer ahead in background thread.
    """

    def __init__(self, repo_id: str, batch_size: int = 16, C: int = 4,
                 path_in_repo: str = None, split: str = "train",
                 prefetch: int = 2, skip_rows: int = 0):
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.C = C
        self.path_in_repo = path_in_repo
        self.split = split
        self.prefetch = prefetch
        self.skip_rows = skip_rows

    def _open_stream(self):
        from datasets import load_dataset

        split = self.split
        kwargs = dict(streaming=True)
        if self.path_in_repo:
            kwargs["data_dir"] = self.path_in_repo
            split = "train"  # data_dir subsets default to "train" split
        kwargs["split"] = split
        return load_dataset(self.repo_id, **kwargs)

    def _list_shard_files(self):
        """Return the ordered list of parquet shard files backing this stream.

        Returns None if the structure is not a simple list of parquet shards
        (in which case we fall back to the naive .skip()).
        """
        from huggingface_hub import HfFileSystem
        try:
            fs = HfFileSystem()
            pattern = f"datasets/{self.repo_id}"
            if self.path_in_repo:
                pattern = f"{pattern}/{self.path_in_repo}"
            paths = fs.ls(pattern, detail=False)
            pq = sorted(p for p in paths if p.endswith(".parquet"))
            return pq if pq else None
        except Exception as e:
            print(f"  [dataloader] Could not list shards for fast skip: {e}")
            return None

    def _shard_row_counts(self, shard_paths):
        """Return (N,) int array of row counts per shard, via parquet metadata.

        Uses the parquet FOOTER only (no data read); typically milliseconds
        per shard via HfFileSystem's cached HEAD requests.
        """
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq
        fs = HfFileSystem()
        counts = []
        for p in shard_paths:
            with fs.open(p, "rb") as f:
                meta = pq.read_metadata(f)
                counts.append(meta.num_rows)
        return counts

    def _iter_stream_with_fast_skip(self):
        """Yield parquet rows starting from position ``self.skip_rows``.

        Strategy: download full shards up to the one that contains the target
        row (fast — one parquet file at a time via pyarrow), drop them, then
        stream the remainder. This is O(shards_before_target) + O(rows_in_target_shard)
        instead of the naive O(rows_before_target) which is 100x-1000x slower
        on a streaming HF Hub connection.

        Falls back to the default .skip() if shard metadata can't be introspected.
        """
        shards = self._list_shard_files()
        if shards is None:
            # Fallback: naive skip
            stream = self._open_stream()
            if self.skip_rows > 0:
                stream = stream.skip(self.skip_rows)
                print(f"  [dataloader] Skipped {self.skip_rows} rows (naive)")
            yield from stream
            return

        counts = self._shard_row_counts(shards)
        print(f"  [dataloader] {len(shards)} shards, "
              f"{sum(counts)} total rows, target skip {self.skip_rows}")

        # Find first shard where cumulative count > skip_rows.
        cum = 0
        start_shard = 0
        for i, c in enumerate(counts):
            if cum + c > self.skip_rows:
                start_shard = i
                break
            cum += c
        else:
            # All shards come before target — skip everything and yield nothing.
            print(f"  [dataloader] skip_rows={self.skip_rows} >= total rows; "
                  f"no data to yield")
            return
        within_shard_skip = self.skip_rows - cum

        print(f"  [dataloader] Fast-skip: starting at shard {start_shard}/"
              f"{len(shards)}, then skipping {within_shard_skip} rows within it")

        # Stream from start_shard onwards, row-by-row, dropping the initial
        # within-shard-skip rows.
        from datasets import load_dataset
        remaining = shards[start_shard:]
        # Point HF at just these files. data_files supports a list of http(s)
        # URLs / repo-relative paths.
        ds = load_dataset(
            "parquet",
            data_files=["hf://" + p for p in remaining],
            split="train",
            streaming=True,
        )
        dropped = 0
        for row in ds:
            if dropped < within_shard_skip:
                dropped += 1
                continue
            yield row

    def _raw_iter(self):
        """Yield batches without prefetching (used by PrefetchIterator)."""
        if self.skip_rows > 0:
            row_iter = self._iter_stream_with_fast_skip()
        else:
            row_iter = iter(self._open_stream())
        buf = []
        target = self.batch_size * self.C

        for row in row_iter:
            # Support both processed bundles ("series") and raw GiftEval ("target")
            series = row.get("series") or row.get("target")
            if series is None:
                continue
            full = np.array(series, dtype=np.float32)

            # Crop non-overlapping T_RAW windows from long series
            for start in range(0, len(full) - T_RAW + 1, T_RAW):
                window = full[start:start + T_RAW].copy()
                if not _forward_fill_nan(window):
                    continue  # skip all-NaN windows
                buf.append(window)

                if len(buf) == target:
                    yield self._flush(buf)
                    buf = []

        # Yield remainder if enough for at least one sample
        if len(buf) >= self.C:
            usable = (len(buf) // self.C) * self.C
            yield self._flush(buf[:usable])

    def __iter__(self):
        return iter(PrefetchIterator(self._raw_iter(), prefetch=self.prefetch))

    def _flush(self, buf: list[np.ndarray]) -> torch.Tensor:
        C = self.C
        B = len(buf) // C
        T = len(buf[0])
        # Stack into [B*C, T] array first, then reshape — avoids Python loops
        stacked = np.stack(buf)                          # [B*C, T]
        stacked = stacked.reshape(B, C, T)               # [B, C, T]
        stacked = stacked.transpose(0, 2, 1)             # [B, T, C]
        return torch.from_numpy(stacked.copy())


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
                          split: str = "train",
                          skip_rows: int = 0) -> HFStreamingLoader:
    """Create a streaming DataLoader from a HuggingFace dataset repo.

    Streams parquet shards on the fly — no full download required.
    Data is already pre-shuffled by the pipeline.

    Args:
        skip_rows: Number of rows to skip at the start of the stream.
            Used for resuming training from a checkpoint so the model
            doesn't re-see data it already trained on.

    Returns an iterable yielding tensors of shape [B, T_raw, C].
    """
    return HFStreamingLoader(
        repo_id=repo_id, batch_size=batch_size, C=C,
        path_in_repo=path_in_repo, split=split, skip_rows=skip_rows,
    )


# ── Mixed loader: HF stream + on-the-fly periodic synthesizer ────────────────

class MixedPeriodicLoader:
    """Half-real, half-synthetic batch stream.

    Each yielded batch is the row-concat of two sub-batches:

    - ``hf_bs`` samples from the underlying ``HFStreamingLoader`` (real data)
    - ``synth_bs`` samples from :func:`src.synthetic_periodic.generate_periodic_batch`
      (on-the-fly, pure numpy, ~1 ms at ``bs=12``)

    The final batch has shape ``[hf_bs + synth_bs, T_raw, C]``. For a 50/50
    split with effective batch ``B=24`` use ``hf_bs=synth_bs=12``.

    The synth generator threads a numpy ``Generator`` seeded from ``seed``
    so runs are reproducible; the generator state advances across batches.
    """

    def __init__(self, hf_loader: "HFStreamingLoader", synth_bs: int,
                 T_raw: int = 1024, C: int = 4,
                 seed: int | None = None,
                 emit_freq_ids: bool = False):
        self.hf_loader = hf_loader
        self.synth_bs = synth_bs
        self.T_raw = T_raw
        self.C = C
        self.emit_freq_ids = emit_freq_ids
        # Synth generator is persistent; each __iter__ seeds a fresh one.
        self._seed = seed

    def __iter__(self):
        # Import locally so dataloader.py doesn't force synthetic_periodic
        # into the test import graph.
        from src.synthetic_periodic import generate_periodic_batch

        rng = np.random.default_rng(self._seed)
        hf_iter = iter(self.hf_loader)

        while True:
            try:
                x_hf = next(hf_iter)                        # [hf_bs, T, C]
            except StopIteration:
                return
            hf_bs = x_hf.shape[0]
            if self.synth_bs > 0:
                if self.emit_freq_ids:
                    x_syn, freq_syn = generate_periodic_batch(
                        batch_size=self.synth_bs, T_raw=self.T_raw,
                        C=self.C, rng=rng, return_freq_ids=True,
                    )
                else:
                    x_syn = generate_periodic_batch(
                        batch_size=self.synth_bs, T_raw=self.T_raw,
                        C=self.C, rng=rng,
                    )
                x = torch.cat([x_hf, x_syn], dim=0)         # [B, T, C]
            else:
                x = x_hf
            if self.emit_freq_ids:
                # HF rows are tagged 0 = unknown (we don't thread real freq
                # metadata through base-bundles yet). Synth rows carry their
                # sampled freq id.
                freq_hf = torch.zeros(hf_bs, dtype=torch.long)
                if self.synth_bs > 0:
                    freq = torch.cat([freq_hf, freq_syn], dim=0)
                else:
                    freq = freq_hf
                yield x, freq
            else:
                yield x


def create_mixed_periodic_dataloader(
    repo_id: str, batch_size: int = 24, C: int = 4,
    mix_ratio: float = 0.5,
    path_in_repo: str = None, split: str = "train",
    skip_rows: int = 0, T_raw: int = 1024, seed: int | None = None,
    emit_freq_ids: bool = False,
) -> "MixedPeriodicLoader":
    """Create a 50/50 (or arbitrary) mix of real HF + on-the-fly periodic synth.

    The effective batch size (HF + synth rows) equals ``batch_size``. With
    ``mix_ratio=0.5`` and ``batch_size=24`` the HF loader yields ``bs=12``
    real rows per step and the synth adds 12 periodic rows; both halves are
    independent draws.

    Args:
        batch_size: Effective batch size (HF + synth combined).
        mix_ratio: Fraction of each batch drawn from the periodic synth.
            0.0 = pure HF (identical to ``create_hf_dataloader``).
            1.0 = pure synth (useful for synth-only smoke tests).
        seed: Seed for the periodic synth generator. Independent from HF
            stream ordering.
    """
    if not 0.0 <= mix_ratio <= 1.0:
        raise ValueError(f"mix_ratio must be in [0, 1], got {mix_ratio}")

    synth_bs = int(round(batch_size * mix_ratio))
    hf_bs = batch_size - synth_bs

    if mix_ratio == 0.0 and not emit_freq_ids:
        # Exact parity with the HF-only path — no synth overhead, and
        # the caller doesn't need freq_ids. When emit_freq_ids is True
        # we fall through to the MixedPeriodicLoader path with
        # synth_bs=0; that path returns (x, freq_ids) tuples (freq=0
        # for HF rows) which downstream training expects.
        return create_hf_dataloader(
            repo_id=repo_id, batch_size=batch_size, C=C,
            path_in_repo=path_in_repo, split=split, skip_rows=skip_rows,
        )

    hf_loader = HFStreamingLoader(
        repo_id=repo_id, batch_size=hf_bs, C=C,
        path_in_repo=path_in_repo, split=split, skip_rows=skip_rows,
    ) if hf_bs > 0 else None

    # Small helper to act as an "iterable-like" HF loader when mix_ratio=1.0.
    class _EmptyHFLoader:
        def __iter__(self):
            while True:
                yield torch.empty(0, T_raw, C, dtype=torch.float32)

    return MixedPeriodicLoader(
        hf_loader=hf_loader if hf_loader is not None else _EmptyHFLoader(),
        synth_bs=synth_bs, T_raw=T_raw, C=C, seed=seed,
        emit_freq_ids=emit_freq_ids,
    )
