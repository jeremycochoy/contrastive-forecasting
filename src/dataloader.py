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
        # Early exit (caller stops iterating, GeneratorExit) used to leave
        # the producer thread blocked on q.put(item) forever. Daemon=True
        # let the interpreter exit, but the dangling thread state at
        # finalization triggered "Fatal Python error: PyGILState_Release:
        # thread state must be current when releasing" and aborted the
        # process. Now we signal the producer via _stop and drain the queue
        # in a finally, so the thread always returns cleanly within ~0.5s
        # of consumer exit.
        q = queue.Queue(maxsize=self._prefetch)
        _sentinel = object()
        _stop = threading.Event()

        def _producer():
            try:
                for item in self._iterable:
                    # Block on queue space, but periodically check _stop so
                    # we can interrupt a full-queue put when the consumer
                    # has gone away.
                    while True:
                        if _stop.is_set():
                            return
                        try:
                            q.put(item, timeout=0.5)
                            break
                        except queue.Full:
                            continue
            except Exception as e:
                if not _stop.is_set():
                    q.put(e)
            finally:
                # Always signal end-of-stream — consumer's q.get() must
                # unblock even if we're exiting via _stop. timeout=1.0
                # handles the case where consumer is gone and queue is full.
                try:
                    q.put(_sentinel, timeout=1.0)
                except queue.Full:
                    pass

        t = threading.Thread(target=_producer, daemon=True)
        t.start()

        try:
            while True:
                item = q.get()
                if item is _sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # Tell producer to stop, drain queue so any pending q.put can
            # unblock immediately, then join. Without this, an early-exit
            # consumer (GeneratorExit, downstream StopIteration in the
            # training loop) leaves the producer hanging on q.put and the
            # daemon thread leaks into interpreter shutdown.
            _stop.set()
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
            t.join(timeout=5.0)


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
        emit_source_ids: If True, yield ``(tensor, source_ids)`` tuples
            where ``source_ids`` is a LongTensor of shape ``[B]`` carrying
            the source_id of the *first* channel of each multi-channel
            sample. Used by :class:`MixedPeriodicLoader` to look up
            ``(freq_id, seasonality_id)`` from ``SOURCE_ID_TO_LABELS``.
            Falls back to 0 (unknown) if a row's source_id column is
            missing — keeps the loader compatible with HF datasets that
            don't expose the column (raw GiftEval).
    """

    def __init__(self, repo_id: str, batch_size: int = 16, C: int = 4,
                 path_in_repo: str = None, split: str = "train",
                 prefetch: int = 2, skip_rows: int = 0,
                 emit_source_ids: bool = False):
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.C = C
        self.path_in_repo = path_in_repo
        self.split = split
        self.prefetch = prefetch
        self.skip_rows = skip_rows
        self.emit_source_ids = emit_source_ids

    def _open_stream(self):
        """Open an iterator over rows of the dataset (skip_rows=0).

        Bypasses ``datasets.load_dataset`` for the common shard-list case,
        because ``load_dataset`` resolves per-file metadata for every shard
        (4274 shards in `gift-pretrain-full-4096`) through a 64-thread pool,
        each call hitting `/api/datasets/<id>/revision/<sha>`. That endpoint
        is intermittently slow (10s default timeout) and 500-prone under load
        — it brought down the τ-sweep arm-2 launch on May 8 2026 even while
        the same dataset was reachable via single curl calls.

        Strategy: list shards via ``HfFileSystem.ls`` (one HTTP call → 4274
        paths in ~2 s) and stream each parquet shard via pyarrow. Only the
        true streaming path (one shard at a time) is exercised, so the
        thundering-herd metadata resolve never happens.

        Falls back to ``load_dataset`` if shard listing fails (e.g. nested
        layouts the simple pattern can't resolve) so behaviour for non-flat
        repos is unchanged.
        """
        shards = self._list_shard_files()
        if shards is None:
            # Fallback: original load_dataset path. Used for repos whose
            # parquet layout isn't a flat list under path_in_repo (e.g.
            # ``contrastive-training-tiny-bundles`` which the unit tests use).
            from datasets import load_dataset

            split = self.split
            kwargs = dict(streaming=True)
            if self.path_in_repo:
                kwargs["data_dir"] = self.path_in_repo
                split = "train"  # data_dir subsets default to "train" split
            kwargs["split"] = split
            return iter(load_dataset(self.repo_id, **kwargs))
        return self._pyarrow_stream_from_shards(shards, within_shard_skip=0)

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

    # Row batch size when reading parquet shards. 256 keeps memory bounded
    # (256 * 4096 floats * 4B = ~4 MB per batch) and matches the typical
    # training batch_size * C, so the prefetch ahead of the consumer is
    # naturally aligned.
    _PYARROW_BATCH_SIZE = 256

    def _pyarrow_stream_from_shards(self, shards, within_shard_skip: int = 0):
        """Yield dict rows from a list of parquet shards via pyarrow + HfFS.

        Each row is ``{"series": [float, ...], "source_id": int}`` matching
        the schema produced by ``datasets.load_dataset``. ``within_shard_skip``
        drops the first N rows of the FIRST shard only (used by the resume
        path); subsequent shards are fully consumed.

        Bypassing ``load_dataset`` here means we never trigger the
        per-file ``resolve_path`` thread pool against
        ``/api/datasets/.../revision/<sha>``.
        """
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq

        fs = HfFileSystem(token=self.token or None)
        for i, shard in enumerate(shards):
            with fs.open(shard, "rb") as f:
                pf = pq.ParquetFile(f)
                # Read only what we need; emit_source_ids gates the
                # source_id column. Always include "series" / fallback
                # "target" if present.
                cols = []
                names = pf.schema_arrow.names
                if "series" in names:
                    cols.append("series")
                elif "target" in names:
                    cols.append("target")
                else:
                    # Unknown schema — read all columns and let _raw_iter
                    # pick the right key.
                    cols = None
                if cols is not None and "source_id" in names:
                    cols.append("source_id")
                rows_seen = 0
                for batch in pf.iter_batches(
                    batch_size=self._PYARROW_BATCH_SIZE, columns=cols
                ):
                    for row in batch.to_pylist():
                        if i == 0 and rows_seen < within_shard_skip:
                            rows_seen += 1
                            continue
                        rows_seen += 1
                        yield row

    @property
    def token(self):
        """HF auth token from the standard env vars; None if not set."""
        import os
        return (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )

    def _shard_row_counts(self, shard_paths):
        """Return row counts per shard from the FIRST shard's metadata.

        Assumes uniform shard sizing: shards 0..N-2 share the same row count
        (read from shard 0's parquet footer); only the last shard may be
        shorter. This is true for our HF datasets which are written by the
        same upload pipeline.

        Cost: ONE parquet-footer fetch (~260ms) instead of N.
        """
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq
        fs = HfFileSystem()
        with fs.open(shard_paths[0], "rb") as f:
            rows_per_shard = pq.read_metadata(f).num_rows
        # All but last assumed equal; last gets the remainder. We only need
        # exact counts for the start-shard search, so leaving the last as
        # rows_per_shard is fine — we only ever skip INTO a shard, never past
        # the end of the dataset. If a future caller needs exact totals, they
        # can sum and trust manifest.json instead.
        return [rows_per_shard] * len(shard_paths)

    def _iter_stream_with_fast_skip(self, skip_rows: int = None):
        """Yield parquet rows starting from absolute position ``skip_rows``.

        Strategy: download full shards up to the one that contains the target
        row (fast — one parquet file at a time via pyarrow), drop them, then
        stream the remainder. This is O(shards_before_target) + O(rows_in_target_shard)
        instead of the naive O(rows_before_target) which is 100x-1000x slower
        on a streaming HF Hub connection.

        Falls back to the default .skip() if shard metadata can't be introspected.

        ``skip_rows`` defaults to ``self.skip_rows`` (the position from which the
        iterator originally opened). Callers may pass a larger value to resume
        further into the stream — used by ``_raw_iter`` to recover from a closed
        httpx client mid-iteration without reading already-consumed rows again.
        """
        if skip_rows is None:
            skip_rows = self.skip_rows
        shards = self._list_shard_files()
        if shards is None:
            # Fallback: naive skip
            stream = self._open_stream()
            if skip_rows > 0:
                stream = stream.skip(skip_rows)
                print(f"  [dataloader] Skipped {skip_rows} rows (naive)")
            yield from stream
            return

        counts = self._shard_row_counts(shards)
        total_rows = sum(counts)
        print(f"  [dataloader] {len(shards)} shards, "
              f"{total_rows} total rows, target skip {skip_rows}")

        # When resuming a long run on a small dataset, hf_rows_consumed can
        # exceed total dataset size (multi-epoch streaming). Wrap modulo so
        # we skip into the position within the next pseudo-epoch instead of
        # yielding nothing. Without this the resumed iterator empties on
        # cycle 1 and the training loop hits StopIteration.
        if total_rows > 0 and skip_rows >= total_rows:
            wrapped = skip_rows % total_rows
            print(f"  [dataloader] skip_rows={skip_rows} >= total "
                  f"({total_rows}); wrapping to {wrapped}")
            skip_rows = wrapped

        # Find first shard where cumulative count > skip_rows.
        cum = 0
        start_shard = 0
        for i, c in enumerate(counts):
            if cum + c > skip_rows:
                start_shard = i
                break
            cum += c
        else:
            # All shards come before target — skip everything and yield nothing.
            print(f"  [dataloader] skip_rows={skip_rows} >= total rows; "
                  f"no data to yield")
            return
        within_shard_skip = skip_rows - cum

        print(f"  [dataloader] Fast-skip: starting at shard {start_shard}/"
              f"{len(shards)}, then skipping {within_shard_skip} rows within it")

        # Stream from start_shard onwards, row-by-row, dropping the initial
        # within-shard-skip rows. Uses pyarrow + HfFileSystem directly so
        # we never hit the ``load_dataset`` per-file metadata-resolve path
        # that brought down the τ-sweep launch on 2026-05-08.
        remaining = shards[start_shard:]
        yield from self._pyarrow_stream_from_shards(
            remaining, within_shard_skip=within_shard_skip
        )

    # Substring matched against the message of any RuntimeError raised inside
    # the HF row-iteration. When httpx's internal client is closed mid-stream
    # (a known race when the connection pool / worker thread is GC'd partway
    # through a long run), the next read raises:
    #   RuntimeError: Cannot send a request, as the client has been closed.
    # The FRESH 167k-step training of #10 died this way at step ~52,400. The
    # underlying state on the wire isn't damaged — re-opening the stream from
    # the row we hadn't yet read recovers cleanly.
    _HF_CLIENT_CLOSED_MSG = "client has been closed"
    _HF_MAX_REOPENS = 5

    def _raw_iter(self):
        """Yield batches without prefetching (used by PrefetchIterator).

        Resilient to HF httpx-client closure mid-stream: we track the absolute
        row position and, on RuntimeError matching ``_HF_CLIENT_CLOSED_MSG``,
        rebuild the row iterator at the current position (no double-feeding
        of rows). Limited to ``_HF_MAX_REOPENS`` consecutive reopens to avoid
        spinning on a permanent fault. Counter resets on every successful
        row pull, so transient closures across hours don't accumulate.
        """
        rows_consumed = self.skip_rows

        def _open_at(start: int):
            if start > 0:
                return self._iter_stream_with_fast_skip(skip_rows=start)
            return iter(self._open_stream())

        row_iter = _open_at(rows_consumed)
        buf = []
        sids: list[int] = []  # parallel source_id per window in buf
        target = self.batch_size * self.C
        reopens = 0

        while True:
            try:
                row = next(row_iter)
            except StopIteration:
                break
            except RuntimeError as e:
                msg = str(e)
                if self._HF_CLIENT_CLOSED_MSG not in msg:
                    raise
                reopens += 1
                if reopens > self._HF_MAX_REOPENS:
                    print(f"  [dataloader] HF httpx client closure: exceeded "
                          f"{self._HF_MAX_REOPENS} consecutive reopens at row "
                          f"{rows_consumed}; giving up", flush=True)
                    raise
                print(f"  [dataloader] HF httpx client closed at row "
                      f"{rows_consumed} (reopen {reopens}/"
                      f"{self._HF_MAX_REOPENS}); rebuilding iterator from this "
                      f"position", flush=True)
                row_iter = _open_at(rows_consumed)
                continue
            reopens = 0
            rows_consumed += 1
            # Support both processed bundles ("series") and raw GiftEval ("target")
            series = row.get("series") or row.get("target")
            if series is None:
                continue
            full = np.array(series, dtype=np.float32)
            row_sid = int(row.get("source_id", 0)) if self.emit_source_ids else 0

            # Crop non-overlapping T_RAW windows from long series
            for start in range(0, len(full) - T_RAW + 1, T_RAW):
                window = full[start:start + T_RAW].copy()
                if not _forward_fill_nan(window):
                    continue  # skip all-NaN windows
                buf.append(window)
                if self.emit_source_ids:
                    sids.append(row_sid)

                if len(buf) == target:
                    yield self._flush(buf, sids)
                    buf = []
                    sids = []

        # Yield remainder if enough for at least one sample
        if len(buf) >= self.C:
            usable = (len(buf) // self.C) * self.C
            yield self._flush(buf[:usable], sids[:usable] if sids else [])

    def __iter__(self):
        return iter(PrefetchIterator(self._raw_iter(), prefetch=self.prefetch))

    def _flush(self, buf: list[np.ndarray],
               sids: list[int] | None = None) -> torch.Tensor:
        C = self.C
        B = len(buf) // C
        T = len(buf[0])
        # Stack into [B*C, T] array first, then reshape — avoids Python loops
        stacked = np.stack(buf)                          # [B*C, T]
        stacked = stacked.reshape(B, C, T)               # [B, C, T]
        stacked = stacked.transpose(0, 2, 1)             # [B, T, C]
        x = torch.from_numpy(stacked.copy())
        if not self.emit_source_ids:
            return x
        # Per-batch-row source_id: first channel of each row group sets the
        # label. Channels of one sample are independent series and can have
        # different source_ids in pre-shuffled bundles, so we tag each row
        # by its primary channel.
        if sids:
            row_sids = [sids[b * C] for b in range(B)]
            sids_t = torch.tensor(row_sids, dtype=torch.long)
        else:
            sids_t = torch.zeros(B, dtype=torch.long)
        return x, sids_t


# ── Batch collation for local shards ─────────────────────────────────────────

class _MultiChannelBatchSampler:
    """Yields index batches where each sample consists of C consecutive rows.

    Sample order is *deterministic and in-order*. The pretraining bundles
    are pre-shuffled at upload time (shard order randomized + samples
    within a shard randomized), so an in-order traversal already gives a
    stochastic-looking sequence. Removing the per-iter np.random.shuffle
    avoids non-determinism across resume — see the May 3 2026 #10-resume
    incident: a torch-RNG-restore bug bled into numpy-driven sampler order
    and caused per-batch loss std to jump +52% at the resume boundary.
    """

    def __init__(self, total_rows: int, C: int, batch_size: int,
                 drop_last: bool = False):
        self.C = C
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = total_rows // C
        self.indices = np.arange(self.num_samples)

    def __iter__(self):
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
                      shuffle: bool | None = None, num_workers: int = 0,
                      drop_last: bool = False) -> _ShardDataLoader:
    """Create a DataLoader from local parquet shards.

    Returns an iterable yielding tensors of shape [B, T_raw, C].

    The traversal is deterministic and in-order. The ``shuffle`` parameter
    is kept for backwards-compat with existing callers but is now a no-op:
    pretraining bundles are pre-shuffled at upload, so re-shuffling at the
    sampler level only adds RNG-driven non-determinism at resume.
    """
    if shuffle is not None:
        # Quiet acceptance — the pre-shuffled-storage assumption makes the
        # in-iter shuffle redundant and harmful (resume breaks RNG state).
        # See May 3 2026 #10-resume incident.
        pass
    dataset = ShardDataset(shard_dir)
    sampler = _MultiChannelBatchSampler(
        total_rows=len(dataset), C=C, batch_size=batch_size,
        drop_last=drop_last,
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
        from src.freq_embedding import SOURCE_ID_TO_LABELS

        rng = np.random.default_rng(self._seed)
        hf_iter = iter(self.hf_loader)

        # When emit_freq_ids is on, the hf_loader yields (x, source_ids)
        # tuples (HFStreamingLoader.emit_source_ids=True or _FakeHFLoader
        # in tests). Otherwise it yields bare tensors. Build a fast lookup
        # from source_id to (freq_id, seasonality_id) once per iter.
        max_sid = max(SOURCE_ID_TO_LABELS.keys())
        sid_to_freq = torch.zeros(max_sid + 1, dtype=torch.long)
        sid_to_seas = torch.zeros(max_sid + 1, dtype=torch.long)
        for sid, (fid, eid) in SOURCE_ID_TO_LABELS.items():
            sid_to_freq[sid] = fid
            sid_to_seas[sid] = eid

        while True:
            try:
                hf_batch = next(hf_iter)
            except StopIteration:
                return
            if self.emit_freq_ids and isinstance(hf_batch, tuple):
                x_hf, hf_source_ids = hf_batch
            else:
                x_hf = hf_batch
                hf_source_ids = None
            hf_bs = x_hf.shape[0]

            if self.synth_bs > 0:
                if self.emit_freq_ids:
                    x_syn, freq_syn, seas_syn = generate_periodic_batch(
                        batch_size=self.synth_bs, T_raw=self.T_raw,
                        C=self.C, rng=rng, return_labels=True,
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
                if hf_source_ids is not None:
                    # Out-of-vocab source_ids fall back to 0 (unknown).
                    safe_sids = hf_source_ids.clamp(min=0, max=max_sid)
                    freq_hf = sid_to_freq[safe_sids]
                    seas_hf = sid_to_seas[safe_sids]
                else:
                    # Backwards compat for callers passing a bare-tensor hf
                    # loader: treat HF rows as unknown (0, 0).
                    freq_hf = torch.zeros(hf_bs, dtype=torch.long)
                    seas_hf = torch.zeros(hf_bs, dtype=torch.long)
                if self.synth_bs > 0:
                    freq = torch.cat([freq_hf, freq_syn], dim=0)
                    seas = torch.cat([seas_hf, seas_syn], dim=0)
                else:
                    freq = freq_hf
                    seas = seas_hf
                yield x, freq, seas
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
        emit_source_ids=emit_freq_ids,
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


# ── Mixed loader: HF stream + on-the-fly composite (TimesFM-style) synth ─────


class MixedCompositeLoader:
    """Sibling of :class:`MixedPeriodicLoader` that uses the composite
    (TimesFM-style: trend + ARIMA + 2 free waves + 1 seas-tied wave)
    on-the-fly synthesizer instead of the clean-periodic one.

    Same row-concat semantics: ``hf_bs`` HF rows + ``synth_bs`` synth rows
    per yielded batch, optional ``(freq_id, seasonality_id)`` plumb. The
    label semantics differ from periodic: when the row's seasonality-tied
    wave is off, the synth emits ``seas_id=0`` (no period info) — see
    :func:`src.synthetic_composite.generate_composite_batch`.

    Extra knobs forwarded to the composite generator (defaults match the
    spec'd recipe):
      * ``p_arma``, ``p_free``, ``p_seas_tied``, ``p_integrate``,
        ``p_trend_mult``, ``p_env``: per-component coinflip probabilities
      * ``slope_std``, ``arma_dimension``, ``arma_target_std_range``,
        ``env_gain_range``, ``scale_range``, ``free_spp_range``: range params
    """

    def __init__(self, hf_loader: "HFStreamingLoader", synth_bs: int,
                 T_raw: int = 1024, C: int = 4,
                 seed: int | None = None,
                 emit_freq_ids: bool = False,
                 synth_kwargs: dict | None = None):
        self.hf_loader = hf_loader
        self.synth_bs = synth_bs
        self.T_raw = T_raw
        self.C = C
        self.emit_freq_ids = emit_freq_ids
        self._seed = seed
        self._synth_kwargs = dict(synth_kwargs or {})

    def __iter__(self):
        from src.synthetic_composite import generate_composite_batch
        from src.freq_embedding import SOURCE_ID_TO_LABELS

        rng = np.random.default_rng(self._seed)
        hf_iter = iter(self.hf_loader)

        max_sid = max(SOURCE_ID_TO_LABELS.keys())
        sid_to_freq = torch.zeros(max_sid + 1, dtype=torch.long)
        sid_to_seas = torch.zeros(max_sid + 1, dtype=torch.long)
        for sid, (fid, eid) in SOURCE_ID_TO_LABELS.items():
            sid_to_freq[sid] = fid
            sid_to_seas[sid] = eid

        while True:
            try:
                hf_batch = next(hf_iter)
            except StopIteration:
                return
            if self.emit_freq_ids and isinstance(hf_batch, tuple):
                x_hf, hf_source_ids = hf_batch
            else:
                x_hf = hf_batch
                hf_source_ids = None
            hf_bs = x_hf.shape[0]

            if self.synth_bs > 0:
                if self.emit_freq_ids:
                    x_syn, freq_syn, seas_syn = generate_composite_batch(
                        batch_size=self.synth_bs, T_raw=self.T_raw,
                        C=self.C, rng=rng, return_labels=True,
                        **self._synth_kwargs,
                    )
                else:
                    x_syn = generate_composite_batch(
                        batch_size=self.synth_bs, T_raw=self.T_raw,
                        C=self.C, rng=rng,
                        **self._synth_kwargs,
                    )
                x = torch.cat([x_hf, x_syn], dim=0)
            else:
                x = x_hf

            if self.emit_freq_ids:
                if hf_source_ids is not None:
                    safe_sids = hf_source_ids.clamp(min=0, max=max_sid)
                    freq_hf = sid_to_freq[safe_sids]
                    seas_hf = sid_to_seas[safe_sids]
                else:
                    freq_hf = torch.zeros(hf_bs, dtype=torch.long)
                    seas_hf = torch.zeros(hf_bs, dtype=torch.long)
                if self.synth_bs > 0:
                    freq = torch.cat([freq_hf, freq_syn], dim=0)
                    seas = torch.cat([seas_hf, seas_syn], dim=0)
                else:
                    freq = freq_hf
                    seas = seas_hf
                yield x, freq, seas
            else:
                yield x


def create_mixed_composite_dataloader(
    repo_id: str, batch_size: int = 24, C: int = 4,
    mix_ratio: float = 0.5,
    path_in_repo: str = None, split: str = "train",
    skip_rows: int = 0, T_raw: int = 1024, seed: int | None = None,
    emit_freq_ids: bool = False,
    synth_kwargs: dict | None = None,
) -> "MixedCompositeLoader":
    """Create an HF + on-the-fly *composite* synth mix, parallel to
    :func:`create_mixed_periodic_dataloader`.

    Same factory contract; ``synth_kwargs`` forwards extra recipe knobs to
    :func:`src.synthetic_composite.generate_composite_batch`.
    """
    if not 0.0 <= mix_ratio <= 1.0:
        raise ValueError(f"mix_ratio must be in [0, 1], got {mix_ratio}")

    synth_bs = int(round(batch_size * mix_ratio))
    hf_bs = batch_size - synth_bs

    if mix_ratio == 0.0 and not emit_freq_ids:
        return create_hf_dataloader(
            repo_id=repo_id, batch_size=batch_size, C=C,
            path_in_repo=path_in_repo, split=split, skip_rows=skip_rows,
        )

    hf_loader = HFStreamingLoader(
        repo_id=repo_id, batch_size=hf_bs, C=C,
        path_in_repo=path_in_repo, split=split, skip_rows=skip_rows,
        emit_source_ids=emit_freq_ids,
    ) if hf_bs > 0 else None

    class _EmptyHFLoader:
        def __iter__(self):
            while True:
                yield torch.empty(0, T_raw, C, dtype=torch.float32)

    return MixedCompositeLoader(
        hf_loader=hf_loader if hf_loader is not None else _EmptyHFLoader(),
        synth_bs=synth_bs, T_raw=T_raw, C=C, seed=seed,
        emit_freq_ids=emit_freq_ids, synth_kwargs=synth_kwargs,
    )


class MixedForkedArmaLoader:
    """HF + on-the-fly *forked-continuation ARIMA* synth mix (#318 follow-up),
    optionally with a third *regime-crossfade* stream (#325).

    Synth half comes from :func:`src.synthetic_forked_arma.generate_forked_arma_batch`,
    which fills ADJACENT synth rows (2k, 2k+1) with a forked pair: identical
    prefix, divergent perturbed-ARMA continuation. When ``cross_bs > 0`` a
    further block of :func:`src.synthetic_crossfade.generate_crossfade_batch`
    rows — each a monotone blend of two distinct *real* windows from the same
    step's HF sub-batch — is appended after the fork block. When
    ``cross_triplets > 0`` a final ADDITIVE block of explicit
    (A_norm, B_norm, C) triplets from
    :func:`src.synthetic_crossfade.generate_crossfade_triplets` (#328) is
    appended on top. The block order is
    ``[HF | forked-arma | crossfade | crossfade-triplets]``, so with
    ``cross_bs == 0`` and ``cross_triplets == 0`` the fork-only path (and its
    RNG draw order) is byte-identical to #318/#322.
    Both synthetic streams carry no canonical frequency/seasonality → label 0.
    """

    def __init__(self, hf_loader, synth_bs, T_raw=1024, C=4, seed=None,
                 emit_freq_ids=False, synth_kwargs=None, cross_bs=0,
                 cross_triplets=0):
        self.hf_loader = hf_loader
        self.synth_bs = synth_bs
        self.cross_bs = cross_bs
        self.cross_triplets = cross_triplets
        self.T_raw = T_raw
        self.C = C
        self.emit_freq_ids = emit_freq_ids
        self._seed = seed
        self._synth_kwargs = dict(synth_kwargs or {})

    def __iter__(self):
        from src.synthetic_forked_arma import generate_forked_arma_batch
        from src.synthetic_crossfade import (
            generate_crossfade_batch,
            generate_crossfade_triplets,
        )
        from src.freq_embedding import SOURCE_ID_TO_LABELS

        rng = np.random.default_rng(self._seed)
        hf_iter = iter(self.hf_loader)
        max_sid = max(SOURCE_ID_TO_LABELS.keys())
        sid_to_freq = torch.zeros(max_sid + 1, dtype=torch.long)
        sid_to_seas = torch.zeros(max_sid + 1, dtype=torch.long)
        for sid, (fid, eid) in SOURCE_ID_TO_LABELS.items():
            sid_to_freq[sid] = fid
            sid_to_seas[sid] = eid

        while True:
            try:
                hf_batch = next(hf_iter)
            except StopIteration:
                return
            if self.emit_freq_ids and isinstance(hf_batch, tuple):
                x_hf, hf_source_ids = hf_batch
            else:
                x_hf = hf_batch
                hf_source_ids = None
            hf_bs = x_hf.shape[0]

            # Block order is [HF | forked-arma | crossfade]; the fork block is
            # built first so cross_bs==0 leaves the fork-only RNG draws (and the
            # output) byte-identical to #318/#322.
            blocks = [x_hf]
            freq_blocks, seas_blocks = [], []
            # Match the synth length to the ACTUAL HF window length
            # (HFStreamingLoader crops to the module constant T_RAW=1024,
            # ignoring T_raw), so the real+synth cat never size-mismatches.
            synth_T = x_hf.shape[1] if hf_bs > 0 else self.T_raw

            if self.synth_bs > 0:
                if self.emit_freq_ids:
                    x_syn, freq_syn, seas_syn = generate_forked_arma_batch(
                        self.synth_bs, T_raw=synth_T, C=self.C, rng=rng,
                        return_labels=True, **self._synth_kwargs,
                    )
                    blocks.append(x_syn)
                    freq_blocks.append(freq_syn)
                    seas_blocks.append(seas_syn)
                else:
                    blocks.append(generate_forked_arma_batch(
                        self.synth_bs, T_raw=synth_T, C=self.C, rng=rng,
                        **self._synth_kwargs,
                    ))

            if self.cross_bs > 0:
                if hf_bs < 2:
                    raise ValueError(
                        f"crossfade needs >=2 real rows in the sub-batch, "
                        f"got hf_bs={hf_bs}")
                if self.emit_freq_ids:
                    x_cross, freq_cross, seas_cross = generate_crossfade_batch(
                        x_hf, self.cross_bs, rng=rng, return_labels=True,
                    )
                    blocks.append(x_cross)
                    freq_blocks.append(freq_cross)
                    seas_blocks.append(seas_cross)
                else:
                    blocks.append(generate_crossfade_batch(
                        x_hf, self.cross_bs, rng=rng,
                    ))

            # Explicit (A_norm, B_norm, C) crossfade triplets (#328), appended
            # last and ADDITIVE — they sit on top of the natural batch rather
            # than consuming HF rows. cross_triplets==0 leaves the order above
            # (and its RNG draws) untouched.
            if self.cross_triplets > 0:
                if hf_bs < 2:
                    raise ValueError(
                        f"crossfade triplets need >=2 real rows in the "
                        f"sub-batch, got hf_bs={hf_bs}")
                if self.emit_freq_ids:
                    x_trip, freq_trip, seas_trip = generate_crossfade_triplets(
                        x_hf, self.cross_triplets, rng=rng, return_labels=True,
                    )
                    blocks.append(x_trip)
                    freq_blocks.append(freq_trip)
                    seas_blocks.append(seas_trip)
                else:
                    blocks.append(generate_crossfade_triplets(
                        x_hf, self.cross_triplets, rng=rng,
                    ))

            x = torch.cat(blocks, dim=0) if len(blocks) > 1 else x_hf

            if self.emit_freq_ids:
                if hf_source_ids is not None:
                    safe_sids = hf_source_ids.clamp(min=0, max=max_sid)
                    freq_hf = sid_to_freq[safe_sids]
                    seas_hf = sid_to_seas[safe_sids]
                else:
                    freq_hf = torch.zeros(hf_bs, dtype=torch.long)
                    seas_hf = torch.zeros(hf_bs, dtype=torch.long)
                freq = torch.cat([freq_hf, *freq_blocks], dim=0) if freq_blocks else freq_hf
                seas = torch.cat([seas_hf, *seas_blocks], dim=0) if seas_blocks else seas_hf
                yield x, freq, seas
            else:
                yield x


def create_mixed_forked_arma_dataloader(
    repo_id: str, batch_size: int = 24, C: int = 4, mix_ratio: float = 0.5,
    path_in_repo: str = None, split: str = "train",
    skip_rows: int = 0, T_raw: int = 1024, seed: int | None = None,
    emit_freq_ids: bool = False, synth_kwargs: dict | None = None,
    crossfade_ratio: float = 0.0, cross_triplets: int = 0,
) -> "MixedForkedArmaLoader":
    """HF + forked-continuation-ARIMA synth mix, optionally plus a regime-
    crossfade stream (#325); same contract as the composite factory.

    The batch splits as ``[hf_bs | synth_bs | cross_bs]`` where
    ``synth_bs = round(batch_size * mix_ratio)`` (forked-arma) and
    ``cross_bs = round(batch_size * crossfade_ratio)`` (crossfade rows blended
    from the ``hf_bs`` real rows). ``cross_triplets`` additionally appends
    ``3 * cross_triplets`` (A_norm, B_norm, C) rows ON TOP (the total batch
    becomes ``batch_size + 3 * cross_triplets``; #328). With
    ``crossfade_ratio == 0`` and ``cross_triplets == 0`` this is the #318/#322
    fork-only loader unchanged. `synth_kwargs` forwards forked-ARMA knobs
    (integrate, perturb_sigma, fork_frac_range, std, dimension)."""
    if not 0.0 <= mix_ratio <= 1.0:
        raise ValueError(f"mix_ratio must be in [0, 1], got {mix_ratio}")
    if not 0.0 <= crossfade_ratio <= 1.0:
        raise ValueError(f"crossfade_ratio must be in [0, 1], got {crossfade_ratio}")
    if cross_triplets < 0:
        raise ValueError(f"cross_triplets must be >= 0, got {cross_triplets}")
    synth_bs = int(round(batch_size * mix_ratio))
    cross_bs = int(round(batch_size * crossfade_ratio))
    hf_bs = batch_size - synth_bs - cross_bs    # triplets are additive, not subtracted
    if hf_bs < 0:
        raise ValueError(
            f"mix_ratio + crossfade_ratio imply synth_bs+cross_bs="
            f"{synth_bs + cross_bs} > batch_size={batch_size}")
    if (cross_bs > 0 or cross_triplets > 0) and hf_bs < 2:
        raise ValueError(
            f"crossfade needs >=2 real rows but hf_bs={hf_bs} "
            f"(batch_size={batch_size}, mix_ratio={mix_ratio}, "
            f"crossfade_ratio={crossfade_ratio}, cross_triplets={cross_triplets})")
    if synth_bs == 0 and cross_bs == 0 and cross_triplets == 0 and not emit_freq_ids:
        return create_hf_dataloader(
            repo_id=repo_id, batch_size=batch_size, C=C,
            path_in_repo=path_in_repo, split=split, skip_rows=skip_rows,
        )
    hf_loader = HFStreamingLoader(
        repo_id=repo_id, batch_size=hf_bs, C=C,
        path_in_repo=path_in_repo, split=split, skip_rows=skip_rows,
        emit_source_ids=emit_freq_ids,
    ) if hf_bs > 0 else None

    class _EmptyHFLoader:
        def __iter__(self):
            while True:
                yield torch.empty(0, T_raw, C, dtype=torch.float32)

    return MixedForkedArmaLoader(
        hf_loader=hf_loader if hf_loader is not None else _EmptyHFLoader(),
        synth_bs=synth_bs, cross_bs=cross_bs, cross_triplets=cross_triplets,
        T_raw=T_raw, C=C, seed=seed,
        emit_freq_ids=emit_freq_ids, synth_kwargs=synth_kwargs,
    )
