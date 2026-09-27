"""
Stream `Salesforce/GiftEvalPretrain` as 1,024-value training windows (#419).

The corpus is 152 sources, 3.3M rows and 231B values in Arrow IPC stream
files (`<source>/data-*.arrow`). A row holds one series, or a multivariate
series (`target` of shape [D, L]) with optional covariates
(`past_feat_dynamic_real`). This module reaches every row of every source and
drops none. It never downloads the corpus: it reads record batches by HTTP
byte range, cuts random windows from them, and keeps a bounded set of record
batches in memory.

The sampler copies the pretraining sampler of Moirai 1.0 in uni2ts
(`cli/conf/pretrain/data/lotsa_v1_weighted.yaml`, `uni2ts/data/dataset.py`):

1. A source k is drawn with probability ceil(rows_k * weight_k) / sum, the
   length uni2ts gives a `TimeSeriesDataset` with `dataset_weight`.
2. A row is drawn uniformly (ERA5, CMIP6: uni2ts builds them `uniform`) or
   with probability proportional to its length (every other source).
3. Per field (target, covariates), n ~ U{1, ..., min(D, 128 * D / D_all)}
   of the field's D variates are drawn, as uni2ts
   `SampleDimension(max_dim=128)` does. Each gives one window, cut at the
   same random position, as uni2ts crops all variates of a sample alike.
   uni2ts reads covariates as context only. A univariate model has no such
   role, so here a covariate is one more series to train on.

A window is 1,024 values. A series shorter than that is padded with zeros on
the left, before its first value. So are the missing values before the first
observed one; later missing values are forward filled. A window with no
observed value is skipped. Each window carries the frequency id and the
seasonality id of its source's `freq`.

Memory and bandwidth stay bounded. A small source (at most
``RESIDENT_BYTES``) is read once and kept. A large one is cut into chunks, runs
of about ``CHUNK_BYTES`` of rows of one record batch, and keeps a few of them
(about ``POOL_BYTES``). A chunk is read with three byte ranges: the record
batch header, the offsets of its rows and their values. Each chunk serves
``BLOCK_USE`` of the windows its values hold, then a chunk drawn uniformly
from the source replaces it, while a background thread already fetches the
next. Drawing chunks uniformly and serving each in proportion to its values
keeps the row probabilities of steps 1 and 2 exact in expectation.

The byte offsets of every record batch come from
`scripts/build_gift_pretrain_index.py`, pinned to one commit of the dataset.
"""

import collections
import gzip
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from src import arrow_ipc

HF_REPO = "Salesforce/GiftEvalPretrain"
# uni2ts `SampleDimension(max_dim=128)` (cli/conf/pretrain/model/moirai_*.yaml).
MAX_DIM = 128
DEFAULT_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "gift_pretrain_index.json.gz")
FIELDS = ("target", "past_feat_dynamic_real")
# A source is read whole and kept when its files hold at most this.
RESIDENT_BYTES = 96 * 2 ** 20
# A large source is read in chunks of about this many bytes: a run of rows
# of one record batch. Some record batches hold 440 MB.
CHUNK_BYTES = 16 * 2 ** 20
# A large source keeps about this much of it in memory, plus one chunk
# fetched ahead.
POOL_BYTES = 48 * 2 ** 20
# A kept chunk serves this share of the windows its values hold
# (values / 1,024), then another one replaces it.
BLOCK_USE = 0.25
# One request reads a record batch header; a longer one reads it again.
HEADER_BYTES = 64 * 1024
# Windows mixed before they reach a batch, so that the variates of one draw
# and the rows of one record batch spread over many batches.
SHUFFLE_WINDOWS = 4096
# uni2ts builds the ERA5 and CMIP6 years with one weight each and uniform
# rows. One pool per family then draws every row exactly as uni2ts does.
FAMILY_PREFIXES = ("era5_", "cmip6_")


def _session_and_headers():
    from huggingface_hub.utils import build_hf_headers, get_session
    token = (os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    return get_session(), build_hf_headers(token=token)


def _local_range(root, path, start, end):
    with open(os.path.join(root, path), "rb") as f:
        f.seek(start)
        return f.read(end - start)


def fetch_range(path, start, end, revision=None, repo=HF_REPO, tries=8,
                root=None):
    """Bytes ``[start, end)`` of one file of the dataset, by HTTP range, or
    from a local copy of the repository under ``root``.

    Retries with a doubling wait on a network error or a non-206 answer
    (a 429 or a 5xx under load), then raises the last error.
    """
    if root is not None:
        return _local_range(root, path, start, end)
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(repo, path, repo_type="dataset", revision=revision)
    session, headers = _session_and_headers()
    rng = {"Range": f"bytes={start}-{end - 1}"}
    for attempt in range(tries):
        try:
            r = session.get(url, headers={**headers, **rng}, timeout=120)
            r.raise_for_status()  # "503 Server Error: ...", as hub_gate reads
            if r.status_code == 206 and len(r.content) == end - start:
                return r.content
            err = RuntimeError(f"range read of {path} [{start}, {end}) gave "
                               f"HTTP {r.status_code}, {len(r.content)} bytes")
        except Exception as e:  # network errors: retry the same range
            err = e
        time.sleep(min(60.0, 2.0 ** attempt))
    raise err


# ── The index ────────────────────────────────────────────────────────────────


def load_index(path=None):
    """The record-batch index `scripts/build_gift_pretrain_index.py` wrote."""
    with gzip.open(path or DEFAULT_INDEX, "rt") as f:
        return json.load(f)


def family_name(source):
    """The pool a source draws from: its ERA5 or CMIP6 family, or itself."""
    for prefix in FAMILY_PREFIXES:
        if source.startswith(prefix):
            return prefix[:-1]
    return source


def draw_slots(src):
    """uni2ts `TimeSeriesDataset.__len__`: ceil(num_ts * dataset_weight)."""
    return math.ceil(src["rows"] * src["weight"])


def _block_refs(name, src):
    """One reference per record batch of a source: where to read it."""
    files = src["files"]
    return [{"source": name, "path": files[b[0]][0],
             "schema_bytes": files[b[0]][1], "offset": b[1], "bytes": b[2],
             "rows": b[3], "values": b[4] + b[5]} for b in src["batches"]]


def build_families(index):
    """The sampling units: every source on its own, ERA5 and CMIP6 as one
    family each. Members of a family must agree on what uni2ts reads."""
    grouped = collections.defaultdict(list)
    for name in sorted(index["sources"]):
        grouped[family_name(name)].append(name)
    families = []
    for fam, members in sorted(grouped.items()):
        srcs = [index["sources"][m] for m in members]
        keys = {(s["freq"], s["uniform"], s["target_dim"], s["cov_dim"],
                 s["weight"]) for s in srcs}
        if len(keys) != 1:
            raise ValueError(f"family {fam} mixes {sorted(keys)}")
        families.append(_family(fam, members, srcs))
    return families


def _family(fam, members, srcs):
    first = srcs[0]
    refs = [r for m, s in zip(members, srcs) for r in _block_refs(m, s)]
    return {"name": fam, "members": members, "freq": first["freq"],
            "uniform": first["uniform"],
            "dims": (first["target_dim"], first["cov_dim"]),
            "slots": sum(draw_slots(s) for s in srcs), "refs": refs,
            "bytes": sum(r["bytes"] for r in refs)}


def family_probabilities(families):
    """Draw probability of every family, from the uni2ts slot counts."""
    slots = np.array([f["slots"] for f in families], dtype=np.float64)
    return slots / slots.sum()


# ── Record batches ───────────────────────────────────────────────────────────


def _field_arrays(column):
    """``(values, starts, ends)`` of a float column: every variate of every
    row is ``values[starts[r, j]:ends[r, j]]``."""
    import pyarrow as pa
    if pa.types.is_fixed_size_list(column.type):
        inner, d = column.values, column.type.list_size
    else:
        inner, d = column, 1
    # Raw offsets index the raw child array; IPC arrays are never sliced.
    offs = inner.offsets.to_numpy()
    values = inner.values.to_numpy(zero_copy_only=False)
    starts = offs[:-1].reshape(-1, d).astype(np.int64)
    ends = offs[1:].reshape(-1, d).astype(np.int64)
    return values.astype(np.float32, copy=False), starts, ends


class Block:
    """Rows of one record batch: per field (the target, then the covariates
    if any), ``(values, starts, ends)`` as :func:`_field_arrays` gives."""

    def __init__(self, fields, uniform, source):
        self.fields, self.source = fields, source
        _, starts, ends = self.fields[0]
        lengths = (ends[:, 0] - starts[:, 0]).astype(np.float64)
        self.rows = len(lengths)
        self.cdf = None if uniform else np.cumsum(lengths)
        self.weight = float(self.rows if uniform else self.cdf[-1])
        self.values = int(sum(len(v) for v, _, _ in self.fields))

    @classmethod
    def from_batch(cls, batch, uniform, source):
        """A Block of every row of a pyarrow record batch."""
        names = batch.schema.names
        return cls([_field_arrays(batch.column(n)) for n in FIELDS
                    if n in names], uniform, source)

    def pick_row(self, rng):
        """A row: uniform, or in proportion to its length."""
        if self.cdf is None:
            return int(rng.integers(self.rows))
        u = rng.random() * self.cdf[-1]
        return min(int(np.searchsorted(self.cdf, u, side="right")),
                   self.rows - 1)

    def series(self, field, row, variate):
        values, starts, ends = self.fields[field]
        return values[starts[row, variate]:ends[row, variate]]

    def length(self, row):
        _, starts, ends = self.fields[0]
        return int(ends[row, 0] - starts[row, 0])


def decode_batches(schema_buf, stream_buf):
    """Every record batch in ``stream_buf``, read with the schema message
    ``schema_buf`` of the file it comes from."""
    import pyarrow as pa
    schema = pa.ipc.read_schema(pa.py_buffer(schema_buf))
    reader = pa.ipc.MessageReader.open_stream(pa.py_buffer(stream_buf))
    out = []
    for message in reader:
        if message.type == "record batch":
            out.append(pa.ipc.read_record_batch(message, schema))
    return out


class BlockLoader:
    """Reads record batches of the index, by byte range or from ``root``:
    whole files of a small source, chunks of rows of a large one. Schemas
    and record batch headers are read once and cached."""

    def __init__(self, revision, root=None):
        self.revision, self.root = revision, root
        self._schemas, self._layouts, self._headers = {}, {}, {}

    def _read(self, ref, start, end):
        return fetch_range(ref["path"], start, end, self.revision,
                           root=self.root)

    def schema(self, ref):
        key = ref["path"]
        if key not in self._schemas:
            self._schemas[key] = fetch_range(
                key, 0, ref["schema_bytes"], self.revision, root=self.root)
        return self._schemas[key]

    def whole_file(self, refs, uniform):
        """Every record batch of one file, read in one request."""
        end = max(r["offset"] + r["bytes"] for r in refs)
        start = refs[0]["schema_bytes"]
        buf = fetch_range(refs[0]["path"], start, end, self.revision,
                          root=self.root)
        return [Block.from_batch(b, uniform, refs[0]["source"])
                for b in decode_batches(self.schema(refs[0]), buf)]


    def layout(self, ref):
        """:func:`arrow_ipc.float_column` of every float field of a file."""
        import pyarrow as pa
        if ref["path"] not in self._layouts:
            schema = pa.ipc.read_schema(pa.py_buffer(self.schema(ref)))
            self._layouts[ref["path"]] = [
                arrow_ipc.float_column(schema, n) for n in FIELDS
                if n in schema.names]
        return self._layouts[ref["path"]]

    def header(self, ref):
        """``(RecordBatch header, absolute body offset)``, read once."""
        key = (ref["path"], ref["offset"])
        if key not in self._headers:
            at = ref["offset"]
            head = self._read(ref, at, at + min(ref["bytes"], HEADER_BYTES))
            mlen, pre = arrow_ipc.split_prefix(head)
            if pre + mlen > len(head):
                head = self._read(ref, at, at + pre + mlen)
            msg = arrow_ipc.message_header(head[pre:pre + mlen])
            self._headers[key] = (msg, at + pre + mlen)
        return self._headers[key]

    def column(self, ref, msg, body, col):
        """``(values, starts, ends)`` of rows ``[rows_from, rows_to)`` of one
        column, from three ranges: offsets, values, and the validity bitmap
        when the column holds nulls."""
        d, node, off_buf, valid_buf = col
        a, b = ref["rows_from"] * d, ref["rows_to"] * d
        off_at = body + msg["buffers"][off_buf][0]
        offs = np.frombuffer(self._read(ref, off_at + 4 * a, off_at + 4 * b + 4),
                             dtype="<i4").astype(np.int64)
        v0, v1 = int(offs[0]), int(offs[-1])
        val_at = body + msg["buffers"][valid_buf + 1][0]
        values = (np.frombuffer(self._read(ref, val_at + 4 * v0, val_at + 4 * v1),
                                dtype="<f4") if v1 > v0 else np.zeros(0, "<f4"))
        if msg["nodes"][node][1] > 0 and v1 > v0:
            values = self._nulls_to_nan(ref, body, msg, valid_buf, values, v0)
        return values, (offs[:-1] - v0).reshape(-1, d), (offs[1:] - v0).reshape(-1, d)

    def _nulls_to_nan(self, ref, body, msg, valid_buf, values, v0):
        at = body + msg["buffers"][valid_buf][0]
        n = len(values)
        raw = self._read(ref, at + v0 // 8, at + (v0 + n + 7) // 8)
        bits = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little")
        out = values.copy()
        out[~bits[v0 % 8:v0 % 8 + n].astype(bool)] = np.nan
        return out

    def chunk(self, ref, uniform):
        """A Block of the rows ``[rows_from, rows_to)`` of one record batch,
        read without the rest of it."""
        msg, body = self.header(ref)
        return Block([self.column(ref, msg, body, col)
                      for col in self.layout(ref)], uniform, ref["source"])


def chunk_refs(refs, chunk_bytes=None):
    """Record batches cut into runs of rows of about ``chunk_bytes``
    (default ``CHUNK_BYTES``). The cut depends on the index alone, so every
    run can be drawn uniformly."""
    chunk_bytes = chunk_bytes or CHUNK_BYTES
    out = []
    for ref in refs:
        n = max(1, min(ref["rows"], round(ref["bytes"] / chunk_bytes)))
        edges = np.linspace(0, ref["rows"], n + 1).round().astype(int)
        out += [dict(ref, rows_from=int(a), rows_to=int(b))
                for a, b in zip(edges[:-1], edges[1:]) if b > a]
    return out


# ── Pools ────────────────────────────────────────────────────────────────────


class ResidentPool:
    """Every record batch of a small source, read once. A draw picks a batch
    in proportion to its rows (uniform sources) or its length.

    ``jobs`` are futures of block lists (one per file); the pool waits for
    them at its first draw, so every family fetches in parallel."""

    def __init__(self, jobs):
        self.jobs, self.blocks, self.cdf = jobs, None, None

    def _resolve(self):
        self.blocks = [b for job in self.jobs for b in job.result()]
        self.cdf = np.cumsum([b.weight for b in self.blocks])
        self.jobs = None

    def pick(self, rng):
        if self.blocks is None:
            self._resolve()
        u = rng.random() * self.cdf[-1]
        i = int(np.searchsorted(self.cdf, u, side="right"))
        return self.blocks[min(i, len(self.blocks) - 1)], None

    def charge(self, slot, n):
        pass


def block_budget(block, block_use=BLOCK_USE, window=1024):
    """Windows a kept record batch serves before it is replaced."""
    return max(1, int(round(block_use * block.values / window)))


class StreamingPool:
    """A few chunks of a large source at a time.

    A draw picks one kept chunk uniformly. Each chunk serves
    :func:`block_budget` windows, in proportion to its values, and then the
    next fetched chunk takes its place. Replacements are drawn uniformly
    from every chunk of the source by the pool's own generator and fetched
    ahead on a thread pool, so the sequence does not depend on network
    timing.
    """

    def __init__(self, refs, load, executor, rng, slots, block_use):
        self.refs, self.load, self.executor = refs, load, executor
        self.rng, self.block_use = rng, block_use
        # The kept batches, then one more fetched ahead. The kept ones are
        # taken at the first draw, so every family fetches in parallel.
        self.pending = collections.deque(
            self._submit() for _ in range(slots + 1))
        self.n_slots, self.slots = slots, None

    def _submit(self):
        ref = self.refs[int(self.rng.integers(len(self.refs)))]
        return self.executor.submit(self.load, ref)

    def _take(self):
        block = self.pending.popleft().result()
        return [block, block_budget(block, self.block_use)]

    def _next(self):
        self.pending.append(self._submit())
        return self._take()

    def pick(self, rng):
        if self.slots is None:
            self.slots = [self._take() for _ in range(self.n_slots)]
        i = int(rng.integers(len(self.slots)))
        if self.slots[i][1] <= 0:
            self.slots[i] = self._next()
        return self.slots[i][0], i

    def charge(self, slot, n):
        self.slots[slot][1] -= n


# ── Windows ──────────────────────────────────────────────────────────────────


def sample_variates(rng, dims, max_dim=MAX_DIM):
    """``(field, variate)`` pairs to cut: per field, n ~ U{1..cap} distinct
    variates with cap = min(D, max_dim * D // sum(dims)), as uni2ts
    `SampleDimension` does with its uniform sampler."""
    total = sum(dims)
    out = []
    for field, d in enumerate(dims):
        if d == 0:
            continue
        cap = max(1, min(d, max_dim * d // total))
        n = int(rng.integers(1, cap + 1))
        out += [(field, int(j)) for j in rng.choice(d, size=n, replace=False)]
    return out


def crop_start(rng, length, window=1024):
    """A uniform start for a window of a series of ``length`` values."""
    return int(rng.integers(0, max(length - window, 0) + 1))


def cut_window(series, start, window=1024):
    """``series[start:start + window]``, left zero-padded to ``window``.

    Missing values before the first observed one become padding, later ones
    are forward filled. None when the piece holds no observed value.
    """
    from src.dataloader import _forward_fill_nan
    piece = series[start:start + window]
    observed = np.isfinite(piece)
    if not observed.any():
        return None
    real = piece[int(observed.argmax()):].astype(np.float32)
    out = np.zeros(window, dtype=np.float32)
    out[window - len(real):] = real
    if not np.isfinite(real).all():
        tail = out[window - len(real):]
        tail[~np.isfinite(tail)] = np.nan
        _forward_fill_nan(tail)
    return out


def draw_windows(rng, block, dims, window=1024):
    """The windows of one draw: one row, its sampled variates, one crop."""
    row = block.pick_row(rng)
    start = crop_start(rng, block.length(row), window)
    cuts = (cut_window(block.series(f, row, j), start, window)
            for f, j in sample_variates(rng, dims))
    return [w for w in cuts if w is not None]


# ── The stream ───────────────────────────────────────────────────────────────


class GiftPretrainStream:
    """Batches of GiftEvalPretrain windows, ``[B, T, C]``, and with
    ``emit_labels`` the frequency id and seasonality id of every batch row
    (of its first channel, as :class:`src.dataloader.HFStreamingLoader`).

    The stream never ends. Its draws are a function of ``seed`` alone: a
    resumed run passes a seed derived from its position, so it continues
    with new draws. ``counts`` holds the windows emitted per family, the
    mix the trainer saw.

    ``root`` reads a local copy of the repository instead of Hugging Face,
    for tests. ``with_covariates=False`` trains on the targets only.
    """

    def __init__(self, batch_size, C=1, seed=0, window=1024, index=None,
                 root=None, freq_vocab="v1", emit_labels=True, prefetch=2,
                 workers=8, block_use=BLOCK_USE, with_covariates=True):
        self.batch_size, self.C, self.window = batch_size, C, window
        self.seed, self.root, self.prefetch = seed, root, prefetch
        self.index = index if index is not None else load_index()
        self.families = build_families(self.index)
        self.freq_vocab, self.emit_labels = freq_vocab, emit_labels
        self.workers, self.block_use = workers, block_use
        self.with_covariates = with_covariates
        self.counts = collections.Counter()

    def __iter__(self):
        from src.dataloader import PrefetchIterator
        return iter(PrefetchIterator(self._batches(), prefetch=self.prefetch))

    def labels(self):
        """``(freq_ids, seasonality_ids)`` of every family, as arrays."""
        from src.freq_embedding import freq_labels
        pairs = [freq_labels(f["freq"], self.freq_vocab) for f in self.families]
        return (np.array([p[0] for p in pairs], dtype=np.int64),
                np.array([p[1] for p in pairs], dtype=np.int64))

    def _dims(self, family):
        dt, dc = family["dims"]
        return (dt, dc) if self.with_covariates else (dt, 0)

    def open_pools(self, rng):
        """One pool per family: resident when small, streaming otherwise."""
        loader = BlockLoader(self.index["revision"], self.root)
        executor = ThreadPoolExecutor(self.workers)
        return [self._pool(f, loader, executor, rng) for f in self.families]

    def _pool(self, family, loader, executor, rng):
        uniform = family["uniform"]
        if family["bytes"] <= RESIDENT_BYTES:
            return ResidentPool(self._whole(family, loader, executor))
        refs = chunk_refs(family["refs"])
        slots = max(1, int(POOL_BYTES * len(refs) // family["bytes"]))
        own = np.random.default_rng(rng.integers(2 ** 62))
        return StreamingPool(refs, lambda r: loader.chunk(r, uniform),
                             executor, own, slots, self.block_use)

    def _whole(self, family, loader, executor):
        by_file = collections.defaultdict(list)
        for ref in family["refs"]:
            by_file[ref["path"]].append(ref)
        return [executor.submit(loader.whole_file, refs, family["uniform"])
                for _, refs in sorted(by_file.items())]

    def rngs(self):
        """Independent generators for the draws and the shuffle buffer, from
        ``seed`` (an int or a list of ints)."""
        draws, shuffle = np.random.SeedSequence(self.seed).spawn(2)
        return np.random.default_rng(draws), np.random.default_rng(shuffle)

    def windows(self, rng):
        """``(window, family index)`` pairs, in draw order, forever."""
        pools = self.open_pools(rng)
        cdf = np.cumsum(family_probabilities(self.families))
        while True:
            k = min(int(np.searchsorted(cdf, rng.random(), side="right")),
                    len(pools) - 1)
            block, slot = pools[k].pick(rng)
            got = draw_windows(rng, block, self._dims(self.families[k]),
                               self.window)
            # A draw with no observed value still spends a window, so a kept
            # batch of empty rows cannot hold its slot forever.
            pools[k].charge(slot, max(1, len(got)))
            yield from ((w, k) for w in got)

    def shuffled(self):
        """:meth:`windows` through a buffer of ``SHUFFLE_WINDOWS``."""
        draws, rng = self.rngs()
        buf = []
        for item in self.windows(draws):
            if len(buf) < SHUFFLE_WINDOWS:
                buf.append(item)
                continue
            i = int(rng.integers(len(buf)))
            yield buf[i]
            buf[i] = item

    def _batches(self):
        freq_ids, seas_ids = self.labels()
        items = self.shuffled()
        n = self.batch_size * self.C
        while True:
            yield self._collate([next(items) for _ in range(n)],
                                freq_ids, seas_ids)

    def _collate(self, items, freq_ids, seas_ids):
        B, C, T = self.batch_size, self.C, self.window
        x = np.stack([w for w, _ in items]).reshape(B, C, T)
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1)))
        self.counts.update(k for _, k in items)
        if not self.emit_labels:
            return x
        first = np.array([items[b * C][1] for b in range(B)])
        return (x, torch.from_numpy(freq_ids[first]),
                torch.from_numpy(seas_ids[first]))
