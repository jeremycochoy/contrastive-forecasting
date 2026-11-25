"""Fewer Hub requests for the GiftEvalPretrain stream, with the same windows.

The Hub limits the range requests of a token. These tests run on the CPU, on
a corpus they write in the layout of `Salesforce/GiftEvalPretrain`, through a
fake Hub session that serves the files and counts the requests:

1. The new loader reads the rows the old loader reads, for every chunk and
   every file, and the stream gives the same windows and ids for one seed:
   with no cache, with a cold disk cache and with a warm one.
2. The requests: a small source costs one request on a cold machine and
   none on a warm one. A chunk of whole record batches costs one request. A
   chunk of part of a record batch costs one request for each float column
   (and one more for a column with nulls) once its layout is in the cache.
3. The cache: one download for concurrent readers, a damaged entry is read
   again, a disk that refuses writes changes no window.
4. The settings: the folder of the cache, the chunk and pool sizes, and
   the bigger reads (an opt-in), which join small record batches.
5. The build script fills the cache with the requests it plans.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import arrow_ipc  # noqa: E402
from src import gift_pretrain as gp  # noqa: E402
from tests.test_419_gift_pretrain import (arrow_schema, builder,  # noqa: E402
                                          load_script)


# ---------------------------------------------------------------------------
# The loader of origin/experiments (ff7dfbe2), kept to compare with
# ---------------------------------------------------------------------------

class OldBlockLoader:
    """The record batch reader before this change: per chunk, the header
    of its batch (once), the schema of its file (once), and for every
    column its offsets, its values and its validity bits when it holds
    nulls."""

    def __init__(self, revision, root=None):
        self.revision, self.root = revision, root
        self._schemas, self._layouts, self._headers = {}, {}, {}

    def _read(self, ref, start, end):
        return gp.fetch_range(ref["path"], start, end, self.revision,
                              root=self.root)

    def schema(self, ref):
        key = ref["path"]
        if key not in self._schemas:
            self._schemas[key] = gp.fetch_range(
                key, 0, ref["schema_bytes"], self.revision, root=self.root)
        return self._schemas[key]

    def whole_file(self, refs, uniform):
        end = max(r["offset"] + r["bytes"] for r in refs)
        start = refs[0]["schema_bytes"]
        buf = gp.fetch_range(refs[0]["path"], start, end, self.revision,
                             root=self.root)
        return [gp.Block.from_batch(b, uniform, refs[0]["source"])
                for b in gp.decode_batches(self.schema(refs[0]), buf)]

    def layout(self, ref):
        import pyarrow as pa
        if ref["path"] not in self._layouts:
            schema = pa.ipc.read_schema(pa.py_buffer(self.schema(ref)))
            self._layouts[ref["path"]] = [
                arrow_ipc.float_column(schema, n) for n in gp.FIELDS
                if n in schema.names]
        return self._layouts[ref["path"]]

    def header(self, ref):
        key = (ref["path"], ref["offset"])
        if key not in self._headers:
            at = ref["offset"]
            head = self._read(ref, at, at + min(ref["bytes"], 64 * 1024))
            mlen, pre = arrow_ipc.split_prefix(head)
            if pre + mlen > len(head):
                head = self._read(ref, at, at + pre + mlen)
            msg = arrow_ipc.message_header(head[pre:pre + mlen])
            self._headers[key] = (msg, at + pre + mlen)
        return self._headers[key]

    def column(self, ref, msg, body, col):
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
        msg, body = self.header(ref)
        return gp.Block([self.column(ref, msg, body, col)
                         for col in self.layout(ref)], uniform, ref["source"])


# ---------------------------------------------------------------------------
# A fake Hub, and a corpus with many files, nulls and covariates
# ---------------------------------------------------------------------------

class Response:
    def __init__(self, data):
        self.status_code, self.content = 206, data

    def raise_for_status(self):
        pass


class LocalHub:
    """Range requests of the Hub served from the files under ``root``.
    ``calls`` holds ``(path, start, end)`` of every request."""

    def __init__(self, root, delay=0.0):
        self.root, self.delay = root, delay
        self.calls, self.lock = [], threading.Lock()

    def get(self, url, timeout, headers):
        path = url.split("/resolve/", 1)[1].split("/", 1)[1]
        a, b = map(int, headers["Range"][len("bytes="):].split("-"))
        time.sleep(self.delay)
        with open(os.path.join(self.root, path), "rb") as f:
            f.seek(a)
            data = f.read(b + 1 - a)
        with self.lock:
            self.calls.append((path, a, b + 1))
        return Response(data)

    def count(self, prefix=""):
        return sum(1 for p, _, _ in self.calls if p.startswith(prefix))


@pytest.fixture(scope="module", autouse=True)
def no_network(corpus):
    """A stream thread that outlives its test reads the local corpus, never
    the Hub: the stream keeps chunks in flight when a test stops it."""
    patch = pytest.MonkeyPatch()
    patch.setattr(gp, "_session_and_headers",
                  lambda: (LocalHub(corpus["root"]), {}))
    yield
    patch.undo()


@pytest.fixture
def hub(corpus, monkeypatch):
    server = LocalHub(corpus["root"])
    monkeypatch.setattr(gp, "_session_and_headers", lambda: (server, {}))
    return server


def float_rows(rng, n, d, lengths, nulls):
    """``n`` rows of ``d`` variates (a list per variate when d > 1). Every
    ``nulls``-th value is None, and a few values are NaN."""
    rows = []
    for i in range(n):
        series = []
        for _ in range(d):
            v = rng.standard_normal(int(lengths[i])).astype(np.float32).tolist()
            v[3::17] = [float("nan")] * len(v[3::17])
            if nulls:
                v[i % nulls::nulls] = [None] * len(v[i % nulls::nulls])
            series.append(v)
        rows.append(series[0] if d == 1 else series)
    return rows


def write_file(root, path, rows, dt, dc, batch_rows):
    """One Arrow IPC stream file, ``batch_rows`` rows a record batch."""
    import pyarrow as pa
    schema = arrow_schema(dt, dc)
    with pa.ipc.new_stream(str(root / path), schema) as w:
        for i in range(0, len(rows), batch_rows):
            part = rows[i:i + batch_rows]
            cols = {"item_id": [f"s{i + j}" for j in range(len(part))],
                    "start": [0] * len(part), "freq": ["H"] * len(part),
                    "target": [r[0] for r in part]}
            if dc:
                cols["past_feat_dynamic_real"] = [r[1] for r in part]
            w.write_batch(pa.record_batch(
                [pa.array(cols[n], type=schema.field(n).type)
                 for n in schema.names], schema=schema))


# name: (files, rows a file, rows a batch, target dims, covariate dims,
#        a None every this many values or 0)
SOURCES = {"small_a": (1, 6, 2, 1, 0, 0),
           "multi_b": (3, 4, 2, 2, 3, 41),
           "era5_2000": (2, 6, 3, 4, 0, 0),
           "era5_2001": (2, 6, 3, 4, 0, 0),
           "uni_c": (2, 9, 3, 1, 0, 23)}


def write_corpus(root):
    rng = np.random.default_rng(11)
    paths = {}
    for name, (files, n, per_batch, dt, dc, nulls) in SOURCES.items():
        (root / name).mkdir(parents=True)
        paths[name] = []
        for f in range(files):
            lengths = rng.integers(30, 3000, n)
            if name.startswith("era5"):
                lengths[:] = 1100
            targets = float_rows(rng, n, dt, lengths, nulls)
            covs = float_rows(rng, n, dc, lengths, nulls) if dc else [None] * n
            if name == "uni_c" and f == 0:
                targets[4] = [None] * int(lengths[4])   # a row with no value
            path = f"{name}/data-{f:05d}-of-{files:05d}.arrow"
            write_file(root, path, list(zip(targets, covs)), dt, dc, per_batch)
            paths[name].append(path)
    weights = {"small_a": 2.0, "multi_b": 1.5, "era5_2000": 1.0,
               "era5_2001": 1.0, "uni_c": 3.0}
    return weights, paths


def index_of(root, weights, paths):
    sources = {}
    for name, files in paths.items():
        walks = [builder.walk_file(p, None, i, (root / p).stat().st_size,
                                   root=str(root)) for i, p in enumerate(files)]
        sources[name] = builder.summarize(name, files, walks, weights)
    return {"repo": gp.HF_REPO, "revision": None, "max_dim": gp.MAX_DIM,
            "sources": sources}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("gift_requests")
    weights, paths = write_corpus(root)
    index = index_of(root, weights, paths)
    small = sum(b[2] for b in index["sources"]["small_a"]["batches"])
    return {"root": str(root), "index": index, "resident_bytes": small}


@pytest.fixture
def sizes(corpus, monkeypatch):
    """small_a is resident. The others stream in chunks of one or a few rows,
    or of a whole record batch."""
    monkeypatch.setattr(gp, "RESIDENT_BYTES", corpus["resident_bytes"])
    monkeypatch.setattr(gp, "CHUNK_BYTES", 16 * 1024)
    monkeypatch.setattr(gp, "POOL_BYTES", 64 * 1024)
    monkeypatch.setattr(gp, "JOIN_BATCHES", False)
    monkeypatch.setattr(gp, "SHUFFLE_WINDOWS", 64)


def every_chunk(index):
    """``(chunk ref, uniform)`` of every chunk of every streaming family."""
    return [(c, f["uniform"]) for f in gp.build_families(index)
            if f["bytes"] > gp.RESIDENT_BYTES for c in gp.chunk_refs(f["refs"])]


def every_file(index):
    """``(refs of one file, uniform)`` of every file of a resident family."""
    out = []
    for f in gp.build_families(index):
        if f["bytes"] <= gp.RESIDENT_BYTES:
            paths = sorted({r["path"] for r in f["refs"]})
            out += [([r for r in f["refs"] if r["path"] == p], f["uniform"])
                    for p in paths]
    return out


def same_block(a, b):
    assert (a.rows, a.values, a.weight, a.source) == (b.rows, b.values, b.weight, b.source)
    assert len(a.fields) == len(b.fields)
    for (va, sa, ea), (vb, sb, eb) in zip(a.fields, b.fields):
        np.testing.assert_array_equal(va, vb)   # NaN equals NaN here
        np.testing.assert_array_equal(sa, sb)
        np.testing.assert_array_equal(ea, eb)


def float_nulls(ref, root):
    """Per float column of the record batch of ``ref``: does it hold a
    null? pyarrow reads the whole file to tell."""
    import pyarrow as pa
    with pa.ipc.open_stream(os.path.join(root, ref["path"])) as reader:
        batch = list(reader)[ref["batch_number"]]
    out = []
    for name in gp.FIELDS:
        if name in batch.schema.names:
            col = batch.column(name)
            nested = pa.types.is_fixed_size_list(col.type)
            out.append((col.values.values if nested else col.values).null_count > 0)
    return out


def warm_cost(ref, root):
    """Requests of a chunk on a warm cache: one for whole record batches.
    For part of a record batch, per float column one for the values and one
    for the validity bits when it holds nulls."""
    if gp.is_whole(ref):
        return 1
    return sum(1 + nulls for nulls in float_nulls(ref, root))


def settle(hub, quiet=0.3):
    """Wait until the Hub got no request for ``quiet`` seconds: the
    chunks fetched ahead are in."""
    n = -1
    while n != len(hub.calls):
        n = len(hub.calls)
        time.sleep(quiet)


# ---------------------------------------------------------------------------
# 1. The same rows, windows and ids
# ---------------------------------------------------------------------------

def test_the_corpus_has_what_the_tests_need(corpus, sizes):
    fams = {f["name"]: f for f in gp.build_families(corpus["index"])}
    assert fams["small_a"]["bytes"] <= gp.RESIDENT_BYTES
    assert all(fams[n]["bytes"] > gp.RESIDENT_BYTES
               for n in ("multi_b", "era5", "uni_c"))
    assert fams["multi_b"]["dims"] == (2, 3)
    assert len({r["path"] for r in fams["multi_b"]["refs"]}) == 3
    chunks = every_chunk(corpus["index"])
    assert len(chunks) > len({(c["path"], c["offset"]) for c, _ in chunks})
    whole = [gp.is_whole(c) for c, _ in chunks]
    assert any(whole) and not all(whole)
    nulls = [float_nulls(c, corpus["root"]) for c, _ in with_batch_numbers(
        corpus["index"]) if not gp.is_whole(c)]
    assert [True, True] in nulls and [False] in nulls and [True] in nulls


@pytest.mark.parametrize("cache", ["off", "cold", "warm"])
def test_every_chunk_and_file_reads_the_old_rows(corpus, sizes, hub, tmp_path,
                                                 cache):
    index, folder = corpus["index"], tmp_path / "cache"
    disk = None if cache == "off" else gp.open_cache(str(folder), None)
    if cache == "warm":
        warm = gp.BlockLoader(None, None, disk)
        for ref, uniform in every_chunk(index):
            warm.chunk(ref, uniform)
        for refs, uniform in every_file(index):
            warm.whole_file(refs, uniform)
    old, new = OldBlockLoader(None), gp.BlockLoader(None, None, disk)
    for ref, uniform in every_chunk(index):
        same_block(new.chunk(ref, uniform), old.chunk(ref, uniform))
    for refs, uniform in every_file(index):
        for a, b in zip(new.whole_file(refs, uniform), old.whole_file(refs, uniform)):
            same_block(a, b)


def test_a_header_longer_than_the_first_read_gives_the_same_rows(
        corpus, sizes, hub, monkeypatch):
    """The header, then the offsets, each need a read of their own."""
    monkeypatch.setattr(gp, "HEADER_BYTES", 16)
    old, new = OldBlockLoader(None), gp.BlockLoader(None, None, None)
    for ref, uniform in every_chunk(corpus["index"]):
        same_block(new.chunk(ref, uniform), old.chunk(ref, uniform))


def stream_batches(corpus, n, cache_dir, seed=3):
    """``n`` batches of a stream, once the chunks it fetched ahead are in."""
    stream = gp.GiftPretrainStream(
        batch_size=8, C=1, seed=[seed, 5], index=corpus["index"],
        freq_vocab="v2", cache_dir=cache_dir, block_use=0.5)
    out = [b for _, b in zip(range(n), stream._batches())]
    settle(gp._session_and_headers()[0])
    return out, stream.counts


def test_the_stream_gives_the_old_windows_and_ids(corpus, sizes, hub, tmp_path,
                                                  monkeypatch):
    """One seed: the same windows, frequency ids, seasonality ids and
    source counts as the old loader, with the cache off, cold and warm."""
    loads, new_loader = [], gp.BlockLoader

    def old_loader(revision, root=None, cache=None):
        loader = OldBlockLoader(revision, root)
        read = loader.chunk
        loader.chunk = lambda ref, uniform: loads.append(ref) or read(ref, uniform)
        return loader
    monkeypatch.setattr(gp, "BlockLoader", old_loader)
    want, want_counts = stream_batches(corpus, 40, "off")
    monkeypatch.setattr(gp, "BlockLoader", new_loader)
    assert len(loads) >= 30, "the stream must replace many chunks"
    folder = str(tmp_path / "cache")
    for cache_dir in ("off", folder, folder):
        got, counts = stream_batches(corpus, 40, cache_dir)
        assert counts == want_counts
        for (x, f, s), (wx, wf, ws) in zip(got, want):
            assert torch.equal(x, wx) and torch.equal(f, wf) and torch.equal(s, ws)


# ---------------------------------------------------------------------------
# 2. The requests
# ---------------------------------------------------------------------------

def with_batch_numbers(index):
    """Every chunk ref gets the number of its record batch in its file."""
    chunks = every_chunk(index)
    numbers = {}
    for src in index["sources"].values():
        for b in src["batches"]:
            path = src["files"][b[0]][0]
            numbers[(path, b[1])] = sum(
                1 for c in src["batches"] if c[0] == b[0] and c[1] < b[1])
    for ref, _ in chunks:
        ref["batch_number"] = numbers[(ref["path"], ref["offset"])]
    return chunks


def chunk_costs(loader, chunks, hub):
    out = []
    for ref, uniform in chunks:
        before = len(hub.calls)
        loader.chunk(ref, uniform)
        out.append(len(hub.calls) - before)
    return out


def test_a_warm_chunk_costs_one_request_a_column(corpus, sizes, hub, tmp_path):
    chunks = with_batch_numbers(corpus["index"])
    disk = gp.open_cache(str(tmp_path / "cache"), None)
    old = chunk_costs(OldBlockLoader(None), chunks, hub)
    cold = chunk_costs(gp.BlockLoader(None, None, disk), chunks, hub)
    warm = chunk_costs(gp.BlockLoader(None, None, disk), chunks, hub)
    assert warm == [warm_cost(ref, corpus["root"]) for ref, _ in chunks]
    assert sum(cold) < sum(old)
    assert all(w < o for w, o in zip(warm, old))
    plain = [w for (ref, _), w in zip(chunks, warm) if ref["source"] == "era5_2000"]
    assert set(plain) == {1}
    assert max(warm) == 4    # values and validity bits of two columns


def test_a_small_source_costs_one_request_then_none(corpus, sizes, hub, tmp_path):
    files = every_file(corpus["index"])
    disk = gp.open_cache(str(tmp_path / "cache"), None)
    for refs, uniform in files:
        OldBlockLoader(None).whole_file(refs, uniform)
    old = hub.count("small_a")
    for refs, uniform in files:
        gp.BlockLoader(None, None, disk).whole_file(refs, uniform)
    cold = hub.count("small_a") - old
    for refs, uniform in files:
        gp.BlockLoader(None, None, disk).whole_file(refs, uniform)
    assert (old, cold, hub.count("small_a") - old - cold) == (2, 1, 0)


def test_a_second_process_reads_only_chunk_values(corpus, sizes, hub, tmp_path,
                                                  monkeypatch):
    """A second stream on a warm cache: no request for the small source,
    and for each chunk it loads, the requests a warm chunk costs."""
    folder = str(tmp_path / "cache")
    stream_batches(corpus, 40, folder)
    settle(hub)
    before = len(hub.calls)
    loads, chunk = [], gp.BlockLoader.chunk

    def spy(self, ref, uniform):
        loads.append(ref)
        return chunk(self, ref, uniform)
    monkeypatch.setattr(gp.BlockLoader, "chunk", spy)
    stream_batches(corpus, 40, folder)
    settle(hub)
    calls = hub.calls[before:]
    numbers = {(c["path"], c["offset"]): c["batch_number"]
               for c, _ in with_batch_numbers(corpus["index"])}
    for ref in loads:
        ref["batch_number"] = numbers[(ref["path"], ref["offset"])]
    assert loads and not [c for c in calls if c[0].startswith("small_a")]
    assert len(calls) == sum(warm_cost(ref, corpus["root"]) for ref in loads)


# ---------------------------------------------------------------------------
# 3. The cache
# ---------------------------------------------------------------------------

def test_concurrent_readers_download_a_small_file_once(corpus, sizes, tmp_path,
                                                       monkeypatch):
    slow = LocalHub(corpus["root"], delay=0.3)
    monkeypatch.setattr(gp, "_session_and_headers", lambda: (slow, {}))
    folder = str(tmp_path / "cache")
    refs, uniform = every_file(corpus["index"])[0]
    blocks = []

    def read():
        loader = gp.BlockLoader(None, None, gp.open_cache(folder, None))
        blocks.append(loader.whole_file(refs, uniform))
    threads = [threading.Thread(target=read) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert slow.count() == 1 and len(blocks) == 3
    for a, b in zip(blocks[0], blocks[2]):
        same_block(a, b)


def test_a_damaged_entry_is_read_again(corpus, sizes, hub, tmp_path):
    folder = tmp_path / "cache"
    disk = gp.open_cache(str(folder), None)
    (ref, uniform), (refs, runiform) = every_chunk(corpus["index"])[0], every_file(corpus["index"])[0]
    gp.BlockLoader(None, None, disk).chunk(ref, uniform)
    gp.BlockLoader(None, None, disk).whole_file(refs, runiform)
    layout = disk.path(f"layouts/{ref['path']}@{ref['offset']}.npz")
    Path(layout).write_bytes(b"not an npz file")
    small = Path(disk.path(f"files/{refs[0]['path']}"))
    small.write_bytes(small.read_bytes()[:100])
    before = len(hub.calls)
    new = gp.BlockLoader(None, None, disk)
    same_block(new.chunk(ref, uniform), OldBlockLoader(None).chunk(ref, uniform))
    for a, b in zip(new.whole_file(refs, runiform),
                    OldBlockLoader(None).whole_file(refs, runiform)):
        same_block(a, b)
    assert gp.unpack_layout(Path(layout).read_bytes())
    assert small.stat().st_size == max(r["offset"] + r["bytes"] for r in refs)
    assert len(hub.calls) > before


def test_a_disk_that_refuses_writes_changes_no_row(corpus, sizes, hub, tmp_path,
                                                   capsys):
    """The cache folder lies under a file: every write fails. The layouts
    stay in memory, and the warning comes once."""
    (tmp_path / "a_file").write_text("x")
    disk = gp.open_cache(str(tmp_path / "a_file" / "cache"), None)
    loader, old = gp.BlockLoader(None, None, disk), OldBlockLoader(None)
    chunks = with_batch_numbers(corpus["index"])
    for ref, uniform in chunks:
        same_block(loader.chunk(ref, uniform), old.chunk(ref, uniform))
    second = chunk_costs(loader, chunks, hub)
    assert second == [warm_cost(ref, corpus["root"]) for ref, _ in chunks]
    assert capsys.readouterr().out.count("disk cache") == 1
    for refs, uniform in every_file(corpus["index"]):
        for a, b in zip(loader.whole_file(refs, uniform), old.whole_file(refs, uniform)):
            same_block(a, b)


def test_a_disk_without_locks_still_reads(corpus, sizes, hub, tmp_path,
                                          monkeypatch, capsys):
    import fcntl

    def no_lock(f, operation):
        raise OSError(37, "No locks available")
    monkeypatch.setattr(fcntl, "flock", no_lock)
    disk = gp.open_cache(str(tmp_path / "cache"), None)
    refs, uniform = every_file(corpus["index"])[0]
    blocks = gp.BlockLoader(None, None, disk).whole_file(refs, uniform)
    for a, b in zip(blocks, OldBlockLoader(None).whole_file(refs, uniform)):
        same_block(a, b)
    assert "No locks available" in capsys.readouterr().out
    assert disk.get(f"files/{refs[0]['path']}") is not None


def test_a_layout_keeps_its_offsets_exactly():
    offs = np.array([7, 7, 1107, 2207, 2208, 2 ** 31 - 1], dtype=np.int32)
    layout = [(2, 10 ** 12, 5, 3, offs), (1, 4, 9, 0, offs[:3])]
    back = gp.unpack_layout(gp.pack_layout(layout))
    for (d, v, n, z, o), (bd, bv, bn, bz, bo) in zip(layout, back):
        assert (d, v, n, z) == (bd, bv, bn, bz)
        np.testing.assert_array_equal(o, bo)


# ---------------------------------------------------------------------------
# 4. The settings
# ---------------------------------------------------------------------------

def test_where_the_cache_lives(monkeypatch, tmp_path):
    monkeypatch.delenv(gp.CACHE_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    auto = gp.open_cache("auto", "abc")
    assert auto.folder == str(tmp_path / ".cache" / "gift_pretrain" / "abc")
    assert gp.open_cache("auto", "abc", root="/a/local/copy") is None
    assert gp.open_cache(None, "abc") is None and gp.open_cache("off", "abc") is None
    assert gp.open_cache("/x", "abc").folder == "/x/abc"
    monkeypatch.setenv(gp.CACHE_ENV, str(tmp_path / "shared"))
    assert gp.open_cache("auto", "abc").folder == str(tmp_path / "shared" / "abc")
    monkeypatch.setenv(gp.CACHE_ENV, "off")
    assert gp.open_cache("auto", "abc") is None
    assert not (tmp_path / ".cache").exists()


def sizes_in_a_new_process(**env):
    clean = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    clean.pop("GIFT_PRETRAIN_CHUNK_MB", None)
    clean.pop("GIFT_PRETRAIN_POOL_MB", None)
    clean.update(env)
    r = subprocess.run(
        [sys.executable, "-c", "from src import gift_pretrain as gp; "
         "print(gp.CHUNK_BYTES, gp.POOL_BYTES)"],
        capture_output=True, text=True, env=clean, cwd=str(REPO_ROOT))
    return r.returncode, r.stdout.split(), r.stderr


def test_the_chunk_and_pool_sizes_are_opt_in():
    assert sizes_in_a_new_process()[1] == [str(16 * 2 ** 20), str(48 * 2 ** 20)]
    code, out, _ = sizes_in_a_new_process(GIFT_PRETRAIN_CHUNK_MB="64",
                                          GIFT_PRETRAIN_POOL_MB="192")
    assert code == 0 and out == [str(64 * 2 ** 20), str(192 * 2 ** 20)]
    code, _, err = sizes_in_a_new_process(GIFT_PRETRAIN_CHUNK_MB="0")
    assert code != 0 and "positive" in err


def test_a_larger_chunk_holds_more_rows(corpus, sizes):
    small = every_chunk(corpus["index"])
    gp.CHUNK_BYTES *= 4
    large = every_chunk(corpus["index"])
    assert len(large) < len(small)
    rows = lambda chunks: sum(c["rows_to"] - c["rows_from"] for c, _ in chunks)
    assert rows(large) == rows(small)


def test_the_join_is_opt_in():
    code = ("from src import gift_pretrain as gp; print(gp.JOIN_BATCHES)")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    env.pop("GIFT_PRETRAIN_CHUNK_MB", None)
    off = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, cwd=str(REPO_ROOT)).stdout
    env["GIFT_PRETRAIN_CHUNK_MB"] = "32"
    on = subprocess.run([sys.executable, "-c", code], capture_output=True,
                        text=True, env=env, cwd=str(REPO_ROOT)).stdout
    assert (off.strip(), on.strip()) == ("False", "True")


def test_joined_batches_give_their_rows_in_one_request(corpus, sizes, hub):
    """Small consecutive record batches of one file join into one chunk,
    read in one request: the rows of each batch, in order."""
    fams = {f["name"]: f for f in gp.build_families(corpus["index"])}
    refs = fams["uni_c"]["refs"]
    big = 4 * max(r["bytes"] for r in refs)
    runs = gp.chunk_refs(refs, big, join=True)
    assert len(runs) < len(refs) and all(gp.is_whole(r) for r in runs)
    assert len({r["path"] for r in runs}) == 2
    old, new = OldBlockLoader(None), gp.BlockLoader(None, None, None)
    for run in runs:
        parts = [dict(r, rows_from=0, rows_to=r["rows"]) for r in refs
                 if r["path"] == run["path"]
                 and run["offset"] <= r["offset"] < run["offset"] + run["bytes"]]
        before = len(hub.calls)
        block = new.chunk(run, False)
        assert len(hub.calls) - before <= 2      # the run, and the schema once
        want = gp.Block.joined([old.chunk(p, False) for p in parts], False)
        same_block(block, want)
        assert block.rows == sum(p["rows"] for p in parts) == run["rows"]
    assert gp.chunk_refs(refs, big) == gp.chunk_refs(refs, big, join=False)


def test_a_stream_with_joined_batches_runs(corpus, sizes, hub, tmp_path,
                                           monkeypatch):
    monkeypatch.setattr(gp, "JOIN_BATCHES", True)
    monkeypatch.setattr(gp, "CHUNK_BYTES", 64 * 1024)
    got, counts = stream_batches(corpus, 20, str(tmp_path / "cache"))
    assert all(x.shape == (8, 1024, 1)
               for x, _, _ in got)
    assert sum(counts.values()) == 20 * 8


# ---------------------------------------------------------------------------
# 5. The build script
# ---------------------------------------------------------------------------

def test_the_build_fills_the_cache_with_the_requests_it_plans(
        corpus, sizes, hub, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gp, "fetch_range", gp.fetch_range)
    script = load_script(REPO_ROOT / "scripts" / "build_gift_pretrain_cache.py",
                         "build_gift_pretrain_cache")
    index_path = tmp_path / "index.json.gz"
    with gzip.open(index_path, "wt") as f:
        json.dump(corpus["index"], f)
    argv = ["--cache-dir", str(tmp_path / "cache"), "--index", str(index_path),
            "--rate", "10000", "--workers", "1"]
    script.main(argv + ["--dry-run"])
    planned = int(capsys.readouterr().out.split("about ")[1].split()[0])
    assert not hub.calls
    script.main(argv)
    assert len(hub.calls) == planned
    disk = gp.open_cache(str(tmp_path / "cache"), None)
    assert script.todo(disk, gp.build_families(corpus["index"])) == ([], [], [])
    chunks = with_batch_numbers(corpus["index"])
    warm = chunk_costs(gp.BlockLoader(None, None, disk), chunks, hub)
    assert warm == [warm_cost(ref, corpus["root"]) for ref, _ in chunks]
