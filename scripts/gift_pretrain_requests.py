#!/usr/bin/env python3
"""
Hub requests, memory and mixing of the GiftEvalPretrain stream, with no data
and no network.

The script runs the sampler of `src/gift_pretrain.py` on the shipped index.
A fake loader gives blocks of the right size, and a fake draw gives the chunk
and the row of each window instead of its values. It counts the chunk loads
and prices them three ways:

- before: the loader before the disk cache. Per chunk, the header of its
  record batch and the schema of its file (each once in a process), and per
  float column the row offsets and the values.
- warm: the loader now, on a machine whose cache holds every layout. A chunk
  of whole record batches costs one request. A part of a record batch costs
  one per float column.
- cold: the same, on a machine with an empty cache: one more read per new
  layout (and one per covariate column), and the schema once per source.

The rows of a record batch get the mean row length of the batch. A column
with nulls costs one more request per chunk in every case. The real record
batches that were checked held none.

Usage:

    python3 scripts/gift_pretrain_requests.py off 32:96 64:192
    python3 scripts/gift_pretrain_requests.py --windows 300000 off

A setting is ``off`` (the chunks of before) or ``CHUNK_MB:POOL_MB``, the
bigger reads of ``GIFT_PRETRAIN_CHUNK_MB`` and ``GIFT_PRETRAIN_POOL_MB``.
"""

import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import gift_pretrain as gp  # noqa: E402

BATCH = 256


class FakeBlock:
    """The size, the rows and the weights of a chunk, with no values."""

    pick_row = gp.Block.pick_row

    def __init__(self, ref, uniform, rows_from, rows_to):
        self.rows = rows_to - rows_from
        self.values = max(1, round(ref["values"] * self.rows / ref["rows"]))
        self.row_len = ref["row_len"]
        lengths = np.full(self.rows, self.row_len, np.float64)
        self.cdf = None if uniform else np.cumsum(lengths)
        self.weight = float(self.rows if uniform else self.cdf[-1])
        self.tag, self.rows_from = (ref["path"], ref["offset"], rows_from), rows_from

    def length(self, row):
        return self.row_len


class FakeLoader:
    def __init__(self):
        self.loads = []

    def chunk(self, ref, uniform):
        self.loads.append(ref)
        return FakeBlock(ref, uniform, ref["rows_from"], ref["rows_to"])

    def whole_file(self, refs, uniform):
        return [FakeBlock(r, uniform, 0, r["rows"]) for r in refs]


def fake_draw(rng, block, dims, window=1024):
    """The random draws of :func:`gp.draw_windows`: ``(chunk, row)`` per
    window."""
    row = block.pick_row(rng)
    gp.crop_start(rng, block.length(row), window)
    n = len(gp.sample_variates(rng, dims))
    return [(block.tag, block.rows_from + row)] * n


def before(loads, columns):
    """Requests of the loader before the disk cache, in one process."""
    files, batches, n = set(), set(), 0
    for ref in loads:
        n += (ref["path"] not in files) + ((ref["path"], ref["offset"]) not in batches)
        files.add(ref["path"])
        batches.add((ref["path"], ref["offset"]))
        n += 2 * columns[ref["source"]]
    return n


def now(loads, columns, warm):
    """Requests of the loader now, on a warm or a cold machine."""
    layouts, sources, n = set(), set(), 0
    for ref in loads:
        if not warm and ref["source"] not in sources:
            n += 1
            sources.add(ref["source"])
        if gp.is_whole(ref):
            n += 1
            continue
        key = (ref["path"], ref["offset"])
        n += columns[ref["source"]] * (1 + (not warm and key not in layouts))
        layouts.add(key)
    return n


def small_files(stream):
    """The files of the small sources: each process reads them at its start."""
    return len({r["path"] for f in stream.families
                if f["bytes"] <= gp.RESIDENT_BYTES for r in f["refs"]})


def pools(stream):
    """``(chunk loads at the start, GB of chunks a process holds)``."""
    loads, total = 0, 0.0
    for f in stream.families:
        if f["bytes"] <= gp.RESIDENT_BYTES:
            continue
        refs = gp.chunk_refs(f["refs"], join=gp.JOIN_BATCHES)
        slots = max(1, int(gp.POOL_BYTES * len(refs) // f["bytes"]))
        loads += slots + 1
        total += (slots + 1) * f["bytes"] / len(refs)
    return loads, total / 2 ** 30


def mixing(seq, streaming, shuffle_rng):
    """Per span of ``SHUFFLE_WINDOWS`` windows: the distinct chunks and the
    windows per chunk of the large sources. Per shuffled batch: the share of
    window pairs from one chunk, and from one row."""
    span = gp.SHUFFLE_WINDOWS
    spans = [[t[0] for t, k in seq[a:a + span] if k in streaming]
             for a in range(0, len(seq) - span + 1, span)]
    chunks = np.mean([len(set(s)) for s in spans])
    per_chunk = np.mean([len(s) / len(set(s)) for s in spans])
    pairs = BATCH * (BATCH - 1) / 2
    same_chunk, same_row = [], []
    shuffled = list(shuffle(seq, shuffle_rng))
    for a in range(0, len(shuffled) - BATCH + 1, BATCH):
        batch = [t for t, k in shuffled[a:a + BATCH] if k in streaming]
        for out, keys in ((same_chunk, [t[0] for t in batch]), (same_row, batch)):
            out.append(sum(n * (n - 1) / 2
                           for n in collections.Counter(keys).values()) / pairs)
    return chunks, per_chunk, 100 * np.mean(same_chunk), 100 * np.mean(same_row)


def shuffle(items, rng):
    """The buffer of :meth:`gp.GiftPretrainStream.shuffled`."""
    buf = []
    for item in items:
        if len(buf) < gp.SHUFFLE_WINDOWS:
            buf.append(item)
            continue
        i = int(rng.integers(len(buf)))
        yield buf[i]
        buf[i] = item


def run(setting, windows, seed):
    """One setting: its requests, memory and mixing."""
    chunk_mb, pool_mb = (16, 48) if setting == "off" else map(float, setting.split(":"))
    gp.CHUNK_BYTES, gp.POOL_BYTES = int(chunk_mb * 2 ** 20), int(pool_mb * 2 ** 20)
    gp.JOIN_BATCHES = setting != "off"
    stream = gp.GiftPretrainStream(batch_size=BATCH, seed=[seed, 0],
                                   freq_vocab="v2", cache_dir=None)
    for f in stream.families:
        for r in f["refs"]:
            r["row_len"] = max(1, int(r["values"] / (r["rows"] * sum(f["dims"]))))
    columns = {m: 1 + (f["dims"][1] > 0) for f in stream.families for m in f["members"]}
    streaming = {i for i, f in enumerate(stream.families) if f["bytes"] > gp.RESIDENT_BYTES}
    loader = FakeLoader()
    gp.BlockLoader, gp.draw_windows = (lambda *a, **k: loader), fake_draw
    draws, shuffle_rng = stream.rngs()
    it = stream.windows(draws)
    seq = [next(it) for _ in range(windows)]
    start, memory = pools(stream)
    first, rest = loader.loads[:start], loader.loads[start:]
    per_k, files = 1000 / windows, small_files(stream)
    return [setting, len(rest) * per_k, before(rest, columns) * per_k,
            now(rest, columns, True) * per_k, now(rest, columns, False) * per_k,
            f"{before(first, columns) + 2 * files}/{now(first, columns, True)}/"
            f"{now(first, columns, False) + files}", memory,
            *mixing(seq, streaming, shuffle_rng)]


HEADER = ("setting", "loads/1k", "before/1k", "warm/1k", "cold/1k",
          "start before/warm/cold", "chunk GB", "chunks/span", "windows/chunk",
          "same chunk %", "same row %")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("settings", nargs="*", default=["off"])
    p.add_argument("--windows", type=int, default=3_000_000,
                   help="Windows of one process.")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    print(" | ".join(HEADER))
    real = (gp.BlockLoader, gp.draw_windows)
    for setting in args.settings:
        row = run(setting, args.windows, args.seed)
        gp.BlockLoader, gp.draw_windows = real
        print(" | ".join(f"{v:.3g}" if isinstance(v, float) else str(v)
                         for v in row), flush=True)


if __name__ == "__main__":
    main()
