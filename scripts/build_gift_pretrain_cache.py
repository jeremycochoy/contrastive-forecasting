#!/usr/bin/env python3
"""
Fill the disk cache of the GiftEvalPretrain stream before a run.

The stream (`src/gift_pretrain.py`) reads the layout of a record batch (its
header and its row offsets) the first time a process of the machine needs
it, and keeps it in the disk cache. So it does with the schema of a source
and the files of the small sources. This script reads all of them in one
pass. After it, a chunk costs one request for each float column from the
first step. Copy the cache folder to another machine to use it there.

Only a record batch that the stream cuts into parts needs a layout. Run the
script with the GIFT_PRETRAIN_CHUNK_MB of the runs, or with none.

Usage (CPU and network only):

    python3 scripts/build_gift_pretrain_cache.py --dry-run
    python3 scripts/build_gift_pretrain_cache.py --rate 5

``--dry-run`` counts the requests and makes none. The folder is
``$GIFT_PRETRAIN_CACHE``, else ``~/.cache/gift_pretrain``. A second run reads
only what the cache does not hold.
"""

import argparse
import contextlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import gift_pretrain as gp  # noqa: E402


def todo(cache, families):
    """What the cache does not hold: ``(record batches that need a layout,
    one record batch of each source that needs its schema, small-source
    files)``. A file is ``(path, end)``, the bytes the stream reads."""
    batches, schemas, files = [], {}, {}
    for fam in families:
        if fam["bytes"] <= gp.RESIDENT_BYTES:
            for r in fam["refs"]:
                files[r["path"]] = max(files.get(r["path"], 0),
                                       r["offset"] + r["bytes"])
            continue
        for r in fam["refs"]:
            schemas.setdefault(r["source"], r)
            if len(gp.chunk_refs([r])) > 1 and not os.path.exists(
                    cache.path(f"layouts/{r['path']}@{r['offset']}.npz")):
                batches.append(r)
    schemas = [r for s, r in sorted(schemas.items())
               if not os.path.exists(cache.path(f"schemas/{s}"))]
    files = [(p, end) for p, end in sorted(files.items())
             if not (os.path.exists(cache.path(f"files/{p}"))
                     and os.path.getsize(cache.path(f"files/{p}")) == end)]
    return batches, schemas, files


def planned_requests(families, batches, schemas, files):
    """Requests of the build: one per schema, one per file, and per layout
    one, plus one for the covariate offsets of a record batch larger than
    ``HEADER_BYTES``. Row offsets beyond the first read cost one more."""
    columns = {m: 1 + (f["dims"][1] > 0) for f in families for m in f["members"]}
    layouts = sum(1 if r["bytes"] <= gp.HEADER_BYTES else columns[r["source"]]
                  for r in batches)
    return layouts + len(schemas) + len(files)


@contextlib.contextmanager
def rate_limited(rate):
    """Every range read waits its turn: at most ``rate`` a second. Gives
    the list of the request times."""
    real, lock, sent = gp.fetch_range, threading.Lock(), []

    def fetch(*args, **kwargs):
        with lock:
            at = max(time.monotonic(), (sent[-1] + 1 / rate) if sent else 0)
            sent.append(at)
        time.sleep(max(0.0, at - time.monotonic()))
        return real(*args, **kwargs)
    gp.fetch_range = fetch
    try:
        yield sent
    finally:
        gp.fetch_range = real


def build(loader, batches, schemas, files, workers):
    """Read and keep every schema, layout and file. The schemas come first:
    a layout needs the schema of its source. Logs every 500 items."""
    for r in schemas:
        loader.schema(r)
    jobs = ([lambda r=r: loader.batch_layout(r) for r in batches]
            + [lambda f=f: loader.file_bytes(*f) for f in files])
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(lambda job: job(), jobs), 1):
            if i % 500 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} done, {time.time() - t0:.0f} s",
                      flush=True)


def folder_bytes(folder):
    return sum(os.path.getsize(os.path.join(d, f))
               for d, _, names in os.walk(folder) for f in names)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache-dir", default="auto",
                   help="Default: $GIFT_PRETRAIN_CACHE or ~/.cache/gift_pretrain.")
    p.add_argument("--index", default=None, help="Default: the shipped index.")
    p.add_argument("--rate", type=float, default=5.0,
                   help="Requests a second, at most. The Hub limit of the "
                        "token is for every run that uses it.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true",
                   help="Count the requests. Make none.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    index = gp.load_index(args.index)
    cache = gp.open_cache(args.cache_dir, index["revision"])
    if cache is None:
        raise SystemExit("The cache is off. Give --cache-dir.")
    families = gp.build_families(index)
    batches, schemas, files = todo(cache, families)
    n = planned_requests(families, batches, schemas, files)
    print(f"{cache.folder}: {len(batches)} layouts, {len(schemas)} schemas "
          f"and {len(files)} files to read, about {n} requests, "
          f"{n / args.rate / 60:.0f} min at {args.rate:g} a second")
    if args.dry_run or not (batches or schemas or files):
        return
    with rate_limited(args.rate) as sent:
        build(gp.BlockLoader(index["revision"], None, cache), batches,
              schemas, files, args.workers)
    print(f"{len(sent)} requests. The cache holds "
          f"{folder_bytes(cache.folder) / 2 ** 20:.0f} MB.")


if __name__ == "__main__":
    main()
