#!/usr/bin/env python3
"""
Build the record-batch index of `Salesforce/GiftEvalPretrain` (#419).

`src/gift_pretrain.py` streams the corpus by HTTP byte range. It needs the
position of every record batch, the frequency of every source and the uni2ts
sampling weight of every source. This script finds them: it walks the Arrow
IPC messages of every data file, reads only the message headers and the
first kB of each body, and writes one gzipped JSON.

Usage (CPU and network only, about 10,000 range requests):

    git clone https://github.com/SalesforceAIResearch/uni2ts /tmp/uni2ts
    python3 scripts/build_gift_pretrain_index.py \
        --uni2ts-yaml /tmp/uni2ts/cli/conf/pretrain/data/lotsa_v1_weighted.yaml \
        --uni2ts-commit "$(git -C /tmp/uni2ts rev-parse HEAD)" \
        --out src/gift_pretrain_index.json.gz

The data files of a source are the ones its `state.json` lists. The folder
`cmip6_1850` also holds two `cache-*.arrow` files that `datasets` left
behind; they are not part of the dataset and the index skips them.

The script stops if a source has no uni2ts weight, holds two frequencies,
changes its schema or its dimensions between files, or compresses a body.
"""

import argparse
import gzip
import json
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import arrow_ipc  # noqa: E402
from src.arrow_ipc import field_starts, type_layout  # noqa: E402
from src.gift_pretrain import HF_REPO, MAX_DIM, fetch_range  # noqa: E402

HEAD_BYTES = 128 * 1024  # one request reads a header and a body head
# uni2ts builds these two families with `uniform = True`
# (ERA5DatasetBuilder, CMIP6DatasetBuilder): their rows are drawn
# uniformly. Every other builder draws a row in proportion to its length.
UNIFORM_PREFIXES = ("era5_", "cmip6_")
TARGET, COVARIATES = "target", "past_feat_dynamic_real"


def column_dims(schema, starts, name, nodes, rows):
    """``(variates per row, values)`` of a float column: 1 for a list of
    floats, D for a [D, L] column, ``(0, 0)`` when the column is absent."""
    if name not in starts:
        return 0, 0
    first, _ = starts[name]
    depth = type_layout(schema.field(name).type)[0]
    values = nodes[first + depth - 1][0]
    if depth == 2:
        return 1, values
    return nodes[first + depth - 2][0] // max(rows, 1), values


def freq_values(reader, body_at, buffers, starts):
    """The distinct strings of the `freq` column of one record batch. Some
    sources store the column last, so this reads it wherever it lies."""
    _, b = starts["freq"]
    (off_at, off_len), (data_at, data_len) = buffers[b + 1], buffers[b + 2]
    offs = struct.unpack(f"<{off_len // 4}i",
                         reader.read(body_at + off_at, off_len))
    data = reader.read(body_at + data_at, data_len)
    return {data[offs[i]:offs[i + 1]].decode() for i in range(len(offs) - 1)}


class Reader:
    """Range reads of one file. A read inside the last fetched window costs
    no request, so a small file costs one. ``root`` reads a local copy."""

    def __init__(self, path, revision, size, root=None):
        self.path, self.revision, self.size = path, revision, size
        self.root, self.start, self.data = root, 0, b""

    def read(self, start, n):
        if not (self.start <= start
                and start + n <= self.start + len(self.data)):
            end = min(self.size, start + max(n, HEAD_BYTES))
            self.start = start
            self.data = fetch_range(self.path, start, end, self.revision,
                                    root=self.root)
        at = start - self.start
        return self.data[at:at + n]


def read_message(reader, pos):
    """``(header, metadata length, body offset)`` of the message at ``pos``,
    or None at the end-of-stream marker."""
    mlen, pre = arrow_ipc.split_prefix(reader.read(pos, 8))
    if mlen == 0:
        return None
    head = arrow_ipc.message_header(reader.read(pos + pre, mlen))
    return head, mlen, pos + pre + mlen


def batch_entry(head, pos, mlen, body_at, ctx):
    """``(index row, dims, frequencies)`` of one record batch. The row is
    ``[file, offset, bytes, rows, target values, covariate values]``."""
    if head["compressed"]:
        raise SystemExit(f"{ctx['path']}: compressed record batch at {pos}")
    rows, nodes = head["rows"], head["nodes"]
    dt, tv = column_dims(ctx["schema"], ctx["starts"], TARGET, nodes, rows)
    dc, cv = column_dims(ctx["schema"], ctx["starts"], COVARIATES, nodes, rows)
    row = [ctx["file"], pos, 8 + mlen + head["body_length"], rows, tv, cv]
    return row, (dt, dc), freq_values(ctx["reader"], body_at, head["buffers"],
                                      ctx["starts"])


def open_file(path, revision, file_idx, size, root=None):
    """A reader on one data file, its context and its first batch offset."""
    import pyarrow as pa
    reader = Reader(path, revision, size, root)
    mlen, pre = arrow_ipc.split_prefix(reader.read(0, 8))
    schema = pa.ipc.read_schema(pa.py_buffer(reader.read(0, pre + mlen)))
    ctx = {"path": path, "file": file_idx, "schema": schema,
           "starts": field_starts(schema), "reader": reader}
    return reader, ctx, pre + mlen


def walk_file(path, revision, file_idx, size, root=None):
    """Schema, record batches, dims and frequencies of one data file."""
    reader, ctx, pos = open_file(path, revision, file_idx, size, root)
    out = {"schema_bytes": pos, "schema": ctx["schema"], "batches": [],
           "dims": set(), "freqs": set()}
    while pos + 8 <= size:
        msg = read_message(reader, pos)
        if msg is None or msg[0]["type"] != arrow_ipc.RECORD_BATCH:
            break
        row, dims, freqs = batch_entry(msg[0], pos, msg[1], msg[2], ctx)
        out["batches"].append(row)
        out["dims"].add(dims)
        out["freqs"] |= freqs
        pos += row[2]
    return out


def uni2ts_weights(yaml_path):
    """The `weight_map` of every builder in a uni2ts pretraining config."""
    import yaml
    conf = yaml.safe_load(open(yaml_path))
    return {name: float(w) for builder in conf["_args_"]
            for name, w in (builder.get("weight_map") or {}).items()}


def list_sources(revision):
    """Source names (folders with a `state.json`) and every file size."""
    from huggingface_hub import HfApi
    tree = HfApi().list_repo_tree(HF_REPO, repo_type="dataset",
                                  revision=revision, recursive=True)
    sizes = {f.path: f.size for f in tree if getattr(f, "size", None)}
    names = sorted({p.split("/")[0] for p in sizes if p.endswith("/state.json")})
    return names, sizes


def data_files(source, revision):
    """The data files `state.json` lists for one source, in order."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HF_REPO, f"{source}/state.json",
                           repo_type="dataset", revision=revision)
    return [f"{source}/{d['filename']}" for d in json.load(open(path))["_data_files"]]


def one_value(values, what, source):
    """The single element of ``values``, or stop with a message."""
    if len(values) != 1:
        raise SystemExit(f"{source}: {len(values)} {what}: {sorted(values)}")
    return next(iter(values))


def summarize(source, files, walks, weights):
    """The index entry of one source."""
    if source not in weights:
        raise SystemExit(f"{source}: no uni2ts weight")
    one_value({str(w["schema"]) for w in walks}, "schemas", source)
    dt, dc = one_value(set().union(*(w["dims"] for w in walks)), "dims", source)
    batches = [b for w in walks for b in w["batches"]]
    return {"freq": one_value(set().union(*(w["freqs"] for w in walks)),
                              "frequencies", source),
            "weight": weights[source],
            "uniform": source.startswith(UNIFORM_PREFIXES),
            "rows": sum(b[3] for b in batches), "target_dim": dt,
            "cov_dim": dc, "files": [[f, w["schema_bytes"]]
                                     for f, w in zip(files, walks)],
            "batches": batches}


def walk_all(names, revision, sizes, workers):
    """Every data file of every source, walked on ``workers`` threads."""
    jobs = [(s, i, f) for s in names
            for i, f in enumerate(data_files(s, revision))]
    with ThreadPoolExecutor(workers) as pool:
        walks = list(pool.map(
            lambda j: walk_file(j[2], revision, j[1], sizes[j[2]]), jobs))
    return jobs, walks


def build_index(args):
    """The whole index as a dict."""
    from huggingface_hub import HfApi
    revision = args.revision or HfApi().dataset_info(HF_REPO).sha
    weights = uni2ts_weights(args.uni2ts_yaml)
    names, sizes = list_sources(revision)
    jobs, walks = walk_all(names, revision, sizes, args.workers)
    sources = {}
    for name in names:
        mine = [(j[2], w) for j, w in zip(jobs, walks) if j[0] == name]
        sources[name] = summarize(name, [m[0] for m in mine],
                                  [m[1] for m in mine], weights)
    return {"repo": HF_REPO, "revision": revision, "max_dim": MAX_DIM,
            "uni2ts": {"commit": args.uni2ts_commit,
                       "file": "cli/conf/pretrain/data/lotsa_v1_weighted.yaml"},
            "batch_columns": ["file", "offset", "bytes", "rows",
                              "target_values", "covariate_values"],
            "sources": sources}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--uni2ts-yaml", required=True,
                   help="uni2ts cli/conf/pretrain/data/lotsa_v1_weighted.yaml")
    p.add_argument("--uni2ts-commit", required=True,
                   help="The uni2ts commit the yaml comes from.")
    p.add_argument("--revision", default=None,
                   help="Dataset commit. Default: the current main.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="src/gift_pretrain_index.json.gz")
    return p.parse_args(argv)


def main():
    args = parse_args()
    index = build_index(args)
    with gzip.open(args.out, "wt") as f:
        json.dump(index, f, separators=(",", ":"), sort_keys=True)
    n = sum(len(s["batches"]) for s in index["sources"].values())
    print(f"{len(index['sources'])} sources, {n} record batches -> {args.out}")


if __name__ == "__main__":
    main()
