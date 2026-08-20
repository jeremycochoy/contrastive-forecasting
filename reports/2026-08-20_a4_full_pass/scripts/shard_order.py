#!/usr/bin/env python3
"""#407 review gap 3 — is the step count confounded with the data mix?

The dataloader walks `small_v1`'s shards in order. At batch 64 and 10,000
rows a shard, steps 0 to 200,000 read shards 0 to 1279 and steps 200,000 to
665,000 read shards 1280 to 4273. The two halves of the card's curve
therefore come from DIFFERENT shards, not from more of the same shards.

That matters only if the shards are grouped by source dataset. If they are,
the second half of the curve measures a change in the data mix, and the
report must not call it "more data helps".

`small_v1/manifest.json` does not answer it: it holds `total_rows`,
`num_shards` and one `source_counts` entry, `{"0": 42571692}`, so every row
carries the same source id and the manifest sees one source. The shard
files carry a `meta` string per row, and that string names the SERIES, not
the dataset: `1990_53_79`, `comstock_amy2018_midwest_G17003411_70027`,
`load_power_unit_198`. So the dataset is read off the family stem of that
name, which is the leading alphabetic token, and every purely numeric name
falls into one family.

This reads the `meta` and `source_id` columns of a sample of shards
straight from the repo. It reads no `series` column, which is 4096 floats a
row, so the whole check moves a few MB rather than a few GB and does not
compete with the training stream.

Reported per shard: the family mix, and the total variation distance from
the FIRST sampled shard's mix. Then the verdict:

  SHUFFLED  every sampled shard holds the same family mix. The step count
            is not confounded with the mix.
  GROUPED   the mix moves along the shard index. The report must state the
            confound beside the figure.

Usage:
  shard_order.py [--shards 0 1 1279 1280 ...] [--json OUT]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

BATCH_SIZE = 64
ROWS_PER_SHARD = 10_000
REPO = "datasets/jeremycochoy/gift-pretrain-full-4096/small_v1"

# The shards the card's boundary makes interesting: the two ends, the shard
# the run had reached at 200,000 steps, its neighbours, and a spread over
# the half the continuation reads.
DEFAULT_SHARDS = [0, 1, 2, 639, 1278, 1279, 1280, 1281, 1900, 2560,
                  3200, 4273]

# Total variation distance between two shards' family mixes. Half the sum
# of the absolute share differences, so it runs 0 (same mix) to 1 (no
# family in common). Two draws of 10,000 rows from one mix sit near 0.01,
# so this threshold is far above the sampling noise and far below a real
# regrouping.
GROUPED_TV = 0.10


def shard_for_step(step: int) -> int:
    """Which shard the run is reading at `step`."""
    return step * BATCH_SIZE // ROWS_PER_SHARD


def read_labels(fs, path):
    """The `meta` and `source_id` columns of one shard. No `series` column."""
    import pyarrow.parquet as pq
    with fs.open(path, "rb") as fh:
        table = pq.ParquetFile(fh).read(columns=["meta", "source_id"])
    meta = [str(v) for v in table.column("meta").to_pylist()]
    source = [int(v) for v in table.column("source_id").to_pylist()]
    return meta, source


def label_of(meta: str) -> str:
    """The dataset family a row belongs to, out of its `meta` string.

    The stem is the leading alphabetic token: `comstock_amy2018_...` gives
    `comstock_amy`, `load_power_unit_198` gives `load_power_unit`. A name
    that starts with a digit carries no stem, so every one of those goes
    into `numeric_id` together. That is the coarsest possible grouping and
    it is the safe direction: a grouping error here can only HIDE a mix
    change inside `numeric_id`, and `distinct_meta` catches a shard that
    holds one series repeated.
    """
    if not meta or meta[0].isdigit():
        return "numeric_id"
    token = re.match(r"[A-Za-z][A-Za-z_]*", meta)
    return token.group(0).rstrip("_") if token else "other"


def mix(counts, total):
    """A family count table as shares."""
    return {k: n / total for k, n in counts.items()} if total else {}


def total_variation(a: dict, b: dict) -> float:
    """Half the sum of |share difference| over the union of two mixes."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def survey(shard_ids, fs=None):
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem() if fs is None else fs
    files = sorted(f for f in fs.ls(REPO, detail=False)
                   if f.endswith(".parquet"))
    out = {"n_shards": len(files), "shards": []}
    for sid in shard_ids:
        if sid >= len(files):
            continue
        meta, source = read_labels(fs, files[sid])
        counts = collections.Counter(label_of(m) for m in meta)
        shares = mix(counts, len(meta))
        reference = out["shards"][0]["mix"] if out["shards"] else shares
        top = counts.most_common(5)
        out["shards"].append({
            "shard": sid,
            "file": files[sid].split("/")[-1],
            "rows": len(meta),
            "distinct_series": len(set(meta)),
            "families": len(counts),
            "distinct_source_id": len(set(source)),
            "mix": shares,
            "tv_from_first": total_variation(shares, reference),
            "top": [[k, n] for k, n in top],
        })
        row = out["shards"][-1]
        head = ", ".join("%s %.1f%%" % (k, n / row["rows"] * 100)
                         for k, n in top[:3])
        print(f"shard {sid:>5}  rows {row['rows']:>6}  series "
              f"{row['distinct_series']:>5}  families {row['families']:>3}  "
              f"TV from first {row['tv_from_first']:.4f}  top: {head}")
    return out


def verdict(out: dict) -> str:
    rows = out["shards"]
    if not rows:
        return "no shard was read"
    worst = max(r["tv_from_first"] for r in rows)
    fewest = min(r["distinct_series"] for r in rows)
    if worst > GROUPED_TV:
        return (f"GROUPED: one sampled shard's family mix sits {worst:.3f} "
                f"in total variation from shard {rows[0]['shard']}'s. The "
                f"step count is confounded with the data mix, and the "
                f"report must say so beside the figure.")
    return (f"SHUFFLED: the family mix moves by at most {worst:.4f} in total "
            f"variation across the sampled shards, and the thinnest shard "
            f"still holds {fewest} distinct series. The half the "
            f"continuation reads carries the same mix as the half #373 "
            f"read, so the step count is not confounded with the data mix.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="*", default=DEFAULT_SHARDS)
    ap.add_argument("--json")
    a = ap.parse_args(argv)

    if not os.environ.get("HF_TOKEN") and \
            not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("WARNING: no HF token, so the read may be rate limited",
              file=sys.stderr)

    for step in (200_000, 300_000, 450_000, 665_000):
        print(f"step {step:>7} reads shard {shard_for_step(step)}")
    out = survey(a.shards)
    out["verdict"] = verdict(out)
    print(out["verdict"])
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
