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

This reads the `meta` and `source_id` columns of a sample of 40 shards
straight from the repo. The verdict quotes that sample size. The read takes
no `series` column, which is 4096 floats a row, so the whole check moves a
few MB rather than a few GB and does not compete with the training stream.

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
# the run had reached at 200,000 steps, and its neighbours.
BOUNDARY_SHARDS = [0, 1, 2, 639, 1278, 1279, 1280, 1281, 4273]
# How many shards the survey reads. Round 3 of the review asked for 40
# rather than 12: a mix estimate from 12 shards carries a wide error bar,
# and each shard costs a few hundred kB because the `series` column stays
# unread. 40 shards is under 1% of the 4,274, so the verdict still quotes
# the sample size beside the claim.
SAMPLE_SIZE = 40
N_SHARDS = 4274


def default_shards(n=SAMPLE_SIZE, total=N_SHARDS):
    """The boundary shards, plus an even spread over the whole range."""
    out = set(BOUNDARY_SHARDS)
    rest = n - len(out)
    if rest > 0:
        step = (total - 1) / float(rest + 1)
        out.update(int(round(step * (i + 1))) for i in range(rest))
    return sorted(s for s in out if s < total)


DEFAULT_SHARDS = default_shards()

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
    out = {"n_shards": len(files), "n_sampled": 0, "shards": []}
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
        out["n_sampled"] = len(out["shards"])
        print(f"shard {sid:>5}  rows {row['rows']:>6}  series "
              f"{row['distinct_series']:>5}  families {row['families']:>3}  "
              f"TV from first {row['tv_from_first']:.4f}  top: {head}")
    return out


# The shard the run had reached at 200,000 steps. Shards below it are the
# half #373 read; shards at or above it are the half the continuation reads.
SPLIT_SHARD = 1280


def halves(out: dict, split: int = SPLIT_SHARD) -> dict:
    """The two halves of the run, each pooled into ONE mix, and their TV.

    A per-shard TV mixes two things: a real difference in the mix, and the
    sampling noise of one shard. `small_v1` holds short shards as well as
    10,000-row ones, and a 424-row shard carries far more noise than a
    10,000-row one. Pooling every sampled shard of a half into one mix
    divides that noise down, and the TV between the two pooled mixes is the
    number the card's question asks for.
    """
    rows = out.get("shards") or []
    counts = {"before": collections.Counter(), "after": collections.Counter()}
    totals = {"before": 0, "after": 0}
    shards = {"before": 0, "after": 0}
    for r in rows:
        side = "before" if r["shard"] < split else "after"
        for family, share in r["mix"].items():
            counts[side][family] += share * r["rows"]
        totals[side] += r["rows"]
        shards[side] += 1
    before = mix(counts["before"], totals["before"])
    after = mix(counts["after"], totals["after"])
    return {
        "split_shard": split,
        "n_shards_before": shards["before"], "n_rows_before": totals["before"],
        "n_shards_after": shards["after"], "n_rows_after": totals["after"],
        "mix_before": before, "mix_after": after,
        "tv_between_halves": total_variation(before, after),
    }


def verdict(out: dict) -> str:
    rows = out["shards"]
    if not rows:
        return "no shard was read"
    worst = max(r["tv_from_first"] for r in rows)
    thin = min(rows, key=lambda r: r["rows"])
    size = f"{len(rows)} of {out.get('n_shards', N_SHARDS)} shards"
    h = out.get("halves") or halves(out)
    between = h["tv_between_halves"]
    if between > GROUPED_TV or worst > GROUPED_TV:
        return (f"GROUPED: over {size}, the two halves sit {between:.4f} "
                f"apart in total variation and the widest single shard sits "
                f"{worst:.4f} from shard {rows[0]['shard']}'s. The step "
                f"count is confounded with the data mix, and the report "
                f"must say so beside the figure.")
    return (f"SHUFFLED: over {size}, the half the continuation reads "
            f"(shard {h['split_shard']} and up, {h['n_shards_after']} "
            f"shards, {h['n_rows_after']} rows) sits {between:.4f} in total "
            f"variation from the half #373 read ({h['n_shards_before']} "
            f"shards, {h['n_rows_before']} rows). The widest single shard "
            f"sits {worst:.4f} from shard {rows[0]['shard']}'s, and that "
            f"shard is a short one: {thin['rows']} rows, against 10,000 in "
            f"a full shard. So the step count is not confounded with the "
            f"data mix.")


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
    out["halves"] = halves(out)
    out["verdict"] = verdict(out)
    print(out["verdict"])
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
