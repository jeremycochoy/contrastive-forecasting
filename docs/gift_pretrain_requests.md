# Hub requests of the GiftEvalPretrain stream

The Hub limits the requests of one token. A PRO account gets 12,000
"resolver" requests in a fixed window of 5 minutes. Each byte range that
`src/gift_pretrain.py` reads is one such request. All the runs share one
token, so all their streams share this limit.

Two changes cut the requests:

- The disk cache. It is on by default and it changes no window: for one seed,
  the stream gives the same windows, ids and labels as before.
- The bigger reads. They are an opt-in, and they change the stream.

## What a read costs

| Read | Before | Now |
|---|---|---|
| A small source at the start of a process (66 files) | 2 requests a file | 1 request a file on a cold machine, 0 on a warm one |
| A chunk of part of a record batch | once in a process: the header of the batch and the schema of the file. For each float column: the row offsets and the values | the values of each float column |
| A chunk of whole record batches | the same | 1 request |
| A column with nulls | 1 more request | 1 more request |

A "warm" machine has the layout of the record batch in its cache: the
header fields and the row offsets. A "cold" machine reads them once, in one
request, the first time a process needs them. The samples of the real record
batches held no nulls.

## The rates

A process that serves 3 million windows (`scripts/gift_pretrain_requests.py`
on the shipped index):

| | Before | Now, cold machine | Now, warm machine |
|---|---|---|---|
| Requests at the start of a process | 580 | 300 | 138 |
| Requests for each 1,000 windows | 3.29 | 1.74 | 1.44 |

A trainer at 2 steps a second of 256 windows serves 154,000 windows in 5
minutes. A head trainer at 10 steps a second serves 768,000. From these
rates, ten trainers and two head trainers make about 10,100 requests in 5
minutes with the loader of before. Now they make about 4,400. The start of
each process adds its requests.

## The disk cache

- The folder is `$GIFT_PRETRAIN_CACHE`, else `~/.cache/gift_pretrain`, with
  one subfolder for each dataset revision. `GIFT_PRETRAIN_CACHE=off` stops
  the cache. A stream on a local copy of the dataset (`root`) uses no cache.
- It holds the 66 files of the small sources (743 MB) and the schemas of the
  86 large sources. It also holds the layouts of the 8,276 record batches
  that the stream cuts into parts, 0.4 to 3.6 KB each.
- Every process of the machine uses it. A write goes to a temporary file,
  and a rename puts it in place. A lock lets one process download a file of
  a small source while the others wait for it.
- The stream reads a damaged entry again. When the disk refuses a write, the
  stream logs it once and reads the Hub.

### Fill the cache before a run

```bash
python3 scripts/build_gift_pretrain_cache.py --dry-run   # counts, no request
python3 scripts/build_gift_pretrain_cache.py --rate 5    # at most 5 a second
```

An empty cache needs about 8,800 requests: 8,276 layouts, 86 schemas and 66
files, 29 minutes at 5 requests a second. Choose a rate that leaves room for
the runs. A second run reads only what the cache does not hold. To use the
cache on another machine, copy the folder of the revision.

## The bigger reads (an opt-in)

- `GIFT_PRETRAIN_CHUNK_MB` sets the size of a chunk. It also joins
  consecutive record batches of one file into one chunk while that brings the
  chunk nearer to this size. A chunk of whole record batches costs one
  request.
- `GIFT_PRETRAIN_POOL_MB` sets `POOL_BYTES`, the memory of the chunks that a
  large source keeps (default 48).

Six sources have record batches of at most 22 MB. Their chunks are whole record
batches, so a larger chunk size alone does not change them. At 16 MB they
make 32 % of the requests of the stream. The join is what cuts them.

| Setting (chunk MB : pool MB) | Requests for each 1,000 windows, warm | Chunk memory of a process | Distinct chunks in 4,096 windows | Windows for each chunk in them | Pairs of a batch from one chunk | Pairs of a batch from one row |
|---|---|---|---|---|---|---|
| unset (the stream of before) | 1.44 | 1.36 GB | 94.7 | 35.9 | 1.91 % | 0.125 % |
| 16 : 48 | 1.13 | 1.41 GB | 62.1 | 54.7 | 2.00 % | 0.126 % |
| 32 : 48 | 0.60 | 1.50 GB | 27.5 | 124 | 4.24 % | 0.157 % |
| 32 : 96 | 0.59 | 2.92 GB | 66.5 | 51.1 | 1.81 % | 0.090 % |
| 64 : 48 | 0.24 | 2.97 GB | 25.7 | 132 | 4.77 % | 0.100 % |
| 64 : 192 | 0.23 | 5.48 GB | 58.1 | 58.5 | 2.03 % | 0.070 % |

- The chunk memory does not include the small sources, 743 MB in each process.
- The 4,096 windows are those of the shuffle buffer (`SHUFFLE_WINDOWS`),
  before the shuffle. The columns count the windows of the large sources
  only.
- A batch has 256 windows. A pair from one row is two windows of one series.
  They have two crops, or the same values for a series of at most 1,024
  values.
- With the pool of before, a larger chunk keeps one chunk of each large
  source in memory. A pool that grows with the chunk keeps the share of pairs
  from one chunk near the value of before. It also holds more rows, so fewer
  pairs come from one row. Its cost is the memory.

Example, for the runs that start after it:

```bash
export GIFT_PRETRAIN_CHUNK_MB=32 GIFT_PRETRAIN_POOL_MB=96
```

The stream logs its settings at its start when the cache or the bigger reads
are on:
`[gift_pretrain] chunks of 32 MB, small record batches joined, pools of 96 MB,
disk cache ...`.

## How the numbers were measured

- `scripts/gift_pretrain_requests.py` runs the sampler of the stream on the
  shipped index, with no data and no network. It counts the chunks that the
  stream loads and the requests of each one. It gives each row of a record
  batch the mean row length of that batch.
- `tests/test_gift_pretrain_requests.py` counts the requests of the loader
  on a small corpus through a fake Hub. It also compares the windows, the ids
  and the labels with those of the loader before.
- On the Hub, the new loader read the same rows as the loader before:

  | Read | Requests before | Requests now |
  |---|---|---|
  | A part of a batch with covariates | 6 | 2 |
  | A part of an ERA5 batch | 4 | 1 |
  | A whole small batch | 4 | 1 |
  | 12 joined small batches | 35 | 1 |
  | The file of a small source | 2 | 1 cold, 0 warm |

  In each of the 25 large sources, one read of 128 KB held the header and the
  row offsets of a batch.
