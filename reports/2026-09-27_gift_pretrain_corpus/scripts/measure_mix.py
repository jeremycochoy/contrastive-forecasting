#!/usr/bin/env python3
"""The mix of windows the GiftEvalPretrain stream feeds the trainer (#419).

Runs the stream the #415 cell reads at batch 256 (254 real rows, C = 1) on
the CPU, for --batches batches, and writes three tables:

    mix_family.tsv   windows per source family: measured, expected, bytes
    mix_freq.tsv     windows per frequency class (v2), measured and expected
    mix_summary.json the rates: batches per second, padded windows, padding

The expected share of a family is its uni2ts draw probability times the mean
number of windows a draw gives (uni2ts SampleDimension: (D + 1) / 2 per
field), normalised. The measured share can fall below it only where windows
hold no observed value and are skipped.

Usage (on the box, CPU only):
    CUDA_VISIBLE_DEVICES= python3 measure_mix.py --batches 2000 --out results
"""

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from src import gift_pretrain as gp  # noqa: E402
from src.freq_embedding import FREQ_NAMES_V2, freq_to_id  # noqa: E402
from src.norm import leading_zero_count  # noqa: E402


def expected_shares(families, with_covariates=True):
    """Share of windows per family, from the draw probabilities."""
    probs = gp.family_probabilities(families)
    per_draw = []
    for f in families:
        dt, dc = f["dims"]
        dc = dc if with_covariates else 0
        caps = [min(d, gp.MAX_DIM * d // (dt + dc)) for d in (dt, dc) if d]
        per_draw.append(sum((c + 1) / 2 for c in caps))
    w = probs * np.array(per_draw)
    return w / w.sum()


def run(stream, batches):
    """Iterate the stream; count windows per family and the padding."""
    padded = pad_values = 0
    it = iter(stream)
    t0 = time.time()
    for i in range(batches):
        x, _, _ = next(it)
        if i == 9:  # the rate leaves out the start-up (the first fetches)
            t0 = time.time()
        z = leading_zero_count(x)[:, 0, 0]
        padded += int((z > 0).sum())
        pad_values += int(z.sum())
    rate = (batches - 10) / (time.time() - t0) if batches > 10 else float("nan")
    # Stop the prefetch thread before the caller reads stream.counts: it
    # counts each batch it fetches, up to `prefetch` batches ahead.
    it.close()
    return rate, padded, pad_values


def family_rows(families, counts, expected):
    total = sum(counts.values())
    for k, f in enumerate(families):
        yield (f["name"], f["freq"], FREQ_NAMES_V2[freq_to_id(f["freq"], "v2")],
               counts.get(k, 0), counts.get(k, 0) / total, expected[k],
               f["bytes"])


def write_tables(out, families, counts, expected):
    rows = list(family_rows(families, counts, expected))
    with open(os.path.join(out, "mix_family.tsv"), "w") as fh:
        fh.write("family\tfreq\tclass\twindows\tshare\texpected\tbytes\n")
        for r in sorted(rows, key=lambda r: -r[5]):
            fh.write("%s\t%s\t%s\t%d\t%.6f\t%.6f\t%d\n" % r)
    by = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for r in rows:
        by[r[2]][0] += r[3]
        by[r[2]][1] += r[4]
        by[r[2]][2] += r[5]
    with open(os.path.join(out, "mix_freq.tsv"), "w") as fh:
        fh.write("class\twindows\tshare\texpected\n")
        for name, (n, s, e) in sorted(by.items(), key=lambda kv: -kv[1][2]):
            fh.write("%s\t%d\t%.6f\t%.6f\n" % (name, n, s, e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=254)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--out", default=os.path.join(HERE, "..", "results"))
    args = p.parse_args()
    stream = gp.GiftPretrainStream(batch_size=args.batch_size, C=1,
                                   seed=[args.seed, 0], freq_vocab="v2")
    rate, padded, pad_values = run(stream, args.batches)
    windows = args.batches * args.batch_size
    os.makedirs(args.out, exist_ok=True)
    write_tables(args.out, stream.families, stream.counts,
                 expected_shares(stream.families))
    summary = {"batches": args.batches, "batch_size": args.batch_size,
               "batches_per_second": rate, "windows": windows,
               "padded_windows": padded / windows,
               "padded_values": pad_values / (windows * 1024)}
    json.dump(summary, open(os.path.join(args.out, "mix_summary.json"), "w"),
              indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
