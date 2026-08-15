#!/usr/bin/env python3
"""#373 — the head FILES behind a same-arm pair, by path and md5.

`pair_identity.py` answers whether two cells hold the same weights. It does
not answer the question the card asked first: is one head file, or one eval
directory, shared between two cells? A shared path would make two "cells" one
measurement by accident, and the tensor comparison could not tell the two
cases apart.

This script prints the resolved path and the file md5 for each side of a
pair. Read it beside `pair_identity.tsv`:

    different path + different md5 + identical tensors
        two heads were trained separately, from separate directories, and
        came out the same. The equal score is the measurement, not a bug.

    same path (or same md5)
        one file serving two cells. That IS the path bug, and the head and
        the eval must be re-run into cell-id paths.

Usage: python3 pair_head_files.py [--out results/pair_head_files.tsv]
"""
import argparse
import hashlib
import os

import pair_identity as P

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

PAIRS = [
    ("A1", "B3", "arm5_combab_alignS"),
    ("A4", "B1", "arm6_v2_combab_alignS"),
    ("A3", "B2", "arm6_v2_combab_alignT"),
    ("A2", "B8", "arm6_v2_nse_alignT"),
]
STOPS = (40, 100)
ENCS = ("student", "teacher")


def md5(path):
    if not path or not os.path.isfile(path):
        return "-"
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RES, "pair_head_files.tsv"))
    args = ap.parse_args()

    rows = [("pair", "arm", "stop_k", "enc", "cell", "head_md5", "head_path")]
    for ca, cb, arm in PAIRS:
        for stop in STOPS:
            for enc in ENCS:
                paths = {c: P.head_ckpt(c, stop, enc) for c in (ca, cb)}
                if not any(paths.values()):
                    continue
                sums = {c: md5(p) for c, p in paths.items()}
                for c in (ca, cb):
                    rows.append((f"{ca}/{cb}", arm, str(stop), enc, c,
                                 sums[c][:8], paths[c] or "-"))
                shared = (paths[ca] == paths[cb]) or (
                    sums[ca] != "-" and sums[ca] == sums[cb])
                verdict = "SHARED FILE — path bug" if shared else "separate files"
                print(f"{ca}/{cb} bb{stop}k {enc:8s} {verdict}: "
                      f"{ca} {sums[ca][:8]} vs {cb} {sums[cb][:8]}")

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
