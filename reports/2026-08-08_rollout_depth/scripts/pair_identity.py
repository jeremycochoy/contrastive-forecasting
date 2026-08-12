#!/usr/bin/env python3
"""#373 — do two cells of the same arm hold the same student encoder?

A1 and B3 scored the same number on the student head at both stops, 1.1305
at bb40k and 1.1676 at bb100k, while their teacher heads scored apart. The
two cells run the same arm and differ only in the EMA regime, so a shared
number reads like a path bug: one head or one eval keyed on the arm and not
on the cell.

This script tests the checkpoints instead of the paths. It loads both
backbones of a pair and compares every tensor, split into the two sides of
the model:

    student   encoder, transformer, channel mixing, embeddings
    teacher   every tensor whose name starts with `teacher_`

The student head reads the student side only. So if the student side is
identical tensor for tensor, one score for both cells is the right answer
and not a collision, and the teacher side is where the regime shows.

Usage:  python3 pair_identity.py [--out results/pair_identity.tsv]
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
RES = os.path.join(STUDY, "results")
R2 = os.environ.get("CF373_R2", "/home/jupyter/cf373_r2")

# The four pairs that run one arm under both EMA regimes. Group A schedules
# the EMA (tau 0.9 -> 1.0 over 100k steps); group B holds alpha at 0.9.
PAIRS = [("A1", "B3", "arm5_combab_alignS"),
         ("A4", "B1", "arm6_v2_combab_alignS"),
         ("A3", "B2", "arm6_v2_combab_alignT"),
         ("A2", "B8", "arm6_v2_nse_alignT")]
STOPS = [40, 100, 200]


def ckpt(cell, stop):
    """The cell's backbone at that stop, under the round-2 sync root."""
    base = os.path.join(R2, cell, "sync")
    if not os.path.isdir(base):
        return None
    hits = []
    for root, _dirs, files in os.walk(base):
        if os.sep + "eval" + os.sep in root + os.sep:
            continue
        for f in files:
            if f.endswith(f"_{stop}k.pth") and "optimizer" not in f:
                hits.append(os.path.join(root, f))
    return sorted(hits)[0] if hits else None


def side(key):
    return "teacher" if key.startswith("teacher_") else "student"


def compare(pa, pb):
    """(n, identical, max abs diff) per side, over the shared keys."""
    A = torch.load(pa, map_location="cpu", weights_only=False)
    B = torch.load(pb, map_location="cpu", weights_only=False)
    out = {"student": [0, 0, 0.0], "teacher": [0, 0, 0.0]}
    for k in sorted(set(A) & set(B)):
        ta, tb = A[k], B[k]
        if not torch.is_tensor(ta) or not torch.is_tensor(tb):
            continue
        s = out[side(k)]
        s[0] += 1
        if ta.shape == tb.shape:
            if torch.equal(ta, tb):
                s[1] += 1
            if ta.is_floating_point():
                s[2] = max(s[2], (ta.float() - tb.float()).abs().max().item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RES, "pair_identity.tsv"))
    a = ap.parse_args()

    rows = [("pair", "arm", "stop_k", "side", "tensors", "identical",
             "max_abs_diff", "verdict")]
    for ca, cb, arm in PAIRS:
        for stop in STOPS:
            pa, pb = ckpt(ca, stop), ckpt(cb, stop)
            if not pa or not pb:
                continue
            res = compare(pa, pb)
            for sd in ("student", "teacher"):
                n, eq, d = res[sd]
                if n == 0:
                    continue
                v = "IDENTICAL" if eq == n else "differs"
                rows.append((f"{ca}/{cb}", arm, str(stop), sd, str(n),
                             str(eq), f"{d:.3e}", v))
                print(f"{ca}/{cb} {arm} bb{stop}k {sd:8s} "
                      f"{eq}/{n} identical  max|diff|={d:.3e}  {v}", flush=True)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
