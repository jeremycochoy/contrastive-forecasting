#!/usr/bin/env python3
"""What zero left padding does to the EWMA normaliser (#419).

A 20-point yearly series at a level of 5,000 (a trend of +40 a year and noise
of 50), padded with 1,004 zeros to the 1,024-value window, normalised by
RevEWMNorm at the #415 cell's span of 128: once by the plain normaliser, once
with ``skip_leading_zeros``, and once without padding at all.

Writes results/padding_norm.tsv: the 20 real positions and the three
normalised values of each.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from src.norm import RevEWMNorm  # noqa: E402

T, N, SPAN, W = 1024, 20, 128, 16


def the_series():
    g = torch.Generator().manual_seed(0)
    trend = 5000 + 40 * torch.arange(N, dtype=torch.float32)
    return (trend + 50 * torch.randn(N, generator=g)).view(1, N, 1)


def normalised(x, skip):
    return RevEWMNorm(1, span=SPAN, patch_size=W, skip_leading_zeros=skip)(x, "norm")


def main():
    s = the_series()
    x = torch.cat([torch.zeros(1, T - N, 1), s], dim=1)
    plain = normalised(x, False)[0, T - N:, 0]
    skip = normalised(x, True)[0, T - N:, 0]
    alone = normalised(s, False)[0, :, 0]
    out = os.path.join(HERE, "..", "results", "padding_norm.tsv")
    with open(out, "w") as fh:
        fh.write("step\tvalue\tplain_padded\tskip_padded\tunpadded\n")
        for i in range(N):
            fh.write(f"{i}\t{s[0, i, 0]:.1f}\t{plain[i]:.3f}\t{skip[i]:.3f}\t{alone[i]:.3f}\n")
    print(open(out).read())


if __name__ == "__main__":
    main()
