#!/usr/bin/env python3
"""Resident memory of the GiftEvalPretrain stream at training speed (#419).

Reads the stream the #415 cell reads (254 real rows a batch, C = 1) at
``rate`` batches a second for ``minutes`` minutes, and prints the resident
size every 30 seconds. A flat line means the pools hold their budget; a
climbing one means memory the stream does not give back.

Usage (on the box, CPU only):
    CUDA_VISIBLE_DEVICES= python3 memory_probe.py 2.5 12
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from src import gift_pretrain as gp  # noqa: E402


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024


def consume(it, rate, minutes):
    """Read ``rate`` batches a second; print the resident size each 30 s."""
    t0 = last = time.time()
    n = 0
    while time.time() - t0 < minutes * 60:
        next(it)
        n += 1
        time.sleep(max(0.0, t0 + n / rate - time.time()))
        if time.time() - last >= 30:
            last = time.time()
            print(f"t={last - t0:5.0f}s batches={n} rate={n / (last - t0):.2f}/s "
                  f"rss={rss_mb():.0f}MB", flush=True)


def main():
    rate, minutes = float(sys.argv[1]), float(sys.argv[2])
    it = iter(gp.GiftPretrainStream(batch_size=254, C=1, seed=[7, 0],
                                    freq_vocab="v2"))
    next(it)
    print(f"start rss={rss_mb():.0f}MB", flush=True)
    consume(it, rate, minutes)


if __name__ == "__main__":
    main()
