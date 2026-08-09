#!/usr/bin/env python3
"""#373 — summarise steptime.sh's windows into the overhead number.

Drops window 1 of every run (CUDA context, autotune, stream fill) and takes
the median over what is left, per k. Median rather than mean: elisa's cards
carry another session's training, so a window that lands next to a burst of
somebody else's work is an outlier, not a measurement.

Usage: steptime_summary.py <steptime_<cell>.csv>
"""
import csv
import statistics
import sys


def main(argv):
    if len(argv) != 2:
        raise SystemExit(__doc__)
    with open(argv[1], newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["window"]) > 1]
    if not rows:
        raise SystemExit("ABORT: no post-warm-up window in " + argv[1])

    fields = ("data_ms", "fwd_ms", "bwd_ms", "total_ms", "sps")
    by_k, out = {}, []
    for r in rows:
        by_k.setdefault(int(r["k"]), []).append(r)

    for k in sorted(by_k):
        rs = by_k[k]
        med = {f: statistics.median(float(r[f]) for r in rs) for f in fields}
        med["compute_ms"] = med["fwd_ms"] + med["bwd_ms"]
        med["n"] = len(rs)
        by_k[k] = med
        out.append(k)

    hdr = f"{'k':>3} {'n':>3} {'data':>8} {'fwd':>8} {'bwd':>8} " \
          f"{'fwd+bwd':>9} {'total':>9} {'sps':>6}"
    print("\nGPU step time, median over post-warm-up windows (ms/step)")
    print(hdr)
    print("-" * len(hdr))
    for k in out:
        m = by_k[k]
        print(f"{k:>3} {m['n']:>3} {m['data_ms']:>8.1f} {m['fwd_ms']:>8.1f} "
              f"{m['bwd_ms']:>8.1f} {m['compute_ms']:>9.1f} "
              f"{m['total_ms']:>9.1f} {m['sps']:>6.2f}")

    if 0 in by_k and 3 in by_k:
        a, b = by_k[0], by_k[3]
        print("\nk=3 against k=0:")
        for label, key in (("fwd+bwd", "compute_ms"), ("total", "total_ms")):
            d = b[key] / a[key] - 1.0
            print(f"  {label:<8} {a[key]:7.1f} -> {b[key]:7.1f} ms "
                  f"({d * 100:+.1f}%)")
        thr = a["sps"] / b["sps"] - 1.0
        print(f"  {'sps':<8} {a['sps']:7.2f} -> {b['sps']:7.2f} "
              f"({-thr / (1 + thr) * 100:+.1f}% throughput, "
              f"{thr * 100:+.1f}% wall clock per step)")
        mem = {k: max(float(r["gpu_mem_mib"]) for r in
                      [x for x in rows if int(x["k"]) == k]) for k in (0, 3)}
        print(f"  {'GPU MiB':<8} {mem[0]:7.0f} -> {mem[3]:7.0f} "
              "(whole card, this run plus its neighbour)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
