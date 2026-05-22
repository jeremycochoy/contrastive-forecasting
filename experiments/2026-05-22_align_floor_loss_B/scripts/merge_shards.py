#!/usr/bin/env python3
"""Merge sharded GIFT-Eval outputs into one canonical summary.txt +
all_results.csv, recomputing the aggregate GM-Relative MASE over the
union of configs. Also derives the triage-11 GM by applying the same
triage regex #309 used (so derive-from-full == a separate triage run).

  merge_shards.py <out_dir> <shard_dir1> <shard_dir2> ...

Per-config relative MASE is computed per config independently of which
shard ran it, so the union's geometric mean is exact.
"""
import csv, math, os, re, sys

TRIAGE = (r'bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|'
          r'ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|'
          r'us_births/D/short')
LEADERBOARD = [("Sundial", 0.673), ("TimesFM", 0.680), ("PatchTST", 0.762),
               ("Chronos", 0.786), ("Moirai", 0.809), ("Naive", 1.000)]
W = 90


def parse_summary(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 4 or "/" not in p[0]:
                continue
            try:
                rows[p[0]] = (float(p[-3]), float(p[-2]), float(p[-1]))
            except ValueError:
                continue
    return rows


def gm(rels):
    return math.exp(sum(math.log(r) for r in rels) / len(rels))


def main():
    out, shards = sys.argv[1], sys.argv[2:]
    merged = {}
    for d in shards:
        for k, v in parse_summary(os.path.join(d, "summary.txt")).items():
            if k in merged:
                print(f"WARN duplicate config across shards: {k}")
            merged[k] = v
    n = len(merged)
    g = gm([v[2] for v in merged.values()])
    tp = re.compile(TRIAGE)
    trig = [v[2] for k, v in merged.items() if tp.search(k)]
    tg = gm(trig) if trig else float("nan")

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("=" * W + "\n" + "GIFT-Eval Official Results".center(W) + "\n" + "=" * W + "\n")
        f.write(f"{'Config':<48}{'MASE':>8}{'SN_MASE':>9}{'Relative':>11}\n")
        f.write("-" * W + "\n")
        for cfg in sorted(merged):
            mase, sn, rel = merged[cfg]
            f.write(f"{cfg:<48}{mase:>8.4f}{sn:>9.4f}{rel:>11.4f}\n")
        f.write("-" * W + "\n\n")
        f.write(f"Aggregate GM-Relative MASE ({n} configs): {g:.4f}\n\n")
        f.write("Leaderboard comparison:\n")
        for name, val in LEADERBOARD:
            f.write(f"  {name + ':':<11} {val:.3f}\n")
        f.write(f"  ** Ours:    {g:.3f} **\n")
        f.write("=" * W + "\n")

    header, rows = None, []
    for d in shards:
        p = os.path.join(d, "all_results.csv")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            rd = csv.reader(f)
            h = next(rd)
            header = header or h
            rows.extend(r for r in rd)
    if header:
        with open(os.path.join(out, "all_results.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    print(f"MERGED full: {n} configs  GM={g:.4f}")
    print(f"TRIAGE:      {len(trig)} configs  GM={tg:.4f}")
    if n != 97:
        print(f"WARNING: expected 97 full configs, got {n}")


if __name__ == "__main__":
    main()
