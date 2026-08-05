#!/usr/bin/env python3
"""Split the 97 GIFT-Eval configs into cost-balanced shards.

One GIFT-Eval is 97 configs run one after another, and their costs span
four orders of magnitude: `m_dense/D/short` takes 0.4 s and
`electricity/15T/long` takes 1537 s. Twenty configs carry 89% of the
work. Splitting the list evenly by count would leave one shard holding
`electricity` and the rest idle, so the split is by measured cost.

The costs come from a completed 97-config eval of the same B4 protocol
(`results/config_costs.csv`, one column per config). They are only used
as weights — a shard's contents are exact, and a mis-weighted config
costs balance, never correctness.

`eval_gift_eval_official.py --config-filter` matches its regex against
the RAW config identifier (`SZ_TAXI/15T/short`), not the pretty name it
prints (`sz_taxi/15T/short`). Emitting an anchored alternation of raw
names is why this is a script and not a `sed` line: a hand-written regex
that silently matches 0 of 97 configs produces an eval that "succeeds"
in two seconds with an empty summary.

Usage:
    shard_configs.py --shards 4 --shard 0     # regex for one shard
    shard_configs.py --shards 4 --plan        # the balance, all shards
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COSTS_CSV = os.path.join(os.path.dirname(HERE), "results", "config_costs.csv")

N_CONFIGS = 97


def read_costs(path: str) -> list[tuple[str, float]]:
    """[(raw config, weight)], heaviest first.

    A missing or short table is fatal rather than defaulted: an unweighted
    split would put `electricity` and `loop_seattle` in one shard and the
    eval would take as long as it did unsharded, which is the failure this
    exists to prevent and would otherwise pass unnoticed.
    """
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != N_CONFIGS:
        raise SystemExit(f"{path}: {len(rows)} configs, want {N_CONFIGS}")
    out = [(r["raw_config"], float(r["seconds"])) for r in rows]
    return sorted(out, key=lambda r: -r[1])


def plan(costs: list[tuple[str, float]], shards: int) -> list[list[str]]:
    """Longest-processing-time-first packing into `shards` bins.

    LPT is within 4/3 of optimal, and on this distribution it does much
    better: the largest single config is 1537 s against an ideal shard of
    3326 s at 4 shards, so no bin is forced long by one item.
    """
    bins: list[list[str]] = [[] for _ in range(shards)]
    loads = [0.0] * shards
    for name, weight in costs:
        i = loads.index(min(loads))
        bins[i].append(name)
        loads[i] += weight
    return bins


def regex_for(names: list[str]) -> str:
    """An anchored alternation, so a shard matches its configs and no others.

    Anchoring matters: `M_DENSE/H/short` is a substring of nothing here,
    but `ETT1/15T/short` and `ETT1/15T/short_x` would both match an
    unanchored `ETT1/15T/short`, and a config evaluated in two shards is
    counted twice in the merge.
    """
    return "^(?:" + "|".join(re.escape(n) for n in names) + ")$"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--shard", type=int, default=None,
                   help="Print this shard's --config-filter regex.")
    p.add_argument("--plan", action="store_true",
                   help="Print the per-shard config count and weight.")
    p.add_argument("--costs", default=COSTS_CSV)
    args = p.parse_args()

    if args.shards < 1:
        raise SystemExit("--shards must be >= 1")
    costs = read_costs(args.costs)
    weight = dict(costs)
    bins = plan(costs, args.shards)

    covered = sorted(n for b in bins for n in b)
    if len(covered) != N_CONFIGS or len(set(covered)) != N_CONFIGS:
        raise SystemExit(f"split covers {len(set(covered))} configs, "
                         f"want {N_CONFIGS}")

    if args.plan:
        total = sum(weight.values())
        for i, b in enumerate(bins):
            w = sum(weight[n] for n in b)
            print(f"shard {i}: {len(b):3d} configs  {w:8.1f} s  "
                  f"{100 * w / total:5.1f}%")
        print(f"total:     {N_CONFIGS} configs  {total:8.1f} s  "
              f"ideal shard {total / args.shards:.1f} s")
        return

    if args.shard is None:
        raise SystemExit("give --shard N or --plan")
    if not 0 <= args.shard < args.shards:
        raise SystemExit(f"--shard {args.shard} outside 0..{args.shards - 1}")
    sys.stdout.write(regex_for(bins[args.shard]) + "\n")


if __name__ == "__main__":
    main()
