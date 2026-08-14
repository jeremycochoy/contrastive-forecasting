#!/usr/bin/env python3
"""#373 — paired dataset-cluster bootstrap on one cell's k = 3 minus k = 0.

The two arms of a cell are evaluated on the SAME 97 configs, so the delta is
paired per config and the pairing must survive the resampling.

The resampling unit is the DATASET, not the config. GIFT-Eval's 97 configs
come from far fewer datasets — `m_dense/H/short`, `m_dense/H/medium` and
`m_dense/H/long` are three configs of one series — and treating them as
independent draws makes the interval too narrow. So a resample draws
datasets with replacement and takes every config of each drawn dataset.

Reported per subset: the observed delta in GM-Relative MASE, the 95%
percentile interval, and the share of resamples on the improving side.

This measures the CONFIG sampling only. It does not measure the head seed
(`ema_sched_ladder.md`'s pooled band is ±0.0384) and it does not measure the spread
between two independent backbone trainings, which no run in this study or
its parents has replicated.

Usage:
  paired_bootstrap.py --k0 <k0 all_results.csv> --k3 <k3 all_results.csv> \\
      --label B5_student [--iters 10000] [--out results/bootstrap.csv]
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
SN_REF = (REPO / "reports" / "2026-07-21_split_pred_rep_small" / "results"
          / "seasonal_naive_all_results.csv")
MASE = "eval_metrics/MASE[0.5]"
TERMS = ("short", "medium", "long")


def read_mase(path):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                v = float(r[MASE])
            except (KeyError, ValueError, TypeError):
                continue
            if v > 0 and math.isfinite(v):
                out[r["dataset"]] = v
    return out


def gm(vals):
    return math.exp(sum(vals) / len(vals)) if vals else float("nan")


def subsets(datasets):
    """`{name: [config, ...]}` — the config groups this reports on."""
    out = {"all": [], "short": [], "medium_long": []}
    for ds in datasets:
        out["all"].append(ds)
        term = ds.rsplit("/", 1)[-1]
        if term == "short":
            out["short"].append(ds)
        elif term in ("medium", "long"):
            out["medium_long"].append(ds)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--k0", required=True)
    p.add_argument("--k3", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--out")
    args = p.parse_args(argv)

    sn = read_mase(SN_REF)
    a, b = read_mase(args.k0), read_mase(args.k3)
    common = sorted(set(a) & set(b) & set(sn))
    if not common:
        raise SystemExit("ABORT: the two arms share no config with the "
                         "seasonal-naive reference")

    # log-ratio against seasonal naive, per config, per arm.
    la = {ds: math.log(a[ds] / sn[ds]) for ds in common}
    lb = {ds: math.log(b[ds] / sn[ds]) for ds in common}

    # Cluster = the dataset the config belongs to, i.e. everything before the
    # final `/<term>`.
    clusters = {}
    for ds in common:
        clusters.setdefault(ds.rsplit("/", 1)[0], []).append(ds)
    keys = sorted(clusters)

    groups = subsets(common)
    rng = random.Random(args.seed)
    rows = []

    for name, members in groups.items():
        member = set(members)
        if not member:
            continue
        obs = gm([lb[d] for d in members]) - gm([la[d] for d in members])

        draws = []
        for _ in range(args.iters):
            pick = [clusters[keys[rng.randrange(len(keys))]]
                    for _ in range(len(keys))]
            sel = [d for grp in pick for d in grp if d in member]
            if not sel:
                continue
            draws.append(gm([lb[d] for d in sel]) - gm([la[d] for d in sel]))
        draws.sort()
        lo = draws[int(0.025 * len(draws))]
        hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
        share = sum(1 for d in draws if d < 0) / len(draws)
        rows.append({"label": args.label, "subset": name, "n": len(members),
                     "delta": f"{obs:.4f}", "ci_lo": f"{lo:.4f}",
                     "ci_hi": f"{hi:.4f}", "p_improved": f"{share:.3f}"})
        print(f"{args.label:<20} {name:<12} n={len(members):3d}  "
              f"Δ={obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"improved in {share * 100:.1f}% of resamples")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["label", "subset", "n", "delta",
                                               "ci_lo", "ci_hi", "p_improved"])
            if new:
                w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
