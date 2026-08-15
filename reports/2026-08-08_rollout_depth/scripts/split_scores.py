#!/usr/bin/env python3
"""#373 — split one stop's 97 configs by horizon term and by domain.

GM-Relative MASE is the geometric mean of `our MASE / seasonal-naive MASE`
over a set of configs. The aggregate over all 97 comes from the eval's own
`summary.txt` and is not recomputed here. This script computes the same
quantity over subsets, which the eval does not publish:

  horizon term  short / medium / long, read off the config name's last
                path element (`m_dense/H/short`). The card's primary
                criterion is medium+long against short, so `medium_long`
                is emitted as its own row.
  domain        the `domain` column of the eval's own CSV.

The seasonal-naive denominator is #379's committed
`seasonal_naive_all_results.csv`, the file every stop of both parent
reports normalised by. One denominator, or the numbers are not on one
scale.

Usage:
  split_scores.py --out results/splits.csv \\
      --stop <label>=<path to gift/all_results.csv> [--stop ...]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
SN_REF = (REPO / "reports" / "2026-07-21_split_pred_rep_small" / "results"
          / "seasonal_naive_all_results.csv")
MASE = "eval_metrics/MASE[0.5]"
TERMS = ("short", "medium", "long")


def read_mase(path):
    """`{dataset: MASE}` for the rows that carry a usable one."""
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                v = float(row[MASE])
            except (KeyError, ValueError, TypeError):
                continue
            if v > 0 and math.isfinite(v):
                out[row["dataset"]] = v
    return out


def read_domains(path):
    with open(path) as fh:
        return {r["dataset"]: r["domain"]
                for r in csv.DictReader(fh) if r.get("domain")}


def gm(values):
    return math.exp(sum(values) / len(values)) if values else None


def term_of(dataset):
    """`m_dense/H/short` -> `short`. Anything else is not a term."""
    tail = dataset.rsplit("/", 1)[-1]
    return tail if tail in TERMS else None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stop", action="append", required=True,
                   metavar="LABEL=CSV")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    if not SN_REF.is_file():
        raise SystemExit(f"ABORT: no seasonal-naive reference at {SN_REF}")
    sn = read_mase(SN_REF)

    rows = []
    for spec in args.stop:
        if "=" not in spec:
            raise SystemExit(f"ABORT: --stop wants LABEL=CSV, got {spec!r}")
        label, path = spec.split("=", 1)
        if not Path(path).is_file():
            print(f"  skip {label}: no {path}", file=sys.stderr)
            continue
        ours, doms = read_mase(path), read_domains(path)

        by_term, by_dom, every = {}, {}, []
        for ds, m in ours.items():
            ref = sn.get(ds)
            if not ref or ref <= 0:
                continue
            lr = math.log(m / ref)
            every.append(lr)
            t = term_of(ds)
            if t:
                by_term.setdefault(t, []).append(lr)
            d = doms.get(ds)
            if d:
                by_dom.setdefault(d, []).append(lr)

        def emit(kind, name, vals):
            if vals:
                rows.append({"stop": label, "split": kind, "name": name,
                             "n": len(vals), "gm_rel_mase": f"{gm(vals):.6f}"})

        emit("all", "all", every)
        for t in TERMS:
            emit("term", t, by_term.get(t, []))
        emit("term", "medium_long",
             by_term.get("medium", []) + by_term.get("long", []))
        for d in sorted(by_dom):
            emit("domain", d, by_dom[d])
        print(f"{label:<28} n={len(every):3d}  all={gm(every):.4f}  "
              + "  ".join(f"{t}={gm(by_term[t]):.4f}"
                          for t in TERMS if by_term.get(t)))

    if not rows:
        raise SystemExit("ABORT: no stop produced a row")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stop", "split", "name", "n",
                                           "gm_rel_mase"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
