#!/usr/bin/env python3
"""#407 review gap 1 — the head-seed band, from the draws on disk.

The card scores one head per (stop, encoder), with head seed 20260722. That
gives no scale for a difference between two stops. `replicate_heads.sh`
draws the same head again on the same backbone under seeds 20260723 and
20260724, and this reads the three numbers back.

Reported per (stop, encoder): every draw, the mean, the sample standard
deviation over the draws, and the range. The range is what the parent study
quotes, so this study can be read next to it.

This measures the HEAD seed only. It does not measure the backbone seed,
which no run in this study or its parents has replicated, and it does not
measure the config sampling, which `stop_bootstrap.py` covers.

Usage:
  head_band.py [--stop STEPS ...] [--results DIR] [--parent DIR]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import full_pass  # noqa: E402

PROTOCOL_SEED = 20260722
REPLICATE_SEEDS = [20260723, 20260724]


def read_score(path):
    try:
        with open(path) as fh:
            return float(fh.read().strip())
    except (OSError, ValueError):
        return None


def draw_path(directory, stop, head, seed):
    """Where one draw's score lives.

    The protocol seed's tag carries no seed, because every other artefact of
    this study and of #373 names it that way. A replicate tag carries one.
    """
    tag = full_pass.tag(stop, head)
    if seed != PROTOCOL_SEED:
        tag = f"{tag}_s{seed}"
    return os.path.join(str(directory), f"score_{tag}.txt")


def draws(stop, head, results, parent, seeds=None):
    """`{seed: score}` for one (stop, head), over every seed on disk.

    Each score is looked for in this study's results first and in #373's
    second. #373's stops wrote only into #373's directory, and `collect.sh`
    copies this card's stops into this study, so both places are real.
    """
    seeds = [PROTOCOL_SEED] + REPLICATE_SEEDS if seeds is None else seeds
    out = {}
    for seed in seeds:
        for directory in (results, parent):
            if directory is None:
                continue
            value = read_score(draw_path(directory, stop, head, seed))
            if value is not None:
                out[seed] = value
                break
    return out


def band(values):
    """`(mean, sample std, range)` of a list of draws, or None below two."""
    if len(values) < 2:
        return None
    return (statistics.fmean(values),
            statistics.stdev(values),
            max(values) - min(values))


def rows(stops, results, parent):
    """One row per (stop, head) that has at least one draw."""
    out = []
    for stop in stops:
        for head in full_pass.HEADS:
            got = draws(stop, head, results, parent)
            if got:
                out.append((stop, head, got))
    return out


def pooled_std(table):
    """The head-seed std pooled over the rows that have two draws or more.

    Root mean square of the per-row sample standard deviations. Same
    quantity `ema_sched_ladder` pools, so the two bands compare.
    """
    parts = [statistics.stdev(list(got.values()))
             for _, _, got in table if len(got) >= 2]
    if not parts:
        return None
    return (sum(s * s for s in parts) / len(parts)) ** 0.5


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", type=int, action="append", dest="stops",
                    help="a stop to report on. Repeatable.")
    ap.add_argument("--results", default=full_pass.RESULTS)
    ap.add_argument("--parent", default=full_pass.PARENT_RESULTS)
    ap.add_argument("--csv", help="write the table here")
    a = ap.parse_args(argv)

    stops = a.stops or ([full_pass.RESUME_STEP] + full_pass.STOPS)
    table = rows(stops, a.results, a.parent)
    if not table:
        print("no draws on disk yet")
        return 0

    print(f"{'stop':>7}  {'head':<8} {'draws':>5}  {'mean':>7}  "
          f"{'std':>7}  {'range':>7}  seeds")
    lines = []
    for stop, head, got in table:
        values = list(got.values())
        stats = band(values)
        mean = f"{statistics.fmean(values):.4f}"
        std = f"{stats[1]:.4f}" if stats else "  -   "
        rng = f"{stats[2]:.4f}" if stats else "  -   "
        seeds = "  ".join(f"s{s}={v:.4f}" for s, v in sorted(got.items()))
        print(f"{stop:>7}  {head:<8} {len(values):>5}  {mean:>7}  "
              f"{std:>7}  {rng:>7}  {seeds}")
        lines.append((stop, head, len(values), mean, std, rng, got))

    pooled = pooled_std(table)
    if pooled is not None:
        print(f"pooled head-seed std over the rows with 2 draws or more: "
              f"{pooled:.4f}")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stop", "head", "n_draws", "mean", "std", "range",
                        "seeds"])
            for stop, head, n, mean, std, rng, got in lines:
                w.writerow([stop, head, n, mean, std, rng,
                            " ".join(f"{s}={v}" for s, v in sorted(got.items()))])
        print(f"wrote {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
