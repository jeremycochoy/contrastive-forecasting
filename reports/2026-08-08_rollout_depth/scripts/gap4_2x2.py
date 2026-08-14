#!/usr/bin/env python3
"""#373 review item 3 — the 2x2, closed by measurement instead of subtraction.

`k = 3` changes two things at once on a cell whose only f-bearing term is
`L_align`. It sums one copy of that term per depth, so the TOTAL weight on
`L_align` goes from 1x to 4x, and the term spreads from horizon t+1 alone to
t+1..t+4. B1 holds all four corners of that 2x2:

    total align weight   horizons     cell                     arm name
    1x                   t+1          B1 k = 0                 k0
    4x                   t+1          B1 k = 0, aw4            w
    1x                   t+1..t+4     B1 k = 3, aw025          h
    4x                   t+1..t+4     B1 k = 3                 k3

The report reads the horizon segment by SUBTRACTION, `k3 - w`, which is only
the horizon effect if the two changes add. This script measures the same
segment directly as `h - k0`, and reports the interaction that separates
them:

    interaction = (k3 - k0) - (w - k0) - (h - k0)
                = k3 - w - h + k0

Zero interaction means the two changes add and the published split stands.

Every number is a difference of geometric means of log-ratios against the one
shared seasonal-naive reference, over the same 97 configs, and every interval
is the study's paired dataset-cluster bootstrap: a resample draws DATASETS
with replacement and takes every config of each drawn dataset. All four arms
resample together on one draw, so the pairing survives for the interaction
too.

Usage: gap4_2x2.py [--head student|teacher|both] [--iters 10000]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent.parent
EVAL = EXP / "results" / "eval"
SN_REF = (REPO / "reports" / "2026-07-21_split_pred_rep_small" / "results"
          / "seasonal_naive_all_results.csv")
MASE = "eval_metrics/MASE[0.5]"

# arm -> cell directory under results/eval/, per head.
CORNERS = {
    "student": {
        "k0": "G6_B1_k0_bb40k_student",
        "w": "G_B1_k0_aw4_bb40k_student",
        "h": "G_B1_k3_aw025_bb40k_student",
        "k3": "B1_k3_bb40k_student",
    },
    "teacher": {
        "k0": "G6_B1_k0_bb40k_teacher",
        "w": "G_B1_k0_aw4_bb40k_teacher",
        "h": "G_B1_k3_aw025_bb40k_teacher",
        "k3": "B1_k3_bb40k_teacher",
    },
}

# label -> callable on a per-arm dict of geometric means.
CONTRASTS = [
    ("weight effect, 4x vs 1x at t+1", lambda g: g["w"] - g["k0"]),
    ("horizon effect, MEASURED at 1x", lambda g: g["h"] - g["k0"]),
    ("horizon effect, INFERRED at 4x", lambda g: g["k3"] - g["w"]),
    ("weight effect at t+1..t+4", lambda g: g["k3"] - g["h"]),
    ("both, k = 3 vs k = 0", lambda g: g["k3"] - g["k0"]),
    ("interaction", lambda g: g["k3"] - g["w"] - g["h"] + g["k0"]),
]


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
    out = {"all": [], "short": [], "medium_long": []}
    for ds in datasets:
        out["all"].append(ds)
        term = ds.rsplit("/", 1)[-1]
        if term == "short":
            out["short"].append(ds)
        elif term in ("medium", "long"):
            out["medium_long"].append(ds)
    return out


def run_head(head, iters, seed, rows):
    corners = CORNERS[head]
    missing = [f"{a}={c}" for a, c in corners.items()
               if not (EVAL / c / "all_results.csv").is_file()]
    if missing:
        print(f"  {head}: SKIP, no eval CSV for {', '.join(missing)}")
        return None

    sn = read_mase(SN_REF)
    arms = {a: read_mase(EVAL / c / "all_results.csv")
            for a, c in corners.items()}

    common = set(sn)
    for v in arms.values():
        common &= set(v)
    common = sorted(common)
    for a, v in arms.items():
        if len(v) != 97:
            print(f"  {head}: ABORT, arm {a} holds {len(v)} configs, not 97")
            return None
    if len(common) != 97:
        print(f"  {head}: ABORT, the four arms share {len(common)} configs, "
              "not 97")
        return None

    # log-ratio against seasonal naive, per config, per arm.
    lr = {a: {ds: math.log(v[ds] / sn[ds]) for ds in common}
          for a, v in arms.items()}

    clusters = {}
    for ds in common:
        clusters.setdefault(ds.rsplit("/", 1)[0], []).append(ds)
    keys = sorted(clusters)

    groups = subsets(common)
    out = {}

    for sub, members in groups.items():
        member = set(members)
        if not member:
            continue
        g_obs = {a: gm([lr[a][d] for d in members]) for a in arms}
        obs = {lab: fn(g_obs) for lab, fn in CONTRASTS}

        draws = {lab: [] for lab, _ in CONTRASTS}
        # One RNG stream, one draw per iteration, shared by every contrast.
        # The four arms therefore move together and the pairing survives.
        rng = random.Random(seed)
        for _ in range(iters):
            pick = [clusters[keys[rng.randrange(len(keys))]]
                    for _ in range(len(keys))]
            sel = [d for grp in pick for d in grp if d in member]
            if not sel:
                continue
            g = {a: gm([lr[a][d] for d in sel]) for a in arms}
            for lab, fn in CONTRASTS:
                draws[lab].append(fn(g))

        for lab, _ in CONTRASTS:
            d = sorted(draws[lab])
            lo = d[int(0.025 * len(d))]
            hi = d[min(len(d) - 1, int(0.975 * len(d)))]
            share = sum(1 for x in d if x < 0) / len(d)
            rows.append({"head": head, "subset": sub, "n": len(members),
                         "contrast": lab, "delta": f"{obs[lab]:+.4f}",
                         "ci_lo": f"{lo:+.4f}", "ci_hi": f"{hi:+.4f}",
                         "p_below_zero": f"{share:.3f}"})
            if sub == "all":
                out[lab] = (obs[lab], lo, hi, share)
        if sub == "all":
            out["_corners"] = {a: g_obs[a] for a in arms}
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="both",
                   choices=["student", "teacher", "both"])
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--out", default=str(EXP / "results" / "gap4_2x2.csv"))
    p.add_argument("--json", default=str(EXP / "results" / "gap4_2x2.json"))
    args = p.parse_args(argv)

    heads = ["student", "teacher"] if args.head == "both" else [args.head]
    rows, summary = [], {}

    for head in heads:
        print(f"== {head} ==")
        res = run_head(head, args.iters, args.seed, rows)
        if res is None:
            continue
        summary[head] = res
        g = res["_corners"]
        print(f"  corners: 1x/t+1 {g['k0']:.4f}   4x/t+1 {g['w']:.4f}   "
              f"1x/t+1..t+4 {g['h']:.4f}   4x/t+1..t+4 {g['k3']:.4f}")
        for lab, _ in CONTRASTS:
            d, lo, hi, share = res[lab]
            print(f"  {lab:<34} {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
                  f"below zero in {share * 100:.1f}%")
        meas = res["horizon effect, MEASURED at 1x"][0]
        infr = res["horizon effect, INFERRED at 4x"][0]
        inter = res["interaction"]
        adds = inter[1] <= 0.0 <= inter[2]
        print(f"  measured horizon segment {meas:+.4f} against the "
              f"inferred {infr:+.4f}")
        print("  the two changes ADD (interaction interval covers zero)"
              if adds else
              "  the two changes DO NOT add (interaction interval excludes "
              "zero)")

    if not rows:
        print("no head had all four corners; nothing written")
        return 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["head", "subset", "n", "contrast",
                                           "delta", "ci_lo", "ci_hi",
                                           "p_below_zero"])
        w.writeheader()
        w.writerows(rows)
    with open(args.json, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
