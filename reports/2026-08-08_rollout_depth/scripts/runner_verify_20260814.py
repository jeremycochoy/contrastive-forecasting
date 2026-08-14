#!/usr/bin/env python3
"""#373 - an independent re-derivation of the two closing items.

This script trusts nothing but the raw per-config eval CSVs and the shared
seasonal-naive reference. It re-computes every published number for:

  item 3, the `L_align` x4 control  - G_B1_k0_aw4 against B1 at k = 0 and k = 3
  item 6, A3's bb200k student redraw - two head seeds against the teacher

It re-derives, and does not read, the `score_*.txt` files. It compares its
own result against them and fails loudly on a mismatch.

Checks, in order:
  1. Every cell holds the same 97 configs.
  2. Every cell divides by the same seasonal-naive denominator.
  3. GM-Relative MASE, re-computed, matches the published `score_*.txt`.
  4. The paired dataset-cluster bootstrap reproduces the published intervals.
  5. The re-weighting's effect splits by forecast term as published.

On checks 4 and 5 a point estimate is exact arithmetic and must match to
four decimals. An interval endpoint is a Monte-Carlo quantity, so this
script re-runs each one under several seeds and asks that the published
endpoint sit within 4 Monte-Carlo standard deviations of this script's mean.
The study's own `paired_bootstrap.py` draws its three subsets off one shared
RNG stream; this script reseeds per subset, so the two agree on the estimate
and differ on the endpoints by the width of that noise.

Usage: runner_verify_20260814.py [--iters 10000]
"""
from __future__ import annotations

import argparse
import csv
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

# The cells this script re-derives, and the number each one publishes.
CELLS = {
    "G6_B1_k0_bb40k_student": "1.2025",
    "G6_B1_k0_bb40k_teacher": "1.2001",
    "B1_k3_bb40k_student": "1.0850",
    "B1_k3_bb40k_teacher": "1.0948",
    "G_B1_k0_aw4_bb40k_student": "1.1513",
    "G_B1_k0_aw4_bb40k_teacher": "1.1482",
    "A3_k3_bb200k_student": "1.3998",
    "A3_k3_bb200k_student_s20260723": "1.4098",
    "A3_k3_bb200k_teacher": "1.2913",
}

# base, arm, published delta and interval. `None` where the report gives none.
PAIRS = [
    ("G6_B1_k0_bb40k_student", "G_B1_k0_aw4_bb40k_student",
     "aw4 vs k=0, student", "-0.0512", "-0.1001", "-0.0023"),
    ("G6_B1_k0_bb40k_teacher", "G_B1_k0_aw4_bb40k_teacher",
     "aw4 vs k=0, teacher", "-0.0519", "-0.0987", "-0.0066"),
    ("G6_B1_k0_bb40k_student", "B1_k3_bb40k_student",
     "k=3 vs k=0, student", "-0.1175", "-0.1801", "-0.0615"),
    ("G6_B1_k0_bb40k_teacher", "B1_k3_bb40k_teacher",
     "k=3 vs k=0, teacher", "-0.1053", "-0.1661", "-0.0515"),
    ("A3_k3_bb200k_student", "A3_k3_bb200k_student_s20260723",
     "A3 draw 2 vs draw 1", "+0.0100", "-0.0163", "+0.0378"),
    ("A3_k3_bb200k_student", "A3_k3_bb200k_teacher",
     "A3 teacher vs draw 1", "-0.1084", "-0.1648", "-0.0671"),
    ("A3_k3_bb200k_student_s20260723", "A3_k3_bb200k_teacher",
     "A3 teacher vs draw 2", "-0.1185", "-0.1819", "-0.0718"),
    # The residual segment: the extra horizons, net of the x4 re-weighting.
    ("G_B1_k0_aw4_bb40k_student", "B1_k3_bb40k_student",
     "k=3 vs aw4, student", "-0.0663", "-0.1070", "-0.0331"),
    ("G_B1_k0_aw4_bb40k_teacher", "B1_k3_bb40k_teacher",
     "k=3 vs aw4, teacher", "-0.0534", None, None),
]

# The re-weighting's own effect, split by forecast term.
SUBSETS = [
    ("G6_B1_k0_bb40k_student", "G_B1_k0_aw4_bb40k_student", "medium_long",
     "aw4 vs k=0, student, medium+long", "-0.1400", "-0.2267", "-0.0662"),
    ("G6_B1_k0_bb40k_student", "G_B1_k0_aw4_bb40k_student", "short",
     "aw4 vs k=0, student, short", "-0.0009", "-0.0483", "+0.0460"),
]


def read_mase(path):
    """`{config: MASE}` for every finite positive row."""
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


def gm_log(vals):
    return math.exp(sum(vals) / len(vals))


def bootstrap(la, lb, configs, iters, seed, member=None):
    """Paired dataset-cluster bootstrap on gm(lb) - gm(la).

    The resampling unit is the dataset, never the config: `m_dense/H/short`,
    `/medium` and `/long` are three configs of one series. A subset draws
    every cluster and then keeps the drawn configs that belong to it, so the
    subset intervals stay on the same draws as the whole.
    """
    clusters = {}
    for ds in configs:
        clusters.setdefault(ds.rsplit("/", 1)[0], []).append(ds)
    keys = sorted(clusters)
    rng = random.Random(seed)
    draws = []
    for _ in range(iters):
        pick = [clusters[keys[rng.randrange(len(keys))]]
                for _ in range(len(keys))]
        sel = [d for grp in pick for d in grp
               if member is None or d in member]
        if not sel:
            continue
        draws.append(gm_log([lb[d] for d in sel]) - gm_log([la[d] for d in sel]))
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return lo, hi, sum(1 for d in draws if d < 0) / len(draws)


def term_of(ds):
    return ds.rsplit("/", 1)[-1]


def mean_sd(vals):
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def sweep(la, lb, configs, iters, seeds, member=None):
    """Bootstrap under several seeds. Returns each endpoint's mean and sd."""
    los, his, shares = [], [], []
    for s in seeds:
        lo, hi, share = bootstrap(la, lb, configs, iters, s, member)
        los.append(lo)
        his.append(hi)
        shares.append(share)
    return mean_sd(los), mean_sd(his), sum(shares) / len(shares)


# How many Monte-Carlo standard deviations a published endpoint may sit from
# this script's mean. The published run is one more draw from the same
# distribution, so 4 sd is the loose end of "the same number".
TOL_SD = 4.0


def endpoint_ok(pub, est):
    """Is the published endpoint a plausible draw from the same bootstrap?"""
    if pub is None:
        return True
    m, sd = est
    return abs(float(pub) - m) <= TOL_SD * sd + 5e-5


def fmt(est):
    m, sd = est
    return f"{m:+.4f}+-{sd:.4f}"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--seeds", type=int, default=5,
                   help="how many bootstrap seeds each endpoint is swept over")
    args = p.parse_args(argv)
    seeds = [args.seed + i for i in range(args.seeds)]

    fails = []

    # ---- load ----------------------------------------------------------
    sn = read_mase(SN_REF)
    arms = {}
    for name in CELLS:
        csv_path = EVAL / name / "all_results.csv"
        if not csv_path.exists():
            fails.append(f"{name}: no all_results.csv")
            continue
        arms[name] = read_mase(csv_path)

    # ---- 1. the same 97 configs everywhere -----------------------------
    print("== 1. config coverage ==")
    ref = None
    for name, m in arms.items():
        keys = frozenset(m)
        print(f"  {name:<34} n={len(keys)}")
        if len(keys) != 97:
            fails.append(f"{name}: {len(keys)} configs, expected 97")
        if ref is None:
            ref = keys
        elif keys != ref:
            fails.append(f"{name}: config set differs from the first cell")
    missing = ref - frozenset(sn) if ref else set()
    if missing:
        fails.append(f"{len(missing)} configs absent from the SN reference")
    print(f"  all cells share one config set: {ref is not None and not fails}")
    print(f"  SN reference covers all of them: {not missing}")

    # ---- 2. one seasonal-naive denominator -----------------------------
    # Each eval writes its own SN_MASE column into summary.txt. Confirm that
    # column equals the shared reference, so no cell divides by its own.
    print("\n== 2. seasonal-naive denominator ==")
    worst = 0.0
    for name in arms:
        summ = EVAL / name / "summary.txt"
        if not summ.exists():
            fails.append(f"{name}: no summary.txt")
            continue
        seen = 0
        for line in summ.read_text().splitlines():
            parts = line.split()
            if len(parts) != 4 or "/" not in parts[0]:
                continue
            ds = parts[0]
            if ds not in sn:
                continue
            try:
                got = float(parts[2])
            except ValueError:
                continue
            seen += 1
            worst = max(worst, abs(got - sn[ds]) / sn[ds])
        if seen != 97:
            fails.append(f"{name}: summary.txt held {seen} SN rows, expected 97")
    print(f"  worst relative gap, per-cell SN_MASE vs shared reference: "
          f"{worst:.2e}")
    if worst > 1e-3:
        fails.append(f"SN denominators disagree by {worst:.2e}")

    # ---- 3. GM-Relative MASE -------------------------------------------
    print("\n== 3. GM-Relative MASE, re-derived ==")
    logs = {}
    for name, m in arms.items():
        common = sorted(set(m) & set(sn))
        logs[name] = {ds: math.log(m[ds] / sn[ds]) for ds in common}
        got = gm_log(list(logs[name].values()))
        want = CELLS[name]
        score_file = EXP / "results" / f"score_{name}.txt"
        on_disk = score_file.read_text().strip() if score_file.exists() else "-"
        ok = f"{got:.4f}" == want and on_disk == want
        print(f"  {name:<34} {got:.4f}  published {want}  "
              f"score file {on_disk}  {'ok' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"{name}: re-derived {got:.4f}, published {want}, "
                         f"score file {on_disk}")

    # ---- 4. paired bootstrap -------------------------------------------
    print(f"\n== 4. paired dataset-cluster bootstrap, {args.iters} resamples x {len(seeds)} seeds ==")
    out_rows = []
    for base, arm, label, d_pub, lo_pub, hi_pub in PAIRS:
        if base not in logs or arm not in logs:
            fails.append(f"{label}: a side is missing")
            continue
        common = sorted(set(logs[base]) & set(logs[arm]))
        obs = gm_log([logs[arm][d] for d in common]) - \
            gm_log([logs[base][d] for d in common])
        lo_rng, hi_rng, share = sweep(logs[base], logs[arm], common,
                                      args.iters, seeds)
        ok = (f"{obs:+.4f}" == d_pub and endpoint_ok(lo_pub, lo_rng)
              and endpoint_ok(hi_pub, hi_rng))
        want = f"{d_pub} [{lo_pub}, {hi_pub}]" if lo_pub else f"{d_pub} (no CI)"
        print(f"  {label:<32} n={len(common)}  d={obs:+.4f} "
              f"[{fmt(lo_rng)}, {fmt(hi_rng)}]  improved in "
              f"{share * 100:5.1f}%  published {want}  "
              f"{'ok' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"{label}: re-derived {obs:+.4f} "
                         f"[{fmt(lo_rng)}, {fmt(hi_rng)}], published {want}")
        out_rows.append({"pair": label, "n": len(common),
                         "delta": f"{obs:+.4f}", "ci_lo": fmt(lo_rng),
                         "ci_hi": fmt(hi_rng), "p_improved": f"{share:.3f}"})

    print(f"\n== 5. the re-weighting split by forecast term ==")
    for base, arm, sub, label, d_pub, lo_pub, hi_pub in SUBSETS:
        common = sorted(set(logs[base]) & set(logs[arm]))
        member = {d for d in common
                  if (term_of(d) == "short" if sub == "short"
                      else term_of(d) in ("medium", "long"))}
        obs = gm_log([logs[arm][d] for d in member]) - \
            gm_log([logs[base][d] for d in member])
        lo_rng, hi_rng, share = sweep(logs[base], logs[arm], common,
                                      args.iters, seeds, member)
        ok = (f"{obs:+.4f}" == d_pub and endpoint_ok(lo_pub, lo_rng)
              and endpoint_ok(hi_pub, hi_rng))
        print(f"  {label:<32} n={len(member)}  d={obs:+.4f} "
              f"[{fmt(lo_rng)}, {fmt(hi_rng)}]  improved in "
              f"{share * 100:5.1f}%  published {d_pub} [{lo_pub}, {hi_pub}]  "
              f"{'ok' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"{label}: re-derived {obs:+.4f} "
                         f"[{fmt(lo_rng)}, {fmt(hi_rng)}], published "
                         f"{d_pub} [{lo_pub}, {hi_pub}]")
        out_rows.append({"pair": label, "n": len(member),
                         "delta": f"{obs:+.4f}", "ci_lo": fmt(lo_rng),
                         "ci_hi": fmt(hi_rng), "p_improved": f"{share:.3f}"})

    out = EXP / "results" / "runner_verify_20260814.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair", "n", "delta", "ci_lo",
                                           "ci_hi", "p_improved"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {out.relative_to(REPO)}")

    print("\n== verdict ==")
    if fails:
        for f in fails:
            print(f"  FAIL {f}")
        return 1
    print("  every re-derived number matches what the report publishes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
