#!/usr/bin/env python3
"""#373 — a third, independent re-derivation of the two flagged results.

This session ran no training and no eval. Every number below comes out of the
per-config `all_results.csv` files already on the branch. The point is to
re-derive the item-3 control and the item-6 redraw with code that shares
nothing with `paired_bootstrap.py` or `independent_recheck.py`:

  * numpy vectorised resampling, not the `random` module, not a Python loop
  * the seasonal-naive denominator read from the canonical GIFT-Eval install
    (`~/workspaces/gift-eval/results/seasonal_naive/all_results.csv`), not the
    repo's copy
  * bootstrap seed 373_1408, a third value
  * the score files are read LAST, only to compare

The estimand is unchanged, or the comparison would mean nothing: the score is
the geometric mean of per-config `MASE / SN_MASE` over the 97 configs, and the
resampling unit is the dataset (everything before the final `/<term>`), so all
configs of one series move together.

Usage:  runner_recheck.py [--iters 20000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RESULTS = EXP / "results"
EVAL = RESULTS / "eval"
SN_REF = Path.home() / "workspaces" / "gift-eval" / "results" / "seasonal_naive" / "all_results.csv"
MASE = "eval_metrics/MASE[0.5]"

# cell -> (eval dir, committed score file)
CELLS = {
    "B1 k=0        student": ("G6_B1_k0_bb40k_student", "score_G6_B1_k0_bb40k_student.txt"),
    "B1 k=0 x4     student": ("G_B1_k0_aw4_bb40k_student", "score_G_B1_k0_aw4_bb40k_student.txt"),
    "B1 k=3        student": ("B1_k3_bb40k_student", "score_B1_k3_bb40k_student.txt"),
    "B1 k=0        teacher": ("G6_B1_k0_bb40k_teacher", "score_G6_B1_k0_bb40k_teacher.txt"),
    "B1 k=0 x4     teacher": ("G_B1_k0_aw4_bb40k_teacher", "score_G_B1_k0_aw4_bb40k_teacher.txt"),
    "B1 k=3        teacher": ("B1_k3_bb40k_teacher", "score_B1_k3_bb40k_teacher.txt"),
    "A3 200k draw1 student": ("A3_k3_bb200k_student", "score_A3_k3_bb200k_student.txt"),
    "A3 200k draw2 student": ("A3_k3_bb200k_student_s20260723", "score_A3_k3_bb200k_student_s20260723.txt"),
    "A3 200k       teacher": ("A3_k3_bb200k_teacher", "score_A3_k3_bb200k_teacher.txt"),
}

# label, from-cell, to-cell  (delta = to - from; negative means the score fell)
CONTRASTS = [
    ("student: the re-weighting", "B1 k=0        student", "B1 k=0 x4     student"),
    ("student: the depth",        "B1 k=0 x4     student", "B1 k=3        student"),
    ("student: total k=0 -> k=3", "B1 k=0        student", "B1 k=3        student"),
    ("teacher: the re-weighting", "B1 k=0        teacher", "B1 k=0 x4     teacher"),
    ("teacher: the depth",        "B1 k=0 x4     teacher", "B1 k=3        teacher"),
    ("teacher: total k=0 -> k=3", "B1 k=0        teacher", "B1 k=3        teacher"),
    ("A3: draw2 vs draw1",        "A3 200k draw1 student", "A3 200k draw2 student"),
    ("A3: draw1 vs teacher",      "A3 200k       teacher", "A3 200k draw1 student"),
    ("A3: draw2 vs teacher",      "A3 200k       teacher", "A3 200k draw2 student"),
]


def read_mase(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    s = pd.to_numeric(df.set_index("dataset")[MASE], errors="coerce")
    return s[np.isfinite(s) & (s > 0)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=373_1408)
    args = ap.parse_args(argv)

    if not SN_REF.exists():
        raise SystemExit(f"ABORT: no seasonal-naive reference at {SN_REF}")
    sn = read_mase(SN_REF)
    print(f"seasonal-naive denominator : {SN_REF}")
    print(f"                             {len(sn)} configs\n")

    # --- per-cell log-ratio vectors, on the configs every cell shares --------
    logr, denom_fingerprint = {}, {}
    for label, (evdir, _) in CELLS.items():
        csv_path = EVAL / evdir / "all_results.csv"
        if not csv_path.exists():
            raise SystemExit(f"ABORT: missing {csv_path}")
        m = read_mase(csv_path)
        common = m.index.intersection(sn.index)
        logr[label] = np.log(m[common] / sn[common]).sort_index()
        denom_fingerprint[label] = tuple(sorted(common))

    shared = sorted(set.intersection(*(set(v.index) for v in logr.values())))
    print(f"configs shared by all {len(CELLS)} cells : {len(shared)}")
    one_denom = len(set(denom_fingerprint.values())) == 1
    print(f"every cell scored on the same config set : {'YES' if one_denom else 'NO'}\n")
    L = {k: v.loc[shared].to_numpy() for k, v in logr.items()}

    # --- 1. do the committed scores reproduce? ------------------------------
    print("=" * 74)
    print("1. every committed score, re-derived from its own 97-config CSV")
    print("=" * 74)
    worst, fails = 0.0, 0
    for label, (_, score_file) in CELLS.items():
        got = float(np.exp(L[label].mean()))
        want = float((RESULTS / score_file).read_text().strip())
        d = abs(got - want)
        worst = max(worst, d)
        ok = d <= 5.0e-5           # the score files print 4 decimals
        fails += (not ok)
        print(f"  {'ok  ' if ok else 'FAIL'} {label}  re-derived {got:.4f}  "
              f"committed {want:.4f}  |diff| {d:.2e}")
    print(f"\n  {len(CELLS) - fails} of {len(CELLS)} reproduce; "
          f"worst deviation {worst:.2e} against a 5.0e-05 allowance\n")

    # --- 2. paired dataset-cluster bootstrap --------------------------------
    clusters: dict[str, list[int]] = {}
    for i, ds in enumerate(shared):
        clusters.setdefault(ds.rsplit("/", 1)[0], []).append(i)
    keys = sorted(clusters)
    idx_of = [np.array(clusters[k]) for k in keys]
    term = np.array([d.rsplit("/", 1)[-1] for d in shared])

    rng = np.random.default_rng(args.seed)
    picks = rng.integers(0, len(keys), size=(args.iters, len(keys)))

    subsets = {
        "all": np.ones(len(shared), bool),
        "short": term == "short",
        "medium+long": np.isin(term, ("medium", "long")),
    }

    print("=" * 74)
    print(f"2. paired dataset-cluster bootstrap  ({len(keys)} clusters, "
          f"{args.iters} resamples, seed {args.seed})")
    print("=" * 74)
    rows = []
    for name, a, b in CONTRASTS:
        diff = L[b] - L[a]
        for sub, mask in subsets.items():
            if sub != "all" and not name.startswith(("student:", "teacher:")):
                continue                       # horizon split only for item 3
            sel = [ix[mask[ix]] for ix in idx_of]
            obs = float(diff[mask].mean())
            draws = np.empty(args.iters)
            for t in range(args.iters):
                take = np.concatenate([sel[j] for j in picks[t]])
                draws[t] = diff[take].mean() if take.size else np.nan
            draws = draws[np.isfinite(draws)]
            lo, hi = np.percentile(draws, [2.5, 97.5])
            share = float((draws < 0).mean())
            # deltas in GM-relative units, not log units
            g_a, g_b = np.exp(L[a][mask].mean()), np.exp(L[b][mask].mean())
            rows.append(dict(contrast=name, subset=sub, n=int(mask.sum()),
                             delta=g_b - g_a, log_lo=lo, log_hi=hi,
                             gm_lo=g_a * np.expm1(lo), gm_hi=g_a * np.expm1(hi),
                             p_lower=share))
            print(f"  {name:<28} {sub:<12} n={int(mask.sum()):3d}  "
                  f"Δ={g_b - g_a:+.4f}  95% [{g_a * np.expm1(lo):+.4f}, "
                  f"{g_a * np.expm1(hi):+.4f}]  lower in {share * 100:5.1f}%")
    print()

    # --- 3. the item-3 verdict ---------------------------------------------
    print("=" * 74)
    print("3. item 3 — does the x4 re-weight alone reproduce the win?")
    print("=" * 74)
    for head in ("student", "teacher"):
        k0 = float(np.exp(L[f"B1 k=0        {head}"].mean()))
        x4 = float(np.exp(L[f"B1 k=0 x4     {head}"].mean()))
        k3 = float(np.exp(L[f"B1 k=3        {head}"].mean()))
        tot, w = k3 - k0, x4 - k0
        print(f"  {head:<8} k=0 {k0:.4f} -> x4 {x4:.4f} -> k=3 {k3:.4f}   "
              f"weight {w:+.4f} ({w / tot * 100:.0f}%)  "
              f"depth {k3 - x4:+.4f} ({(k3 - x4) / tot * 100:.0f}%)  "
              f"total {tot:+.4f}")
    print()

    out = RESULTS / "runner_recheck.csv"
    pd.DataFrame(rows).round(6).to_csv(out, index=False)
    print(f"wrote {out}")
    return 1 if fails or not one_denom else 0


if __name__ == "__main__":
    sys.exit(main())
