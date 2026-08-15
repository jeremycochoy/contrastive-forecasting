#!/usr/bin/env python3
"""#373 — a second, independent derivation of the two flagged results.

`verify_close.sh` re-derives the study with the scripts that produced it. A
script can agree with itself. This file re-derives the two numbers the review
put at risk with separate code: numpy instead of the `random` module, a
different bootstrap seed, and vectorised cluster resampling.

It re-derives, from `all_results.csv` and the seasonal-naive reference alone:

  item 3   B1 k=0, B1 k=0 with `L_align` x4, B1 k=3, on both heads.
           The re-weighting segment, the depth segment, and the total.
  item 6   A3 bb200k student draw 1, draw 2, and the teacher.

It reads no score file and no bootstrap.csv until the end, when it compares.

The statistic matches `paired_bootstrap.py`'s definition:
  score   = geometric mean over configs of MASE / seasonal-naive MASE
  delta   = GM(right) - GM(left), paired on the same configs
  resample unit = the DATASET, i.e. the config name without its final /term
  interval = 95% percentile over 10,000 cluster resamples

Agreement to Monte-Carlo noise means two implementations agree, not one.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent.parent
EVAL = EXP / "results" / "eval"
SN_REF = (REPO / "reports" / "2026-07-21_split_pred_rep_small" / "results"
          / "seasonal_naive_all_results.csv")
MASE = "eval_metrics/MASE[0.5]"

ITERS = 10_000
SEED = 7_919_373          # deliberately not paired_bootstrap.py's 20260809

# eval directory -> the name the report prints it under
CELLS = {
    "G6_B1_k0_bb40k_student": "B1 k=0 student",
    "G6_B1_k0_bb40k_teacher": "B1 k=0 teacher",
    "G_B1_k0_aw4_bb40k_student": "B1 k=0 L_align x4 student",
    "G_B1_k0_aw4_bb40k_teacher": "B1 k=0 L_align x4 teacher",
    "G6_B1_k3_bb40k_student": "B1 k=3 student",
    "G6_B1_k3_bb40k_teacher": "B1 k=3 teacher",
    "A3_k3_bb200k_student": "A3 bb200k student draw 1",
    "A3_k3_bb200k_student_s20260723": "A3 bb200k student draw 2",
    "A3_k3_bb200k_teacher": "A3 bb200k teacher",
}

# label -> (left arm, right arm). delta = right - left.
CONTRASTS = [
    ("item 3 re-weighting student", "G6_B1_k0_bb40k_student",
     "G_B1_k0_aw4_bb40k_student"),
    ("item 3 depth       student", "G_B1_k0_aw4_bb40k_student",
     "G6_B1_k3_bb40k_student"),
    ("item 3 total       student", "G6_B1_k0_bb40k_student",
     "G6_B1_k3_bb40k_student"),
    ("item 3 re-weighting teacher", "G6_B1_k0_bb40k_teacher",
     "G_B1_k0_aw4_bb40k_teacher"),
    ("item 3 depth       teacher", "G_B1_k0_aw4_bb40k_teacher",
     "G6_B1_k3_bb40k_teacher"),
    ("item 3 total       teacher", "G6_B1_k0_bb40k_teacher",
     "G6_B1_k3_bb40k_teacher"),
    ("item 6 draw 2 vs draw 1", "A3_k3_bb200k_student",
     "A3_k3_bb200k_student_s20260723"),
    ("item 6 draw 1 vs teacher", "A3_k3_bb200k_teacher",
     "A3_k3_bb200k_student"),
    ("item 6 draw 2 vs teacher", "A3_k3_bb200k_teacher",
     "A3_k3_bb200k_student_s20260723"),
]


def read_mase(path):
    """config -> MASE, dropping non-finite and non-positive entries."""
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                v = float(row[MASE])
            except (KeyError, ValueError, TypeError):
                continue
            if v > 0 and np.isfinite(v):
                out[row["dataset"]] = v
    return out


def subset_mask(configs, name):
    terms = {"short": ("short",), "medium_long": ("medium", "long")}
    if name == "all":
        return np.ones(len(configs), dtype=bool)
    want = terms[name]
    return np.array([c.rsplit("/", 1)[-1] in want for c in configs])


def main():
    sn = read_mase(SN_REF)

    logratio, scores = {}, {}
    configs = None
    for d in CELLS:
        path = EVAL / d / "all_results.csv"
        if not path.exists():
            raise SystemExit(f"ABORT: no all_results.csv for {d}")
        arm = read_mase(path)
        common = sorted(set(arm) & set(sn))
        if configs is None:
            configs = common
        elif common != configs:
            raise SystemExit(f"ABORT: {d} does not share the config set")
        logratio[d] = np.array([np.log(arm[c] / sn[c]) for c in configs])
        scores[d] = float(np.exp(logratio[d].mean()))

    n = len(configs)
    print("=== re-derived scores  (GM of MASE / seasonal-naive MASE) ===")
    print(f"configs shared by every arm and the reference : {n}\n")
    print(f"{'cell':<32}{'re-derived':>12}{'score file':>12}{'diff':>10}")
    worst = 0.0
    for d, pretty in CELLS.items():
        printed = float((EXP / "results" / f"score_{d}.txt").read_text())
        diff = abs(scores[d] - printed)
        worst = max(worst, diff)
        print(f"{pretty:<32}{scores[d]:>12.4f}{printed:>12.4f}{diff:>10.1e}")
    print(f"\nworst |re-derived - score file| : {worst:.2e}  "
          f"(4-decimal print, allowance 5.0e-05)")
    print("SCORES REPRODUCE" if worst <= 5.0e-05 else "SCORES DO NOT REPRODUCE")

    # cluster = the dataset, i.e. everything before the final /term
    keys, cluster_of = [], {}
    for i, c in enumerate(configs):
        k = c.rsplit("/", 1)[0]
        cluster_of.setdefault(k, []).append(i)
    keys = sorted(cluster_of)
    members = [np.array(cluster_of[k]) for k in keys]
    nk = len(keys)
    print(f"\n=== paired dataset-cluster bootstrap  "
          f"({nk} datasets, {ITERS} resamples, seed {SEED}) ===\n")

    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, nk, size=(ITERS, nk))
    # one flat index array per resample, built once and reused by every contrast
    resamples = [np.concatenate([members[j] for j in row]) for row in picks]

    rows = []
    print(f"{'contrast':<30}{'subset':<13}{'n':>4}{'delta':>10}"
          f"{'95% interval':>22}{'p_impr':>9}")
    for label, left, right in CONTRASTS:
        dl = logratio[right] - logratio[left]
        for sub in ("all", "short", "medium_long"):
            m = subset_mask(configs, sub)
            obs = (np.exp(logratio[right][m].mean())
                   - np.exp(logratio[left][m].mean()))
            draws = []
            for sel in resamples:
                s = sel[m[sel]]
                if s.size == 0:
                    continue
                draws.append(np.exp(logratio[right][s].mean())
                             - np.exp(logratio[left][s].mean()))
            draws = np.sort(np.array(draws))
            lo = float(draws[int(0.025 * draws.size)])
            hi = float(draws[min(draws.size - 1, int(0.975 * draws.size))])
            p = float((draws < 0).mean())
            rows.append((label, sub, int(m.sum()), obs, lo, hi, p))
            print(f"{label:<30}{sub:<13}{int(m.sum()):>4}{obs:>+10.4f}"
                  f"   [{lo:>+8.4f}, {hi:>+8.4f}]{p * 100:>8.1f}%")
        _ = dl  # paired per config by construction above

    out = EXP / "results" / "independent_recheck.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["contrast", "subset", "n", "delta", "ci_lo", "ci_hi",
                    "p_improved"])
        for label, sub, k, obs, lo, hi, p in rows:
            w.writerow([label.strip(), sub, k, f"{obs:.4f}", f"{lo:.4f}",
                        f"{hi:.4f}", f"{p:.3f}"])
    print(f"\nwrote {out}")

    # the segment split the review asked for
    print("\n=== item 3 — the split the review asked for ===\n")
    for head in ("student", "teacher"):
        k0 = scores[f"G6_B1_k0_bb40k_{head}"]
        aw4 = scores[f"G_B1_k0_aw4_bb40k_{head}"]
        k3 = scores[f"G6_B1_k3_bb40k_{head}"]
        total, rew, dep = k3 - k0, aw4 - k0, k3 - aw4
        print(f"{head:<9} k=0 {k0:.4f}  ->  x4 {aw4:.4f}  ->  k=3 {k3:.4f}")
        print(f"{'':<9} total {total:+.4f}   re-weighting {rew:+.4f} "
              f"({rew / total * 100:.0f}%)   depth {dep:+.4f} "
              f"({dep / total * 100:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
