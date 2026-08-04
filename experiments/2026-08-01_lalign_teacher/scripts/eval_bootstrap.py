#!/usr/bin/env python3
"""Eval-sampling uncertainty for the teacher-vs-student comparison.

Every GM-Relative MASE in this report is one number with no interval, so a
teacher cell beating its student counterpart by 0.01 reads the same as one
beating it by 0.2. The 97 per-config rows behind each aggregate are enough
to fix that without spending any GPU.

GM-Relative MASE is ``exp(mean_i log(MASE_i / naive_i))``. For a teacher
cell and its student counterpart the seasonal-naive denominator is the same
per config, so the paired per-config difference

    d_i = log MASE_teacher_i - log MASE_student_i

has ``exp(mean d) = GM_teacher / GM_student`` exactly. Resampling the d_i
gives a confidence interval for that ratio directly.

**Resample datasets, not configs.** The 97 configs are (dataset, frequency,
term) triples over 28 base datasets — ``electricity`` alone contributes 8.
Configs from one dataset share a series, so they are not independent draws;
a config-level bootstrap treats 97 correlated rows as 97 independent ones
and returns an interval that is too narrow. The cluster bootstrap resamples
the 28 datasets with replacement and takes every config of each draw. Both
are reported, the config-level one only to show the size of that error.

Across the 10 arms at a fixed backbone step, the per-arm ratios are 10
paired observations of the same intervention. A sign test and a Wilcoxon
signed-rank test on those say whether the direction holds across arms,
which no single-arm interval can.

Usage:
    python3 eval_bootstrap.py --teacher-results <dir> --student-results <dir> \
        --out-ci ci.csv --out-tests tests.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats

MASE_COL = "eval_metrics/MASE[0.5]"

# The 10 cells #390 retrained. The other 20 carry no L_align term, so they
# are the same measurement on both sides and have no teacher/student pair.
ARMS = ("arm5", "arm5_tr1", "arm5_nse", "arm5_ncpc", "arm5_combab",
        "arm6_v2", "arm6_v2_tr1", "arm6_v2_nse", "arm6_v2_ncpc",
        "arm6_v2_combab")

# (backbone step k, head steps). 200k is deliberately absent: only 3 arms
# reached it on the teacher side and they were promoted by their own
# trajectory, so the 200k row is a different selection on each side.
WAVES = ((40, 15000), (100, 30000))

N_BOOT = 10000
BOOT_SEED = 20260804


def read_mase(path: str) -> dict[str, float]:
    """{config: MASE} for one cell's all_results.csv."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {r["dataset"]: float(r[MASE_COL]) for r in rows}
    if len(out) != len(rows):
        raise SystemExit(f"{path}: duplicate config rows")
    return out


def dataset_of(config: str) -> str:
    """`electricity/15T/short` -> `electricity`."""
    return config.split("/")[0]


def gm_relative(mase: dict[str, float], naive: dict[str, float],
                configs: list[str]) -> float:
    return math.exp(sum(math.log(mase[c] / naive[c]) for c in configs)
                    / len(configs))


def cluster_bootstrap(d: np.ndarray, groups: np.ndarray, n_boot: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Ratios from resampling whole groups with replacement."""
    uniq = np.unique(groups)
    idx_by_group = [np.flatnonzero(groups == g) for g in uniq]
    picks = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    out = np.empty(n_boot)
    for b in range(n_boot):
        take = np.concatenate([idx_by_group[i] for i in picks[b]])
        out[b] = math.exp(d[take].mean())
    return out


def simple_bootstrap(d: np.ndarray, n_boot: int,
                     rng: np.random.Generator) -> np.ndarray:
    picks = rng.integers(0, len(d), size=(n_boot, len(d)))
    return np.exp(d[picks].mean(axis=1))


def sign_test(d: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact binomial test on the signs. Zeros dropped."""
    neg = int((d < 0).sum())
    pos = int((d > 0).sum())
    n = neg + pos
    if n == 0:
        return neg, pos, 1.0
    p = stats.binomtest(min(neg, pos), n, 0.5, alternative="two-sided").pvalue
    return neg, pos, float(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-results", required=True,
                    help="reports/2026-08-04_lalign_teacher/results")
    ap.add_argument("--student-results", required=True,
                    help="reports/2026-07-21_split_pred_rep_small/results")
    ap.add_argument("--out-ci", required=True)
    ap.add_argument("--out-tests", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    naive = read_mase(os.path.join(args.teacher_results,
                                   "seasonal_naive_all_results.csv"))
    naive_student = read_mase(os.path.join(args.student_results,
                                           "seasonal_naive_all_results.csv"))
    if naive != naive_student:
        raise SystemExit(
            "the two reports' seasonal-naive baselines differ — the "
            "GM-Relative MASEs are not on one scale and nothing below is "
            "comparable.")

    rng = np.random.default_rng(BOOT_SEED)
    ci_rows = []
    per_wave_ratios: dict[int, list[tuple[str, float]]] = defaultdict(list)

    for bb_k, hd in WAVES:
        for arm in ARMS:
            cell = f"{arm}_bb{bb_k}k_hd{hd}s"
            t_path = os.path.join(args.teacher_results, "eval_gm_mase", cell,
                                  "all_results.csv")
            s_path = os.path.join(args.student_results, "eval_gm_mase", cell,
                                  "all_results.csv")
            if not (os.path.isfile(t_path) and os.path.isfile(s_path)):
                continue
            t = read_mase(t_path)
            s = read_mase(s_path)
            configs = sorted(set(t) & set(s))
            if set(t) != set(s) or set(t) - set(naive):
                raise SystemExit(
                    f"{cell}: teacher/student/naive config sets differ "
                    f"({len(t)}/{len(s)}/{len(naive)}) — a paired difference "
                    "would silently compare different configs.")

            d = np.array([math.log(t[c]) - math.log(s[c]) for c in configs])
            groups = np.array([dataset_of(c) for c in configs])
            ratio = math.exp(d.mean())

            boot_ds = cluster_bootstrap(d, groups, args.n_boot, rng)
            boot_cfg = simple_bootstrap(d, args.n_boot, rng)
            lo_ds, hi_ds = np.percentile(boot_ds, [2.5, 97.5])
            lo_cf, hi_cf = np.percentile(boot_cfg, [2.5, 97.5])

            gm_t = gm_relative(t, naive, configs)
            gm_s = gm_relative(s, naive, configs)
            # exp(mean d) must equal gm_t / gm_s to floating point; if it
            # does not, the two aggregates are over different config sets.
            assert abs(ratio - gm_t / gm_s) < 1e-9, (cell, ratio, gm_t / gm_s)

            per_wave_ratios[bb_k].append((arm, ratio))
            ci_rows.append({
                "arm_slug": arm, "bb_steps": bb_k * 1000, "head_steps": hd,
                "cell": cell,
                "gm_rel_mase_teacher": f"{gm_t:.4f}",
                "gm_rel_mase_student": f"{gm_s:.4f}",
                "ratio_teacher_over_student": f"{ratio:.4f}",
                "ci95_lo_dataset_boot": f"{lo_ds:.4f}",
                "ci95_hi_dataset_boot": f"{hi_ds:.4f}",
                "ci95_lo_config_boot": f"{lo_cf:.4f}",
                "ci95_hi_config_boot": f"{hi_cf:.4f}",
                "ci95_width_dataset_boot": f"{hi_ds - lo_ds:.4f}",
                "ci95_width_config_boot": f"{hi_cf - lo_cf:.4f}",
                "teacher_better": "yes" if ratio < 1.0 else "no",
                "ci_excludes_1": "yes" if (hi_ds < 1.0 or lo_ds > 1.0)
                                 else "no",
                "n_configs": len(configs),
                "n_datasets": int(len(set(groups))),
                "n_boot": args.n_boot,
            })

    with open(args.out_ci, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ci_rows[0]))
        w.writeheader()
        w.writerows(ci_rows)

    # --- paired tests over the arms at a fixed backbone step --------------
    test_rows = []
    for bb_k in sorted(per_wave_ratios):
        pairs = per_wave_ratios[bb_k]
        logr = np.array([math.log(r) for _, r in pairs])
        neg, pos, p_sign = sign_test(logr)
        # Wilcoxon on the paired log-ratios. Exact for n=10, no ties here;
        # `zsplit` keeps it defined if a future rerun produces an exact tie.
        wil = stats.wilcoxon(logr, alternative="two-sided",
                             zero_method="zsplit")
        test_rows.append({
            "bb_steps": bb_k * 1000,
            "n_arms": len(pairs),
            "n_teacher_better": neg,
            "n_student_better": pos,
            "median_ratio": f"{float(np.exp(np.median(logr))):.4f}",
            "mean_log_ratio": f"{float(logr.mean()):.4f}",
            "gm_ratio_over_arms": f"{float(np.exp(logr.mean())):.4f}",
            "sign_test_p": f"{p_sign:.4f}",
            "wilcoxon_stat": f"{float(wil.statistic):.1f}",
            "wilcoxon_p": f"{float(wil.pvalue):.4f}",
        })

    with open(args.out_tests, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(test_rows[0]))
        w.writeheader()
        w.writerows(test_rows)

    for r in ci_rows:
        print(f"{r['cell']:34s} ratio={r['ratio_teacher_over_student']} "
              f"CI_ds=[{r['ci95_lo_dataset_boot']}, "
              f"{r['ci95_hi_dataset_boot']}] "
              f"CI_cfg=[{r['ci95_lo_config_boot']}, "
              f"{r['ci95_hi_config_boot']}] "
              f"excl_1={r['ci_excludes_1']}")
    print()
    for r in test_rows:
        print(f"bb={r['bb_steps']:>6} n={r['n_arms']} "
              f"teacher_better={r['n_teacher_better']}/{r['n_arms']} "
              f"median_ratio={r['median_ratio']} "
              f"sign_p={r['sign_test_p']} wilcoxon_p={r['wilcoxon_p']}")


if __name__ == "__main__":
    main()
