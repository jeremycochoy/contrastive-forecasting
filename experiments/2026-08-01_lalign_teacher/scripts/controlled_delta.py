#!/usr/bin/env python3
"""The controlled teacher-minus-student delta at backbone 40k.

`eval_bootstrap.py` compares a #390 teacher cell against #379's student cell
of the same name. That comparison spans a code boundary. arm5 at 40k, same
arm, same `--align-target student`, same seed 20260520, same command line:

    #379's sweep, older code   1.5478
    this branch, re-run        1.4501

0.0977 of the number is the code snapshot. The teacher cell reads 1.3515, so
the cross-experiment delta of -0.1963 is roughly half snapshot and half flag.
Every cross-experiment delta in the wave tables carries the same term.

This script forms the delta that does not. Both sides at backbone 40k are
measured on this branch under `run_arm_student.sh` / `run_arm.sh`, which
differ by `--align-target` and nothing else, so

    delta_controlled = GM(teacher, #390) - GM(student, #390)

is attributable to the flag. The cross-experiment delta and the snapshot
shift are carried beside it, in their own columns, because the issue asks
for #379's number and because the size of the shift is itself the finding.

Intervals are the dataset-level paired cluster bootstrap of
`eval_bootstrap.py`, imported rather than re-implemented: resample the 28
base datasets with replacement, take every config of each draw, recompute
both aggregates on that draw. The paired difference and the paired ratio
come out of the same resample, so they agree by construction.

Only 40k is controlled. Nothing was re-run past step 40 000 with the student
target, so the 100k and 200k rows have no same-branch student and stay
cross-experiment — `--bb-steps-k` refuses to pretend otherwise.

Usage:
    python3 controlled_delta.py \
        --results reports/2026-08-04_lalign_teacher/results \
        --student-379-results reports/2026-07-21_split_pred_rep_small/results \
        --out-delta  <out>/controlled_delta_40k.csv \
        --out-tests  <out>/controlled_paired_tests_40k.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eb = _load(HERE / "eval_bootstrap.py", "cf390_eval_bootstrap_for_delta")

ARMS = eb.ARMS
N_BOOT = eb.N_BOOT
BOOT_SEED = 20260805   # not eval_bootstrap's, so the two tables are
                       # independent draws rather than accidentally coupled


def paired_cluster_boot(t: dict[str, float], s: dict[str, float],
                        naive: dict[str, float], configs: list[str],
                        n_boot: int, rng: np.random.Generator,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """(delta, ratio) per bootstrap draw, resampling whole datasets.

    One draw of datasets gives one config index set; both aggregates are
    recomputed on it, so the delta and the ratio describe the same draw.
    """
    lt = np.array([math.log(t[c] / naive[c]) for c in configs])
    ls = np.array([math.log(s[c] / naive[c]) for c in configs])
    groups = np.array([eb.dataset_of(c) for c in configs])
    uniq = np.unique(groups)
    idx_by_group = [np.flatnonzero(groups == g) for g in uniq]
    picks = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    delta = np.empty(n_boot)
    ratio = np.empty(n_boot)
    for b in range(n_boot):
        take = np.concatenate([idx_by_group[i] for i in picks[b]])
        gm_t = math.exp(lt[take].mean())
        gm_s = math.exp(ls[take].mean())
        delta[b] = gm_t - gm_s
        ratio[b] = gm_t / gm_s
    return delta, ratio


def cell_path(results: str, cell: str) -> str:
    return os.path.join(results, "eval_gm_mase", cell, "all_results.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="reports/2026-08-04_lalign_teacher/results")
    ap.add_argument("--student-379-results", required=True,
                    help="reports/2026-07-21_split_pred_rep_small/results")
    ap.add_argument("--bb-steps-k", type=int, default=40,
                    choices=[40],
                    help="only 40k has a same-branch student control")
    ap.add_argument("--head-steps", type=int, default=15000)
    ap.add_argument("--out-delta", required=True)
    ap.add_argument("--out-tests", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the table with fewer than all 10 arms "
                         "controlled (for a mid-run look, not for the report)")
    args = ap.parse_args()

    bb_k, hd = args.bb_steps_k, args.head_steps
    naive = eb.read_mase(os.path.join(args.results,
                                      "seasonal_naive_all_results.csv"))
    naive379 = eb.read_mase(os.path.join(args.student_379_results,
                                         "seasonal_naive_all_results.csv"))
    if naive != naive379:
        raise SystemExit(
            "the two reports' seasonal-naive baselines differ — the "
            "GM-Relative MASEs are not on one scale.")

    rng = np.random.default_rng(BOOT_SEED)
    rows, missing = [], []

    for arm in ARMS:
        t_cell = f"{arm}_bb{bb_k}k_hd{hd}s"
        s_cell = f"{arm}_alignstudent_bb{bb_k}k_hd{hd}s"
        s379_cell = f"{arm}_bb{bb_k}k_hd{hd}s"
        t_path = cell_path(args.results, t_cell)
        s_path = cell_path(args.results, s_cell)
        s379_path = cell_path(args.student_379_results, s379_cell)
        if not os.path.isfile(t_path):
            missing.append(f"{arm}: no teacher cell {t_cell}")
            continue
        if not os.path.isfile(s_path):
            missing.append(f"{arm}: no same-branch student control {s_cell}")
            continue

        t = eb.read_mase(t_path)
        s = eb.read_mase(s_path)
        if set(t) != set(s) or set(t) - set(naive):
            raise SystemExit(
                f"{arm}: teacher/student/naive config sets differ "
                f"({len(t)}/{len(s)}/{len(naive)}) — a paired difference "
                "would silently compare different configs.")
        configs = sorted(t)

        gm_t = eb.gm_relative(t, naive, configs)
        gm_s = eb.gm_relative(s, naive, configs)
        delta_b, ratio_b = paired_cluster_boot(t, s, naive, configs,
                                               args.n_boot, rng)
        d_lo, d_hi = np.percentile(delta_b, [2.5, 97.5])
        r_lo, r_hi = np.percentile(ratio_b, [2.5, 97.5])
        # The point ratio is exp(mean of the paired log differences); it must
        # equal the quotient of the two aggregates, or the aggregates are
        # over different config sets.
        ratio = math.exp(np.array([math.log(t[c]) - math.log(s[c])
                                   for c in configs]).mean())
        assert abs(ratio - gm_t / gm_s) < 1e-9, (arm, ratio, gm_t / gm_s)

        # #379's student number for the same arm and step, for the two
        # columns the issue asks to see side by side.
        if os.path.isfile(s379_path):
            s379 = eb.read_mase(s379_path)
            gm_s379 = eb.gm_relative(s379, naive, sorted(s379))
            cross = gm_t - gm_s379
            shift = gm_s - gm_s379
        else:
            gm_s379 = cross = shift = None
            missing.append(f"{arm}: no #379 student cell {s379_cell}")

        rows.append({
            "arm_slug": arm,
            "bb_steps": bb_k * 1000,
            "head_steps": hd,
            "teacher_cell": t_cell,
            "student_cell_390": s_cell,
            "student_cell_379": s379_cell,
            "gm_teacher_390": f"{gm_t:.4f}",
            "gm_student_390": f"{gm_s:.4f}",
            "gm_student_379": "" if gm_s379 is None else f"{gm_s379:.4f}",
            "delta_controlled": f"{gm_t - gm_s:+.4f}",
            "delta_ci95_lo": f"{d_lo:+.4f}",
            "delta_ci95_hi": f"{d_hi:+.4f}",
            "delta_ci_excludes_0": "yes" if (d_hi < 0 or d_lo > 0) else "no",
            "ratio_controlled": f"{ratio:.4f}",
            "ratio_ci95_lo": f"{r_lo:.4f}",
            "ratio_ci95_hi": f"{r_hi:.4f}",
            "ratio_ci_excludes_1": "yes" if (r_hi < 1.0 or r_lo > 1.0)
                                   else "no",
            "delta_cross_experiment": "" if cross is None else f"{cross:+.4f}",
            "code_snapshot_shift": "" if shift is None else f"{shift:+.4f}",
            "teacher_better_controlled": "yes" if gm_t < gm_s else "no",
            "n_configs": len(configs),
            "n_datasets": int(len({eb.dataset_of(c) for c in configs})),
            "n_boot": args.n_boot,
            "boot_seed": BOOT_SEED,
        })

    for m in missing:
        print("MISSING: " + m)
    if not rows:
        print("no controlled pair yet — nothing to write")
        return 1
    if len(rows) != len(ARMS) and not args.allow_partial:
        print(f"REFUSING to write: {len(rows)}/{len(ARMS)} arms controlled. "
              "The headline table is all ten cells or it is not the headline. "
              "Pass --allow-partial for a mid-run look.")
        return 2

    with open(args.out_delta, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # --- across the arms, at this one backbone step ------------------------
    # Ten paired observations of the same intervention. Two directions are
    # tested: the controlled delta (what the flag is worth) and the
    # cross-experiment delta (what the wave tables reported), so the
    # difference between the two conclusions is on the record.
    test_rows = []
    for label, key in (("controlled", "delta_controlled"),
                       ("cross_experiment", "delta_cross_experiment")):
        vals = [float(r[key]) for r in rows if r[key] != ""]
        if not vals:
            continue
        a = np.array(vals)
        neg, pos, p_sign = eb.sign_test(a)
        wil = stats.wilcoxon(a, alternative="two-sided", zero_method="zsplit")
        test_rows.append({
            "comparison": label,
            "bb_steps": bb_k * 1000,
            "n_arms": len(vals),
            "n_teacher_better": neg,
            "n_student_better": pos,
            "mean_delta": f"{a.mean():+.4f}",
            "median_delta": f"{float(np.median(a)):+.4f}",
            "min_delta": f"{a.min():+.4f}",
            "max_delta": f"{a.max():+.4f}",
            "sign_test_p": f"{p_sign:.4f}",
            "wilcoxon_stat": f"{float(wil.statistic):.1f}",
            "wilcoxon_p": f"{float(wil.pvalue):.4f}",
        })
    shifts = [float(r["code_snapshot_shift"]) for r in rows
              if r["code_snapshot_shift"] != ""]
    if shifts:
        a = np.array(shifts)
        neg, pos, p_sign = eb.sign_test(a)
        test_rows.append({
            "comparison": "code_snapshot_shift",
            "bb_steps": bb_k * 1000,
            "n_arms": len(shifts),
            "n_teacher_better": neg,   # here: n snapshots that moved down
            "n_student_better": pos,
            "mean_delta": f"{a.mean():+.4f}",
            "median_delta": f"{float(np.median(a)):+.4f}",
            "min_delta": f"{a.min():+.4f}",
            "max_delta": f"{a.max():+.4f}",
            "sign_test_p": f"{p_sign:.4f}",
            "wilcoxon_stat": "", "wilcoxon_p": "",
        })

    with open(args.out_tests, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(test_rows[0]))
        w.writeheader()
        w.writerows(test_rows)

    print(f"{len(rows)} controlled cells at bb={bb_k}k, head={hd}\n")
    print(f"{'arm':16s} {'teach':>7s} {'stud390':>8s} {'stud379':>8s} "
          f"{'d_ctrl':>8s} {'CI95':>19s} {'d_cross':>8s} {'shift':>8s}")
    for r in rows:
        print(f"{r['arm_slug']:16s} {r['gm_teacher_390']:>7s} "
              f"{r['gm_student_390']:>8s} {r['gm_student_379']:>8s} "
              f"{r['delta_controlled']:>8s} "
              f"[{r['delta_ci95_lo']}, {r['delta_ci95_hi']}] "
              f"{r['delta_cross_experiment']:>8s} "
              f"{r['code_snapshot_shift']:>8s}")
    print()
    for r in test_rows:
        print(f"{r['comparison']:20s} n={r['n_arms']} "
              f"mean={r['mean_delta']} median={r['median_delta']} "
              f"neg/pos={r['n_teacher_better']}/{r['n_student_better']} "
              f"sign_p={r['sign_test_p']} wilcoxon_p={r['wilcoxon_p']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
