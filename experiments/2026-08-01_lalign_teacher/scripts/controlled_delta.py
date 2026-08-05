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

**The head seed is a second source of movement, and it is measured on two
of the ten arms only.** Every cell is one q-head trained once. `arm5 base`
and `arm5 combab` now carry three head seeds on both sides, so their delta
comes with a mean over seeds and the range it moves across them. On arm5
base that range is 0.1159 and the sign changes. The other eight arms have
one head seed each: they get `n_head_seeds=1`,
`head_seed_spread_measured=no`, and empty spread columns. Nothing is
imputed onto them — an unmeasured spread is not a small one.

`delta_controlled` stays the seed-20260722 delta on every arm, so it is one
comparable column across all ten and the bootstrap interval beside it
describes that pair. `delta_seed_mean` is the extra column, present only
where more than one seed ran.

Intervals are the dataset-level paired cluster bootstrap of
`eval_bootstrap.py`, imported rather than re-implemented: resample the 28
base datasets with replacement, take every config of each draw, recompute
both aggregates on that draw. The paired difference and the paired ratio
come out of the same resample, so they agree by construction. They cover
eval sampling only; the head seed is a separate axis and lives in its own
columns.

The aggregate over the ten arms is reported on both bases — seed 20260722
alone, and the three-seed mean where one exists — because the two answer
different questions and neither subsumes the other.

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
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
import eval_cell_identity as cid  # noqa: E402


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
# The seed every wave cell was measured under, from the cell-identity
# library: the same constant decides whether a cell name carries a
# `_s<seed>` token, so a second copy here would read the wave's own cells as
# a replicate seed's the moment the two drifted.
WAVE_HEAD_SEED = cid.DEFAULT_HEAD_SEED


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
    """The per-config file of an already-named cell — a join, not a name."""
    return os.path.join(results, "eval_gm_mase", cell, "all_results.csv")


def seed_cells(results: str, arm: str, bb_k: int, hd: int,
               ) -> dict[str, tuple[str, str]]:
    """{head seed: (teacher cell, student cell)} for the seeds that ran BOTH.

    A per-seed difference needs both sides at that seed. A seed present on
    one side only would otherwise pair a teacher head against a student head
    trained from a different seed, which is not the comparison.

    Names are read with `scripts/eval_cell_identity.py`, so a cell measured
    on a resumed backbone is seen rather than skipped — and two replicates of
    one (side, seed) is the choice `resolve_eval_checkpoint.sh` refuses to
    make, refused here the same way.
    """
    root = Path(results) / "eval_gm_mase"
    sides = {arm: "teacher", f"{arm}_alignstudent": "student"}
    found: dict[str, dict[str, list[str]]] = {}
    for p in sorted(root.glob("*")):
        parsed = cid.parse_cell(p.name)
        if not p.is_dir() or parsed is None:
            continue
        if (parsed.bb_k, parsed.head_steps) != (bb_k, hd):
            continue
        slug, seed = cid.split_head_seed(parsed.slug)
        if slug in sides:
            found.setdefault(seed, {}).setdefault(sides[slug], []).append(p.name)
    for seed, by_side in sorted(found.items()):
        for side, cells in sorted(by_side.items()):
            if len(cells) > 1:
                raise SystemExit(
                    f"{arm} {side} at seed {seed}: {len(cells)} replicate "
                    f"cells measured ({', '.join(cells)}) — name the one the "
                    "delta should be over.")
    return {seed: (by_side["teacher"][0], by_side["student"][0])
            for seed, by_side in sorted(found.items())
            if "teacher" in by_side and "student" in by_side}


def fmt_list(vals: list[float]) -> str:
    return " ".join(f"{v:+.4f}" for v in vals)


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
        # Each side asked for by coordinate, not by a name spelled here: a
        # cell measured on a resumed backbone carries `_r<N>`, and this
        # loop reading it as missing takes the arm out of a table that is
        # then refused for having nine rows. `eval_bootstrap.cell_results`
        # is the shared answer, and it refuses two replicates rather than
        # picking one. The names below are what was looked for, for the
        # MISSING lines.
        t_name = cid.cell_name(arm, bb_k, "", hd, WAVE_HEAD_SEED)
        s_name = cid.cell_name(f"{arm}_alignstudent", bb_k, "", hd,
                               WAVE_HEAD_SEED)
        t_path = eb.cell_results(args.results, arm, bb_k, hd, WAVE_HEAD_SEED)
        s_path = eb.cell_results(args.results, f"{arm}_alignstudent", bb_k, hd,
                                 WAVE_HEAD_SEED)
        s379_path = eb.cell_results(args.student_379_results, arm, bb_k, hd,
                                    WAVE_HEAD_SEED)
        if t_path is None:
            missing.append(f"{arm}: no teacher cell {t_name}")
            continue
        if s_path is None:
            missing.append(f"{arm}: no same-branch student control {s_name}")
            continue
        t_cell, s_cell = t_path.parent.name, s_path.parent.name
        s379_cell = s379_path.parent.name if s379_path else t_name

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

        # --- the head-seed axis, where it was measured ---------------------
        # Only seeds that ran on BOTH sides form a paired per-seed delta.
        pairs = seed_cells(args.results, arm, bb_k, hd)
        seeds = sorted(pairs)
        per_seed: list[tuple[str, float, float, float]] = []
        for seed in seeds:
            tc, sc = pairs[seed]
            ts = eb.read_mase(cell_path(args.results, tc))
            ss = eb.read_mase(cell_path(args.results, sc))
            if set(ts) != set(configs) or set(ss) != set(configs):
                raise SystemExit(
                    f"{arm} seed {seed}: config set differs from the wave "
                    "cell — a per-seed difference would compare eval sets.")
            g_t = eb.gm_relative(ts, naive, configs)
            g_s = eb.gm_relative(ss, naive, configs)
            per_seed.append((seed, g_t, g_s, g_t - g_s))
        if WAVE_HEAD_SEED not in seeds:
            raise SystemExit(f"{arm}: the wave seed {WAVE_HEAD_SEED} is not "
                             f"among the paired seeds {seeds}")
        deltas = [d for _, _, _, d in per_seed]
        multi = len(per_seed) > 1
        # Eight of the ten arms ran one head seed. They get no spread, not a
        # spread of zero: a range column filled in from a single measurement
        # would read as "this cell barely moves".
        spread = {
            "gm_teacher_seed_mean": f"{np.mean([g for _, g, _, _ in per_seed]):.4f}",
            "gm_student_seed_mean": f"{np.mean([g for _, _, g, _ in per_seed]):.4f}",
            "teacher_seed_range": f"{max(g for _, g, _, _ in per_seed) - min(g for _, g, _, _ in per_seed):.4f}",
            "student_seed_range": f"{max(g for _, _, g, _ in per_seed) - min(g for _, _, g, _ in per_seed):.4f}",
            "delta_per_seed": fmt_list(deltas),
            "delta_seed_mean": f"{float(np.mean(deltas)):+.4f}",
            "delta_seed_min": f"{min(deltas):+.4f}",
            "delta_seed_max": f"{max(deltas):+.4f}",
            "delta_seed_range": f"{max(deltas) - min(deltas):.4f}",
            "delta_sign_stable": "yes" if (min(deltas) > 0 or max(deltas) < 0)
                                 else "no",
        } if multi else {k: "" for k in (
            "gm_teacher_seed_mean", "gm_student_seed_mean",
            "teacher_seed_range", "student_seed_range", "delta_per_seed",
            "delta_seed_mean", "delta_seed_min", "delta_seed_max",
            "delta_seed_range", "delta_sign_stable")}

        # #379's student number for the same arm and step, for the two
        # columns the issue asks to see side by side.
        if s379_path is not None:
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
            "n_head_seeds": len(per_seed),
            "head_seeds": " ".join(seeds),
            "head_seed_spread_measured": "yes" if multi else "no",
            **spread,
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
    # Ten paired observations of the same intervention. Four directions are
    # tested: the controlled delta on the wave's own head seed, the same
    # delta with the three-seed mean substituted where one exists, the
    # cross-experiment delta the wave tables reported, and the snapshot
    # shift on its own — so the difference between the conclusions is on the
    # record rather than left to the reader.
    n_multi = sum(r["head_seed_spread_measured"] == "yes" for r in rows)
    seed_mean_vals = [float(r["delta_seed_mean"] or r["delta_controlled"])
                      for r in rows]
    test_specs = [
        ("controlled", [float(r["delta_controlled"]) for r in rows],
         f"head seed {WAVE_HEAD_SEED} on all 10 arms"),
        ("controlled_seed_mean", seed_mean_vals,
         f"3-seed mean on {n_multi}/10 arms, head seed {WAVE_HEAD_SEED} "
         f"on the other {len(rows) - n_multi}"),
        ("cross_experiment",
         [float(r["delta_cross_experiment"]) for r in rows
          if r["delta_cross_experiment"] != ""],
         f"teacher #390-branch vs student #379-sweep, head seed "
         f"{WAVE_HEAD_SEED}"),
    ]
    test_rows = []
    for label, vals, basis in test_specs:
        if not vals:
            continue
        a = np.array(vals)
        neg, pos, p_sign = eb.sign_test(a)
        wil = stats.wilcoxon(a, alternative="two-sided", zero_method="zsplit")
        test_rows.append({
            "comparison": label,
            "head_seed_basis": basis,
            "bb_steps": bb_k * 1000,
            "n_arms": len(vals),
            "n_arms_multi_seed": n_multi if label.startswith("controlled")
                                 else 0,
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
            "head_seed_basis": f"head seed {WAVE_HEAD_SEED}",
            "bb_steps": bb_k * 1000,
            "n_arms": len(shifts),
            "n_arms_multi_seed": 0,
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

    print(f"\nhead seeds, per arm ({n_multi}/{len(rows)} replicated):")
    for r in rows:
        if r["head_seed_spread_measured"] == "yes":
            print(f"  {r['arm_slug']:16s} {r['n_head_seeds']} seeds "
                  f"[{r['head_seeds']}]  delta {r['delta_per_seed']}  "
                  f"mean {r['delta_seed_mean']}  range "
                  f"{r['delta_seed_range']}  sign stable "
                  f"{r['delta_sign_stable']}")
        else:
            print(f"  {r['arm_slug']:16s} 1 seed  [{r['head_seeds']}]  "
                  f"delta {r['delta_controlled']}  "
                  f"NO SPREAD MEASURED")
    print()
    for r in test_rows:
        print(f"{r['comparison']:22s} n={r['n_arms']} "
              f"mean={r['mean_delta']} median={r['median_delta']} "
              f"neg/pos={r['n_teacher_better']}/{r['n_student_better']} "
              f"sign_p={r['sign_test_p']} wilcoxon_p={r['wilcoxon_p']}  "
              f"({r['head_seed_basis']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
