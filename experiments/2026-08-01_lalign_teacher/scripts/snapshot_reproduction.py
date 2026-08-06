#!/usr/bin/env python3
"""Does the same command line on this branch reproduce #379's student number?

`controlled_delta.py` carries the answer in a `code_snapshot_shift` column
rounded to four decimals, where a difference of 0.000028 prints as `-0.0000`
and a run that reproduces exactly prints the same thing. The report needs to
say how many of the ten reproduce and name the one that does not, so the
difference has to be readable below the fourth decimal.

For each of the ten retrained arms at backbone 40 000 this compares

    #379's sweep, older code   eval_gm_mase/<arm>_bb40k_hd15000s
    this branch, re-run        eval_gm_mase/<arm>_alignstudent_bb40k_hd15000s

Same arm, same `--align-target student`, same backbone seed 20260520, same
head seed 20260722, same 97 configs. Anything left is the code snapshot.

Two columns beyond the aggregate difference, because an aggregate can agree
while the configs behind it do not: `n_configs_identical` counts configs
whose MASE matches to 1e-9, and `max_abs_rel_config_diff` is the largest
per-config relative move. An arm that re-ran to the identical trajectory
scores 97 and 0.0; an arm that genuinely trained differently scores neither.

`reproduces` is |aggregate difference| <= --tol, default 0.0002, which is
the resolution the report's 4-decimal tables can distinguish.

Usage:
    python3 snapshot_reproduction.py \
        --results reports/2026-08-04_lalign_teacher/results \
        --student-379-results reports/2026-07-21_split_pred_rep_small/results \
        --out <out>/snapshot_reproduction_40k.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
import eval_cell_identity as cid  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eb = _load(HERE / "eval_bootstrap.py", "cf390_eval_bootstrap_for_snapshot")

ARMS = eb.ARMS
# The seed both sides were measured under, from the cell-identity library:
# the same constant decides whether a cell name carries a `_s<seed>` token.
HEAD_SEED = cid.DEFAULT_HEAD_SEED


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--student-379-results", required=True)
    ap.add_argument("--bb-steps-k", type=int, default=40, choices=[40],
                    help="only 40k has a same-branch student control")
    ap.add_argument("--head-steps", type=int, default=15000)
    ap.add_argument("--tol", type=float, default=0.0002)
    ap.add_argument("--out", required=True)
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

    rows, missing = [], []
    for arm in ARMS:
        # By coordinate, not by a name spelled here: a cell measured on a
        # resumed backbone carries `_r<N>`, and reading it as missing drops
        # an arm from a claim the script then refuses to publish for having
        # nine of ten. The names are what was looked for, for the MISSING
        # lines.
        new_name = cid.cell_name(f"{arm}_alignstudent", bb_k, "", hd,
                                 HEAD_SEED)
        old_name = cid.cell_name(arm, bb_k, "", hd, HEAD_SEED)
        new_path = eb.cell_results(args.results, f"{arm}_alignstudent", bb_k,
                                   hd, HEAD_SEED)
        old_path = eb.cell_results(args.student_379_results, arm, bb_k, hd,
                                   HEAD_SEED)
        if new_path is None:
            missing.append(f"{arm}: no same-branch student control {new_name}")
            continue
        if old_path is None:
            missing.append(f"{arm}: no #379 student cell {old_name}")
            continue
        new_cell, old_cell = new_path.parent.name, old_path.parent.name

        new = eb.read_mase(new_path)
        old = eb.read_mase(old_path)
        if set(new) != set(old) or set(new) - set(naive):
            raise SystemExit(
                f"{arm}: config sets differ ({len(new)}/{len(old)}/"
                f"{len(naive)}) — the two numbers are not comparable.")
        configs = sorted(new)

        gm_new = eb.gm_relative(new, naive, configs)
        gm_old = eb.gm_relative(old, naive, configs)
        diff = gm_new - gm_old
        identical = sum(1 for c in configs
                        if abs(new[c] - old[c]) <= 1e-9 * max(1.0, abs(old[c])))
        max_rel = max(abs(new[c] - old[c]) / old[c] for c in configs)

        rows.append({
            "arm_slug": arm,
            "bb_steps": bb_k * 1000,
            "head_steps": hd,
            "cell_379": old_cell,
            "cell_390": new_cell,
            "gm_student_379": f"{gm_old:.6f}",
            "gm_student_390": f"{gm_new:.6f}",
            "snapshot_diff": f"{diff:+.6f}",
            "abs_snapshot_diff": f"{abs(diff):.6f}",
            "reproduces": "yes" if abs(diff) <= args.tol else "no",
            "n_configs": len(configs),
            "n_configs_identical": identical,
            "max_abs_rel_config_diff": f"{max_rel:.2e}",
        })

    for m in missing:
        print("MISSING: " + m)
    if len(rows) != len(ARMS):
        print(f"REFUSING to write: {len(rows)}/{len(ARMS)} arms have both "
              "sides. The reproduction claim is all ten arms or it is not a "
              "claim.")
        return 2

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["reproduces"] == "yes"]
    bad = [r for r in rows if r["reproduces"] == "no"]
    print(f"{len(ok)}/{len(rows)} arms reproduce #379's student number at "
          f"bb={bb_k}k within {args.tol}\n")
    print(f"{'arm':16s} {'sweep379':>10s} {'branch390':>10s} {'diff':>11s} "
          f"{'ident/97':>9s} {'max_rel':>9s}  repro")
    for r in rows:
        print(f"{r['arm_slug']:16s} {r['gm_student_379']:>10s} "
              f"{r['gm_student_390']:>10s} {r['snapshot_diff']:>11s} "
              f"{r['n_configs_identical']:>6}/{r['n_configs']:<3} "
              f"{r['max_abs_rel_config_diff']:>9s}  {r['reproduces']}")
    worst_ok = max((float(r["abs_snapshot_diff"]) for r in ok), default=0.0)
    print(f"\nlargest difference among the {len(ok)} that reproduce: "
          f"{worst_ok:.6f}")
    for r in bad:
        print(f"does NOT reproduce: {r['arm_slug']} — "
              f"{r['gm_student_379']} -> {r['gm_student_390']} "
              f"({r['snapshot_diff']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
