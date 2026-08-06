#!/usr/bin/env python3
"""Gate the twelve 40k head-seed measurements before they enter any table.

The controlled comparison lives at backbone 40 000. Two of its ten cells now
carry three head seeds on both sides — `arm5 base` (which moved) and
`arm5 combab` (which did not) — so twelve aggregates have to be trusted
before a mean or a range is taken over them.

A summary line is not evidence. This script re-reads each cell's
`all_results.csv` and checks four things:

* the cell has exactly 97 config rows;
* all twelve cells cover the *same* 97 configs, and every one of them is in
  the seasonal-naive denominator — otherwise a difference between two cells
  is partly a difference of eval set;
* the aggregate recomputed as `exp(mean log(MASE / naive))` reproduces the
  committed `_summary.txt` line to four decimals;
* the twelve aggregates match the values this report publishes.

Exit status is non-zero on any failure, so the collect step can gate on it.

Usage:
    python3 verify_head_seeds_40k.py --results <experiment eval_gm_mase dir> \
        --naive <seasonal_naive_all_results.csv>
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import re
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


eb = _load(HERE / "eval_bootstrap.py", "cf390_eval_bootstrap_for_verify")

N_CONFIGS = 97
SEEDS = (cid.DEFAULT_HEAD_SEED, "20260723", "20260724")
BB_STEPS_K = 40
HEAD_STEPS = 15000

# (arm, align target) -> (cell slug, the aggregate each seed published). The
# name is built from the slug by `scripts/eval_cell_identity.py`, which is
# what `eval_arm.sh` built it with: the wave's own seed carries no `_s<seed>`
# token and the two replicates do, and a cell measured on a resumed backbone
# carries `_r<N>` — spelled here it would read as absent.
CELLS = {
    ("arm5", "teacher"): ("arm5", (1.3515, 1.3765, 1.3927)),
    ("arm5", "student"): ("arm5_alignstudent", (1.4501, 1.4248, 1.3754)),
    ("arm5_combab", "teacher"): ("arm5_combab", (1.2728, 1.2739, 1.2746)),
    ("arm5_combab", "student"): ("arm5_combab_alignstudent",
                                 (1.2868, 1.2873, 1.2976)),
}

VAL_RE = re.compile(r"\((?P<n>\d+) configs\):\s*(?P<v>[0-9.]+)")


def cell_dir(results: Path, slug: str, seed: str) -> Path | None:
    """The one directory that measured (slug, 40k, 15 000 steps, seed).

    Two replicates of one coordinate is the choice
    `resolve_eval_checkpoint.sh` refuses to make, and a gate that picked one
    of them would verify a number the report does not publish.
    """
    hits = cid.cell_paths(results, slug, BB_STEPS_K, HEAD_STEPS, seed)
    if len(hits) > 1:
        raise SystemExit(
            f"{slug} at seed {seed}: {len(hits)} replicate cells measured "
            f"({', '.join(p.name for p in hits)}) — name the one the report "
            "publishes.")
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="directory holding <cell>/ and <cell>_summary.txt; "
                         "the experiment's eval_gm_mase or the report's copy")
    ap.add_argument("--naive", required=True,
                    help="seasonal_naive_all_results.csv")
    ap.add_argument("--gift-subdir", default="",
                    help="'gift' for the raw experiment layout, '' for the "
                         "collected report layout")
    args = ap.parse_args()

    res = Path(args.results)
    naive = eb.read_mase(args.naive)
    problems: list[str] = []
    config_sets: dict[str, frozenset[str]] = {}
    print(f"{'cell':52s} {'n':>4s} {'summary':>8s} {'recomputed':>11s} "
          f"{'expected':>9s}")

    for (arm, target), (slug, expected) in sorted(CELLS.items()):
        for seed, want in zip(SEEDS, expected):
            found = cell_dir(res, slug, seed)
            cell = found.name if found else cid.cell_name(
                slug, BB_STEPS_K, "", HEAD_STEPS, seed)
            csv_path = res / cell / args.gift_subdir / "all_results.csv"
            sum_path = res / f"{cell}_summary.txt"
            if not csv_path.is_file():
                problems.append(f"{cell}: no {csv_path}")
                continue
            if not sum_path.is_file():
                problems.append(f"{cell}: no {sum_path.name}")
                continue

            with open(csv_path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            mase = eb.read_mase(str(csv_path))
            config_sets[cell] = frozenset(mase)
            if len(rows) != N_CONFIGS:
                problems.append(f"{cell}: {len(rows)} config rows, "
                                f"not {N_CONFIGS}")
            missing = sorted(set(mase) - set(naive))
            if missing:
                problems.append(f"{cell}: {len(missing)} configs absent from "
                                f"the seasonal-naive denominator, e.g. "
                                f"{missing[0]}")
                continue

            gm = eb.gm_relative(mase, naive, sorted(mase))
            vm = VAL_RE.search(sum_path.read_text())
            if not vm:
                problems.append(f"{cell}: no aggregate line in the summary")
                continue
            said_n, said_v = int(vm.group("n")), float(vm.group("v"))
            if said_n != len(rows):
                problems.append(f"{cell}: summary says {said_n} configs, "
                                f"CSV has {len(rows)}")
            if abs(said_v - gm) > 5e-5:
                problems.append(f"{cell}: summary {said_v:.4f} but "
                                f"recomputed {gm:.4f}")
            if abs(want - gm) > 5e-5:
                problems.append(f"{cell}: expected {want:.4f} but "
                                f"recomputed {gm:.4f}")
            print(f"{cell:52s} {len(rows):>4d} {said_v:>8.4f} {gm:>11.4f} "
                  f"{want:>9.4f}")

    # One eval set, or the differences are partly differences of eval set.
    if len(set(config_sets.values())) > 1:
        ref_cell, ref = next(iter(config_sets.items()))
        for cell, cs in config_sets.items():
            if cs != ref:
                problems.append(
                    f"{cell}: config set differs from {ref_cell} "
                    f"(+{len(cs - ref)} / -{len(ref - cs)})")

    print(f"\n{len(config_sets)}/12 cells read, "
          f"{len(set(config_sets.values()))} distinct config set(s), "
          f"{N_CONFIGS} configs each")
    for p in problems:
        print("PROBLEM: " + p)
    if problems:
        return 1
    print("all twelve verified: 97 configs, one shared config set, "
          "aggregates reproduce the summaries and the published values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
