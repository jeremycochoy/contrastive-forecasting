#!/usr/bin/env python3
"""One tidy row per measured cell, so the report does not parse txt files.

Reads every ``eval_gm_mase/<cell>_summary.txt`` in a results directory,
re-counts the configs from that cell's ``all_results.csv`` rather than
trusting the summary line, and writes

    arm_slug,variant,align_target,bb_steps,head_steps,bb_seed,head_seed,
    cell,gm_rel_mase,n_configs,source

The report's central comparison is a teacher cell against its student
counterpart at a fixed backbone step, so both have to be rebuildable from
this one file (review item 6). Cell names carry the disambiguating tags:

    arm6_v2_nse_bb100k_hd30000s                  teacher, seed 20260722
    arm6_v2_nse_alignstudent379_bb100k_hd30000s  student, copied from #379
    arm5_alignstudent_bb40k_hd15000s             student, measured here
    arm5_nse_s20260723_bb200k_hd30000s           teacher, replicate seed

The `379` matters: the student control of review item 3 is the same arm and
the same target measured on THIS branch, so it has to be distinguishable
from the copied #379 measurement it is compared against.

``align_target`` is ``none`` for the twenty arms that carry no L_align term
at all — arm1, arm3, arm4 and bimoco — where the flag names nothing.

The two seeds are properties of the pipeline, not of a cell: the backbone
seed is the ``SEED=`` line of both launchers and the head seed is
``eval_arm.sh``'s ``HEAD_SEED`` default, which the argv test pins to #379's
literal. Both are READ from those scripts here, so this table cannot drift
from what actually ran. A ``_s<seed>`` cell tag overrides the head seed.

    make_gm_table.py <results-dir> <out.csv>
"""
import csv
import re
import sys
from pathlib import Path

CELL_RE = re.compile(r"^(?P<arm>.+?)_bb(?P<bb>\d+)k_hd(?P<hd>\d+)s$")
VAL_RE = re.compile(r"\((?P<n>\d+) configs\):\s*(?P<v>[0-9.]+)")
VARIANTS = ("tr1", "nse", "ncpc", "combab")
RETRAINED_PREFIX = ("arm5", "arm6_v2")

REPO = Path(__file__).resolve().parents[3]
LAUNCHER_390 = REPO / "experiments/2026-08-01_lalign_teacher/scripts/run_arm.sh"
LAUNCHER_379 = (REPO / "experiments/2026-07-21_split_pred_rep_small"
                / "scripts/run_arm.sh")
EVAL_390 = REPO / "experiments/2026-08-01_lalign_teacher/scripts/eval_arm.sh"

SEED_TAG_RE = re.compile(r"_s(?P<seed>\d{6,})$")
TARGET_TAG_RE = re.compile(r"_align(?P<target>student|teacher)(?P<src>379)?$")


def read_seeds() -> tuple[str, str]:
    """(backbone seed, default head seed), read off the scripts that ran."""
    seeds = set()
    for p in (LAUNCHER_390, LAUNCHER_379):
        m = re.search(r"(?m)^SEED=(\d+)", p.read_text())
        if not m:
            sys.exit(f"no SEED= line in {p}")
        seeds.add(m.group(1))
    if len(seeds) != 1:
        sys.exit(f"#379 and #390 launchers disagree on the backbone seed: "
                 f"{sorted(seeds)}")
    m = re.search(r'HEAD_SEED="\$\{HEAD_SEED:-(\d+)\}"', EVAL_390.read_text())
    if not m:
        sys.exit(f"no HEAD_SEED default in {EVAL_390}")
    return seeds.pop(), m.group(1)


def strip_tags(arm: str,
               default_head_seed: str) -> tuple[str, str | None, bool, str]:
    """(bare arm, explicit align target or None, copied-from-#379, head seed)."""
    target, copied, head_seed = None, False, default_head_seed
    changed = True
    while changed:
        changed = False
        m = SEED_TAG_RE.search(arm)
        if m:
            head_seed = m.group("seed")
            arm = arm[: m.start()]
            changed = True
            continue
        m = TARGET_TAG_RE.search(arm)
        if m:
            target = m.group("target")
            copied = m.group("src") == "379"
            arm = arm[: m.start()]
            changed = True
    return arm, target, copied, head_seed


def split_arm(arm: str) -> tuple[str, str]:
    for v in VARIANTS:
        if arm.endswith("_" + v):
            return arm[: -len(v) - 1], v
    return arm, "base"


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    res, out = Path(sys.argv[1]), Path(sys.argv[2])
    bb_seed, default_head_seed = read_seeds()
    rows, bad = [], []
    for f in sorted((res / "eval_gm_mase").glob("*_summary.txt")):
        cell = f.name[: -len("_summary.txt")]
        m = CELL_RE.match(cell)
        if not m:
            bad.append(f"{cell}: unparseable cell name")
            continue
        vm = VAL_RE.search(f.read_text())
        if not vm:
            bad.append(f"{cell}: no aggregate line in {f.name}")
            continue
        csv_path = res / "eval_gm_mase" / cell / "all_results.csv"
        n_rows = sum(1 for _ in open(csv_path)) - 1 if csv_path.exists() else 0
        if n_rows != int(vm.group("n")):
            bad.append(f"{cell}: summary says {vm.group('n')}, CSV has {n_rows}")
        arm, tag_target, copied, head_seed = strip_tags(m.group("arm"),
                                                        default_head_seed)
        base, variant = split_arm(arm)
        if base not in RETRAINED_PREFIX:
            # arm1 / arm3 / arm4 / bimoco carry no L_align term.
            align_target, source = "none", "#379"
        else:
            # A bare cell is the wave's teacher measurement. `_alignstudent379`
            # is #379's copy; a plain `_alignstudent` is the control measured
            # here, and the whole point of it is that it is NOT #379's number.
            align_target = tag_target or "teacher"
            source = "#379" if copied else "#390"
        rows.append({
            "arm_slug": arm, "variant": variant,
            "align_target": align_target,
            "bb_steps": int(m.group("bb")) * 1000,
            "head_steps": int(m.group("hd")),
            "bb_seed": bb_seed, "head_seed": head_seed,
            "cell": cell,
            "gm_rel_mase": vm.group("v"), "n_configs": n_rows,
            "source": source,
        })

    rows.sort(key=lambda r: (r["arm_slug"], r["align_target"], r["bb_steps"],
                             r["head_seed"]))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n390 = sum(r["source"] == "#390" for r in rows)
    n_student = sum(r["align_target"] == "student" for r in rows)
    print(f"wrote {out}: {len(rows)} cells "
          f"({n390} from #390, {len(rows) - n390} from #379, of which "
          f"{n_student} are the student counterparts), "
          f"{len({r['arm_slug'] for r in rows})} arms")
    # The comparison the report is built on: every teacher cell needs a
    # student cell at the same arm and backbone step, or a delta cannot be
    # formed from this file.
    teacher = {(r["arm_slug"], r["bb_steps"]) for r in rows
               if r["align_target"] == "teacher"
               and r["head_seed"] == default_head_seed}
    student = {(r["arm_slug"], r["bb_steps"]) for r in rows
               if r["align_target"] == "student"}
    unpaired = sorted(teacher - student)
    print(f"teacher cells at the default head seed: {len(teacher)}; "
          f"paired with a student cell: {len(teacher & student)}")
    if unpaired:
        print("UNPAIRED (no student counterpart): "
              + " ".join(f"{a}@{s}" for a, s in unpaired))
    short = [r["cell"] for r in rows if r["n_configs"] != 97]
    if short:
        print("NOT 97 CONFIGS: " + " ".join(short))
    for b in bad:
        print("PROBLEM: " + b)
    return 1 if (short or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
