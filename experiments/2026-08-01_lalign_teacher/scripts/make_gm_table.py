#!/usr/bin/env python3
"""One tidy row per measured cell, so the report does not parse txt files.

Reads every ``eval_gm_mase/<cell>_summary.txt`` in a results directory,
re-counts the configs from that cell's ``all_results.csv`` rather than
trusting the summary line, and writes

    arm_slug,variant,align_target,code_snapshot,bb_steps,head_steps,
    bb_seed,head_seed,cell,gm_rel_mase,n_configs,source

``code_snapshot`` is the column that keeps the report honest. arm5 at 40k
with ``--align-target student`` and seed 20260520 reads 1.5478 in #379's
sweep and 1.4501 when the identical command line is re-run on this branch:
0.0977 of movement that belongs to the code, not to the flag. So a delta is
only a statement about ``--align-target`` when both sides carry the same
value here. ``#379-sweep`` and ``#390-branch`` are the two snapshots;
``source`` records the same split from the artefact's point of view.

The report's central comparison is a teacher cell against its student
counterpart at a fixed backbone step, so both have to be rebuildable from
this one file (review item 6). Cell names carry the disambiguating tags:

    arm6_v2_nse_bb100k_hd30000s                  teacher, seed 20260722
    arm6_v2_nse_alignstudent379_bb100k_hd30000s  student, copied from #379
    arm5_alignstudent_bb40k_hd15000s             student, measured here
    arm5_nse_s20260723_bb200k_hd30000s           teacher, replicate seed
    arm5_nse_bb200k_r3_hd30000s                  teacher, r3 backbone

The `_r<N>` token beside the backbone step names the backbone *replicate* the
cell was measured on: a resumed run keeps its own `_r<N>` run name, so one
(arm, step) pair can leave several different backbones. It qualifies the
backbone, not the arm, so the arm slug is read from the part before `_bb`.

Two replicates of one arm at one step are two measurements on two models, so
the pairing below keys on the replicate as well as on (arm, backbone step).
Keyed without it, a teacher `_r2` cell with no student counterpart is
answered by the base run's student cell and reads as paired.

The `379` matters: the student control of review item 3 is the same arm and
the same target measured on THIS branch, so it has to be distinguishable
from the copied #379 measurement it is compared against.

``align_target`` is ``none`` for the twenty arms that carry no L_align term
at all — arm1, arm3, arm4 and bimoco — where the flag names nothing.

The two seeds are properties of the pipeline, not of a cell: the backbone
seed is the ``SEED=`` line of both launchers, READ from them here so this
table cannot drift from what actually ran. The head seed is the cell-identity
library's ``DEFAULT_HEAD_SEED`` — the same constant that decides whether a
cell name carries a ``_s<seed>`` token, so the column and the name cannot
disagree. A ``_s<seed>`` cell tag overrides it.

The cell-name grammar comes from ``scripts/eval_cell_identity.py``, which is
what ``eval_arm.sh`` builds the names from. A second copy here is how the
parse starts disagreeing with the thing it parses.

    make_gm_table.py <results-dir> <out.csv>
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import eval_cell_identity as cid  # noqa: E402

VAL_RE = re.compile(r"\((?P<n>\d+) configs\):\s*(?P<v>[0-9.]+)")
VARIANTS = ("tr1", "nse", "ncpc", "combab")
RETRAINED_PREFIX = ("arm5", "arm6_v2")

# The two code snapshots any number in this directory can come from.
SNAP_379 = "#379-sweep"      # experiments/2026-07-21_split_pred_rep_small
SNAP_390 = "#390-branch"     # feature/contrastive-forecasting-390

REPO = Path(__file__).resolve().parents[3]
LAUNCHER_390 = REPO / "experiments/2026-08-01_lalign_teacher/scripts/run_arm.sh"
LAUNCHER_379 = (REPO / "experiments/2026-07-21_split_pred_rep_small"
                / "scripts/run_arm.sh")
EVAL_390 = REPO / "experiments/2026-08-01_lalign_teacher/scripts/eval_arm.sh"

TARGET_TAG_RE = re.compile(r"_align(?P<target>student|teacher)(?P<src>379)?$")


def read_seeds() -> tuple[str, str]:
    """(backbone seed, default head seed), from the code that ran."""
    seeds = set()
    for p in (LAUNCHER_390, LAUNCHER_379):
        m = re.search(r"(?m)^SEED=(\d+)", p.read_text())
        if not m:
            sys.exit(f"no SEED= line in {p}")
        seeds.add(m.group(1))
    if len(seeds) != 1:
        sys.exit(f"#379 and #390 launchers disagree on the backbone seed: "
                 f"{sorted(seeds)}")
    return seeds.pop(), cid.DEFAULT_HEAD_SEED


def strip_tags(arm: str,
               default_head_seed: str) -> tuple[str, str | None, bool, str]:
    """(bare arm, explicit align target or None, copied-from-#379, head seed)."""
    target, copied, head_seed = None, False, default_head_seed
    changed = True
    while changed:
        changed = False
        stripped, seed = cid.split_head_seed(arm)
        if stripped != arm:
            head_seed, arm = seed, stripped
            changed = True
            continue
        m = TARGET_TAG_RE.search(arm)
        if m:
            target = m.group("target")
            copied = m.group("src") == "379"
            arm = arm[: m.start()]
            changed = True
    return arm, target, copied, head_seed


def bb_replicate(row: dict) -> str:
    """The backbone replicate a row's cell was measured on, read back out of
    its name: `""` for the base run, `_r<N>` for a resume.

    Two replicates of one arm at one step are two models, so every set keyed
    on (arm, backbone step) has to carry this too — without it the two
    collapse into one entry and an unpaired replicate reads as paired.
    """
    parsed = cid.parse_cell(row["cell"])
    return parsed.replicate if parsed else ""


def pair_key(row: dict) -> tuple[str, int, str, str]:
    """What two rows have to share to be two sides of one comparison.

    The head seed belongs here for the same reason the backbone replicate
    does: it decides the number. Without it a student cell measured under
    another seed answers a default-seed teacher cell's pairing, and an arm
    whose only same-seed student was never run reads as controlled.
    """
    return (row["arm_slug"], row["bb_steps"], bb_replicate(row),
            row["head_seed"])


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
        parsed = cid.parse_cell(cell)
        if parsed is None:
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
        arm, tag_target, copied, head_seed = strip_tags(parsed.slug,
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
        # Which code produced the number. Copied cells were measured by the
        # #379 sweep and were never re-run; everything else ran here.
        code_snapshot = SNAP_379 if source == "#379" else SNAP_390
        rows.append({
            "arm_slug": arm, "variant": variant,
            "align_target": align_target,
            "code_snapshot": code_snapshot,
            "bb_steps": parsed.bb_k * 1000,
            "head_steps": parsed.head_steps,
            "bb_seed": bb_seed, "head_seed": head_seed,
            "cell": cell,
            "gm_rel_mase": vm.group("v"), "n_configs": n_rows,
            "source": source,
        })

    # Stable, over rows already in sorted-filename order, so two replicates
    # of one arm at one step — identical on every key here — keep the cell
    # order the glob gave them.
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
    # student cell at the same arm, backbone step, backbone replicate AND
    # head seed, or a delta cannot be formed from this file. Dropping the
    # replicate from the key answers a teacher `_r2` cell with the base run's
    # student cell; dropping the seed answers it with another seed's.
    teacher = {pair_key(r) for r in rows
               if r["align_target"] == "teacher"
               and r["head_seed"] == default_head_seed}
    student = {pair_key(r) for r in rows if r["align_target"] == "student"}
    unpaired = sorted(teacher - student)
    print(f"teacher cells at the default head seed: {len(teacher)}; "
          f"paired with a student cell: {len(teacher & student)}")
    if unpaired:
        print("UNPAIRED (no student counterpart): "
              + " ".join(f"{a}{rp}{cid.head_seed_tag(hs)}@{s}"
                         for a, s, rp, hs in unpaired))
    # The pairing that actually licenses a statement about the flag: both
    # sides measured on THIS branch. A teacher cell whose only student
    # counterpart is #379's carries the code-snapshot shift as well.
    same_snap = {pair_key(r) for r in rows
                 if r["align_target"] == "student"
                 and r["code_snapshot"] == SNAP_390}
    print(f"of those, controlled (student measured on {SNAP_390} too): "
          f"{len(teacher & same_snap)}")
    cross_only = sorted((teacher & student) - same_snap)
    if cross_only:
        print("CROSS-EXPERIMENT ONLY (student is " + SNAP_379 + "): "
              + " ".join(f"{a}{rp}{cid.head_seed_tag(hs)}@{s}"
                         for a, s, rp, hs in cross_only))
    short = [r["cell"] for r in rows if r["n_configs"] != 97]
    if short:
        print("NOT 97 CONFIGS: " + " ".join(short))
    for b in bad:
        print("PROBLEM: " + b)
    return 1 if (short or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
