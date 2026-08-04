#!/usr/bin/env python3
"""One tidy row per measured cell, so the report does not parse 68 txt files.

Reads every ``eval_gm_mase/<cell>_summary.txt`` in a results directory,
re-counts the configs from that cell's ``all_results.csv`` rather than
trusting the summary line, and writes

    arm_slug,variant,bb_steps,head_steps,cell,gm_rel_mase,n_configs,source

``source`` is ``#390`` for the ten arms this experiment retrained with the
teacher-targeted L_align and ``#379`` for the twenty it did not.

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


def split_arm(arm: str) -> tuple[str, str]:
    for v in VARIANTS:
        if arm.endswith("_" + v):
            return arm[: -len(v) - 1], v
    return arm, "base"


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    res, out = Path(sys.argv[1]), Path(sys.argv[2])
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
        base, variant = split_arm(m.group("arm"))
        rows.append({
            "arm_slug": m.group("arm"), "variant": variant,
            "bb_steps": int(m.group("bb")) * 1000,
            "head_steps": int(m.group("hd")), "cell": cell,
            "gm_rel_mase": vm.group("v"), "n_configs": n_rows,
            "source": "#390" if base in RETRAINED_PREFIX else "#379",
        })

    rows.sort(key=lambda r: (r["arm_slug"], r["bb_steps"]))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n390 = sum(r["source"] == "#390" for r in rows)
    print(f"wrote {out}: {len(rows)} cells "
          f"({n390} from #390, {len(rows) - n390} from #379), "
          f"{len({r['arm_slug'] for r in rows})} arms")
    short = [r["cell"] for r in rows if r["n_configs"] != 97]
    if short:
        print("NOT 97 CONFIGS: " + " ".join(short))
    for b in bad:
        print("PROBLEM: " + b)
    return 1 if (short or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
