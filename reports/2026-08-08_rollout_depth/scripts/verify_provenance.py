#!/usr/bin/env python3
"""#373 — read the training machine of every head from its own log.

A head log names the backbone path it loaded. The rented box mounts its run
tree under `/root/`, elisa under `/home/jupyter/`. So the path says which
machine trained the head, and no separate bookkeeping can drift from it.

The study compares columns to each other. A comparison that crosses machines
holds one thing less than a comparison that does not, so every such pair must
be named. This prints, per eval, the machine, and then lists the pairs the
report divides that do not hold it.

Usage:
  verify_provenance.py --results results [--tsv results/provenance.tsv]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

BOX, ELISA = "box", "elisa"


def machine_of(d: Path):
    """Return (machine, backbone_path) for one eval directory, or (None, None).

    Reads whichever log the round wrote: round 3 used `head.log`, the gap
    rounds `stop.log`. Both carry the same `Backbone loaded from` line.
    """
    for name in ("head.log", "stop.log"):
        p = d / name
        if not p.exists():
            continue
        m = re.search(r"Backbone loaded from (\S+)", p.read_text(errors="replace"))
        if not m:
            continue
        path = m.group(1)
        return (BOX if path.startswith("/root/") else ELISA), path
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tsv")
    a = ap.parse_args(argv)
    res = Path(a.results)

    rows = []
    for d in sorted((res / "eval").iterdir()):
        if not d.is_dir():
            continue
        mach, path = machine_of(d)
        rows.append((d.name, mach or "unknown", path or ""))

    n_box = sum(1 for _t, m, _p in rows if m == BOX)
    n_el = sum(1 for _t, m, _p in rows if m == ELISA)
    n_un = sum(1 for _t, m, _p in rows if m == "unknown")
    print(f"eval directories : {len(rows)}")
    print(f"head trained on the box   : {n_box}")
    print(f"head trained on elisa     : {n_el}")
    print(f"machine not readable      : {n_un}")
    if n_un:
        print("  (those directories hold `backbone.txt` and the eval log but no")
        print("   head log, so the head's machine is not in the eval directory.")
        print("   Rounds 1 and 2 wrote it to their launch logs instead.)")
    print()

    by = {t: m for t, m, _p in rows}

    # The pairs the report divides one column by another. A pair that crosses
    # machines is reported as crossing; the report must not call it held.
    PAIRS = [
        ("item 3, re-weighting, student",
         "G6_B1_k0_bb40k_student", "G_B1_k0_aw4_bb40k_student"),
        ("item 3, depth, student",
         "G_B1_k0_aw4_bb40k_student", "G6_B1_k3_bb40k_student"),
        ("item 3, total, student",
         "G6_B1_k0_bb40k_student", "G6_B1_k3_bb40k_student"),
        ("item 3, re-weighting, teacher",
         "G6_B1_k0_bb40k_teacher", "G_B1_k0_aw4_bb40k_teacher"),
        ("item 3, depth, teacher",
         "G_B1_k0_aw4_bb40k_teacher", "G6_B1_k3_bb40k_teacher"),
        ("item 3, total, teacher",
         "G6_B1_k0_bb40k_teacher", "G6_B1_k3_bb40k_teacher"),
        ("item 6, head seed, student",
         "A3_k3_bb200k_student", "A3_k3_bb200k_student_s20260723"),
        ("item 6, student vs teacher, draw 1",
         "A3_k3_bb200k_student", "A3_k3_bb200k_teacher"),
        ("item 6, student vs teacher, draw 2",
         "A3_k3_bb200k_student_s20260723", "A3_k3_bb200k_teacher"),
        ("A3 depth control, student",
         "A3_k0_bb40k_student", "A3_k3_bb40k_student"),
        ("A3 re-weighting control, student",
         "A3_k0_bb40k_student", "G3_A3_k0_aw4_bb40k_student"),
    ]
    print("| pair | left | right | machine |")
    print("|---|---|---|---|")
    crossing = []
    for label, left, right in PAIRS:
        ml, mr = by.get(left, "missing"), by.get(right, "missing")
        held = ml == mr and ml not in ("missing", "unknown")
        print(f"| {label} | {ml} | {mr} | "
              f"{'HELD' if held else 'CROSSES' if ml != mr else '?'} |")
        if not held:
            crossing.append(label)
    print()
    print(f"pairs that cross machines: {crossing or 'none'}")

    if a.tsv:
        out = Path(a.tsv)
        out.write_text("tag\tmachine\tbackbone_path\n"
                       + "".join(f"{t}\t{m}\t{p}\n" for t, m, p in rows))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
