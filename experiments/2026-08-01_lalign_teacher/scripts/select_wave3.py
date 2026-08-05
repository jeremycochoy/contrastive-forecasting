#!/usr/bin/env python3
"""#390 wave-3 gate.

The issue's rule: a cell goes on to 200 000 backbone steps only if its
GM-Relative MASE at 100 000 is LOWER than its value at 40 000. A cell whose
value did not fall stops at 100 000.

Reads the two summary lines each cell already wrote:

    eval_gm_mase/<arm>_bb40k_hd15000s_summary.txt
    eval_gm_mase/<arm>_bb100k_hd30000s_summary.txt

A cell measured on a resumed backbone carries the replicate in its name
(`<arm>_bb100k_r2_hd30000s`), so both shapes are looked for — the wave-2 and
wave-3 backbones of this experiment are all resumes. Two replicates measured
at one step is the choice `resolve_eval_checkpoint.sh` refuses to make, and
this refuses it too rather than promote an arm on one of them.

Prints the surviving arms space-separated on stdout (the value `ARMS=` wants),
and the full comparison table on stderr. A cell missing either measurement is
NOT promoted — an unmeasured cell has not shown a fall.

    python3 select_wave3.py <eval_gm_mase dir> [arm ...]
"""
import os
import re
import sys

AGG = re.compile(r"Aggregate GM-Relative MASE \((\d+) configs\):\s*([0-9.]+)")
# Returned in place of a value when two replicates of one cell are measured.
AMBIGUOUS = "ambiguous"
DEFAULT_ARMS = ["arm5", "arm5_tr1", "arm5_nse", "arm5_ncpc", "arm5_combab",
                "arm6_v2", "arm6_v2_tr1", "arm6_v2_nse", "arm6_v2_ncpc",
                "arm6_v2_combab"]


def cell_summaries(root, arm, bb_k, head_steps):
    """Every summary for this (arm, backbone step, head steps), whichever
    replicate measured it. More than one is an ambiguity, not a choice."""
    pattern = re.compile(
        rf"^{re.escape(arm)}_bb{bb_k}k(_r\d+)?_hd{head_steps}s_summary\.txt$")
    try:
        names = os.listdir(root)
    except OSError:
        return []
    return sorted(os.path.join(root, n) for n in names if pattern.match(n))


def read_value(root, arm, bb_k, head_steps):
    """(n_configs, value) for one cell, or (None, None).

    (None, AMBIGUOUS) when two replicates of the cell are measured.
    """
    paths = cell_summaries(root, arm, bb_k, head_steps)
    if len(paths) > 1:
        return None, AMBIGUOUS
    if not paths:
        return None, None
    try:
        with open(paths[0]) as fh:
            m = AGG.search(fh.read())
    except OSError:
        return None, None
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = sys.argv[1]
    arms = sys.argv[2:] or DEFAULT_ARMS

    survivors = []
    print(f"{'arm':<18} {'bb40k':>8} {'bb100k':>8}  verdict", file=sys.stderr)
    for arm in arms:
        n40, v40 = read_value(root, arm, 40, 15000)
        n100, v100 = read_value(root, arm, 100, 30000)
        if AMBIGUOUS in (v40, v100):
            verdict = "STOP (ambiguous: two replicate cells measured)"
            v40 = None if v40 == AMBIGUOUS else v40
            v100 = None if v100 == AMBIGUOUS else v100
        elif v40 is None or v100 is None:
            verdict = "STOP (missing measurement)"
        elif n40 != 97 or n100 != 97:
            verdict = f"STOP (partial: {n40}/{n100} configs)"
        elif v100 < v40:
            verdict = "GO to 200k"
            survivors.append(arm)
        else:
            verdict = "STOP (did not fall)"
        f40 = f"{v40:.4f}" if v40 is not None else "--"
        f100 = f"{v100:.4f}" if v100 is not None else "--"
        print(f"{arm:<18} {f40:>8} {f100:>8}  {verdict}", file=sys.stderr)

    print(" ".join(survivors))


if __name__ == "__main__":
    main()
