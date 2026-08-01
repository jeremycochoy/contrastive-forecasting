#!/usr/bin/env python3
"""#390 wave-3 gate.

The issue's rule: a cell goes on to 200 000 backbone steps only if its
GM-Relative MASE at 100 000 is LOWER than its value at 40 000. A cell whose
value did not fall stops at 100 000.

Reads the two summary lines each cell already wrote:

    eval_gm_mase/<arm>_bb40k_hd15000s_summary.txt
    eval_gm_mase/<arm>_bb100k_hd30000s_summary.txt

Prints the surviving arms space-separated on stdout (the value `ARMS=` wants),
and the full comparison table on stderr. A cell missing either measurement is
NOT promoted — an unmeasured cell has not shown a fall.

    python3 select_wave3.py <eval_gm_mase dir> [arm ...]
"""
import os
import re
import sys

AGG = re.compile(r"Aggregate GM-Relative MASE \((\d+) configs\):\s*([0-9.]+)")
DEFAULT_ARMS = ["arm5", "arm5_tr1", "arm5_nse", "arm5_ncpc", "arm5_combab",
                "arm6_v2", "arm6_v2_tr1", "arm6_v2_nse", "arm6_v2_ncpc",
                "arm6_v2_combab"]


def read_value(path):
    """(n_configs, value) from a cell summary, or (None, None)."""
    try:
        with open(path) as fh:
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
        n40, v40 = read_value(os.path.join(root, f"{arm}_bb40k_hd15000s_summary.txt"))
        n100, v100 = read_value(os.path.join(root, f"{arm}_bb100k_hd30000s_summary.txt"))
        if v40 is None or v100 is None:
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
