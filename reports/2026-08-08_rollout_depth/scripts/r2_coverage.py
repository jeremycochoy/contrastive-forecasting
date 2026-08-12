#!/usr/bin/env python3
"""#373 — the coverage table: deliverable against done.

Rows are the card's 14 cells. Columns are the three stops, 40k / 100k /
200k, each split into the two heads the protocol trains, student and
teacher. Every entry is one GM-Relative MASE number over the 97 configs.

A number counts as `done` only when its score file holds a value. Anything
else is missing, and the table names the stage that blocks it, so a missing
number can never read as a covered one:

    done    the number is in hand
    run     training or eval is running now
    MISS-e  head trained, GIFT-Eval not run
    MISS-h  backbone at that stop exists, head not trained
    MISS-t  backbone not trained to that stop
    stop    the extend rule ended this head, so 200k is not a deliverable

Usage: python3 r2_coverage.py [--md]
"""
import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
RES = os.path.join(STUDY, "results")
SYNC = os.environ.get("CF373_R2_SYNC", "/home/jupyter/cf373_r2")
K = 3
CELLS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4",
         "B5", "B6", "B7", "B8", "B9", "B10"]
STOPS = [40, 100, 200]
ENCS = ["student", "teacher"]

# Round 1 scored four cells at bb40k before round 2 renamed the tags. Those
# runs are this study's own bb40k numbers: round 2 resumed the very same
# k = 3 checkpoints rather than retraining them, so the head and the score
# belong to the cell. The alias records where each number was written.
ALIAS = {("B1", 40, "student"): "G6_B1_k3_bb40k_student",
         ("B1", 40, "teacher"): "G6_B1_k3_bb40k_teacher"}

# The card's published k = 0 baselines. Group A's ladder table gives both
# heads. Group B's parents publish the student-encoder head only, so its
# teacher column is empty and no delta is formed against it.
BASELINE = {
    "A1": {40: (1.2596, 1.2347), 100: (1.2102, 1.2407), 200: (1.1910, None)},
    "A2": {40: (1.4238, 1.4177), 100: (1.3913, 1.3746), 200: (1.3586, 1.3459)},
    "A3": {40: (1.1895, 1.1793), 100: (1.1921, 1.1963), 200: (None, None)},
    "A4": {40: (1.1603, 1.1544), 100: (1.1945, 1.1837), 200: (None, None)},
    "B1": {40: (1.2025, None), 100: (1.1616, None), 200: (1.1652, None)},
    "B2": {40: (1.2765, None), 100: (1.2514, None), 200: (1.1850, None)},
    "B3": {40: (1.2868, None), 100: (1.2456, None), 200: (1.2034, None)},
    "B4": {40: (1.2728, None), 100: (1.3678, None), 200: (None, None)},
    "B5": {40: (1.2748, None), 100: (1.3219, None), 200: (None, None)},
    "B6": {40: (1.3623, None), 100: (1.2978, None), 200: (1.3011, None)},
    "B7": {40: (1.3159, None), 100: (1.3012, None), 200: (1.3325, None)},
    "B8": {40: (1.3074, None), 100: (1.3368, None), 200: (None, None)},
    "B9": {40: (1.5579, None), 100: (1.4548, None), 200: (1.3308, None)},
    "B10": {40: (1.3791, None), 100: (1.3914, None), 200: (None, None)},
}


def tag(cell, stop, enc):
    return f"{cell}_k{K}_bb{stop}k_{enc}"


def score_path(cell, stop, enc):
    """The score file for one deliverable, following the alias if there is one."""
    direct = os.path.join(RES, f"score_{tag(cell, stop, enc)}.txt")
    if os.path.exists(direct) and os.path.getsize(direct) > 0:
        return direct
    a = ALIAS.get((cell, stop, enc))
    if a:
        p = os.path.join(RES, f"score_{a}.txt")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def score(cell, stop, enc):
    p = score_path(cell, stop, enc)
    if not p:
        return None
    try:
        return float(open(p).read().strip().split()[0])
    except (ValueError, IndexError):
        return None


def head_exists(cell, stop, enc):
    d = os.path.join(SYNC, cell, "sync", "eval", tag(cell, stop, enc))
    return bool(glob.glob(os.path.join(d, "qhead_*_final.pth")))


def bb_exists(cell, stop):
    hits = glob.glob(os.path.join(SYNC, cell, "sync", "**", f"*_{stop}k.pth"),
                     recursive=True)
    return any("optimizer" not in h for h in hits)


def running():
    """Tags with an eval or a cell with training running on this machine."""
    try:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True,
                            text=True, timeout=20).stdout
    except (subprocess.SubprocessError, OSError):
        return set(), set()
    evals, trains = set(), set()
    for line in ps.splitlines():
        if "eval_local.sh " in line and "grep" not in line:
            parts = line.split("eval_local.sh ", 1)[1].split()
            if parts:
                evals.add(parts[0])
        if "r2_launch_cell.sh" in line or "r2_cell_worker.sh" in line:
            for c in CELLS:
                if f" {c} " in line or line.rstrip().endswith(f" {c}"):
                    trains.add(c)
    return evals, trains


def state(cell, stop, enc, evals, trains, stopped):
    if score(cell, stop, enc) is not None:
        return "done"
    t = tag(cell, stop, enc)
    if t in evals or ALIAS.get((cell, stop, enc)) in evals:
        return "run"
    if (cell, stop, enc) in stopped:
        return "stop"
    if head_exists(cell, stop, enc):
        return "MISS-e"
    if bb_exists(cell, stop):
        return "MISS-h"
    if cell in trains:
        return "run"
    return "MISS-t"


def read_stopped():
    """Heads the extend rule ended, from results/r2_extend.tsv if it exists."""
    out = set()
    p = os.path.join(RES, "r2_extend.tsv")
    if not os.path.exists(p):
        return out
    for line in open(p):
        f = line.split()
        if len(f) >= 4 and f[3] == "stop":
            out.add((f[0], int(f[1]), f[2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="Markdown table")
    args = ap.parse_args()

    evals, trains = running()
    stopped = read_stopped()
    rows, miss = [], 0
    for c in CELLS:
        cells = []
        for s in STOPS:
            for e in ENCS:
                st = state(c, s, e, evals, trains, stopped)
                v = score(c, s, e)
                cells.append(f"{v:.4f}" if v is not None else st)
                if st not in ("done", "stop"):
                    miss += 1
        rows.append((c, cells))

    hdr = ["cell", "40k S", "40k T", "100k S", "100k T", "200k S", "200k T"]
    if args.md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join(["---"] * len(hdr)) + "|")
        for c, cs in rows:
            print("| " + " | ".join([c] + cs) + " |")
    else:
        print("  ".join(f"{h:<8}" for h in hdr))
        for c, cs in rows:
            print("  ".join([f"{c:<8}"] + [f"{x:<8}" for x in cs]))
    print()
    total = len(CELLS) * len(STOPS) * len(ENCS)
    print(f"deliverables {total}   done {total - miss}   MISSING {miss}")
    print("done=number in hand  run=running  MISS-e=eval not run  "
          "MISS-h=head not trained  MISS-t=backbone not trained  "
          "stop=extend rule ended this head")
    return 0


if __name__ == "__main__":
    sys.exit(main())
