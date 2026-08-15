#!/usr/bin/env python3
"""#373 — the coverage table: deliverable against done.

Rows are the card's 14 cells. Columns are the three stops, 40k / 100k /
200k, each split into the two heads the protocol trains, student and
teacher. Every entry is one GM-Relative MASE number over the 97 configs.

A number counts as `done` only when its score file holds a value. Anything
else is missing, and the table names the stage that blocks it, so a missing
number can never read as a covered one:

    done    the number is in hand
    run     this number's own head or eval is running now
    bb-run  the backbone this number waits on is training now
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


# Round 3 replaced the per-cell fleet with ONE box and ONE flat durable root
# on elisa. Both trees are searched: round 2's per-cell trees hold every 40k
# and 100k artefact, round 3's flat tree holds everything from 200k on.
R3 = os.environ.get("CF373_R3", "/home/jupyter/cf373_r3/sync")


def _r3_bb(cell, stop):
    """The round-3 tree's checkpoint for this stop, or None.

    Resolved by `cell_paths.sh`, the one place that knows how each of the
    three launchers lays a run out. Re-deriving it here would be a second
    implementation that can drift from the launchers.
    """
    cmd = (f'. "{HERE}/cell_paths.sh"; '
           f'cf373_bb_ckpt "{cell}" "{K}" "{stop * 1000}"')
    env = dict(os.environ, CF373_ROOT=R3)
    try:
        out = subprocess.run(["bash", "-c", cmd], env=env, timeout=30,
                             capture_output=True, text=True).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    return out if out and os.path.exists(out) else None


def head_exists(cell, stop, enc):
    t = tag(cell, stop, enc)
    for d in (os.path.join(SYNC, cell, "sync", "eval", t),
              os.path.join(R3, "eval", t)):
        if glob.glob(os.path.join(d, "qhead_*_final.pth")):
            return True
    return False


def bb_exists(cell, stop):
    hits = glob.glob(os.path.join(SYNC, cell, "sync", "**", f"*_{stop}k.pth"),
                     recursive=True)
    if any("optimizer" not in h for h in hits):
        return True
    return _r3_bb(cell, stop) is not None


def _job(jid):
    """(kind, cell, stop_k, enc) for a queue job id, or None.

    Ids are `bb_B8_100k`, `hd_B8_100k_student`, `ev_B8_100k_student`.
    """
    p = jid.split("_")
    if len(p) < 3 or not p[2].endswith("k"):
        return None
    try:
        stopk = int(p[2][:-1])
    except ValueError:
        return None
    return p[0], p[1], stopk, (p[3] if len(p) >= 4 else None)


def planned():
    """Every deliverable the round-3 queue covers, and what is in flight.

    Returns (running, backbone_running, queued) as sets of (cell, stop_k, enc).

    A cell whose backbone runs now does NOT make its every stop `running`.
    Round 2's version read only the training cell, so B8 — queued to 100k
    and no further — reported 40k and 200k as running, and a stop nobody
    was going to produce read like one in progress. Coverage is read off
    the job that produces the number: the head and the eval.

    A number whose head has not started, but whose backbone is on a card
    right now, is still work in flight, and reading it as `plan` hides four
    busy GPUs. So the dependency chain of each head is walked: if a running
    backbone job sits anywhere above it, the number is `bb-run`. The chain
    is what decides, not the cell — B8's 40k head hangs off `bb_B8_100k`,
    which does produce that stop's checkpoint on its way past it.
    """
    run, bb_run, plan = set(), set(), set()
    dep, kind = {}, {}
    q = os.path.join(HERE, "q_queue.tsv")
    if os.path.exists(q):
        for line in open(q):
            if line.startswith("#") or not line.strip():
                continue
            f = [x.strip() for x in line.split("\t")]
            if len(f) < 6:
                continue
            dep[f[0]], kind[f[0]] = f[5], f[1]
            j = _job(f[0])
            if j and j[0] in ("hd", "ev") and j[3]:
                plan.add((j[1], j[2], j[3]))
    live = set()
    for p in glob.glob(os.path.join(RES, "queue", "*.state")):
        if open(p).read().strip() == "running":
            live.add(os.path.basename(p)[: -len(".state")])
    for jid in live:
        j = _job(jid)
        if j and j[0] in ("hd", "ev") and j[3]:
            run.add((j[1], j[2], j[3]))
    for jid in dep:
        j = _job(jid)
        if not (j and j[0] == "hd" and j[3]) or jid in live:
            continue
        seen, cur = set(), dep.get(jid, "-")
        while cur and cur != "-" and cur not in seen:
            seen.add(cur)
            if cur in live and kind.get(cur) == "bb":
                bb_run.add((j[1], j[2], j[3]))
                break
            cur = dep.get(cur, "-")
    return run, bb_run, plan


def state(cell, stop, enc, run, bb_run, plan, stopped):
    key = (cell, stop, enc)
    if score(cell, stop, enc) is not None:
        return "done"
    # The queue beats the rule. `stopped` is what `r2_extend.tsv` recorded
    # when the rule last ran; a job that is running NOW is running, whatever
    # the rule said before. A4's teacher head at bb200k is the case: the rule
    # read +0.0019 — 5% of the head-seed band — and returned `stop`, and the
    # card extended the head by hand. Reading `stopped` first printed `stop`
    # for a head that was training at that moment.
    if key in run:
        return "run"
    if key in bb_run:
        return "bb-run"
    if key in plan:
        return "plan"
    if key in stopped:
        return "stop"
    if head_exists(cell, stop, enc):
        return "MISS-e"
    if bb_exists(cell, stop):
        return "MISS-h"
    return "MISS-t"


def read_stopped():
    """Heads the extend rule ended, from results/r2_extend.tsv if it exists.

    The file records the stop the rule *read*. The verdict governs the stop
    after it, so a `stop` read at bb100k marks bb200k as not a deliverable.
    """
    out = set()
    p = os.path.join(RES, "r2_extend.tsv")
    if not os.path.exists(p):
        return out
    for line in open(p):
        f = line.split()
        if len(f) >= 4 and f[3] == "stop":
            out.add((f[0], int(f[1]) + 100, f[2]))
    # A head the card extends anyway is a deliverable again, so it must not
    # read as ended. results/r3_extend_override.tsv carries the reason.
    o = os.path.join(RES, "r3_extend_override.tsv")
    if os.path.exists(o):
        for line in open(o):
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) >= 3:
                out.discard((f[0].strip(), int(f[1]) + 100, f[2].strip()))
    # Cells the round leaves at a stop for a reason the extend rule cannot
    # express — B8 had no bb100k number when the round was planned, so the
    # rule had nothing to read. results/r3_no_extend.tsv carries each one.
    n = os.path.join(RES, "r3_no_extend.tsv")
    if os.path.exists(n):
        for line in open(n):
            if line.startswith("#") or not line.strip():
                continue
            f = line.split("\t")
            if len(f) >= 3:
                out.add((f[0].strip(), int(f[1]), f[2].strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="Markdown table")
    # The round's status table asks for the STAGE of each deliverable, not
    # its value. A number reads as covered whether it landed this hour or
    # last night, so a table of numbers cannot say what is still moving.
    ap.add_argument("--state", action="store_true",
                    help="print the stage of each deliverable, never its value")
    args = ap.parse_args()

    run, bb_run, plan = planned()
    stopped = read_stopped()
    rows, miss = [], 0
    for c in CELLS:
        cells = []
        for s in STOPS:
            for e in ENCS:
                st = state(c, s, e, run, bb_run, plan, stopped)
                v = score(c, s, e)
                if args.state:
                    cells.append(st)
                else:
                    cells.append(f"{v:.4f}" if v is not None else st)
                if st not in ("done", "stop"):
                    miss += 1
        rows.append((c, cells))

    # A1 and B3 hold ONE student model. Their 110 student tensors match to
    # 0.000e+00 at both stops and their two head trainings run step for step
    # (`results/pair_identity.tsv`). The grid prints that model twice per
    # stop, so the table marks both printings and the count below says how
    # many distinct models the 72 deliverables hold.
    SHARED = ("A1", "B3")
    shared_dupes = 0
    if not args.state:
        for c, cs in rows:
            if c not in SHARED:
                continue
            for i, h in enumerate(("40k S", "40k T", "100k S", "100k T",
                                   "200k S", "200k T")):
                if h.endswith(" S") and cs[i] not in ("stop",) and \
                        not cs[i].startswith("MISS"):
                    cs[i] = cs[i] + "‡"
                    if c == SHARED[1]:
                        shared_dupes += 1

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
    # A `stop` is not a deliverable, so it belongs in neither the numerator
    # nor the denominator. Counting the 13 stops as done read 65/84 when the
    # round holds 52 numbers against 71 asked for.
    grid = len(CELLS) * len(STOPS) * len(ENCS)
    n_stop = sum(1 for _c, cs in rows for x in cs if x == "stop")
    n_run = sum(1 for _c, cs in rows for x in cs if x in ("run", "bb-run"))
    n_plan = sum(1 for _c, cs in rows for x in cs if x == "plan")
    total = grid - n_stop
    n_done = total - miss
    print(f"deliverables {total}   done {n_done}   "
          f"running {n_run}   queued {n_plan}   NOT STARTED {miss - n_run - n_plan}"
          f"   (+{n_stop} stops, not deliverables)")
    print("done=number in hand  run=own head/eval running  "
          "bb-run=backbone training now  plan=queued, not started  "
          "MISS-e=eval not run  MISS-h=head not trained  "
          "MISS-t=backbone not trained  stop=not a deliverable this round")
    if shared_dupes:
        print(f"‡ A1 and B3 hold one student model, printed twice per stop. "
              f"The {total} deliverables therefore hold "
              f"{total - shared_dupes} distinct measurements. Their teacher "
              f"columns are two models and are counted twice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
