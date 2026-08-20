#!/usr/bin/env python3
"""#407 — A4's continuation to one full pass over `small_v1`, in one module.

#373 trained A4 to 200,000 steps and stopped there. At `batch_size = 64`
that is 12.8M of the 42.57M rows in `small_v1`, so the run saw 30% of the
data. The card asks one question: does A4 keep improving when it sees all
of it once?

Vocabulary:

  A4        `arm6_v2_combab_alignS` at `--train-rollout-depth 3`. L_rep with
            MoCo keys at tau_rep 1, L_align on the student latent, no CPC,
            EMA momentum 0.9 to 1.0 over steps 0..100k.
  stop      a backbone step count at which both heads are trained and
            scored: 300k, 450k and 665k.
  head      `student` or `teacher`, the encoder the quantile head reads.
            Each head is trained and evaluated on its own encoder.
  score     GM-Relative MASE over the 97 GIFT-Eval configs, official B4
            strategy, forecast horizon 16. Lower is better.

The driver, the collector and the figure all import this module, so the
card's constants have one home. A constant copied into a driver and again
into a plot is a constant with two values.

Usage:
  full_pass.py                       # print the plan and the curve so far
  full_pass.py --check-resume <root> # verify the checkpoint the card pins
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
RESULTS = os.path.join(STUDY, "results")
PARENT_RESULTS = os.path.join(os.path.dirname(STUDY),
                              "2026-08-08_rollout_depth", "results")

# The cell, as #373's launcher and #373's cell table name it. `CELL` is the
# argument `run_leg_k.sh` takes, `CELL_ID` the id `stop_k.sh` takes.
CELL = "arm6_v2_combab_alignS"
CELL_ID = "A4"
K = 3
RUN_NAME = f"cf393_{CELL}_cf373k{K}"

BATCH_SIZE = 64

# The card's three stops, and the three #373 already published.
STOPS = [300_000, 450_000, 665_000]
PARENT_STOPS = [40_000, 100_000, 200_000]

# The checkpoint the continuation resumes, and the two md5 sums the card
# pins on it. Both files matter. Without the optimizer sidecar a resume
# loses the step counter, the RNG state and AdamW's momentum, and the run
# that comes out is not a continuation of this one.
RESUME_STEP = 200_000
RESUME_NAME = f"{RUN_NAME}_r2_{RESUME_STEP // 1000}k"
RESUME_MD5 = {
    f"{RESUME_NAME}.pth": "f477c03525bf5e169704715511f1c6d7",
    f"{RESUME_NAME}_optimizer.pth": "740891276637ff7bce744b1d9109d57a",
}

# The head protocol, from #373. `stop_k.sh` carries the same defaults. The
# driver passes them anyway, so the card's numbers are visible in the
# command line rather than inherited from another study's fallback.
HEADS = ["student", "teacher"]
HEAD_STEPS = 30_000
HEAD_SEED = 20260722

# The project's best GM-Relative MASE before this card: A4's student head
# at 200,000 steps. The figure draws it as a rule, and reads it off the
# file #373 committed rather than repeating the digits.
BEST_BEFORE_TAG = "A4_k3_bb200k_student"


def steps_for_one_pass(rows: int, batch_size: int = BATCH_SIZE) -> int:
    """How many optimizer steps consume `rows` rows once."""
    return rows // batch_size


def tag(stop: int, head: str) -> str:
    """The name every artefact of one (stop, head) carries."""
    return f"{CELL_ID}_k{K}_bb{stop // 1000}k_{head}"


def leg_dir(root, stop: int) -> str:
    """Where the leg that targets `stop` saves. Matches `leg_paths.sh`."""
    return os.path.join(str(root), CELL, f"leg_{stop // 1000}k")


def ckpt_name(stop: int) -> str:
    """The checkpoint file the leg that targets `stop` writes.

    `train.py` names a periodic snapshot `<run name>_<step // 1000>k.pth`,
    and each leg saves into a fresh directory, so the run name carries no
    `_rN` branch infix. The 665k leg is off the 20,000-step save cadence
    and lands only because `run_leg_k.sh` passes `--extra-save-steps`.
    """
    return f"{RUN_NAME}_{stop // 1000}k.pth"


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_resume(root, expect: dict | None = None) -> list[str]:
    """What is wrong with the checkpoint the continuation resumes.

    Returns one line per problem, and an empty list when both files are
    there and both digests match. The driver refuses to train on anything
    else: a leg that silently resumes an earlier checkpoint, or starts
    from step 0, still produces a score at every stop.
    """
    expect = RESUME_MD5 if expect is None else expect
    here = leg_dir(root, RESUME_STEP)
    problems = []
    for name in sorted(expect):
        path = os.path.join(here, name)
        if not os.path.isfile(path):
            problems.append(f"missing: {path}")
            continue
        got = md5(path)
        if got != expect[name]:
            problems.append(f"md5 {got} != {expect[name]}: {path}")
    return problems


def score(stop: int, head: str, results=RESULTS):
    """One stop's GM-Relative MASE, or None when it has not been scored.

    `eval_local.sh` writes exactly one number per (stop, head), into
    `score_<tag>.txt`. Reading that file is how #373's own tables read a
    score, so this study cannot disagree with the parent about what a
    number means.
    """
    if results is None:
        return None
    path = os.path.join(str(results), f"score_{tag(stop, head)}.txt")
    try:
        with open(path) as fh:
            return float(fh.read().strip())
    except (OSError, ValueError):
        return None


def curve(head: str, results=RESULTS, parent=PARENT_RESULTS) -> dict:
    """`{backbone step: score}` for one head, #373's stops and this card's.

    One run, one curve. #373's three points and this card's three points
    come off the same backbone trajectory, so the figure draws them as one
    line and does not invite the reader to compare two runs.
    """
    out = {}
    for stop in PARENT_STOPS:
        value = score(stop, head, parent)
        if value is not None:
            out[stop] = value
    for stop in STOPS:
        value = score(stop, head, results)
        if value is not None:
            out[stop] = value
    return out


def best_before(parent=PARENT_RESULTS) -> float:
    """The project's best GM-Relative MASE before this card."""
    with open(os.path.join(str(parent),
                           f"score_{BEST_BEFORE_TAG}.txt")) as fh:
        return float(fh.read().strip())


BEST_BEFORE = best_before()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-resume", metavar="ROOT",
                    help="verify the checkpoint the card pins, under ROOT")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--parent", default=PARENT_RESULTS)
    a = ap.parse_args()

    if a.check_resume:
        problems = check_resume(a.check_resume)
        for line in problems:
            print(f"ABORT: {line}", file=sys.stderr)
        if not problems:
            print(f"resume OK: {leg_dir(a.check_resume, RESUME_STEP)}")
        return 1 if problems else 0

    print(f"{CELL_ID}  {CELL}  k={K}  seed 20260520")
    print(f"resume  {RESUME_NAME}.pth  (+ optimizer)")
    print(f"stops   {', '.join(str(s) for s in STOPS)}")
    print(f"heads   {', '.join(HEADS)}  {HEAD_STEPS} steps  seed {HEAD_SEED}")
    print(f"best before this card: {BEST_BEFORE:.4f} ({BEST_BEFORE_TAG})")
    for head in HEADS:
        points = curve(head, a.results, a.parent)
        row = "  ".join(f"bb{s // 1000}k={v:.4f}"
                        for s, v in sorted(points.items()))
        print(f"{head:<8} {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
