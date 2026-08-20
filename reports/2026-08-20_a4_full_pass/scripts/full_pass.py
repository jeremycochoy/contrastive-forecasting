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
  leg       one call of #373's `run_leg_k.sh`, which trains from the run's
            furthest checkpoint up to one stop.
  head      `student` or `teacher`, the encoder the quantile head reads.
            Each head is trained and evaluated on its own encoder.
  score     GM-Relative MASE over the 97 GIFT-Eval configs, official B4
            strategy, forecast horizon 16. Lower is better.

The driver, the collector and the figure all import this module, so the
card's constants have one home. A constant copied into a driver and again
into a plot is a constant with two values. Nothing here reads a file at
import time, because the plot runs on a box that holds no copy of #373's
results.

Usage:
  full_pass.py                        # print the plan and the curve so far
  full_pass.py --check-resume <root>  # verify the checkpoint the card pins
  full_pass.py --check-leg <stop> --root <root>          # before a leg
  full_pass.py --check-leg-done <stop> --root <root> ... # after a leg
  full_pass.py --check-score <stop> --head <head> --wt <checkout>
  full_pass.py --log-paths --wt <checkout>               # the two leg logs
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(os.path.dirname(STUDY))
RESULTS = os.path.join(STUDY, "results")

# The study this card continues. Its launcher, its stop script and its
# three published points all come from there.
PARENT_STUDY = "2026-08-08_rollout_depth"

# The cell, as #373's launcher and #373's cell table name it. `CELL` is the
# argument `run_leg_k.sh` takes, `CELL_ID` the id `stop_k.sh` takes.
CELL = "arm6_v2_combab_alignS"
CELL_ID = "A4"
K = 3
# What `run_leg_k.sh` calls the run. Every checkpoint of it starts with
# this, so it is how this study finds the run on disk.
RUN_NAME = f"cf393_{CELL}_cf373k{K}"

BATCH_SIZE = 64

# The card's three stops, and the three #373 already published.
STOPS = [300_000, 450_000, 665_000]
PARENT_STOPS = [40_000, 100_000, 200_000]

# Every name a stop appears in carries it in thousands, so a stop off the
# 1000 grid would share a name with the stop below it.
STOP_UNIT = 1000

# ---- how long one pass is -------------------------------------------------
#
# Three counts of `small_v1` are in play, and they disagree. All three are
# written down here, because the card's own line ("4274 shards x 10,000
# rows, so 42.57M rows") does not add up: 4274 x 10,000 is 42.74M.
#
#   manifest           42,571,692 rows. `small_v1/manifest.json` on
#                      `jeremycochoy/gift-pretrain-full-4096`, read on
#                      2026-08-04: {"total_rows": 42571692, "num_shards":
#                      4274}. `experiments/2026-08-04_ema_sched_ladder/
#                      scripts/ladder.py` and `tests/test_393_ladder.py`
#                      both hold this number. One pass is 665,182 steps.
#   shard arithmetic   42,740,000 rows. 4274 shards at the nominal 10,000
#                      rows each. `experiments/2026-05-08_exp_tau_sweep/
#                      scripts/eval_multisample.py` and
#                      `experiments/2026-05-10_exp_transformer_encoder/
#                      scripts/eval_held_out.py` hold this number. One pass
#                      is 667,812 steps.
#   card               665,156 steps. Issue #407 gives a step count, not a
#                      row count, and it agrees with neither of the above.
#
# This module uses the manifest count. The manifest is the dataset's own
# record, and the other two are estimates. The choice does not change the
# card: 665,000 steps is between 99.5% and 100% of one pass under all
# three, and the lowest of the three ratios is 0.9958.
ROW_COUNTS = {"manifest": 42_571_692, "shard_arithmetic": 42_740_000}
CARD_PASS_STEPS = 665_156
ROWS = ROW_COUNTS["manifest"]

# The checkpoint the continuation resumes, and the two md5 sums the card
# pins on it. Both files matter. Without the optimizer sidecar a resume
# loses the step counter, the RNG state and AdamW's momentum, and the run
# that comes out is not a continuation of this one.
RESUME_STEP = 200_000
RESUME_NAME = f"{RUN_NAME}_r2_{RESUME_STEP // STOP_UNIT}k"
RESUME_MD5 = {
    f"{RESUME_NAME}.pth": "f477c03525bf5e169704715511f1c6d7",
    f"{RESUME_NAME}_optimizer.pth": "740891276637ff7bce744b1d9109d57a",
}
# The test seam that lets the whole driver run off elisa. See `resume_md5`.
ENV_MD5 = "CF407_RESUME_MD5"

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

# What each leg writes into the two logs #373's scripts keep. The launcher
# says what it decided to resume. `train.py` says which step the resume
# really started at, and the two disagree when the optimizer sidecar is
# missing: `load_training_state` prints "starting fresh" and returns step 0.
LEG_FRESH = "FRESH start at step 0"
LEG_SKIP = "SKIP:"
LEG_RESUME = re.compile(r"RESUME from \S+ \(step (\d+)k\)")
TRAIN_RESUME = re.compile(r"Resumed from \S+ at step (\d+)")

# The step field of a checkpoint name, in thousands. Same field
# `leg_paths.sh` reads, and the only one that means what it says: a
# checkpoint set copied between machines carries the copy order in its
# mtimes.
CKPT_STEP = re.compile(r"_(\d+)k\.pth$")


def parent_results(wt) -> str:
    """#373's results directory, inside the checkout `wt`."""
    return os.path.join(str(wt), "reports", PARENT_STUDY, "results")


PARENT_RESULTS = parent_results(REPO_ROOT)


def cell_log(wt) -> str:
    """The launcher's own log. One RESUME, FRESH or SKIP line per leg."""
    return os.path.join(parent_results(wt), f"leg_{CELL}.log")


def train_log(wt) -> str:
    """`train.py`'s log. It prints the step the resume really started at."""
    return os.path.join(parent_results(wt), f"run_{RUN_NAME}.log")


def steps_for_one_pass(rows: int, batch_size: int = BATCH_SIZE) -> int:
    """How many optimizer steps consume `rows` rows once."""
    return rows // batch_size


def pass_steps() -> dict:
    """One pass in optimizer steps, as each of the three sources counts it."""
    out = {name: steps_for_one_pass(rows)
           for name, rows in ROW_COUNTS.items()}
    out["card"] = CARD_PASS_STEPS
    return out


def check_stop(stop: int) -> list[str]:
    """What is wrong with a stop, as a list of problems."""
    problems = []
    if stop <= 0:
        problems.append(f"stop {stop} is not above zero")
    if stop % STOP_UNIT:
        near = stop // STOP_UNIT * STOP_UNIT
        problems.append(
            f"stop {stop} is not a whole number of thousands. `tag`, "
            f"`leg_dir` and `ckpt_name` all name a stop in thousands, so "
            f"{stop} and {near} would write one name.")
    return problems


def stop_k(stop: int) -> int:
    """A stop in thousands, which is how every name carries it."""
    problems = check_stop(stop)
    if problems:
        raise ValueError(", ".join(problems))
    return stop // STOP_UNIT


def tag(stop: int, head: str) -> str:
    """The name every artefact of one (stop, head) carries."""
    return f"{CELL_ID}_k{K}_bb{stop_k(stop)}k_{head}"


def leg_dir(root, stop: int) -> str:
    """Where the leg that targets `stop` saves. Matches `leg_paths.sh`."""
    return os.path.join(str(root), CELL, f"leg_{stop_k(stop)}k")


def ckpt_name(stop: int) -> str:
    """The checkpoint file a fresh leg that targets `stop` writes.

    `train.py` names a periodic snapshot `<run name>_<step // 1000>k.pth`,
    and each leg saves into a fresh directory, so the run name carries no
    `_rN` branch infix. A re-fired leg that finds its own earlier files
    does pick one up, so every LOOKUP goes through `ckpt_path`, which
    globs. The 665k leg is off the 20,000-step save cadence and lands only
    because `run_leg_k.sh` passes `--extra-save-steps`.
    """
    return f"{RUN_NAME}_{stop_k(stop)}k.pth"


def sidecar(path) -> str:
    """The optimizer file that goes with a checkpoint.

    Same rule as `src.checkpoint.get_optimizer_state_path`, which is what
    `train.py` writes and reads.
    """
    root, ext = os.path.splitext(str(path))
    return f"{root}_optimizer{ext}"


def ckpt_step(path) -> int | None:
    """The train step a checkpoint filename carries, or None."""
    match = CKPT_STEP.search(os.path.basename(str(path)))
    return int(match.group(1)) * STOP_UNIT if match else None


def ckpt_path(root, stop: int) -> str | None:
    """The checkpoint the leg that targets `stop` wrote, or None.

    Mirrors `ckpt_at_step` in `leg_paths.sh`. The `*` tolerates `train.py`'s
    `_rN` infix, which a re-fired leg picks up.
    """
    pattern = os.path.join(leg_dir(root, stop),
                           f"{RUN_NAME}*_{stop_k(stop)}k.pth")
    found = sorted(p for p in glob.glob(pattern)
                   if not p.endswith("_optimizer.pth"))
    return found[0] if found else None


def resume_source(root) -> str | None:
    """The checkpoint `run_leg_k.sh` resumes, or None.

    Mirrors `newest_ckpt` in `leg_paths.sh`: the furthest checkpoint across
    the cell's legs, chosen by the step in its name and never by mtime. On
    a tie the path breaks it, which is what the shell's `sort` does.
    """
    pattern = os.path.join(str(root), CELL, "leg_*",
                           f"{RUN_NAME}*_[0-9]*k.pth")
    found = [p for p in glob.glob(pattern) if ckpt_step(p) is not None]
    if not found:
        return None
    return sorted(found, key=lambda p: (ckpt_step(p), p))[-1]


def prior_stop(stop: int) -> int:
    """The step the leg that targets `stop` must start from.

    The first leg starts at the checkpoint the card pins. Each later leg
    starts at the stop before it, which the leg before it wrote.
    """
    if stop not in STOPS:
        raise ValueError(f"{stop} is not one of this card's stops {STOPS}")
    index = STOPS.index(stop)
    return RESUME_STEP if index == 0 else STOPS[index - 1]


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resume_md5(env=None) -> dict:
    """The digests the resume gate wants.

    The card's two digests, unless `CF407_RESUME_MD5` holds others. That
    variable takes `<name>=<digest>` pairs, separated by commas. It exists
    so the driver's end-to-end test runs on a machine that holds no copy of
    a 5 MB checkpoint pair.

    The seam cannot weaken the gate. The variable must name the card's two
    files and no others, so it changes the digests and nothing else.
    """
    env = os.environ if env is None else env
    raw = (env.get(ENV_MD5) or "").strip()
    if not raw:
        return dict(RESUME_MD5)
    out = {}
    for item in raw.split(","):
        name, sep, digest = item.strip().partition("=")
        if not sep:
            raise ValueError(
                f"{ENV_MD5}: {item.strip()!r} is not <name>=<md5>")
        out[name.strip()] = digest.strip()
    if set(out) != set(RESUME_MD5):
        raise ValueError(
            f"{ENV_MD5} names {sorted(out)}, want {sorted(RESUME_MD5)}")
    return out


def check_resume(root, expect: dict | None = None) -> list[str]:
    """What is wrong with the checkpoint the continuation resumes.

    Returns one line per problem, and an empty list when both files are
    there and both digests match. The driver refuses to train on anything
    else: a leg that silently resumes an earlier checkpoint, or starts
    from step 0, still produces a score at every stop.
    """
    expect = resume_md5() if expect is None else expect
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


def check_chain(root, stop: int) -> list[str]:
    """What is wrong with the trajectory behind a leg that is already done.

    `check_leg_done` verifies a leg by reading the log lines THAT leg wrote,
    between two byte offsets the driver took before it started. A driver
    the watchdog re-fired has no such window: the leg ran under an earlier
    process, and the offsets went with it. So this checks the disk instead.

    A trajectory that reaches `stop` honestly leaves one checkpoint at every
    stop before it, each with its optimizer sidecar. Every leg saves into
    its own directory and no leg deletes another's, so a gap in that chain
    means the checkpoint at `stop` did not come from this run.
    """
    problems = []
    for step in [RESUME_STEP] + [s for s in STOPS if s < stop]:
        found = ckpt_path(root, step)
        if found is None:
            problems.append(
                f"the leg to {stop} is already done, but no checkpoint at "
                f"step {step} is under {leg_dir(root, step)}: the "
                f"trajectory behind it is not on disk")
        elif not os.path.isfile(sidecar(found)):
            problems.append(
                f"no optimizer state at {sidecar(found)}: the chain behind "
                f"the leg to {stop} is broken at step {step}")
    return problems


def check_leg_start(root, stop: int) -> list[str]:
    """What is wrong before the leg that targets `stop` starts.

    The md5 gate pins the FIRST leg's checkpoint. This one covers all
    three, and it is the same failure each time: a leg that starts at step
    0, or resumes a checkpoint from before the stop it must continue,
    trains to the target, writes a checkpoint and scores. Nothing
    downstream can tell that apart from a continuation.

    A resume from INSIDE the leg is legal. A leg that died at 380k on its
    way to 450k resumes 380k, and re-running it is how the driver recovers.
    """
    if ckpt_path(root, stop) is not None:
        # `run_leg_k.sh` skips a leg it already ran, so there is no resume
        # to check. The watchdog can re-fire the driver onto a checkpoint no
        # process of its own watched land, so the chain behind it is checked
        # instead of nothing.
        return check_chain(root, stop)
    want = prior_stop(stop)
    source = resume_source(root)
    if source is None:
        return [f"no checkpoint under {os.path.join(str(root), CELL)}: the "
                f"leg to {stop} would start at step 0, not {want}"]
    problems = []
    step = ckpt_step(source)
    if step < want or step >= stop:
        problems.append(f"the leg to {stop} would resume {source} at step "
                        f"{step}, want {want} or more and under {stop}")
    if not os.path.isfile(sidecar(source)):
        problems.append(
            f"no optimizer state at {sidecar(source)}: train.py loads the "
            f"weights, prints 'starting fresh' and counts from step 0")
    return problems


def check_leg_done(root, stop: int, cell_tail: str,
                   train_tail: str) -> list[str]:
    """What is wrong after the leg that targets `stop` exits 0.

    `cell_tail` and `train_tail` are the two logs' new text, from before
    the leg started. Three outcomes reach here. The launcher skipped a leg
    it had already run. The launcher resumed, and `train.py` reports the
    step it started at. Or the launcher found no checkpoint and started at
    step 0, which is the one outcome the card must not have.
    """
    want = prior_stop(stop)
    problems = []
    produced = ckpt_path(root, stop)
    if produced is None:
        problems.append(f"no checkpoint at step {stop} under "
                        f"{leg_dir(root, stop)}")
    elif not os.path.isfile(sidecar(produced)):
        problems.append(f"no optimizer state at {sidecar(produced)}: the "
                        f"next leg would restart at step 0")

    if LEG_FRESH in cell_tail:
        problems.append(f"the leg to {stop} started at step 0, not {want}")
        return problems
    if LEG_SKIP in cell_tail:
        return problems
    if LEG_RESUME.search(cell_tail) is None:
        problems.append("the leg wrote no RESUME, FRESH or SKIP line into "
                        "the launcher log")
        return problems
    started = [int(x) for x in TRAIN_RESUME.findall(train_tail)]
    if not started:
        problems.append("train.py printed no 'Resumed from ... at step' "
                        "line, so the step it started at is unknown")
        return problems
    step = started[-1]
    if step < want or step >= stop:
        problems.append(f"the leg to {stop} started at step {step}, want "
                        f"{want} or more and under {stop}")
    return problems


def read_since(path, offset: int = 0) -> str:
    """The text a file gained after byte `offset`. Empty when it is absent."""
    try:
        with open(path, "rb") as fh:
            fh.seek(int(offset))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def score_path(results, stop: int, head: str) -> str:
    """The file one (stop, head) writes its number into.

    `stop_k.sh` builds the same path from `$RES` and its own `$TAG`, and
    hands it to `eval_local.sh` as `SCORE_OUT`.
    """
    return os.path.join(str(results), f"score_{tag(stop, head)}.txt")


def score(stop: int, head: str, results=RESULTS):
    """One stop's GM-Relative MASE, or None when it has not been scored.

    `eval_local.sh` writes exactly one number per (stop, head), into
    `score_<tag>.txt`. Reading that file is how #373's own tables read a
    score, so this study cannot disagree with the parent about what a
    number means.
    """
    if results is None:
        return None
    try:
        with open(score_path(results, stop, head)) as fh:
            return float(fh.read().strip())
    except (OSError, ValueError):
        return None


def check_score(wt, stop: int, head: str) -> list[str]:
    """What is wrong after `stop_k.sh` exits 0 for one (stop, head).

    A clean exit code is not a score. `eval_local.sh` writes the number
    last, and it stops before that line when the merged CSV is short of the
    97 configs. The pair then reaches `collect.sh`, which drops it, and the
    figure draws a shorter line that reads as a finished study.

    The file is the one `stop_k.sh` writes, in #373's results directory.
    `collect.sh` copies it into this study later, with the eval beside it.
    """
    if score(stop, head, parent_results(wt)) is not None:
        return []
    return [f"no GM-Relative MASE at "
            f"{score_path(parent_results(wt), stop, head)}: "
            "the head exited 0 and scored nothing"]


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


def report(problems: list[str], ok: str) -> int:
    """Print a gate's verdict and give the exit code that goes with it."""
    for line in problems:
        print(f"ABORT: {line}", file=sys.stderr)
    if not problems:
        print(ok)
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-resume", metavar="ROOT",
                    help="verify the checkpoint the card pins, under ROOT")
    ap.add_argument("--check-leg", type=int, metavar="STOP",
                    help="verify that the leg to STOP would continue the run")
    ap.add_argument("--check-leg-done", type=int, metavar="STOP",
                    help="verify that the leg to STOP did continue the run")
    ap.add_argument("--check-score", type=int, metavar="STOP",
                    help="verify that STOP's head wrote a score")
    ap.add_argument("--head", choices=HEADS, help="which head to ask about")
    ap.add_argument("--log-paths", action="store_true",
                    help="print the launcher log and the train log")
    ap.add_argument("--ckpt-at", type=int, metavar="STOP",
                    help="print the checkpoint STOP wrote, under --root")
    ap.add_argument("--root", help="the durable root that holds the legs")
    ap.add_argument("--wt", help="the checkout that holds #373's results")
    ap.add_argument("--since-cell", type=int, default=0,
                    help="byte offset into the launcher log")
    ap.add_argument("--since-train", type=int, default=0,
                    help="byte offset into the train log")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--parent", default=PARENT_RESULTS)
    a = ap.parse_args()

    if a.log_paths:
        if not a.wt:
            print("ABORT: --log-paths needs --wt", file=sys.stderr)
            return 2
        print(cell_log(a.wt))
        print(train_log(a.wt))
        return 0

    if a.ckpt_at is not None:
        if not a.root:
            print("ABORT: --ckpt-at needs --root", file=sys.stderr)
            return 2
        found = ckpt_path(a.root, a.ckpt_at)
        if not found or not os.path.isfile(found):
            print(f"ABORT: no checkpoint at step {a.ckpt_at} under "
                  f"{leg_dir(a.root, a.ckpt_at)}", file=sys.stderr)
            return 3
        print(found)
        return 0

    if a.check_resume:
        return report(check_resume(a.check_resume),
                      f"resume OK: {leg_dir(a.check_resume, RESUME_STEP)}")

    if a.check_leg is not None:
        if not a.root:
            print("ABORT: --check-leg needs --root", file=sys.stderr)
            return 2
        stop = a.check_leg
        source = resume_source(a.root)
        return report(check_leg_start(a.root, stop),
                      f"leg {stop} will continue from "
                      f"{os.path.basename(source) if source else 'nothing'}")

    if a.check_leg_done is not None:
        if not a.root or not a.wt:
            print("ABORT: --check-leg-done needs --root and --wt",
                  file=sys.stderr)
            return 2
        stop = a.check_leg_done
        return report(
            check_leg_done(a.root, stop,
                           read_since(cell_log(a.wt), a.since_cell),
                           read_since(train_log(a.wt), a.since_train)),
            f"leg {stop} continued the run")

    if a.check_score is not None:
        if not a.wt or not a.head:
            print("ABORT: --check-score needs --wt and --head",
                  file=sys.stderr)
            return 2
        stop = a.check_score
        got = score(stop, a.head, parent_results(a.wt))
        return report(check_score(a.wt, stop, a.head),
                      f"{a.head} scored {got} at {stop}")

    print(f"{CELL_ID}  {CELL}  k={K}  seed 20260520")
    print(f"resume  {RESUME_NAME}.pth  (+ optimizer)")
    print(f"stops   {', '.join(str(s) for s in STOPS)}")
    print(f"heads   {', '.join(HEADS)}  {HEAD_STEPS} steps  seed {HEAD_SEED}")
    try:
        print(f"best before this card: {best_before(a.parent):.4f} "
              f"({BEST_BEFORE_TAG})")
    except (OSError, ValueError):
        print(f"best before this card: not on disk ({BEST_BEFORE_TAG})")
    for head in HEADS:
        points = curve(head, a.results, a.parent)
        row = "  ".join(f"bb{s // STOP_UNIT}k={v:.4f}"
                        for s, v in sorted(points.items()))
        print(f"{head:<8} {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
