#!/usr/bin/env python3
"""#393 backbone-ladder driver.

Ten cells, one per row of the issue's union table. Each cell is ONE
continuous run: train to a stop, checkpoint, evaluate, then resume from
that checkpoint with its optimizer state and carry on. The stops are 40k
and 100k unconditionally, then 100k at a time.

At every stop the checkpoint gets two quantile heads, trained and
evaluated separately: one on the student encoder, one on the EMA
teacher's. `--encoder-source` (#393) selects which, and the head's
recorded source is checked at eval time, so a teacher head can never
produce a student number.

The extend rule reads the two GM-Relative MASE values against the same
head's value at the previous stop:

  * both heads down  -> extend, keep evaluating both;
  * one head down    -> extend, evaluate only that head from then on;
  * neither down     -> stop the run.

A run also stops when one pass over the dataset is exhausted. That cap is
derived, not assumed: `small_v1` holds 42,571,692 rows and the trainer
consumes `(batch_size - synth_bs - cross_bs) * C` of them per step.

Everything above the shell boundary is a pure function so the decisions
are checkable without a GPU — see `tests/test_393_ladder.py`.

Usage:
    WT=<checkout> RUNS=<checkpoint dir> BB_GPU=0 \
      python3 ladder.py --cells arm6_v2_combab_alignS,arm6_v2_combab_alignT
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPTS_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import ema_tau_at_step

# --- protocol constants ---------------------------------------------------

# The ten runs. `arm` is the #379 recipe (experiments/2026-07-21_
# split_pred_rep_small/scripts/run_arm.sh); `align` is L_align's target,
# None for the two cells that carry no L_align term. Runs 1 and 2 are the
# `arm6_v2 combab` head-to-head, first in the queue because that cell leads
# both parent reports.
CELLS = (
    {"slug": "arm6_v2_combab_alignS", "arm": "arm6_v2_combab", "align": "student"},
    {"slug": "arm6_v2_combab_alignT", "arm": "arm6_v2_combab", "align": "teacher"},
    {"slug": "arm5_combab_alignS",    "arm": "arm5_combab",    "align": "student"},
    {"slug": "arm5_combab_alignT",    "arm": "arm5_combab",    "align": "teacher"},
    {"slug": "arm6_v2_ncpc_alignS",   "arm": "arm6_v2_ncpc",   "align": "student"},
    {"slug": "arm6_v2_ncpc_alignT",   "arm": "arm6_v2_ncpc",   "align": "teacher"},
    {"slug": "arm6_v2_nse_alignS",    "arm": "arm6_v2_nse",    "align": "student"},
    {"slug": "arm6_v2_nse_alignT",    "arm": "arm6_v2_nse",    "align": "teacher"},
    {"slug": "arm4_combab",           "arm": "arm4_combab",    "align": None},
    {"slug": "arm1_nse",              "arm": "arm1_nse",       "align": None},
)

HEADS = ("student", "teacher")

# Durable root for everything a run produces that cost GPU time. Kept
# equal to `RUNS_DEFAULT` in scripts/leg_paths.sh, which the two launchers
# read; tests/test_393_ladder.py pins the two together.
RUNS_DEFAULT = "/home/jupyter/checkpoints_backup/cf-393"

STOP_FIRST = 40_000
STOP_SECOND = 100_000
STOP_INCREMENT = 100_000
HEAD_STEPS_FIRST = 15_000
HEAD_STEPS_LATER = 30_000

# α rises linearly 0.9 -> 1.0 over steps 0..100k, then holds. Anchored to
# the step count, not to any run's budget.
EMA_TAU_START = 0.9
EMA_TAU_END = 1.0
EMA_TAU_RAMP_STEPS = 100_000

# Backbone data recipe, mirroring run_leg.sh. Used to derive the step cap.
BATCH_SIZE = 64
MIX_RATIO = 0.0078125
CROSSFADE_RATIO = 0.0
CROSS_TRIPLETS = 1
N_CHANNELS = 1
# small_v1/manifest.json on jeremycochoy/gift-pretrain-full-4096, confirmed
# 2026-08-04: {"total_rows": 42571692, "num_shards": 4274}. NOT the 167k-step
# figure from the 2026-05-03 run, which was that run's batch size of 256.
SMALL_V1_ROWS = 42_571_692

LADDER_COLUMNS = ["cell", "arm", "align", "stop", "head", "head_steps",
                  "ema_tau", "gm_rel_mase"]
DECISION_COLUMNS = ["cell", "stop", "branch", "extend", "heads_next"]


# --- the ladder, as pure functions ----------------------------------------


def next_stop(step: int) -> int:
    """The stop after `step`: 40k, then 100k, then 100k at a time."""
    if step < STOP_FIRST:
        return STOP_FIRST
    if step < STOP_SECOND:
        return STOP_SECOND
    return step + STOP_INCREMENT


def head_steps_for(stop: int) -> int:
    """Head budget at a stop: 15k at bb40k, 30k from bb100k on."""
    return HEAD_STEPS_FIRST if stop < STOP_SECOND else HEAD_STEPS_LATER


def batch_composition(batch_size: int, mix_ratio: float,
                      crossfade_ratio: float, cross_triplets: int,
                      channels: int) -> dict:
    """How one step's batch splits, and how many real rows it costs.

    Mirrors train.py (`hf_bs = batch_size - synth_bs - cross_bs`,
    `hf_rows_per_step = hf_bs * C`) and `create_mixed_forked_arma_dataloader`.

    Two details decide this experiment's cap, and neither is guessable from
    the flag values:

    * `synth_bs = int(round(batch_size * mix_ratio))`. At `batch_size=64`
      and `mix_ratio=1/128` that product is exactly 0.5, and Python rounds
      half to EVEN, so `synth_bs` is 0. The nominal 1/128 synthetic
      fraction rounds away and every row in the batch is a real one.
    * Crossfade TRIPLETS are blended from the real sub-batch and appended
      on top (`3 * cross_triplets` rows), so they widen the batch the model
      sees without consuming another dataset row.

    Crossfade ROWS (`crossfade_ratio`) do shrink the real sub-batch, and
    are counted.
    """
    synth_bs = int(round(batch_size * mix_ratio))
    cross_bs = int(round(batch_size * crossfade_ratio))
    hf_bs = batch_size - synth_bs - cross_bs
    triplet_rows = 3 * cross_triplets
    return {
        "batch_size": batch_size,
        "hf_bs": hf_bs,
        "synth_bs": synth_bs,
        "cross_bs": cross_bs,
        "triplet_rows": triplet_rows,
        "total_batch_rows": batch_size + triplet_rows,
        "channels": channels,
        "hf_rows_per_step": hf_bs * channels,
    }


def step_cap(total_rows: int, rows_per_step: int) -> int:
    """Steps in one pass over the dataset — the point where a run has shown
    every sample once. Extending past it repeats data."""
    return total_rows // rows_per_step


def experiment_batch_composition() -> dict:
    """The batch this experiment's recipe actually builds."""
    return batch_composition(BATCH_SIZE, MIX_RATIO, CROSSFADE_RATIO,
                             CROSS_TRIPLETS, N_CHANNELS)


def experiment_step_cap() -> int:
    """The cap for this experiment's recipe."""
    return step_cap(SMALL_V1_ROWS,
                    experiment_batch_composition()["hf_rows_per_step"])


def ladder_decision(stop: int, previous: dict | None, current: dict,
                    step_cap: int | None = None) -> dict:
    """Whether to extend past `stop`, and which heads carry on.

    `previous` / `current` map head name -> GM-Relative MASE. Lower is
    better, so a head "went down" when its value decreased. Only heads
    present in both are compared: a head dropped at an earlier stop stays
    dropped. `previous=None` is the 40k stop, which extends unconditionally.

    Returns {"extend": bool, "heads": tuple, "branch": str}.
    """
    heads = tuple(h for h in HEADS if h in current)
    if step_cap is not None and next_stop(stop) > step_cap:
        return {"extend": False, "heads": heads, "branch": "data_exhausted"}
    if previous is None or stop < STOP_SECOND:
        return {"extend": True, "heads": heads, "branch": "unconditional"}
    compared = tuple(h for h in heads if h in previous)
    down = tuple(h for h in compared if current[h] < previous[h])
    if not down:
        return {"extend": False, "heads": heads, "branch": "none_down"}
    branch = "both_down" if len(down) == len(compared) > 1 else "one_down"
    return {"extend": True, "heads": down, "branch": branch}


# --- CSV state ------------------------------------------------------------


def read_rows(path: str, columns: list[str]) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if rows and list(rows[0]) != columns:
        raise SystemExit(f"{path}: schema mismatch — expected {columns}, "
                         f"found {list(rows[0])}")
    return rows


def append_row(path: str, columns: list[str], row: list) -> None:
    fresh = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if fresh:
            writer.writerow(columns)
        writer.writerow(row)


def scores_at(rows: list[dict], cell: str, stop: int) -> dict:
    """{head: GM-Relative MASE} already recorded for a cell at a stop."""
    return {r["head"]: float(r["gm_rel_mase"]) for r in rows
            if r["cell"] == cell and int(r["stop"]) == stop
            and r["gm_rel_mase"] != ""}


# --- shell boundary -------------------------------------------------------


def runs_root() -> str:
    """The durable root, refusing the two paths that lose work.

    `git worktree remove --force` deletes every untracked file under the
    checkout (CLAUDE.md checkpoint safety rule 4) and /tmp does not survive
    a reboot. Mirrors `runs_root()` in scripts/leg_paths.sh.
    """
    root = os.environ.get("RUNS") or RUNS_DEFAULT
    checkout = os.environ.get("WT")
    inside = checkout and (root == checkout
                           or root.startswith(checkout.rstrip("/") + "/"))
    if root == "/tmp" or root.startswith("/tmp/") or inside:
        raise SystemExit(f"RUNS={root} is under /tmp or inside the checkout. "
                         f"Point it at a durable path, e.g. {RUNS_DEFAULT}.")
    return root


def run(cmd: list[str], env: dict) -> int:
    print(f"[ladder] $ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, env=env)


def train_leg(cell: str, target: int, env: dict) -> None:
    """Train `cell` up to `target`, resuming its own newest checkpoint."""
    rc = run(["bash", os.path.join(SCRIPTS_DIR, "run_leg.sh"), cell,
              str(target)], env)
    if rc != 0:
        raise SystemExit(f"[ladder] {cell}: run_leg.sh rc={rc} at {target}")


def evaluate(cell: str, stop: int, head: str, env: dict) -> float:
    """Train one head on the stop's checkpoint and GIFT-Eval it.

    eval_stop.sh writes the aggregate GM-Relative MASE, and nothing else,
    to the score path. That path sits on the durable root next to the head
    it came from, so the number survives a worktree teardown even if
    `results/ladder.csv` does not.
    """
    score_path = os.path.join(runs_root(), cell, "eval",
                              f"score_bb{stop // 1000}k_{head}.txt")
    rc = run(["bash", os.path.join(SCRIPTS_DIR, "eval_stop.sh"), cell,
              str(stop), head, str(head_steps_for(stop)), score_path], env)
    if rc != 0:
        raise SystemExit(f"[ladder] {cell}: eval_stop.sh rc={rc} "
                         f"at {stop} ({head})")
    with open(score_path) as fh:
        return float(fh.read().strip())


def alpha_at(step: int) -> float:
    """α at a global step, from the function the trainer calls every step.

    The budget argument is dead once the anchor is set; passing the anchor
    keeps the call reading as the fixed-step schedule it is.
    """
    return ema_tau_at_step(step, EMA_TAU_RAMP_STEPS, EMA_TAU_START,
                           EMA_TAU_END, EMA_TAU_RAMP_STEPS)


def record_decision(path: str, slug: str, stop: int, branch: str,
                    extend: bool, heads) -> None:
    """One row per stop, including the two that end a session early.

    `--max-stop` and the data cap both leave the cell mid-ladder, and a
    reader of decisions.csv has to be able to tell either from a cell that
    is still running.
    """
    append_row(path, DECISION_COLUMNS,
               [slug, stop, branch, int(extend), " ".join(heads)])


def stop_scores(cell: dict, stop: int, heads: tuple, recorded: dict,
                env: dict, ladder_csv: str) -> dict:
    """{head: GM-Relative MASE} at a stop, evaluating only what is missing.

    `recorded` is what ladder.csv already holds for this cell and stop, so
    a crashed cell does not re-train a 30k-step head it already has.
    """
    scores = {}
    for head in heads:
        if head in recorded:
            scores[head] = recorded[head]
            continue
        scores[head] = evaluate(cell["slug"], stop, head, env)
        append_row(ladder_csv, LADDER_COLUMNS,
                   [cell["slug"], cell["arm"], cell["align"] or "", stop,
                    head, head_steps_for(stop), f"{alpha_at(stop):.6f}",
                    f"{scores[head]:.6f}"])
    return scores


def session_hold(results_dir: str) -> int | None:
    """A ceiling on how far any cell may climb this session, or None.

    `--max-stop` does the same thing, but it is fixed when the driver
    starts. This is read from `results/HOLD_ABOVE` on every stop, so the
    spend order can be changed under drivers that are already running —
    "all ten cells to bb100k first, extensions with what is left" was
    decided after five cells were already climbing.

    Stopping the cell here rather than raising keeps the driver alive for
    the cells behind it in a `--cells a,b` invocation.
    """
    path = os.path.join(results_dir, "HOLD_ABOVE")
    try:
        with open(path) as fh:
            return int("".join(c for c in fh.read() if c.isdigit()))
    except (OSError, ValueError):
        return None


def climb(cell: dict, env: dict, ladder_csv: str, decisions_csv: str,
          max_stop: int | None) -> None:
    """Walk one cell up the ladder until the rule or the data stops it.

    Resumable: the walk replays from step 0 every time, and each stop whose
    checkpoint and scores are already on disk costs nothing — run_leg.sh
    skips a leg whose target checkpoint exists, and `stop_scores` skips a
    head ladder.csv already holds. That replay is also what rebuilds
    `previous` and `heads` after a crash.
    """
    cap = experiment_step_cap()
    rows = read_rows(ladder_csv, LADDER_COLUMNS)
    slug = cell["slug"]
    stop, previous, heads = 0, None, HEADS
    while True:
        target = next_stop(stop)
        if target > cap:
            record_decision(decisions_csv, slug, stop, "data_exhausted",
                            False, heads)
            print(f"[ladder] {slug}: step cap {cap} reached")
            return
        if max_stop is not None and target > max_stop:
            record_decision(decisions_csv, slug, stop, "session_end",
                            True, heads)
            print(f"[ladder] {slug}: --max-stop {max_stop} reached")
            return
        hold = session_hold(os.path.dirname(ladder_csv))
        if hold is not None and target > hold:
            record_decision(decisions_csv, slug, stop, "session_end",
                            True, heads)
            print(f"[ladder] {slug}: HOLD_ABOVE {hold} reached")
            return
        train_leg(slug, target, env)
        current = stop_scores(cell, target, heads,
                             scores_at(rows, slug, target), env, ladder_csv)
        decision = ladder_decision(target, previous, current, step_cap=cap)
        record_decision(decisions_csv, slug, target, decision["branch"],
                        decision["extend"], decision["heads"])
        print(f"[ladder] {slug} @{target}: {current} -> {decision}")
        if not decision["extend"]:
            return
        stop, previous, heads = target, current, decision["heads"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--cells", default=None,
                   help="Comma-separated cell slugs; default all ten, in the "
                        "issue's order (the arm6_v2 combab pair first).")
    p.add_argument("--max-stop", type=int, default=None,
                   help="Stop the ladder at this backbone step even when the "
                        "rule would extend. For splitting a cell across "
                        "machines or sessions; the run resumes from its own "
                        "checkpoint on the next invocation.")
    p.add_argument("--print-cap", action="store_true",
                   help="Print the derived step cap and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.print_cap:
        comp = experiment_batch_composition()
        print(f"small_v1 rows: {SMALL_V1_ROWS}")
        print(f"batch:         {comp['hf_bs']} real + {comp['synth_bs']} synth "
              f"+ {comp['cross_bs']} crossfade + {comp['triplet_rows']} "
              f"triplet = {comp['total_batch_rows']} rows")
        print(f"rows/step:     {comp['hf_rows_per_step']} real "
              f"(batch {BATCH_SIZE}, mix {MIX_RATIO}, C {N_CHANNELS})")
        print(f"step cap:      {experiment_step_cap()}")
        return
    wanted = args.cells.split(",") if args.cells else [c["slug"] for c in CELLS]
    by_slug = {c["slug"]: c for c in CELLS}
    unknown = [s for s in wanted if s not in by_slug]
    if unknown:
        raise SystemExit(f"unknown cell(s): {unknown}; "
                         f"valid: {sorted(by_slug)}")
    results = os.path.join(EXP_DIR, "results")
    os.makedirs(results, exist_ok=True)
    env = os.environ.copy()
    for slug in wanted:
        climb(by_slug[slug], env,
              os.path.join(results, "ladder.csv"),
              os.path.join(results, "decisions.csv"),
              args.max_stop)


if __name__ == "__main__":
    main()
