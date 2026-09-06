#!/bin/bash
# #412 — the whole card on ONE machine, in the order it runs.
#
#   1. size     the parameter count of each candidate shape, and of the model
#               today. No GPU, seconds. It is the measurement the width rests
#               on.
#   2. smoke    each arm for a few tens of steps, through the same wrapper and
#               the same runner. It reads the shape back off the trainer's
#               command line, counts the depth columns, and records the step
#               time and the memory the run plan is sized from.
#   3. trial    one arm end to end: the backbone leg, its head, and ONE
#               GIFT-Eval config. The head trainer and the evaluation rebuild
#               the backbone from flags, so this is where a wrong width shows.
#   4. phase1   the five arms to each stop in turn, one 30,000-step student
#               head for each stop, then that head's 97 GIFT-Eval configs.
#   5. collect  every score in one CSV.
#
# Each stage is idempotent. An arm whose checkpoint is on disk is a no-op, a
# head whose score file is written is a no-op, and a GIFT-Eval resumes for
# each shard. So a re-run after a crash costs only what did not finish.
#
# Both GPUs of elisa carry other work, so an arm waits for a free card through
# #373's `gpu_gate.sh`. Run the stages on two cards by giving each its own
# `ARMS` list and its own `BB_GPU`.
#
# Usage:  BB_GPU=0 bash run.sh              # everything
#         BB_GPU=0 bash run.sh size smoke   # some stages
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGES="${*:-size smoke trial phase1 collect}"

for stage in $STAGES; do
  case "$stage" in
    size)    bash "$HERE/scripts/size_table.sh" ;;
    smoke)   bash "$HERE/scripts/smoke.sh" "${SMOKE_STEPS:-60}" ;;
    trial)   bash "$HERE/scripts/trial.sh" ;;
    phase1)  bash "$HERE/scripts/phase1.sh" ;;
    collect) bash "$HERE/scripts/collect.sh" ;;
    *) echo "ABORT: unknown stage '$stage'" >&2
       echo "  (size smoke trial phase1 collect)" >&2; exit 2 ;;
  esac
  rc=$?
  [ $rc -eq 0 ] || { echo "ABORT: stage '$stage' rc=$rc" >&2; exit $rc; }
done
