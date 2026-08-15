#!/bin/bash
# #401 — the whole study, in the order the card runs it.
#
#   1. smoke     k = 0 and k = 16, a few hundred steps each. Records the step
#                time and the peak GPU memory of THIS run's process tree. The
#                run plan below depends on both, so it runs first and the
#                numbers land in results/smoke_k16.csv.
#   2. phase 1   three arms, k = 16 then 8 then 32, each to 40k / 100k / 200k
#                backbone steps. One student head at 30,000 steps per stop,
#                then that head's 97-config GIFT-Eval.
#   3. phase 2   the two best arms again, with the head budget matched to the
#                backbone stop: 40k / 100k / 200k head steps.
#
# Each stage is idempotent. A stop whose checkpoint is on disk is a no-op, a
# head whose score file is written is a no-op, and a GIFT-Eval resumes per
# shard. So a re-run after a crash costs only what did not finish.
#
# Usage:  BB_GPU=0 bash run.sh              # everything
#         BB_GPU=0 bash run.sh phase1       # one stage
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGES="${*:-smoke phase1 phase2}"

for stage in $STAGES; do
  case "$stage" in
    smoke)  bash "$HERE/scripts/smoke_k16.sh" "${SMOKE_STEPS:-300}" ;;
    phase1) bash "$HERE/scripts/phase1.sh" ;;
    phase2) bash "$HERE/scripts/phase2.sh" ;;
    *) echo "ABORT: unknown stage '$stage' (smoke phase1 phase2)" >&2; exit 2 ;;
  esac
  rc=$?
  [ $rc -eq 0 ] || { echo "ABORT: stage '$stage' rc=$rc" >&2; exit $rc; }
done
