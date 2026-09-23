#!/bin/bash
# #415 — the whole leg: train the value-space reference to each stop, then
# score that stop.
#
# Usage:  BB_GPU=0 bash reports/2026-09-23_value_space_reference/run.sh
#         CF415_DRY_RUN=1 bash .../run.sh        # print every command
#         bash .../run.sh 40000 100000           # a subset of the stops
#
# The card trains ONE model and scores it eight times. Each stop resumes the
# previous stop's checkpoint with its optimizer state, so the eight points sit
# on one trajectory — the shape #414 measures. 665,000 steps is one pass over
# the data.
#
# Both halves are idempotent: a stop whose checkpoint is on disk trains
# nothing, and a stop whose score file exists scores nothing. So a re-fire
# after a crash costs only what was lost.
#
# The head and the eval run AFTER their stop's backbone, on the same box. The
# eval is CPU work (#393 measured it: 32 cores beat two contended GPUs), so it
# does not hold the card the next leg needs.
set -uo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/scripts/paths.sh"

# The eight stops live in `paths.sh`, because `head_eval_value.sh` validates
# against them. A second list here would let a stop train and then fail to
# score.
STOPS="$CF415_STOPS"
[ "$#" -gt 0 ] && STOPS="$*"

BB_GPU="${BB_GPU:-0}"
# A dry run prints and writes nothing, so the guards can run the whole ladder
# without leaving a log behind in a shared checkout.
if [ -n "${CF415_DRY_RUN:-}" ]; then
  log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#415] $*"; }
else
  mkdir -p "$CF415_RESULTS"
  log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#415] $*" \
    | tee -a "$CF415_RESULTS/run.log"; }
fi

for stop in $STOPS; do
  log "stop $stop: train"
  BB_GPU="$BB_GPU" bash "$CF415_SCRIPTS/run_leg_value.sh" "$stop" || {
    rc=$?; log "stop $stop: train rc=$rc — stopping the ladder"; exit $rc; }
  log "stop $stop: head + 97-config GIFT-Eval"
  BB_GPU="$BB_GPU" bash "$CF415_SCRIPTS/head_eval_value.sh" "$stop" || {
    rc=$?; log "stop $stop: score rc=$rc — stopping the ladder"; exit $rc; }
  log "stop $stop: GM-Relative MASE $(cat "$CF415_RESULTS/score_$(cf415_tag "$stop").txt" 2>/dev/null)"
done
log "DONE — scores in $CF415_RESULTS/score_*.txt"
