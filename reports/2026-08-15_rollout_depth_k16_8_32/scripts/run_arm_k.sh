#!/bin/bash
# #401 — one leg of one arm: train the cell to <stop> steps at depth <k>.
#
# This is a wrapper, on purpose. The trainer command line for this
# configuration lives in ONE place, #373's `run_leg_k.sh`, and a copy of it
# here would be a second protocol that drifts. The wrapper supplies six
# things #373's runner takes from the environment:
#
#   K              the rollout depth, 8 or 32
#   GAP_ARGS       `--train-rollout-reduce <sum|mean>`, appended LAST to the
#                  trainer command line. #373's runner already forwards this
#                  variable for its own control runs, so the reduction rides
#                  in without a second trainer invocation.
#   RUN_SUFFIX     the reduction, in the run name, so the mean arm's
#                  checkpoints and losses CSV cannot be read as the summed
#                  arm's
#   RUNS           this study's durable root, so no checkpoint mixes with a
#                  #373 one
#   CF_STUDY_DIR   this study's directory, so the leg's log lands here
#   BB_GPU         the card
#
# The runner is idempotent per leg: a stop whose checkpoint is on disk is a
# no-op, and a leg resumes the cell's furthest checkpoint with its optimizer
# state. So a re-fired stop after a crash costs nothing.
#
# ---- The reduction has to reach the trainer -----------------------------------
#
# GAP_ARGS travels through an environment, a wrapper and #373's runner before
# it becomes a trainer argument. Nothing in the artefacts says whether it
# arrived: a mean leg that trained the sum writes the same file names, the same
# CSV columns and the same log lines. So this script starts the leg, waits for
# the trainer's own command line to land in the leg log, and reads the
# reduction off it. A leg with the wrong objective is stopped in its first
# minute instead of at hour 33.
#
# The command line is the trainer's FIRST log line, so this costs no window.
# The count of those lines before the start is what tells this leg's line from
# the lines of the legs below it — the runner appends to one log per cell.
#
# Usage:  run_arm_k.sh <k> <stop steps>
#         BB_GPU=0 bash run_arm_k.sh 8 40000
#         CF401_DRY_RUN=1 bash run_arm_k.sh 8 40000      # print, do not run
set -uo pipefail

K="${1:?usage: run_arm_k.sh <k> <stop steps>}"
STOP="${2:?usage: run_arm_k.sh <k> <stop steps>}"

. "$(dirname "${BASH_SOURCE[0]}")/study.sh"
cf401_require_depth "$K" || exit $?
cf401_require_stop "$STOP" || exit $?

RUNNER="$CF401_PARENT/scripts/run_leg_k.sh"
[ -f "$RUNNER" ] || { echo "ABORT: no runner at $RUNNER" >&2; exit 2; }

BB_GPU="${BB_GPU:-0}"
mkdir -p "$CF401_RESULTS"

ARM_ROOT="$(cf401_arm_root "$K")"

# The one trainer flag this study adds to #373's command line. It is stated
# on every leg, including `sum`, so the leg's log names the objective it
# trained rather than leaving the reader to infer the default.
REDUCE_ARGS="--train-rollout-reduce $CF401_REDUCE"

if [ -n "${CF401_DRY_RUN:-}" ]; then
  echo "arm cell=$CF401_CELL K=$K steps=$STOP gpu=$BB_GPU reduce=$CF401_REDUCE"
  echo "  runner=$RUNNER"
  echo "  GAP_ARGS=$REDUCE_ARGS RUN_SUFFIX=$CF401_RUN_SUFFIX"
  echo "  RUNS=$ARM_ROOT CF_STUDY_DIR=$CF401_STUDY CF_RESULTS=$CF401_RESULTS"
  echo "  ckpt=$(cf401_leg_dir "$K" "$STOP")/$(cf401_run_name "$K")_$(( STOP / 1000 ))k.pth"
  exit 0
fi

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#401] $*" \
  | tee -a "$CF401_RESULTS/arms.log"; }

TLOG="$(cf401_leg_log "$K")"
CHECK_TIMEOUT="${CF401_REDUCE_CHECK_TIMEOUT:-1800}"
cmdlines_before="$(cf401_cmdlines "$TLOG")"

log "arm k=$K reduce=$CF401_REDUCE -> ${STOP} steps on gpu $BB_GPU"
K="$K" RUNS="$ARM_ROOT" CF_STUDY_DIR="$CF401_STUDY" WT="$CF401_WT" \
  CF_RESULTS="$CF401_RESULTS" \
  GAP_ARGS="$REDUCE_ARGS" RUN_SUFFIX="$CF401_RUN_SUFFIX" \
  BB_GPU="$BB_GPU" \
  bash "$RUNNER" "$CF401_CELL" "$STOP" &
runner=$!

# Wait for THIS leg's command line. The loop also ends when the runner does:
# `run_leg_k.sh` exits without a trainer on a stop already on disk, on a cell
# another machine claims and on a session HOLD, and none of those is a wrong
# objective.
waited=0
while [ "$(cf401_cmdlines "$TLOG")" -le "$cmdlines_before" ]; do
  kill -0 "$runner" 2>/dev/null || break
  if [ "$waited" -ge "$CHECK_TIMEOUT" ]; then
    log "arm k=$K WARNING: no trainer command line in ${CHECK_TIMEOUT}s —" \
        "the reduction is unchecked. See $TLOG"
    break
  fi
  sleep 5; waited=$(( waited + 5 ))
done

got="$(cf401_leg_reduce "$TLOG" 2>/dev/null)"
if [ -n "$got" ] && [ "$got" != "$CF401_REDUCE" ]; then
  cf401_kill_tree "$runner"
  wait "$runner" 2>/dev/null
  log "arm k=$K stop=$STOP STOPPED — trained '$got', not '$CF401_REDUCE'"
  echo "ABORT: this leg's trainer runs --train-rollout-reduce '$got'," >&2
  echo "  and this arm is '$CF401_REDUCE'. The two objectives write the" >&2
  echo "  same file names, so the leg is stopped rather than left to" >&2
  echo "  climb. Its command line is the last 'Command line:' in" >&2
  echo "  $TLOG" >&2
  exit 3
fi
[ -z "$got" ] || log "arm k=$K reduce=$got OK — the flag reached the trainer"

wait "$runner"; rc=$?
log "arm k=$K stop=$STOP rc=$rc"
exit $rc
