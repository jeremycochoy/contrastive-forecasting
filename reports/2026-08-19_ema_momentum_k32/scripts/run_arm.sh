#!/bin/bash
# #404 — one arm's backbone: train the cell to <stop> steps at this arm's
# EMA momentum.
#
# This is a wrapper, on purpose. The trainer command line for this
# configuration lives in ONE place, #373's `run_leg_k.sh`, and a copy of it
# here would be a second protocol that drifts. The wrapper supplies six things
# that runner takes from the environment:
#
#   K              the rollout depth, 32
#   EMA_ARGS       this arm's momentum, which REPLACES the runner's own
#                  schedule. Two of the four arms hold alpha fixed and pass no
#                  --ema-tau-end at all, and no repeated flag can remove one.
#   GAP_ARGS       `--train-rollout-reduce mean`, appended LAST to the trainer
#                  command line
#   RUN_SUFFIX     the reduction and the arm, in the run name, so no arm's
#                  checkpoints and losses CSV can be read as another's
#   RUNS           this arm's durable root
#   CF_RESULTS     this study's results/, so the leg's log lands here
#
# The runner is idempotent: a stop whose checkpoint is on disk is a no-op, and
# a leg resumes the cell's furthest checkpoint with its optimizer state. So a
# re-fired leg after a crash costs nothing.
#
# ---- The momentum has to reach the trainer -----------------------------------
#
# The four arms share a configuration and differ in alpha alone. So an arm
# whose alpha did not arrive is a DUPLICATE of another arm, under a name that
# says otherwise: same file names, same CSV columns, same log lines. The card's
# result is four numbers, and two of them would be one number twice.
#
# The trainer's own command line is the one place that names alpha. So this
# script starts the leg, waits for that line to land in the leg log, and reads
# both alpha and the reduction off it. A leg with the wrong objective stops in
# its first minute instead of at hour five.
#
# The command line is the trainer's FIRST log line, so this costs no window.
# The count of those lines before the start is what tells this leg's line from
# the lines of the legs below it — the runner appends to one log per cell.
#
# Usage:  run_arm.sh <arm> <stop steps>
#         BB_GPU=0 bash run_arm.sh a08 40000
#         CF404_DRY_RUN=1 bash run_arm.sh a08 40000     # print, do not run
set -uo pipefail

ARM="${1:?usage: run_arm.sh <arm> <stop steps>}"
STOP="${2:?usage: run_arm.sh <arm> <stop steps>}"

. "$(dirname "${BASH_SOURCE[0]}")/study.sh"
cf404_require_arm "$ARM" || exit $?
cf404_require_stop "$STOP" || exit $?

RUNNER="$CF404_PARENT/scripts/run_leg_k.sh"
[ -f "$RUNNER" ] || { echo "ABORT: no runner at $RUNNER" >&2; exit 2; }

BB_GPU="${BB_GPU:-0}"
mkdir -p "$CF404_RESULTS"

ARM_ROOT="$(cf404_arm_root "$ARM")"
EMA_ARGS="$(cf404_ema_args "$ARM")"
# The reduction is stated on every leg, so the log names the objective it
# trained rather than leaving the reader to infer the trainer's default.
REDUCE_ARGS="--train-rollout-reduce $CF404_REDUCE"

# Fault injection, for the test that proves the check below fires. It hands
# the trainer a momentum this arm does not carry, which is what a wiring
# defect does. Nothing in the study sets it.
[ -n "${CF404_FORCE_EMA:-}" ] && EMA_ARGS="$CF404_FORCE_EMA"

if [ -n "${CF404_DRY_RUN:-}" ]; then
  echo "arm $ARM cell=$CF404_CELL k=$CF404_K steps=$STOP gpu=$BB_GPU"
  echo "  ema=$EMA_ARGS"
  echo "  reduce=$CF404_REDUCE runner=$RUNNER"
  echo "  RUN_SUFFIX=$(cf404_run_suffix "$ARM") RUNS=$ARM_ROOT"
  echo "  CF_RESULTS=$CF404_RESULTS"
  echo "  ckpt=$(cf404_leg_dir "$ARM" "$STOP")/$(cf404_run_name "$ARM")_$(( STOP / 1000 ))k.pth"
  exit 0
fi

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404] $*" \
  | tee -a "$CF404_RESULTS/arms.log"; }

TLOG="$(cf404_leg_log "$ARM")"
CHECK_TIMEOUT="${CF404_CHECK_TIMEOUT:-1800}"
cmdlines_before="$(cf404_cmdlines "$TLOG")"

log "arm $ARM ema='$EMA_ARGS' reduce=$CF404_REDUCE -> ${STOP} steps on gpu $BB_GPU"
K="$CF404_K" RUNS="$ARM_ROOT" CF_RESULTS="$CF404_RESULTS" WT="$CF404_WT" \
  EMA_ARGS="$EMA_ARGS" GAP_ARGS="$REDUCE_ARGS" \
  RUN_SUFFIX="$(cf404_run_suffix "$ARM")" \
  BB_GPU="$BB_GPU" \
  bash "$RUNNER" "$CF404_CELL" "$STOP" &
runner=$!

# Wait for THIS leg's command line. The loop also ends when the runner does:
# `run_leg_k.sh` exits without a trainer on a stop already on disk, on a cell
# another machine claims and on a session HOLD, and none of those is a wrong
# objective.
waited=0
while [ "$(cf404_cmdlines "$TLOG")" -le "$cmdlines_before" ]; do
  kill -0 "$runner" 2>/dev/null || break
  if [ "$waited" -ge "$CHECK_TIMEOUT" ]; then
    log "arm $ARM WARNING: no trainer command line in ${CHECK_TIMEOUT}s —" \
        "the momentum is unchecked. See $TLOG"
    break
  fi
  sleep 5; waited=$(( waited + 5 ))
done

line="$(cf404_last_cmdline "$TLOG" 2>/dev/null)"
if [ -n "$line" ]; then
  got_ema="$(printf '%s' "$line" | cf404_ema_of_cmdline)"
  got_red="$(printf '%s' "$line" | cf404_reduce_of_cmdline)"
  want_ema="$(cf404_ema_sig "$ARM")"
  if [ "$got_ema" != "$want_ema" ] || [ "$got_red" != "$CF404_REDUCE" ]; then
    cf404_kill_tree "$runner"
    wait "$runner" 2>/dev/null
    log "arm $ARM STOPPED — trained '$got_ema' / '$got_red'," \
        "not '$want_ema' / '$CF404_REDUCE'"
    echo "ABORT: this leg's trainer runs the momentum '$got_ema' under the" >&2
    echo "  reduction '$got_red', and arm '$ARM' is '$want_ema' under" >&2
    echo "  '$CF404_REDUCE'. The values are <tau> <end> <ramp>, with '-' for" >&2
    echo "  a flag the command line does not carry. Every arm of this card" >&2
    echo "  writes the same file names, so the leg is stopped rather than" >&2
    echo "  left to climb. Its command line is the last 'Command line:' in" >&2
    echo "  $TLOG" >&2
    exit 3
  fi
  log "arm $ARM ema='$got_ema' reduce=$got_red OK — both reached the trainer"
fi

wait "$runner"; rc=$?
log "arm $ARM stop=$STOP rc=$rc"
exit $rc
