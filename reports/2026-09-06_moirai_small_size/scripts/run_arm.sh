#!/bin/bash
# #412 — one arm's backbone: train the cell to <stop> steps at 11.4M
# parameters.
#
# This is a wrapper, on purpose. The trainer command line for this
# configuration lives in ONE place, #373's `run_leg_k.sh`, and a copy of it
# here would be a second protocol that drifts. The wrapper supplies seven
# things that runner takes from the environment:
#
#   K              this arm's rollout depth, from the arms table
#   EMA_ARGS       this arm's momentum, which REPLACES the runner's schedule
#   SEED           this arm's backbone seed
#   GAP_ARGS       the WIDTH, the depth of the two stacks, the reduction and
#                  the batch size, appended LAST to the trainer command line.
#                  The runner states the width and the batch size earlier, so
#                  a repeat here is what moves them: argparse keeps the last
#                  value.
#   RUN_SUFFIX     the card and the arm, in the run name, so no arm's
#                  checkpoints and losses CSV read as another's
#   RUNS           this arm's durable root
#   CF_RESULTS     this study's results/, so the leg's log lands here
#
# The runner is idempotent: a stop whose checkpoint is on disk is a no-op, and
# a leg resumes the cell's furthest checkpoint with its optimizer state. So a
# re-fired leg after a crash costs nothing.
#
# ---- Six values have to reach the trainer -----------------------------------
#
# The WIDTH is the whole card. `run_leg_k.sh` states `--d-model 64` in its own
# block, and this wrapper repeats the flag at 384 at the end of the line. A
# width that did not arrive gives a 1.1M run under an 11.4M name, and nothing
# in the checkpoint says so at a glance.
#
# The DEPTH, the REDUCTION, the MOMENTUM and the SEED separate the five arms
# from each other. They share one cell, so they write the same file names, the
# same CSV columns and the same log lines. An arm whose flags did not arrive
# is a DUPLICATE of another arm under a name that says otherwise.
#
# The ALIGN TARGET is the trap #409 fell into. It comes from the cell name,
# and this card is worth nothing if it trains the student target.
#
# The trainer's own command line names all six. So this script starts the leg,
# waits for that line to land in the leg log, and reads them back off it. A
# leg with the wrong objective stops in its first minute rather than at hour
# twenty.
#
# The command line is the trainer's FIRST log line, so this costs no window.
# The count of those lines before the start is what tells this leg's line from
# the lines of the legs below it — the runner appends to one log for each cell.
#
# Usage:  run_arm.sh <arm> <stop steps>
#         BB_GPU=0 bash run_arm.sh k3_r100_09 40000
#         CF412_DRY_RUN=1 bash run_arm.sh k3_r100_09 40000   # print, do not run
set -uo pipefail

ARM="${1:?usage: run_arm.sh <arm> <stop steps>}"
STOP="${2:?usage: run_arm.sh <arm> <stop steps>}"

. "$(dirname "${BASH_SOURCE[0]}")/study.sh"
cf412_require_arm "$ARM" || exit $?
cf412_require_stop "$STOP" || exit $?

RUNNER="$CF412_PARENT/scripts/run_leg_k.sh"
[ -f "$RUNNER" ] || { echo "ABORT: no runner at $RUNNER" >&2; exit 2; }

BB_GPU="${BB_GPU:-0}"
mkdir -p "$CF412_RESULTS"

ARM_ROOT="$(cf412_arm_root "$ARM")"
ARM_K="$(cf412_depth "$ARM")"
ARM_REDUCE="$(cf412_reduce "$ARM")"
EMA_ARGS="$(cf412_ema_args "$ARM")"
ARM_SEED="$(cf412_seed "$ARM")"
ARCH_ARGS="$(cf412_arch_args)"
# GAP_ARGS is the LAST thing on the trainer command line. Every value here is
# stated by the runner earlier, so the repeat is what moves it. An arm at the
# runner's own value repeats it unchanged, which costs nothing and makes every
# leg log name the objective and the shape it trained.
GAP_ARGS="$ARCH_ARGS --train-rollout-reduce $ARM_REDUCE"
GAP_ARGS="$GAP_ARGS --batch-size $CF412_BATCH_SIZE"

# Fault injection, for the test that proves the check below fires. It hands
# the trainer a shape this card does not carry, which is what a wiring defect
# does. Nothing in the study sets it.
[ -n "${CF412_FORCE_ARCH:-}" ] && \
  GAP_ARGS="$CF412_FORCE_ARCH --train-rollout-reduce $ARM_REDUCE --batch-size $CF412_BATCH_SIZE"

if [ -n "${CF412_DRY_RUN:-}" ]; then
  echo "arm $ARM cell=$CF412_CELL k=$ARM_K steps=$STOP gpu=$BB_GPU"
  echo "  arch=$ARCH_ARGS"
  echo "  gap=$GAP_ARGS"
  echo "  ema=$EMA_ARGS"
  echo "  seed=$ARM_SEED reduce=$ARM_REDUCE batch=$CF412_BATCH_SIZE"
  echo "  runner=$RUNNER"
  echo "  RUN_SUFFIX=$(cf412_run_suffix "$ARM") RUNS=$ARM_ROOT"
  echo "  CF_RESULTS=$CF412_RESULTS"
  echo "  ckpt=$(cf412_leg_dir "$ARM" "$STOP")/$(cf412_run_name "$ARM")_$(( STOP / 1000 ))k.pth"
  exit 0
fi

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412] $*" \
  | tee -a "$CF412_RESULTS/arms.log"; }

TLOG="$(cf412_leg_log "$ARM")"
CHECK_TIMEOUT="${CF412_CHECK_TIMEOUT:-1800}"
cmdlines_before="$(cf412_cmdlines "$TLOG")"

log "arm $ARM arch='$ARCH_ARGS' k=$ARM_K reduce=$ARM_REDUCE" \
    "ema='$EMA_ARGS' seed=$ARM_SEED -> ${STOP} steps on gpu $BB_GPU"
K="$ARM_K" RUNS="$ARM_ROOT" CF_RESULTS="$CF412_RESULTS" WT="$CF412_WT" \
  EMA_ARGS="$EMA_ARGS" GAP_ARGS="$GAP_ARGS" SEED="$ARM_SEED" \
  RUN_SUFFIX="$(cf412_run_suffix "$ARM")" \
  BB_GPU="$BB_GPU" \
  bash "$RUNNER" "$CF412_CELL" "$STOP" &
runner=$!

# Wait for THIS leg's command line. The loop also ends when the runner does:
# `run_leg_k.sh` exits without a trainer on a stop already on disk, on a cell
# another machine claims and on a session HOLD, and none of those is a wrong
# objective.
waited=0
while [ "$(cf412_cmdlines "$TLOG")" -le "$cmdlines_before" ]; do
  kill -0 "$runner" 2>/dev/null || break
  if [ "$waited" -ge "$CHECK_TIMEOUT" ]; then
    log "arm $ARM WARNING: no trainer command line in ${CHECK_TIMEOUT}s —" \
        "the shape is unchecked. See $TLOG"
    break
  fi
  sleep 5; waited=$(( waited + 5 ))
done

line="$(cf412_last_cmdline "$TLOG" 2>/dev/null)"
if [ -n "$line" ]; then
  got_arch="$(printf '%s' "$line" | cf412_arch_of_cmdline)"
  got_ema="$(printf '%s' "$line" | cf412_ema_of_cmdline)"
  got_k="$(printf '%s' "$line" | cf412_depth_of_cmdline)"
  got_red="$(printf '%s' "$line" | cf412_reduce_of_cmdline)"
  got_seed="$(printf '%s' "$line" | cf412_seed_of_cmdline)"
  got_batch="$(printf '%s' "$line" | cf412_batch_of_cmdline)"
  got_align="$(printf '%s' "$line" | cf412_align_target_of_cmdline)"
  want_arch="$(cf412_arch_sig)"
  want_ema="$(cf412_ema_sig "$ARM")"
  if [ "$got_arch" != "$want_arch" ] || [ "$got_ema" != "$want_ema" ] \
     || [ "$got_k" != "$ARM_K" ] || [ "$got_red" != "$ARM_REDUCE" ] \
     || [ "$got_seed" != "$ARM_SEED" ] || [ "$got_align" != "teacher" ] \
     || ! cf412_num_eq "$got_batch" "$CF412_BATCH_SIZE"; then
    cf412_kill_tree "$runner"
    wait "$runner" 2>/dev/null
    log "arm $ARM STOPPED — trained arch '$got_arch' / k $got_k /" \
        "'$got_red' / ema '$got_ema' / seed $got_seed / batch $got_batch /" \
        "align $got_align"
    echo "ABORT: this leg's trainer runs the shape '$got_arch' at depth" >&2
    echo "  '$got_k' under '$got_red', momentum '$got_ema', seed" >&2
    echo "  '$got_seed', batch '$got_batch' and align target '$got_align'." >&2
    echo "  Arm '$ARM' is '$want_arch' at depth '$ARM_K' under" >&2
    echo "  '$ARM_REDUCE', momentum '$want_ema', seed '$ARM_SEED', batch" >&2
    echo "  '$CF412_BATCH_SIZE' and align target 'teacher'. The shape reads" >&2
    echo "  <d_model> <num_layers> <num_encoder_layers> and the momentum" >&2
    echo "  reads <tau> <end> <ramp>, with '-' for a flag the command line" >&2
    echo "  does not carry. Every arm of this card writes the same file" >&2
    echo "  names, so the leg is stopped rather than left to climb. Its" >&2
    echo "  command line is the last 'Command line:' in $TLOG" >&2
    exit 3
  fi
  log "arm $ARM arch=$got_arch k=$got_k reduce=$got_red ema='$got_ema'" \
      "seed=$got_seed batch=$got_batch align=$got_align OK —" \
      "all six reached the trainer"
fi

wait "$runner"; rc=$?
log "arm $ARM stop=$STOP rc=$rc"
exit $rc
