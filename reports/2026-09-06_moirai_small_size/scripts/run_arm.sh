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
#   GAP_ARGS       the WIDTH, the depth of the two stacks, the reduction, the
#                  batch size, the learning rate and the L_rep decay, appended
#                  LAST to the trainer command line. The runner states the
#                  width, the batch size and the rate earlier, so a repeat here
#                  is what moves them: argparse keeps the last value.
#   RUN_SUFFIX     the card and the arm, in the run name, so no arm's
#                  checkpoints and losses CSV read as another's
#   RUNS           this arm's durable root
#   CF_RESULTS     this study's results/, so the leg's log lands here
#
# The runner is idempotent: a stop whose checkpoint is on disk is a no-op, and
# a leg resumes the cell's furthest checkpoint with its optimizer state. So a
# re-fired leg after a crash costs nothing.
#
# ---- Eight values have to reach the trainer ----------------------------------
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
# The DECAY is the whole difference between configuration 5 and configuration
# 2. Two arms whose decay did not arrive are one arm at two seeds.
#
# The RATE separates the two bracket arms from configuration 1.
# `run_leg_k.sh` states 1e-3 for every cell, so a rate that did not arrive
# gives a THIRD copy of configuration 1 under a name that says otherwise.
#
# The ALIGN TARGET is the trap #409 fell into. It comes from the cell name,
# and this card is worth nothing if it trains the student target.
#
# The trainer's own command line names all eight. So this script starts the
# leg, waits for that line to land in the leg log, and reads them back off it.
# A leg with the wrong objective stops in its first minute rather than at hour
# twenty.
#
# The command line is the trainer's FIRST log line, so this costs no window.
# The count of those lines before the start is what tells this leg's line from
# the lines of the legs below it — the runner appends to one log for each cell.
#
# NO LINE IS NOT A PASS. Three things end the wait without a trainer: the stop
# is already on disk, another machine claims the cell, or the session holds
# above this stop. Each one exits the runner with a code this script gives
# back. Anything else — a timeout, or a clean exit that left no checkpoint —
# stops the leg. An unchecked leg is not a checked one.
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

RUNNER="$CF412_RUNNER"
[ -f "$RUNNER" ] || { echo "ABORT: no runner at $RUNNER" >&2; exit 2; }

BB_GPU="${BB_GPU:-0}"
mkdir -p "$CF412_RESULTS"

ARM_ROOT="$(cf412_arm_root "$ARM")"
ARM_K="$(cf412_depth "$ARM")"
ARM_REDUCE="$(cf412_reduce "$ARM")"
EMA_ARGS="$(cf412_ema_args "$ARM")"
ARM_SEED="$(cf412_seed "$ARM")"
DECAY_ARGS="$(cf412_decay_args "$ARM")"
ARM_LR="$(cf412_lr "$ARM")"
ARCH_ARGS="$(cf412_arch_args)"
# GAP_ARGS is the LAST thing on the trainer command line. Every value here is
# stated by the runner earlier, so the repeat is what moves it. An arm at the
# runner's own value repeats it unchanged, which costs nothing and makes every
# leg log name the objective and the shape it trained.
GAP_ARGS="$ARCH_ARGS --train-rollout-reduce $ARM_REDUCE"
GAP_ARGS="$GAP_ARGS --batch-size $CF412_BATCH_SIZE --lr $ARM_LR"
# Empty for an arm with no decay, whose objective is then its plain twin's.
[ -n "$DECAY_ARGS" ] && GAP_ARGS="$GAP_ARGS $DECAY_ARGS"

# Fault injection, for the test that proves the check below fires. It hands
# the trainer a shape this card does not carry, which is what a wiring defect
# does. Nothing in the study sets it.
[ -n "${CF412_FORCE_ARCH:-}" ] && \
  GAP_ARGS="$CF412_FORCE_ARCH --train-rollout-reduce $ARM_REDUCE --batch-size $CF412_BATCH_SIZE --lr $ARM_LR"

if [ -n "${CF412_DRY_RUN:-}" ]; then
  echo "arm $ARM cell=$CF412_CELL k=$ARM_K steps=$STOP gpu=$BB_GPU"
  echo "  arch=$ARCH_ARGS"
  echo "  gap=$GAP_ARGS"
  echo "  ema=$EMA_ARGS"
  echo "  decay=${DECAY_ARGS:-none}"
  echo "  seed=$ARM_SEED reduce=$ARM_REDUCE batch=$CF412_BATCH_SIZE" \
       "lr=$ARM_LR"
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
    "ema='$EMA_ARGS' decay='${DECAY_ARGS:-none}' seed=$ARM_SEED" \
    "lr=$ARM_LR -> ${STOP} steps on gpu $BB_GPU"
# The checkpoint of THIS stop, before the leg starts. It is what tells a
# no-trainer exit that is the idempotent path from one that trained nothing.
ckpt_before="$(cf412_bb_ckpt "$ARM" "$STOP")"
# A collapse note of an EARLIER leg of this arm would make this leg read as
# stopped the moment it finishes. Delete it: this leg writes its own.
rm -f "$(cf412_collapse_file "$ARM")"
K="$ARM_K" RUNS="$ARM_ROOT" CF_RESULTS="$CF412_RESULTS" WT="$CF412_WT" \
  EMA_ARGS="$EMA_ARGS" GAP_ARGS="$GAP_ARGS" SEED="$ARM_SEED" \
  RUN_SUFFIX="$(cf412_run_suffix "$ARM")" \
  BB_GPU="$BB_GPU" \
  bash "$RUNNER" "$CF412_CELL" "$STOP" &
runner=$!

# Wait for THIS leg's command line, or for the runner to end without one.
waited=0
timed_out=0
while [ "$(cf412_cmdlines "$TLOG")" -le "$cmdlines_before" ]; do
  kill -0 "$runner" 2>/dev/null || break
  if [ "$waited" -ge "$CHECK_TIMEOUT" ]; then timed_out=1; break; fi
  sleep 5; waited=$(( waited + 5 ))
done

# A timeout leaves the leg running and unchecked, which is exactly the state
# this script exists to prevent. Stop it.
if [ "$timed_out" -eq 1 ]; then
  cf412_kill_tree "$runner"
  wait "$runner" 2>/dev/null
  log "arm $ARM STOPPED — no trainer command line in ${CHECK_TIMEOUT}s"
  echo "ABORT: this leg named no objective in ${CHECK_TIMEOUT}s, so the" >&2
  echo "  width, the depth, the reduction, the momentum, the decay, the" >&2
  echo "  seed, the rate and the align target are unchecked. Every arm of" >&2
  echo "  this card writes the same file names, so the leg is stopped" >&2
  echo "  rather than left to climb. Raise CF412_CHECK_TIMEOUT for a" >&2
  echo "  slower box, and read $TLOG" >&2
  exit 3
fi

if [ "$(cf412_cmdlines "$TLOG")" -le "$cmdlines_before" ]; then
  # The runner ended without a trainer. Three exits do that legitimately: the
  # stop is on disk (0), the session holds above it (9), another machine
  # claims the cell (10). A clean exit that left no checkpoint is none of
  # them, and it is a leg this script never checked.
  wait "$runner"; rc=$?
  if [ "$rc" -eq 0 ] && [ -z "$ckpt_before" ]; then
    log "arm $ARM STOPPED — the runner exited 0 with no command line and no" \
        "checkpoint at $STOP steps"
    echo "ABORT: this leg wrote no trainer command line and left no" >&2
    echo "  checkpoint at $STOP steps, so nothing proves which objective" >&2
    echo "  it ran. See $TLOG" >&2
    exit 3
  fi
  log "arm $ARM no trainer this leg (rc=$rc) —" \
      "${ckpt_before:-no checkpoint}"
  exit $rc
fi

line="$(cf412_last_cmdline "$TLOG")"
got_arch="$(printf '%s' "$line" | cf412_arch_of_cmdline)"
got_ema="$(printf '%s' "$line" | cf412_ema_of_cmdline)"
got_k="$(printf '%s' "$line" | cf412_depth_of_cmdline)"
got_red="$(printf '%s' "$line" | cf412_reduce_of_cmdline)"
got_seed="$(printf '%s' "$line" | cf412_seed_of_cmdline)"
got_batch="$(printf '%s' "$line" | cf412_batch_of_cmdline)"
got_lr="$(printf '%s' "$line" | cf412_lr_of_cmdline)"
got_align="$(printf '%s' "$line" | cf412_align_target_of_cmdline)"
got_decay="$(printf '%s' "$line" | cf412_decay_of_cmdline)"
want_arch="$(cf412_arch_sig)"
want_ema="$(cf412_ema_sig "$ARM")"
want_decay="$(cf412_decay_sig "$ARM")"
if [ "$got_arch" != "$want_arch" ] || [ "$got_ema" != "$want_ema" ] \
   || [ "$got_k" != "$ARM_K" ] || [ "$got_red" != "$ARM_REDUCE" ] \
   || [ "$got_seed" != "$ARM_SEED" ] || [ "$got_align" != "teacher" ] \
   || [ "$got_decay" != "$want_decay" ] \
   || ! cf412_num_eq "$got_batch" "$CF412_BATCH_SIZE" \
   || ! cf412_num_eq "$got_lr" "$ARM_LR"; then
  cf412_kill_tree "$runner"
  wait "$runner" 2>/dev/null
  log "arm $ARM STOPPED — trained arch '$got_arch' / k $got_k /" \
      "'$got_red' / ema '$got_ema' / decay '$got_decay' / seed $got_seed /" \
      "batch $got_batch / lr $got_lr / align $got_align"
  echo "ABORT: this leg's trainer runs the shape '$got_arch' at depth" >&2
  echo "  '$got_k' under '$got_red', momentum '$got_ema', decay" >&2
  echo "  '$got_decay', seed '$got_seed', batch '$got_batch', rate" >&2
  echo "  '$got_lr' and align target '$got_align'. Arm '$ARM' is" >&2
  echo "  '$want_arch' at depth '$ARM_K' under '$ARM_REDUCE', momentum" >&2
  echo "  '$want_ema', decay '$want_decay', seed '$ARM_SEED', batch" >&2
  echo "  '$CF412_BATCH_SIZE', rate '$ARM_LR' and align target 'teacher'." >&2
  echo "  The shape reads <d_model> <num_layers>" >&2
  echo "  <num_encoder_layers>, the momentum reads <tau> <end> <ramp> and" >&2
  echo "  the decay reads <start> <end> <ramp>, with '-' for a flag the" >&2
  echo "  command line does not carry. Every arm of this card writes the" >&2
  echo "  same file names, so the leg is stopped rather than left to" >&2
  echo "  climb. Its command line is the last 'Command line:' in $TLOG" >&2
  exit 3
fi
log "arm $ARM arch=$got_arch k=$got_k reduce=$got_red ema='$got_ema'" \
    "decay='$got_decay' seed=$got_seed batch=$got_batch lr=$got_lr" \
    "align=$got_align OK — all eight reached the trainer"

# ---- The AUC gate ------------------------------------------------------------
#
# A backbone that lost the contrastive task climbs to a checkpoint whose score
# is already known to be bad. One k = 32 arm costs 32 GPU-hours to 200,000
# steps. `auc_guard.sh` reads the trainer's own `auc` column while the leg
# runs and stops the arm on a `lost` verdict. See its header for the reading
# and the warm-up.
#
# It starts AFTER the objective check, so a leg stopped for the wrong
# objective is never read as a collapse.
guard=""
if [ "${CF412_AUC_WATCH:-1}" = "1" ]; then
  bash "$CF412_SCRIPTS/auc_guard.sh" "$ARM" "$STOP" "$runner" \
    >>"$CF412_RESULTS/auc_guard_${ARM}.out" 2>&1 &
  guard=$!
  log "arm $ARM AUC gate pid $guard — window $CF412_AUC_WINDOW," \
      "threshold $CF412_AUC_THRESHOLD," \
      "warmup $(cf412_auc_warmup "$ARM")"
else
  log "arm $ARM AUC gate OFF (CF412_AUC_WATCH=0) — a human must watch the" \
      "auc column of $(cf412_live_losses_csv "$ARM" "$STOP")"
fi

wait "$runner"; rc=$?
# The tree, not the guard alone: the guard sleeps between reads, and that
# `sleep` would outlive a signal sent to its parent.
[ -n "$guard" ] && { cf412_kill_tree "$guard"; wait "$guard" 2>/dev/null; }

# The gate stops the leg, so the leg's own exit code says "killed", which
# phase1 would read as a crash. A collapse is not a crash: a re-fire trains
# the same collapse.
if [ -f "$(cf412_collapse_file "$ARM")" ]; then
  log "arm $ARM STOPPED by the AUC gate — see $(cf412_collapse_file "$ARM")"
  exit "$CF412_RC_COLLAPSED"
fi
log "arm $ARM stop=$STOP rc=$rc"
exit $rc
