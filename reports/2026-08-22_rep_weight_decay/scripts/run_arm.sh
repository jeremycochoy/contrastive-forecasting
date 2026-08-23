#!/bin/bash
# #409 — one arm's backbone: train the cell to <stop> steps, with the decay,
# at this arm's backbone seed.
#
# This is a wrapper, on purpose. The trainer command line for this
# configuration lives in ONE place, #373's `run_leg_k.sh`, and a copy of it
# here would be a second protocol that drifts. The wrapper supplies eight
# things that runner takes from the environment:
#
#   K              the rollout depth, 32
#   EMA_ARGS       this arm's EMA schedule, which REPLACES the runner's own
#                  three momentum flags. It replaces and does not append: a
#                  fixed arm passes `--ema-tau` alone, and no repeated flag can
#                  remove `--ema-tau-end`.
#   GAP_ARGS       the reduction, the L_align target and the decay flags,
#                  appended LAST to the trainer command line. The cell states
#                  `--align-target` earlier, so a repeat here is what keeps it:
#                  argparse keeps the last value.
#   SEED           this arm's backbone seed
#   RUN_SUFFIX     the study and the arm, in the run name, so no arm's
#                  checkpoints and losses CSV can be read as another's
#   RUNS           this arm's durable root
#   CF_STUDY_DIR   this study's directory, so the leg's log lands here
#   CF_RESULTS     this study's results/
#
# The decay is NOT an arm's own. Every arm carries the card's one shape.
#
# The runner is idempotent: a stop whose checkpoint is on disk is a no-op, and
# a leg resumes the cell's furthest checkpoint with its optimizer state. So a
# re-fired leg after a crash costs nothing.
#
# ---- The schedule and the decay have to reach the trainer --------------------
#
# Every arm shares one configuration and one decay, and differs in the EMA
# schedule. So an arm whose schedule did not arrive is a DUPLICATE of arm 1,
# under a name that says otherwise: same file names, same CSV columns, same log
# lines. The card's result is one number per schedule, and two of them would be
# one number twice. An arm whose decay did not arrive repeats a number the
# sweep already published.
#
# The `rep_w` column of the losses CSV would show the decay, but only after the
# run. The trainer's own command line names all five values in its first log
# line. So this script starts the leg, waits for that line, and reads the
# decay, the seed, the target, the reduction and the momentum off it. A leg
# with the wrong objective stops in its first minute, not at hour five.
#
# Usage:  run_arm.sh <arm> <stop steps>
#         BB_GPU=0 bash run_arm.sh dec_s20 40000
#         CF409_DRY_RUN=1 bash run_arm.sh dec_s20 40000   # print, do not run
set -uo pipefail

ARM="${1:?usage: run_arm.sh <arm> <stop steps>}"
STOP="${2:?usage: run_arm.sh <arm> <stop steps>}"

. "$(dirname "${BASH_SOURCE[0]}")/study.sh"
cf409_require_arm "$ARM" || exit $?
cf409_require_stop "$STOP" || exit $?

RUNNER="$CF409_PARENT/scripts/run_leg_k.sh"
[ -f "$RUNNER" ] || { echo "ABORT: no runner at $RUNNER" >&2; exit 2; }

BB_GPU="${BB_GPU:-0}"
mkdir -p "$CF409_RESULTS"

ARM_ROOT="$(cf409_arm_root "$ARM")"
ARM_SEED="$(cf409_seed "$ARM")"
ARM_EMA="$(cf409_ema_args "$ARM")"
DECAY_ARGS="$(cf409_decay_args)"
# The reduction and the L_align target are stated on every leg, so the log
# names the objective it trained rather than leaving the reader to infer the
# cell's own values. The decay rides the same block, which is the LAST thing on
# the trainer command line.
PROBE_ARGS="--latent-drift-probe-batch-size $CF409_PROBE_BS"
GAP="--train-rollout-reduce $CF409_REDUCE --align-target $CF409_ALIGN_TARGET $DECAY_ARGS $PROBE_ARGS"

# Fault injection, for the tests that prove the check below fires. Each hands
# the trainer something this arm does not carry, which is what a wiring defect
# does. Nothing in the study sets them.
[ -n "${CF409_FORCE_DECAY:-}" ] && \
  GAP="--train-rollout-reduce $CF409_REDUCE --align-target $CF409_ALIGN_TARGET $CF409_FORCE_DECAY $PROBE_ARGS"
[ -n "${CF409_FORCE_EMA:-}" ] && ARM_EMA="$CF409_FORCE_EMA"

if [ -n "${CF409_DRY_RUN:-}" ]; then
  echo "arm $ARM cell=$CF409_CELL k=$CF409_K steps=$STOP gpu=$BB_GPU"
  echo "  decay=$DECAY_ARGS"
  echo "  rep_w at 0 / ramp / stop = $(cf409_rep_w_at 0)" \
       "$(cf409_rep_w_at "$(cf409_ramp)")" "$(cf409_rep_w_at "$STOP")"
  echo "  ema=$ARM_EMA"
  echo "  momentum at 0 / stop = $(cf409_momentum_at "$ARM" 0)" \
       "$(cf409_momentum_at "$ARM" "$STOP") ($(cf409_ema_label "$ARM"))"
  echo "  seed=$ARM_SEED align_target=$CF409_ALIGN_TARGET"
  echo "  auc gate=${CF409_AUC_WATCH:-1} window=$CF409_AUC_WINDOW" \
       "threshold=$CF409_AUC_THRESHOLD warmup=$CF409_AUC_WARMUP"
  echo "  reduce=$CF409_REDUCE runner=$RUNNER"
  echo "  RUN_SUFFIX=$(cf409_run_suffix "$ARM") RUNS=$ARM_ROOT"
  echo "  CF_RESULTS=$CF409_RESULTS"
  echo "  ckpt=$(cf409_leg_dir "$ARM" "$STOP")/$(cf409_run_name "$ARM")_$(( STOP / 1000 ))k.pth"
  exit 0
fi

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#409] $*" \
  | tee -a "$CF409_RESULTS/arms.log"; }

TLOG="$(cf409_leg_log "$ARM")"
CHECK_TIMEOUT="${CF409_CHECK_TIMEOUT:-1800}"
cmdlines_before="$(cf409_cmdlines "$TLOG")"

log "arm $ARM ema='$ARM_EMA' decay='$DECAY_ARGS' seed=$ARM_SEED" \
    "reduce=$CF409_REDUCE target=$CF409_ALIGN_TARGET -> ${STOP} steps" \
    "on gpu $BB_GPU"
# A collapse note of an EARLIER leg of this arm would make this leg read as
# stopped the moment it finishes. Delete it: this leg writes its own.
rm -f "$(cf409_collapse_file "$ARM")"
K="$CF409_K" RUNS="$ARM_ROOT" CF_STUDY_DIR="$CF409_STUDY" \
  CF_RESULTS="$CF409_RESULTS" WT="$CF409_WT" \
  EMA_ARGS="$ARM_EMA" GAP_ARGS="$GAP" SEED="$ARM_SEED" \
  RUN_SUFFIX="$(cf409_run_suffix "$ARM")" \
  BB_GPU="$BB_GPU" \
  bash "$RUNNER" "$CF409_CELL" "$STOP" &
runner=$!

# Wait for THIS leg's command line. The loop also ends when the runner does:
# `run_leg_k.sh` exits without a trainer on a stop already on disk, on a cell
# another machine claims and on a session HOLD, and none of those is a wrong
# objective.
waited=0
while [ "$(cf409_cmdlines "$TLOG")" -le "$cmdlines_before" ]; do
  kill -0 "$runner" 2>/dev/null || break
  if [ "$waited" -ge "$CHECK_TIMEOUT" ]; then
    log "arm $ARM WARNING: no trainer command line in ${CHECK_TIMEOUT}s —" \
        "the decay is unchecked. See $TLOG"
    break
  fi
  sleep 5; waited=$(( waited + 5 ))
done

line="$(cf409_last_cmdline "$TLOG" 2>/dev/null)"
if [ -n "$line" ]; then
  got_decay="$(printf '%s' "$line" | cf409_decay_of_cmdline)"
  got_seed="$(printf '%s' "$line" | cf409_seed_of_cmdline)"
  got_target="$(printf '%s' "$line" | cf409_align_target_of_cmdline)"
  got_red="$(printf '%s' "$line" | cf409_reduce_of_cmdline)"
  got_ema="$(printf '%s' "$line" | cf409_ema_of_cmdline)"
  want_decay="$(cf409_decay_sig)"
  want_ema="$(cf409_ema_sig "$ARM")"
  if [ "$got_decay" != "$want_decay" ] || [ "$got_seed" != "$ARM_SEED" ] \
     || [ "$got_target" != "$CF409_ALIGN_TARGET" ] \
     || [ "$got_red" != "$CF409_REDUCE" ] \
     || [ "$got_ema" != "$want_ema" ]; then
    cf409_kill_tree "$runner"
    wait "$runner" 2>/dev/null
    log "arm $ARM STOPPED — trained decay '$got_decay' / seed $got_seed /" \
        "target $got_target / reduce $got_red / ema '$got_ema', not" \
        "'$want_decay' / seed $ARM_SEED / target $CF409_ALIGN_TARGET /" \
        "reduce $CF409_REDUCE / ema '$want_ema'"
    echo "ABORT: this leg's trainer runs the decay '$got_decay' at seed" >&2
    echo "  '$got_seed', against the '$got_target' target, under the" >&2
    echo "  reduction '$got_red', at the momentum '$got_ema'. This card is" >&2
    echo "  '$want_decay' at seed '$ARM_SEED' against" >&2
    echo "  '$CF409_ALIGN_TARGET' under '$CF409_REDUCE' at" >&2
    echo "  '$want_ema'. The decay reads <start> <end> <ramp> and the" >&2
    echo "  momentum reads <tau> <end> <ramp>, with '-' for a flag the" >&2
    echo "  command line does not carry. Every arm of this card writes the" >&2
    echo "  same file names, so the leg is stopped rather than left to" >&2
    echo "  climb. Its command line is the last 'Command line:' in" >&2
    echo "  $TLOG" >&2
    exit 3
  fi
  log "arm $ARM decay='$got_decay' seed=$got_seed target=$got_target" \
      "reduce=$got_red ema='$got_ema' OK — all five reached the trainer"
fi

# ---- The AUC gate ------------------------------------------------------------
#
# The decay ends at step 10,000 and the leg trains to 40,000. An arm that lost
# the contrastive task has nothing left to train, so it would climb about
# 30,000 dead steps to a checkpoint whose score is already known to be bad.
# Past the ramp nothing pushes the representations apart.
# `auc_guard.sh` reads the trainer's own `auc` column while the leg runs and
# stops the leg on a `lost` verdict. See its header for the reading and the
# warmup.
#
# It starts AFTER the decay check, so a leg stopped for the wrong objective is
# never read as a collapse.
guard=""
if [ "${CF409_AUC_WATCH:-1}" = "1" ]; then
  bash "$CF409_SCRIPTS/auc_guard.sh" "$ARM" "$STOP" "$runner" \
    >>"$CF409_RESULTS/auc_guard_${ARM}.out" 2>&1 &
  guard=$!
  log "arm $ARM AUC gate pid $guard — window $CF409_AUC_WINDOW," \
      "threshold $CF409_AUC_THRESHOLD, warmup $CF409_AUC_WARMUP"
else
  log "arm $ARM AUC gate OFF (CF409_AUC_WATCH=0) — a human must watch the" \
      "auc column of $(cf409_live_losses_csv "$ARM" "$STOP")"
fi

wait "$runner"; rc=$?
# The tree, not the guard alone: the guard sleeps between reads, and that
# `sleep` would outlive a signal sent to its parent.
[ -n "$guard" ] && { cf409_kill_tree "$guard"; wait "$guard" 2>/dev/null; }

# The gate stops the leg, so the leg's own exit code says "killed", which a
# lane would read as a crash and re-fire. A collapse is not a crash: the
# re-fire trains the same collapse.
if [ -f "$(cf409_collapse_file "$ARM")" ]; then
  log "arm $ARM STOPPED by the AUC gate — see" \
      "$(cf409_collapse_file "$ARM")"
  exit "$CF409_RC_COLLAPSED"
fi
log "arm $ARM stop=$STOP rc=$rc"
exit $rc
