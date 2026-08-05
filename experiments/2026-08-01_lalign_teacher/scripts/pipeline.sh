#!/bin/bash
# #390 — the whole experiment, end to end, in one background process.
#
#   wave 1 : backbone → 40 000   →  15 000-step head  →  GIFT-Eval (97 cfgs)
#   wave 2 : backbone → 100 000  →  30 000-step head  →  GIFT-Eval
#   gate   : keep the cells whose GM-Relative MASE fell from wave 1 to wave 2
#   wave 3 : backbone → 200 000  →  30 000-step head  →  GIFT-Eval
#
# No cell trains past 200 000.
#
#   WT=/home/jupyter/wt-cf-390-train \
#     nohup setsid bash pipeline.sh > /dev/null 2>&1 &
#
# Resumable. Every stage is idempotent: `run_arm.sh` short-circuits on a
# checkpoint already at the wave's target, `eval_arm.sh` short-circuits on a
# head already trained and resumes a partial GIFT-Eval from its CSV. Re-run
# this script after any crash and it picks up where it stopped.
#
# START_WAVE skips ahead when a wave is already known good:
#   START_WAVE=2 bash pipeline.sh
set -uo pipefail

WT="${WT:-$HOME/wt-cf-390-train}"
case "$WT" in
  /tmp/*|/tmp) echo "ABORT: WT=$WT is under /tmp — refusing." >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=arm_names.sh
source "$HERE/arm_names.sh"

# The presence check below asks the same question the eval stage asks, so it
# asks it the same way. Taken from this script's own checkout ($HERE), not
# from $WT, for the reason spelled out in eval_arm.sh.
CKPT_RESOLVER="$(cd "$HERE/../../.." && pwd)/scripts/resolve_eval_checkpoint.sh"
[ -f "$CKPT_RESOLVER" ] || { echo "ABORT: no checkpoint resolver at $CKPT_RESOLVER" >&2; exit 2; }

EXP="$WT/experiments/2026-08-01_lalign_teacher"
RES="$EXP/results"; mkdir -p "$RES"
SCRIPTS="$EXP/scripts"
RUNS="$EXP/runs"
EVALS="$EXP/eval_gm_mase"
LOG="$RES/pipeline.log"
START_WAVE="${START_WAVE:-1}"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [pipeline-390] $*" | tee -a "$LOG"; }

PIDFILE="$RES/pipeline.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ABORT: pipeline already running as PID $(cat "$PIDFILE")" >&2
  exit 3
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

# A run that hit NaN before its target is dropped: the trainer prints
# `*** NaN/Inf DETECTED at step N ***` and writes `_EMERGENCY_<step>.pth`
# before exiting 1, so both signals are on disk. The remaining runs continue —
# that is the issue's stop rule.
live_arms(){ # arms... -> the ones with no NaN evidence
  local arm name out=()
  for arm in "$@"; do
    name="$(bb_name "$arm")" || continue
    if ls "$RUNS/${name}"*_EMERGENCY_*.pth >/dev/null 2>&1; then
      log "  DROP $arm — NaN emergency checkpoint on disk" >&2
      continue
    fi
    if grep -ql "NaN/Inf DETECTED" "$RES/run_${name}.log" 2>/dev/null; then
      log "  DROP $arm — 'NaN/Inf DETECTED' in its trainer log" >&2
      continue
    fi
    out+=("$arm")
  done
  echo "${out[*]-}"
}

# Arms that actually have this wave's backbone snapshot on disk. An arm that
# died for any other reason is not handed to the eval stage, where it would
# only produce a confusing ABORT.
arms_at_step(){ # step_k arms...
  local k="$1"; shift
  local arm name hit rc errf out=()
  errf="$(mktemp)"
  for arm in "$@"; do
    name="$(bb_name "$arm")" || continue
    # A resume writes its own `_r<N>` run name, so this pair can match two
    # backbones. They are different models and mtime does not say which one a
    # cell should cite, so the resolver refuses — and so does this. The arm is
    # dropped from the wave with both paths in the log rather than measured
    # under an arbitrary replicate. Re-run it by hand once you know which:
    #   BB_CHECKPOINT=<path> ARM=<arm> BB_STEP_K=<k> ... bash eval_arm.sh
    hit=$(bash "$CKPT_RESOLVER" "$RUNS" "$name" "$k" 2>"$errf")
    rc=$?
    case "$rc" in
      0) log "  READY $arm — $hit" >&2; out+=("$arm") ;;
      3) log "  SKIP $arm — no ${k}k backbone on disk" >&2 ;;
      5) log "  DROP $arm — ${k}k backbone is ambiguous, not choosing:" >&2
         while IFS= read -r line; do log "    $line" >&2; done <"$errf" ;;
      # Any other code is the resolver itself failing. The arm is still
      # dropped — nothing here guesses — but it must not be filed under the
      # one message an operator reads as routine.
      *) log "  DROP $arm — checkpoint resolver failed rc=$rc:" >&2
         while IFS= read -r line; do log "    $line" >&2; done <"$errf" ;;
    esac
  done
  rm -f "$errf"
  echo "${out[*]-}"
}

# Slots per GPU, re-read before every stage so the pool can be resized on a
# live multi-day run without restarting the pipeline. Two is the measured
# optimum for the backbone (results/probe_concurrency.txt); drop to 1 by
# writing `BB_SLOTS_PER_GPU=1` into results/slots.env if memory gets tight.
slots(){ # which -> slots per GPU
  BB_SLOTS_PER_GPU=2; EVAL_SLOTS_PER_GPU=2
  # shellcheck source=/dev/null
  [ -f "$RES/slots.env" ] && source "$RES/slots.env"
  case "$1" in bb) echo "$BB_SLOTS_PER_GPU" ;; *) echo "$EVAL_SLOTS_PER_GPU" ;; esac
}

run_wave(){ # wave_number step_k arms...
  local wave="$1" step_k="$2"; shift 2
  local arms="$*"
  log "=== WAVE $wave: backbone → ${step_k}k — arms: $arms"
  WT="$WT" WAVE="$wave" ARMS="$arms" SLOTS_PER_GPU="$(slots bb)" \
    bash "$SCRIPTS/orchestrate_pool.sh" >>"$LOG" 2>&1
  log "=== WAVE $wave backbone stage rc=$?"

  local alive; alive="$(live_arms $arms)"
  [ "$alive" = "$arms" ] || log "=== WAVE $wave: NaN drop applied — continuing with: $alive"
  local ready; ready="$(arms_at_step "$step_k" $alive)"
  if [ -z "$ready" ]; then
    log "=== WAVE $wave: no arm reached ${step_k}k — stopping"
    return 1
  fi
  log "=== WAVE $wave: head + GIFT-Eval — arms: $ready"
  WT="$WT" WAVE="$wave" ARMS="$ready" SLOTS_PER_GPU="$(slots eval)" \
    bash "$SCRIPTS/eval_wave.sh" >>"$LOG" 2>&1
  log "=== WAVE $wave eval stage rc=$?"
  echo "$ready" > "$RES/wave${wave}_arms.txt"
  return 0
}

log "pipeline start — WT=$WT start_wave=$START_WAVE arms=${#CF390_ARMS[@]}"
ARMS_ALL="${CF390_ARMS[*]}"

# ---- wave 1: 40k backbone, 15k head --------------------------------------
if [ "$START_WAVE" -le 1 ]; then
  run_wave 1 40 $ARMS_ALL || { log "pipeline stops: wave 1 produced no backbone"; exit 1; }
fi

# ---- wave 2: 100k backbone, fresh 30k head -------------------------------
# The same 10 cells, minus any the NaN rule dropped.
W2_ARMS="$(live_arms $ARMS_ALL)"
if [ "$START_WAVE" -le 2 ]; then
  run_wave 2 100 $W2_ARMS || { log "pipeline stops: wave 2 produced no backbone"; exit 1; }
fi

# ---- gate: only the cells whose value fell from 40k to 100k ---------------
W3_ARMS="$(python3 "$SCRIPTS/select_wave3.py" "$EVALS" $W2_ARMS 2>>"$LOG")"
log "wave-3 gate: promoted arms = ${W3_ARMS:-<none>}"
echo "$W3_ARMS" > "$RES/wave3_selected.txt"

# ---- wave 3: 200k backbone, fresh 30k head -------------------------------
if [ -z "$W3_ARMS" ]; then
  log "no cell improved from 40k to 100k — wave 3 is empty, pipeline done"
else
  run_wave 3 200 $W3_ARMS || log "wave 3 produced no backbone"
fi

log "PIPELINE DONE"
