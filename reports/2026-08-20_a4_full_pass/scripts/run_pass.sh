#!/bin/bash
# #407 — give A4 one full pass over `small_v1`, and score three stops.
#
# Usage: run_pass.sh [stop ...]        default: 300000 450000 665000
#
#   WT=<checkout> RUNS=<durable root> BB_GPU=0 HEAD_GPU=0 \
#     nohup setsid bash run_pass.sh > run_pass.out 2>&1 &
#
# The card asks for one thing: more steps for the run #373 stopped at
# 200,000. So this driver trains nothing of its own. It calls #373's
# launcher and #373's stop script, and it decides only the order:
#
#   leg to <stop>  ->  student head + GIFT-Eval  ->  teacher head + GIFT-Eval
#
# Every training flag stays in
# `reports/2026-08-08_rollout_depth/scripts/run_leg_k.sh`, cell
# `arm6_v2_combab_alignS`. A copy of them here would be a second place for
# the recipe to drift, and the card's contract is that the recipe does not
# change. `tests/test_407_full_pass.py` refuses a training flag in this
# file for that reason.
#
# One root, two names. `run_leg_k.sh` reads `RUNS`, `stop_k.sh` reads
# `CF373_ROOT`, and both must resolve to the same `<root>/<cell>/leg_<N>k`.
# They are set from one variable below.
#
# A remote box takes the same command. Point `RUNS` at the box's durable
# root and run #373's sync loop against it from elisa:
#
#   REMOTE_HOST=<host> REMOTE_PORT=<port> \
#   REMOTE_DIR=<checkout>/reports/2026-08-08_rollout_depth \
#   REMOTE_RUNS=<root> LOCAL_DIR=<local checkout>/reports/2026-08-08_rollout_depth \
#     bash reports/2026-08-08_rollout_depth/sync/sync_loop.sh
#
# CLAUDE.md § Remote Machine Monitoring: every remote run carries a sync
# loop for its full duration. This leg is 40 GPU-hours.
#
# `$WT/experiments/hf_token.txt` must hold the read-only HF token. Both
# child scripts refuse to start without it, because an anonymous stream
# from HF throttles to 0.5 sps and idles the card.
#
# Exit codes: 1 a leg failed, 2 bad input, 3 the run did not continue,
# 4 a head never scored.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
# Where this study's evidence lands. It defaults into the checkout, which
# is right on elisa. On a rented box the checkout is disposable and the
# durable root is what the sync loop pulls, so point it there.
RES="${CF407_RESULTS:-$STUDY/results}"
mkdir -p "$RES"

WT="${WT:-$HOME/workspaces/contrastive-forecasting}"
PARENT="$WT/reports/2026-08-08_rollout_depth/scripts"
LEG_SH="$PARENT/run_leg_k.sh"
STOP_SH="$PARENT/stop_k.sh"

# The durable root that holds the 200k checkpoint the card pins, and every
# leg this driver adds to it.
RUNS="${RUNS:-/home/jupyter/cf373_r3/sync}"

CELL="arm6_v2_combab_alignS"
CELL_ID="A4"
DEPTH=3
BB_GPU="${BB_GPU:-0}"
HEAD_GPU="${HEAD_GPU:-$BB_GPU}"

# Measured on this study's own cells: a k = 3 leg holds 5,585 MiB
# (`reports/2026-08-08_rollout_depth/results/gpu_mem_B5.csv`). elisa's two
# 4090s carry other work, so the leg waits for room instead of dying in
# `.to(device)` several seconds after launch.
BB_VRAM_MIB="${BB_VRAM_MIB:-6500}"
VRAM_POLL="${VRAM_POLL:-60}"
VRAM_TIMEOUT="${VRAM_TIMEOUT:-86400}"

STOPS=("$@")
[ "${#STOPS[@]}" -gt 0 ] || STOPS=(300000 450000 665000)

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [cf407] $*" | tee -a "$RES/run_pass.log"; }

for f in "$LEG_SH" "$STOP_SH" "$HERE/full_pass.py" "$HERE/collect.sh"; do
  [ -f "$f" ] || { log "ABORT: missing $f"; exit 2; }
done

# The two logs the continuity gates read. `full_pass.py` names them, so the
# run name is spelled once and this file never re-derives it.
mapfile -t LOG_PATHS < <(python3 "$HERE/full_pass.py" --log-paths --wt "$WT")
[ "${#LOG_PATHS[@]}" -eq 2 ] || { log "ABORT: full_pass.py gave no log paths"; exit 2; }
CELL_LOG="${LOG_PATHS[0]}"
TRAIN_LOG="${LOG_PATHS[1]}"
bytes_of(){ stat -c %s "$1" 2>/dev/null || echo 0; }

# The one preflight that cannot be skipped. A leg that resumes an earlier
# checkpoint, or starts at step 0, still writes a checkpoint at every stop
# and still scores. The md5 sums are the card's.
if ! python3 "$HERE/full_pass.py" --check-resume "$RUNS" >>"$RES/run_pass.log" 2>&1; then
  tail -5 "$RES/run_pass.log" >&2
  log "ABORT: the checkpoint under $RUNS is not the one the card pins"
  exit 3
fi

# Block until the card has room for a leg. `gpu_gate` inside the launcher
# returns at once on a `Default`-mode GPU, which is what elisa runs, so it
# does not cover a card that another session already filled.
wait_vram(){ # <gpu index> <MiB needed>
  local gpu="$1" need="$2" waited=0 free
  while :; do
    free=$(nvidia-smi --id="$gpu" --query-gpu=memory.free \
             --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$free" ] || return 0            # no nvidia-smi: proceed ungated
    [ "$free" -ge "$need" ] && break
    if [ "$waited" -ge "$VRAM_TIMEOUT" ]; then
      log "TIMEOUT after ${waited}s: GPU $gpu has ${free} MiB free, need $need"
      return 1
    fi
    [ $(( waited % 600 )) -eq 0 ] && \
      log "waiting for VRAM on GPU $gpu: ${free} MiB free, need $need"
    sleep "$VRAM_POLL"; waited=$(( waited + VRAM_POLL ))
  done
  [ "$waited" -gt 0 ] && log "GPU $gpu has ${free} MiB free after ${waited}s"
  return 0
}

log "start stops=${STOPS[*]} runs=$RUNS wt=$WT bb_gpu=$BB_GPU head_gpu=$HEAD_GPU"

# Heads that ran twice and never wrote a score. The driver keeps going, so
# one dead head costs one point rather than every stop behind it, and it
# exits non-zero at the end. A gap in the curve under a zero exit code
# reads as a finished study.
MISSING=""

for stop in "${STOPS[@]}"; do
  # A typo here trains to the wrong target and scores it as a stop, so it
  # is rejected rather than turned into 0 by the arithmetic below. A stop
  # off the 1000 grid is rejected too: every name carries a stop in
  # thousands, so 300500 and 300000 would both be `bb300k`.
  case "$stop" in ''|*[!0-9]*) log "ABORT: bad stop '$stop'"; exit 2;; esac
  [ $(( stop % 1000 )) -eq 0 ] || { log "ABORT: stop '$stop' is not a whole number of thousands"; exit 2; }
  stop_k=$(( stop / 1000 ))

  # Continuity, before the leg. The md5 gate above pins the FIRST leg's
  # checkpoint. This one covers all three, and it is the same failure each
  # time: a leg that starts at step 0, or resumes a checkpoint from before
  # the stop it must continue, still trains to the target and still scores.
  if ! python3 "$HERE/full_pass.py" --check-leg "$stop" --root "$RUNS" \
       >>"$RES/run_pass.log" 2>&1; then
    tail -5 "$RES/run_pass.log" >&2
    log "ABORT: the ${stop_k}k leg would not continue the run"
    exit 3
  fi

  # Where each log ends now, so the check after the leg reads only what
  # this leg wrote. Both logs are appended across every leg of the study.
  before_cell=$(bytes_of "$CELL_LOG")
  before_train=$(bytes_of "$TRAIN_LOG")

  wait_vram "$BB_GPU" "$BB_VRAM_MIB" || exit 1
  log "LEG -> ${stop_k}k"
  # `run_leg_k.sh` is idempotent: a leg whose checkpoint is already on disk
  # exits 0 without touching the GPU, so a re-fired driver costs nothing.
  RUNS="$RUNS" WT="$WT" BB_GPU="$BB_GPU" \
    bash "$LEG_SH" "$CELL" "$stop"
  rc=$?
  if [ $rc -ne 0 ]; then
    log "ABORT: leg to ${stop_k}k rc=$rc"
    exit 1
  fi

  # Continuity, after the leg. This reads what `train.py` printed, not what
  # the launcher meant to do. The two disagree when the optimizer sidecar
  # is missing: the launcher resumes, `load_training_state` finds no state,
  # and the run counts from step 0 with the weights loaded.
  if ! python3 "$HERE/full_pass.py" --check-leg-done "$stop" --root "$RUNS" \
       --wt "$WT" --since-cell "$before_cell" --since-train "$before_train" \
       >>"$RES/run_pass.log" 2>&1; then
    tail -5 "$RES/run_pass.log" >&2
    log "ABORT: the ${stop_k}k leg did not continue the run"
    exit 3
  fi
  log "LEG done ${stop_k}k"

  for head in student teacher; do
    scored=0
    # Retried once. `stop_k.sh` skips a head that is already trained and
    # resumes the eval per shard, so a retry costs only what died. Not
    # retrying costs 30,000 GPU steps for a transient.
    for attempt in 1 2; do
      log "HEAD ${stop_k}k $head (attempt $attempt)"
      CF373_ROOT="$RUNS" WT="$WT" BB_GPU="$HEAD_GPU" \
      HEAD_STEPS=30000 HEAD_SEED=20260722 \
        bash "$STOP_SH" "$CELL_ID" "$DEPTH" "$stop" "$head"
      rc=$?
      log "HEAD ${stop_k}k $head rc=$rc"
      [ $rc -eq 0 ] && { scored=1; break; }
    done
    # A clean exit code is not a score. `eval_local.sh` writes
    # `score_<tag>.txt` last, and it stops before that line when the merged
    # CSV is short of the 97 configs. The pair then reaches `collect.sh`,
    # which drops it, and the figure draws a shorter line that reads as a
    # finished study. So the point counts only when the number is on disk.
    if [ "$scored" -eq 1 ] && ! python3 "$HERE/full_pass.py" \
         --check-score "$stop" --head "$head" --wt "$WT" \
         >>"$RES/run_pass.log" 2>&1; then
      log "NO SCORE for ${stop_k}k $head after a clean exit"
      scored=0
    fi
    if [ "$scored" -ne 1 ]; then
      log "GIVING UP on ${stop_k}k $head — the curve loses this point"
      MISSING="$MISSING ${stop_k}k/$head"
    fi
  done

  # After every stop, not once at the end: the numbers reach the report
  # even if the next leg never finishes.
  RUNS="$RUNS" WT="$WT" bash "$HERE/collect.sh" >>"$RES/collect.log" 2>&1 \
    || log "WARN: collect.sh rc=$? — see $RES/collect.log"
  python3 "$HERE/full_pass.py" --results "$RES" | tee -a "$RES/run_pass.log"
done

if [ -n "$MISSING" ]; then
  log "INCOMPLETE: no score for$MISSING"
  exit 4
fi

log "drained: ${STOPS[*]}"
