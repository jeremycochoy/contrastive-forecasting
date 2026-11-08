#!/bin/bash
# freq_family — follow several backbone runs at once: for each stop of a list,
# one wave trains the control head of each run in one trainer process, then
# each run gets its B4 score.
#
# The heads of one stop read the same batches of one data stream, so the runs
# are compared on the same head training. One wave reads the stream at a time
# (the stream lock of run_wave.sh): a wave for N runs reads it once, where the
# waves of N followers (follow_abc_gift.sh) read it N times.
#
# For each stop, in the order of the list:
#   1. Wait for the checkpoint of each run, `<ckpt folder>/<run>*_<stop>k.pth`:
#      the file of the last start, when it loads with its optimizer file. Copy
#      it to `<base>/ckpt/<run>/<run>_<stop>k.pth`; results/checkpoints.tsv
#      keeps the source and the SHA-256 of each copy.
#   2. Train the control head of each run with no final head, in one process:
#      the flags of each head come from head_eval_bb.sh (CF_HEAD_ARGV_TO) into
#      one jobs file. The wave holds the stream lock, and waits for
#      FF_WAVE_VRAM_MIB free on its GPU.
#   3. Start the score of each run in the background (run_wave.sh with
#      FF_TRAIN=0; its log: results/follow_<run>.log), and go to the next stop.
# A second start skips the work that is done: a copy, a final head, a score.
# The script ends with code 1 when a wait ended, or a wave or a score failed.
#
# Usage, from the code folder:
#   FF_BASE=<base> FF_MULTI_RUNS="<run>=<ckpt folder> ..." FF_STOPS="0 10 25" \
#     setsid nohup bash follow_multi.sh >>"<base>/results/follow_multi.log" 2>&1 </dev/null &
# The settings of run_wave.sh (FF_GPU, FF_BB_SHAPE, FF_STREAM_LOCK,
# FF_HEAD_STEPS, FF_EVAL_DEVICE, FF_EVAL_SLOTS, FF_EVAL_FORECAST_BATCH_SIZE,
# FF_EVAL_FILTER, ...) reach the training here and the scores alike.
#   FF_WAIT_MAX=172800   the longest wait for the checkpoints of one stop, in s
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${FF_BASE:?FF_BASE: the base folder of the waves}"
. "$HERE/code_folder.sh"
RUNS="${FF_MULTI_RUNS:?FF_MULTI_RUNS: <run>=<ckpt folder> ...}"
STOPS="${FF_STOPS:?FF_STOPS: the stops, in thousands of steps}"
WAIT_MAX="${FF_WAIT_MAX:-172800}"
POLL="${FF_POLL:-60}"
WAVE="${FF_WAVE:-$HERE/run_wave.sh}"
ROOT="$BASE/heads"; RES="$BASE/results"; LOCKS="$BASE/locks"
GPU="${FF_GPU:-1}"
HEAD_STEPS="${FF_HEAD_STEPS:-30000}"
STREAM_LOCK="${FF_STREAM_LOCK:-$LOCKS/stream.lock}"
STREAM_WAIT="${FF_STREAM_WAIT:-86400}"
VRAM="${FF_WAVE_VRAM_MIB:-8000}"
VRAM_WAIT="${FF_VRAM_WAIT:-7200}"
SHAPE="${FF_BB_SHAPE:---d-model 384 --n-heads 8 --num-layers 3}"
RUNNER="${FF_RUNNER:-$CODE/reports/2026-08-08_rollout_depth/scripts/head_eval_bb.sh}"
TRAINER="${FF_TRAINER:-$CODE/experiments/2026-04-13_gift-eval/scripts/train_forecasting_heads_shared.py}"
GIFT_EVAL="${GIFT_EVAL:-$HOME/workspaces/gift-eval-data}"
if [ $(( HEAD_STEPS % 1000 )) -eq 0 ]; then HK="$(( HEAD_STEPS / 1000 ))k"; else HK="$HEAD_STEPS"; fi
# The knobs of head_eval_bb.sh that make another head than the control.
unset CF_FREQ_FAMILY CF_FREQ_FAMILY_RULE CF_FREQ_FAMILY_MEMBERS \
  CF_RECONSTRUCTION CF_HEAD_ARCH CF_HEAD_ARGV_TO CF_SKIP_EVAL EVAL_STRATEGY HEAD_SEED

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [freq_family multi] $*"; }
tag_of(){ echo "${1}_bb${2}k_h${HK}_control"; }                     # <run> <stop k>
has_head(){ compgen -G "$ROOT/eval/$1/*_final.pth" >/dev/null; }    # <tag>
gpu_free(){  # MiB, or nothing with no nvidia-smi
  nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits \
    2>/dev/null | head -1 | tr -dc 0-9
}

# A checkpoint and its optimizer file both load.
loads(){  # <checkpoint>
  python3 - "$1" <<'PY' >/dev/null 2>&1
import sys, torch
path = sys.argv[1]
torch.load(path, map_location="cpu", weights_only=False)
torch.load(path[:-4] + "_optimizer.pth", map_location="cpu", weights_only=False)
PY
}

# The checkpoint of a run at a stop, in CKPT: the file of the last start, when
# it loads with its optimizer file. Until then the trainer may write it.
find_ckpt(){  # <run> <ckpt folder> <stop k>
  local f
  f="$(ls "$2/$1"*_"$3"k.pth 2>/dev/null | sort -rV | head -1)"
  [ -n "$f" ] && [ -s "${f%.pth}_optimizer.pth" ] && loads "$f" && CKPT="$f"
}

# The copy that the wave reads, in COPY, after the wait for the checkpoint.
copy_ckpt(){  # <run> <ckpt folder> <stop k>
  local bytes sum tmp start=$SECONDS said=-1
  COPY="$BASE/ckpt/$1/${1}_${3}k.pth"
  [ -s "$COPY" ] && return 0
  until find_ckpt "$1" "$2" "$3"; do
    (( SECONDS - start < WAIT_MAX )) || { log "STOP: no checkpoint of $1 at ${3}k after $WAIT_MAX s"; return 1; }
    if (( (SECONDS - start) / 1800 > said )); then
      said=$(( (SECONDS - start) / 1800 ))
      log "waiting for the checkpoint of $1 at ${3}k in $2 ($(( SECONDS - start )) s of $WAIT_MAX)"
    fi
    sleep "$POLL"
  done
  mkdir -p "$BASE/ckpt/$1" && bytes=$(stat -c %s "$CKPT") && tmp="$COPY.tmp.$$" || return 1
  cp "$CKPT" "$tmp" && [ "$(stat -c %s "$tmp")" = "$bytes" ] && mv -f "$tmp" "$COPY" \
    || { rm -f "$tmp"; log "ABORT: the copy of $CKPT is not $bytes bytes"; return 1; }
  sum=$(sha256sum "$COPY" | cut -d' ' -f1)
  [ -s "$RES/checkpoints.tsv" ] || printf 'run\tstop_k\tsource\tbytes\tsha256\n' >"$RES/checkpoints.tsv"
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$3" "$CKPT" "$bytes" "$sum" >>"$RES/checkpoints.tsv"
  log "checkpoint ${3}k of $1: $CKPT ($bytes bytes, sha256 $sum)"
}

# The environment of head_eval_bb.sh for the control head of a stop, as
# run_wave.sh gives it (job_env).
job_env(){  # <stop k>
  JOB_ENV=(WT="$CODE" CF373_ROOT="$ROOT" CF_RESULTS="$RES" CF_STOP_K="$1"
           GIFT_EVAL="$GIFT_EVAL" EVAL_SHARDS="${FF_EVAL_SHARDS:-4}" EVAL_DEVICE="${FF_EVAL_DEVICE:-cpu}"
           CF393_EVAL_SLOTS="${FF_EVAL_SLOTS:-2}" CF393_EVAL_SLOTDIR="$RES/evalslots"
           BB_GPU="$GPU" GPU_GATE_LOCKDIR="$LOCKS" CF_BB_SHAPE="$SHAPE")
  [ -z "${FF_HEAD_SAVE_EVERY:-}" ] || JOB_ENV+=(HEAD_SAVE_EVERY="$FF_HEAD_SAVE_EVERY")
  [ -z "${FF_HEAD_LOG_EVERY:-}" ] || JOB_ENV+=(HEAD_LOG_EVERY="$FF_HEAD_LOG_EVERY")
}

# Wait for VRAM free on the GPU of the wave. Returns 1 after VRAM_WAIT s.
wait_vram(){
  local start=$SECONDS free
  while :; do
    free=$(gpu_free)
    { [ -z "$free" ] || [ "$free" -ge "$VRAM" ]; } && return 0
    (( SECONDS - start < VRAM_WAIT )) || { log "ABORT: GPU $GPU has $free MiB free after ${VRAM_WAIT} s"; return 1; }
    (( (SECONDS - start) % 600 < 30 )) && log "waiting: GPU $GPU has $free MiB free, a wave needs $VRAM"
    sleep 30
  done
}

# The runs of a stop with no final head, in TODO_RUNS.
todo_runs(){  # <stop k> <run> ...
  local k="$1" run; shift
  TODO_RUNS=()
  for run in "$@"; do has_head "$(tag_of "$run" "$k")" || TODO_RUNS+=("$run"); done
}

# The jobs file of the heads of TODO_RUNS, in a new wave folder DIR.
write_jobs(){  # <stop k>
  local run
  mkdir -p "$RES/waves" && DIR=$(mktemp -d "$RES/waves/multi_bb${1}k_$(date '+%m%d_%H%M%S')_XXXX") || return 1
  job_env "$1"
  for run in "${TODO_RUNS[@]}"; do
    env "${JOB_ENV[@]}" CF_HEAD_ARGV_TO="$DIR/jobs.jsonl" \
      bash "$RUNNER" "$(tag_of "$run" "$1")" "$BASE/ckpt/$run/${run}_${1}k.pth" student "$HEAD_STEPS" \
      >>"$RES/heads.log" 2>&1 </dev/null || { log "ABORT: no head flags for $run at ${1}k. See $RES/heads.log."; return 1; }
  done
  [ "$(grep -c . "$DIR/jobs.jsonl" 2>/dev/null)" = "${#TODO_RUNS[@]}" ] \
    || { log "ABORT: $DIR/jobs.jsonl does not hold ${#TODO_RUNS[@]} jobs"; return 1; }
}

# The training of the jobs of DIR, in one process. Returns 1 when a run has no
# final head after it.
train_jobs(){  # <stop k>
  local rc run
  log "train: ${#TODO_RUNS[@]} heads at ${1}k, $HEAD_STEPS steps, GPU $GPU, code $(cat "$CODE/DEPLOYED_COMMIT" 2>/dev/null): ${TODO_RUNS[*]} -> $DIR/train.log"
  env PYTHONPATH="$CODE" CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS="${FF_TRAIN_THREADS:-4}" MKL_NUM_THREADS="${FF_TRAIN_THREADS:-4}" \
    HF_TOKEN="$(cat "$CODE/experiments/hf_token.txt")" \
    HUGGING_FACE_HUB_TOKEN="$(cat "$CODE/experiments/hf_token.txt")" \
    python3 -u "$TRAINER" --jobs "$DIR/jobs.jsonl" >>"$DIR/train.log" 2>&1 </dev/null
  rc=$?
  log "train rc=$rc"
  for run in "${TODO_RUNS[@]}"; do
    has_head "$(tag_of "$run" "$1")" || { log "no final head for $run at ${1}k after the trainer"; rc=1; }
  done
  return "$rc"
}

# The heads of a stop, trained under the stream lock. The runs with no head are
# counted again after the wait: another start may have trained some.
train_stop(){  # <stop k> <run> ...
  local lk rc=0
  todo_runs "$@"
  [ "${#TODO_RUNS[@]}" -gt 0 ] || { log "train SKIP at ${1}k: each run has its final head"; return 0; }
  exec {lk}>>"$STREAM_LOCK"
  flock -n "$lk" || log "waiting: another wave reads the data stream ($STREAM_LOCK)"
  if flock -w "$STREAM_WAIT" "$lk"; then
    todo_runs "$@"
    if [ "${#TODO_RUNS[@]}" -gt 0 ]; then
      write_jobs "$1" && wait_vram && train_jobs "$1"; rc=$?
    fi
  else
    log "ABORT: another wave reads the data stream after ${STREAM_WAIT} s"; rc=1
  fi
  exec {lk}>&-
  return "$rc"
}

[ -f "$WAVE" ] && [ -f "$RUNNER" ] && [ -f "$TRAINER" ] || { log "ABORT: no wave, runner or trainer in $CODE"; exit 2; }
mkdir -p "$RES" "$LOCKS" || exit 2
declare -A DIRS; NAMES=()
for pair in $RUNS; do NAMES+=("${pair%%=*}"); DIRS["${pair%%=*}"]="${pair#*=}"; done
log "start: runs ${NAMES[*]}, stops $STOPS, code $CODE"
rc=0; pids=(); what=()
for stop in $STOPS; do
  for run in "${NAMES[@]}"; do copy_ckpt "$run" "${DIRS[$run]}" "$stop" || { rc=1; break 2; }; done
  train_stop "$stop" "${NAMES[@]}" || { log "the heads of ${stop}k FAILED. The next stop goes on."; rc=1; continue; }
  for run in "${NAMES[@]}"; do
    FF_TRAIN=0 bash "$WAVE" "$run" "$stop" "$BASE/ckpt/$run/${run}_${stop}k.pth" control \
      >>"$RES/follow_$run.log" 2>&1 </dev/null &
    pids+=($!); what+=("$run ${stop}k")
  done
done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { log "the score of ${what[$i]} FAILED"; rc=1; }
done
log "FOLLOW_END rc=$rc"
exit "$rc"
