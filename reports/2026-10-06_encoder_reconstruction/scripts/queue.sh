#!/bin/bash
# #425 — the reconstruction queue on the box: one reconstruction head and one
# reconstruction score for each row of jobs.tsv.
#
# The heads train in waves. One process trains the heads of one wave
# together (`train_forecasting_heads_shared.py`): it reads one data stream,
# and it gives each batch to the frozen backbone and the head of each job in
# turn. A solo head read about 80 GB of stream, at $0.005 for each GB. Each
# job keeps the B4 settings of `head_eval_bb.sh` (the head architecture,
# 30,000 steps, batch 256, lr 1e-3, head seed 20260722, the scaling of the
# run, one head per patch size), its own files, and the batches and random
# draws of its solo run. The heads write no snapshot every 5,000 steps (the
# disk is small), and the score reads the true horizon (strategy R).
#
# One lane for each data stream (the `data` column): the old-data runs, and
# the GiftEvalPretrain runs. The two lanes run at the same time, each with
# one wave at a time. A lane takes the first CF425_OLD_WAVE_SIZE or
# CF425_WAVE_SIZE jobs of its stream with no head and no score, trains them,
# starts their scores in the background, and takes the next wave. A score
# waits for one of CF393_EVAL_SLOTS slots and runs on the GPU
# (CF425_EVAL_DEVICE). A job with a head and no score gets its score only.
#
# Order: tier, then row of jobs.tsv. Tier 1 holds the first and the last
# stop of each run, so each run has its two ends before the stops between.
#
# The sizes come from probe.sh on the box (10-07): a wave of 12
# GiftEvalPretrain jobs trains 30.0 job steps/s and uses 9,612 MiB of GPU
# memory, so it ends in about 3.3 h. The 9 old-data jobs train 25.5 job
# steps/s in 7,140 MiB, so they end in about 2.9 h.
#
# A job is done when its score file exists. A wave locks each of its jobs
# (`flock` on the job's lock file) and passes the lock to the job's score,
# so no two processes run one job, and a new queue skips the jobs that an
# older wave or score still runs. A job that fails CF425_TRIES times stays
# failed. A lane stops when it can lock no job and none of its scores runs.
#
# Before a wave, the lane waits for CF425_MIN_FREE_GB of free disk (the sync
# frees the box copies that elisa holds), then for CF425_WAVE_VRAM_MIB of
# free GPU memory. One lock, for both lanes, holds from that GPU check until
# the trainer ends its first step, so two waves never count the same free
# memory. The queue starts after forecast_scores.sh, which uses the same
# GPU with other locks, and only when the seasonal-naive reference of the
# score is on the box.
#
# The linear queue (CF425_HEAD_ARCH=linear) gives each job a linear head: one
# linear map from each encoder latent to the quantiles of the values of its
# patch, with the other B4 settings and the same score. Its tags end in
# `_recon_lin`, and its code, its heads and its results have their own
# folders (cf-425-lin). So it runs beside the queue of the transformer heads.
# The two queues share the GPU lock, the eval slots and the disk floor.
#
# Usage, on the box:
#   nohup setsid bash queue.sh >>/workspace/results/cf-425/queue.log 2>&1 &
#   CF425_DRY_RUN=1 bash queue.sh     # the waves and the input check only
#   CF425_SCORE=0 ...                 # train the heads, and score nothing
#   CF425_GPU=1 ...                   # the GPU of the waves and the scores
#   CF425_HEAD_ARCH=linear nohup setsid bash queue.sh \
#     >>/workspace/results/cf-425-lin/queue.log 2>&1 &     # the linear queue
set -uo pipefail

# The head of the queue: its folders, the end of its tags, the jobs of a
# wave of each stream, and the free GPU memory that a wave needs.
ARCH="${CF425_HEAD_ARCH:-transformer}"
case "$ARCH" in
  transformer) NAME=cf-425; SUFFIX=recon; WHO="queue"
               OLD_WAVE=9; GIFT_WAVE=12; WAVE_VRAM=12000 ;;
  linear) NAME=cf-425-lin; SUFFIX=recon_lin; WHO="linear queue"
          OLD_WAVE=9; GIFT_WAVE=12; WAVE_VRAM=12000 ;;
  *) echo "[#425 queue] ABORT: CF425_HEAD_ARCH=$ARCH. Use transformer or linear."
     exit 2 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${CF425_JOBS:-$HERE/jobs.tsv}"
CODE="${CF425_CODE:-/workspace/$NAME}"
CK="${CF425_CK:-/workspace/ckpt}"
ROOT="${CF425_ROOT:-$CK/$NAME/recon}"
RES="${CF425_RES:-/workspace/results/$NAME}"
TRIES="${CF425_TRIES:-2}"
STAGGER="${CF425_LANE_STAGGER:-120}"
HEAD_STEPS="${CF425_HEAD_STEPS:-30000}"
SCORE="${CF425_SCORE:-1}"
GPU="${CF425_GPU:-0}"
RUNNER="${CF425_RUNNER:-$CODE/reports/2026-08-08_rollout_depth/scripts/head_eval_bb.sh}"
TRAINER="${CF425_TRAINER:-$CODE/experiments/2026-04-13_gift-eval/scripts/train_forecasting_heads_shared.py}"
GPU_LOCK="${CF425_GPU_LOCK:-/tmp/cf425_gpu_start.lock}"
AFTER="${CF425_AFTER:-forecast_scores\.sh}"
# The seasonal-naive reference of the GM-Relative MASE: the first path that
# eval_gift_eval_official.py reads. Git does not hold it.
SN_REF="${CF425_SN_REF:-$HOME/workspaces/gift-eval/results/seasonal_naive/all_results.csv}"
SN_BYTES=24831
STREAMS="old gift_pretrain"
# The backbone shape of every run of this card. One word for `env`.
SHAPE="--d-model 384 --n-heads 8 --num-layers 3"
LOCKS="$RES/locks"
FAILED="$RES/failed"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#425 $WHO] $*"; }

tag_of(){ echo "${1}_bb${2}k_h30k_$SUFFIX"; }   # <arm> <stop k>

wave_size(){  # <stream>
  if [ "$1" = old ]; then echo "${CF425_OLD_WAVE_SIZE:-$OLD_WAVE}"
  else echo "${CF425_WAVE_SIZE:-$GIFT_WAVE}"; fi
}

# The jobs in queue order:
# "<tier> <row> <code> <arm> <stop k> <ckpt> <bytes> <stream>".
ordered_jobs(){
  awk -F'\t' '!/^#/ && NF >= 7 { print $4, NR, $1, $2, $3, $5, $6, $7 }' "$JOBS" \
    | sort -k1,1n -k2,2n
}

# Each input on the box with the byte size of its elisa copy, or ABORT.
check_inputs(){
  local bad=0 tier row code arm stop ckpt bytes data size
  while read -r tier row code arm stop ckpt bytes data; do
    size=$(stat -c %s "$CK/$ckpt" 2>/dev/null || echo missing)
    [ "$size" = "$bytes" ] && continue
    log "INPUT $code ${stop}k: $CK/$ckpt is $size bytes, want $bytes"
    bad=$(( bad + 1 ))
  done < <(ordered_jobs)
  size=$(stat -c %s "$SN_REF" 2>/dev/null || echo missing)
  if [ "$size" != "$SN_BYTES" ]; then
    log "INPUT the seasonal-naive reference $SN_REF is $size bytes, want $SN_BYTES"
    bad=$(( bad + 1 ))
  fi
  [ "$bad" -eq 0 ] || { log "ABORT: $bad input(s) not on the box. Run stage_inputs.sh on elisa."; return 1; }
}

# Wait while a process of the card that uses the GPU with other locks runs.
wait_for_others(){
  while pgrep -f "$AFTER" >/dev/null; do
    log "waiting: a process that matches '$AFTER' runs"
    sleep "${CF425_AFTER_POLL:-300}"
  done
}

# The waves of a queue that starts with no head and no score:
# "<stream> <wave> <tier> <code> <stop k> <tag>".
plan(){
  local stream n tier row code arm stop ckpt bytes data
  for stream in $STREAMS; do
    n=0
    while read -r tier row code arm stop ckpt bytes data; do
      [ "$data" = "$stream" ] || continue
      echo "$stream $(( n / $(wave_size "$stream") + 1 )) $tier $code ${stop}k $(tag_of "$arm" "$stop")"
      n=$(( n + 1 ))
    done < <(ordered_jobs)
  done
}

fail_count(){ ls "$FAILED/$1".* 2>/dev/null | wc -l; }

mark_failed(){  # <tag> <what>
  mkdir -p "$FAILED"
  echo "$2 $(date '+%m-%d %H:%M:%S')" >"$FAILED/$1.$(( $(fail_count "$1") + 1 ))"
}

has_head(){ compgen -G "$ROOT/eval/$1/*_final.pth" >/dev/null; }   # <tag>

# The environment of one head_eval_bb.sh call. The queue trains no head in
# head_eval_bb.sh, and its gates share one lock folder for both lanes.
job_env(){  # <stop k>
  echo WT="$CODE" CF373_ROOT="$ROOT" CF_RESULTS="$RES" CF_STOP_K="$1" \
    GIFT_EVAL="${GIFT_EVAL:-/workspace/gift-eval-data}" \
    EVAL_SHARDS="${CF425_EVAL_SHARDS:-2}" \
    EVAL_DEVICE="${CF425_EVAL_DEVICE:-cuda}" \
    CF393_EVAL_SLOTS="${CF425_EVAL_SLOTS:-2}" \
    CF393_EVAL_SLOTDIR=/tmp/cf425_evalslots \
    BB_GPU="$GPU" HEAD_VRAM_MIB="${CF425_HEAD_VRAM_MIB:-9000}" \
    GPU_GATE_LOCKDIR=/tmp/cf425_gpu \
    CF_RECONSTRUCTION=encoder CF_HEAD_ARCH="$ARCH" HEAD_SAVE_EVERY=1000000
}

# The lock of each job of the lane: job tag -> file descriptor.
declare -A FD=()

lock_job(){  # <tag>
  local fd
  exec {fd}>>"$LOCKS/$1.lock" || return 1
  if ! flock -n "$fd"; then exec {fd}>&-; return 1; fi
  FD[$1]=$fd
}

unlock_all(){
  local tag fd
  for tag in "${!FD[@]}"; do fd=${FD[$tag]}; exec {fd}>&-; done
  FD=()
}

# The score of one job, in the background. The lock of the job passes to
# the score, and the score holds no other lock.
start_score(){  # <tag> <ckpt> <stop k>
  local tag="$1" other fd rc
  [ "$SCORE" = 1 ] || return 0
  (
    for other in "${!FD[@]}"; do
      [ "$other" = "$tag" ] || { fd=${FD[$other]}; exec {fd}>&-; }
    done
    env $(job_env "$3") CF_BB_SHAPE="$SHAPE" \
      bash "$RUNNER" "$tag" "$CK/$2" student "$HEAD_STEPS" \
      >>"$RES/scores.log" 2>&1 </dev/null
    rc=$?
    [ "$rc" -eq 0 ] || mark_failed "$tag" "score rc=$rc"
    log "score $tag rc=$rc"
  ) &
  fd=${FD[$tag]}; exec {fd}>&-; unset "FD[$tag]"
}

# The next wave of a stream: up to its wave size of jobs with no head, and
# each job with a head and no score, all locked. "<tag> <ckpt> <stop k>" in
# WAVE and in SCORE_ONLY.
pick_wave(){  # <stream>
  local size tier row code arm stop ckpt bytes data tag
  size=$(wave_size "$1"); WAVE=(); SCORE_ONLY=()
  while read -r tier row code arm stop ckpt bytes data; do
    [ "$data" = "$1" ] || continue
    tag="$(tag_of "$arm" "$stop")"
    [ -s "$RES/score_$tag.txt" ] && continue
    [ "$(fail_count "$tag")" -lt "$TRIES" ] || continue
    if has_head "$tag"; then
      [ "$SCORE" = 1 ] || continue
      lock_job "$tag" && SCORE_ONLY+=("$tag $ckpt $stop")
      continue
    fi
    [ "${#WAVE[@]}" -lt "$size" ] || continue
    lock_job "$tag" && WAVE+=("$tag $ckpt $stop")
  done < <(ordered_jobs)
}

wait_for_disk(){  # <stream>
  local need="${CF425_MIN_FREE_GB:-8}" free
  while :; do
    free=$(df -BG --output=avail "$RES" | tail -1 | tr -dc 0-9)
    [ "${free:-0}" -ge "$need" ] && return 0
    log "lane $1: ${free} GB free, a wave needs $need GB. The sync frees the box copies."
    sleep 300
  done
}

gpu_free(){  # MiB, or nothing with no nvidia-smi
  nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits \
    2>/dev/null | head -1 | tr -dc 0-9
}

# Start the trainer of a wave in the background (TRAINER_PID) when the GPU
# has the free memory of a wave. The GPU lock holds until the trainer ends
# its first step, or ends, and the trainer does not inherit it.
start_trainer(){  # <stream> <wave dir>
  local need="${CF425_WAVE_VRAM_MIB:-$WAVE_VRAM}" lk free waited=0
  exec {lk}>>"$GPU_LOCK"
  flock "$lk"
  while :; do
    free=$(gpu_free)
    { [ -z "$free" ] || [ "$free" -ge "$need" ]; } && break
    [ $(( waited % 600 )) -eq 0 ] && log "lane $1: $free MiB free on the GPU, a wave needs $need"
    sleep 30; waited=$(( waited + 30 ))
  done
  ( exec {lk}>&-
    exec env PYTHONPATH="$CODE" CUDA_VISIBLE_DEVICES="$GPU" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      HF_TOKEN="$(cat "$CODE/experiments/hf_token.txt" 2>/dev/null)" \
      python3 -u "$TRAINER" --jobs "$2/jobs.jsonl" >>"$2/train.log" 2>&1 \
      </dev/null ) &
  TRAINER_PID=$!
  waited=$SECONDS
  until grep -q '^\[shared\] 1 steps' "$2/train.log" 2>/dev/null \
      || ! kill -0 "$TRAINER_PID" 2>/dev/null \
      || (( SECONDS - waited >= ${CF425_GPU_SETTLE_MAX:-1800} )); do
    sleep "${CF425_GPU_POLL:-5}"
  done
  exec {lk}>&-
}

# One wave: the flags of each job from head_eval_bb.sh, then one shared
# trainer. Returns the trainer's exit code.
train_wave(){  # <stream>
  local dir job tag ckpt stop
  mkdir -p "$RES/waves"
  dir=$(mktemp -d "$RES/waves/${1}_$(date '+%m%d_%H%M%S')_XXXX") || return 1
  for job in "${WAVE[@]}"; do
    read -r tag ckpt stop <<<"$job"
    env $(job_env "$stop") CF_BB_SHAPE="$SHAPE" \
      CF_HEAD_ARGV_TO="$dir/jobs.jsonl" \
      bash "$RUNNER" "$tag" "$CK/$ckpt" student "$HEAD_STEPS" \
      >>"$RES/heads.log" 2>&1 </dev/null || log "lane $1: no flags for $tag"
  done
  log "lane $1: wave $dir: ${#WAVE[@]} jobs: $(printf '%s ' "${WAVE[@]%% *}")"
  start_trainer "$1" "$dir"
  wait "$TRAINER_PID"
}

run_lane(){  # <stream>
  local stream="$1" job tag ckpt stop rc
  # The queue lock stays with the queue: a wave or a score that outlives it
  # must not stop a new queue. The job locks keep each job single.
  exec 9>&-
  mkdir -p /tmp/cf425_gpu
  while :; do
    pick_wave "$stream"
    if [ "${#WAVE[@]}" -eq 0 ] && [ "${#SCORE_ONLY[@]}" -eq 0 ]; then
      # A score of the lane that fails gets its next try from the lane. So
      # the lane stops only when none of its scores runs.
      [ -n "$(jobs -pr)" ] || break
      wait -n
      continue
    fi
    for job in "${SCORE_ONLY[@]}"; do
      read -r tag ckpt stop <<<"$job"
      start_score "$tag" "$ckpt" "$stop"
    done
    if [ "${#WAVE[@]}" -gt 0 ]; then
      wait_for_disk "$stream"
      train_wave "$stream"
      rc=$?
      log "lane $stream: wave rc=$rc"
      for job in "${WAVE[@]}"; do
        read -r tag ckpt stop <<<"$job"
        if has_head "$tag"; then
          start_score "$tag" "$ckpt" "$stop"
        else
          mark_failed "$tag" "wave rc=$rc"
        fi
      done
    fi
    unlock_all
  done
  log "lane $stream: done"
}

mkdir -p "$RES" "$LOCKS" "$FAILED" "$ROOT"
[ -f "$JOBS" ] || { log "ABORT: no job table at $JOBS"; exit 2; }
[ -f "$RUNNER" ] || { log "ABORT: no runner at $RUNNER"; exit 2; }
[ -f "$TRAINER" ] || { log "ABORT: no shared trainer at $TRAINER"; exit 2; }
check_inputs || exit 3
if [ -n "${CF425_DRY_RUN:-}" ]; then
  plan
  exit 0
fi

exec 9>>"$RES/queue.lock"
flock -n 9 || { log "ABORT: another queue holds $RES/queue.lock"; exit 4; }
wait_for_others
log "start: $(ordered_jobs | wc -l) jobs, $ARCH heads, lanes: $STREAMS, code $CODE ($(cat "$CODE/DEPLOYED_COMMIT" 2>/dev/null))"
for stream in $STREAMS; do
  run_lane "$stream" &
  sleep "$STAGGER"
done
wait
log "QUEUE_END: $(ls "$RES"/score_*_"$SUFFIX".txt 2>/dev/null | wc -l) scores"
