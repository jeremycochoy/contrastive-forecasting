#!/bin/bash
# #393 — the OTHER end of the paired delta: bb40k heads at two more seeds.
#
# Usage:  WT=<checkout> [RUNS=<durable root>] [CF393_BB40K_JOBS=4] \
#           bash scripts/seed_replicates_bb40k.sh [--list] [--status]
#         bash scripts/seed_replicates_bb40k.sh --one <cell> <stop> <enc> <seed> <tag> <gpu>
#
# WHY. `scripts/seed_replicates.sh` replicated the bb100k end of every
# delta and `results/seed_spread.csv` reports its sd. The extend rule reads
# a DIFFERENCE, bb100k minus bb40k, and the bb40k term is one head seed,
# 20260722, with no replicate. Dividing that difference by the spread of
# the bb100k end alone states a significance the data does not contain: the
# #390 parent measured 15,000-step bb40k heads with sd up to 0.0380
# (results/parent_seed_spread.csv), which is 2x to 20x the in-study bb100k
# sds now serving as the denominator. Propagating a bb40k sd of 0.02 takes
# arm6_v2_combab_alignS's student from 13.6 sigma to 1.6.
#
# So the bb40k end is measured here, at the same two seeds, and the delta
# gets a standard error from BOTH of its ends.
#
# WHAT IS HELD FIXED. Everything except the head seed: the same bb40k
# backbone checkpoint (backbones are NOT retrained), the same encoder
# pairing rule, 15,000 head steps — the bb40k budget, not bb100k's 30,000 —
# 97 GIFT-Eval configs, B4, forecast horizon 16, --grad-clip 1.0, and the
# same seasonal-naive denominator.
#
# THREE TIERS, in the order they run.
#
#   t1  Required. Six cells x two heads x two seeds at bb40k. These are the
#       six cells that already have bb100k replicates, so completing t1
#       gives every one of their twelve deltas an sd at both ends.
#
#   hw  The GPU control. Three of the six cells trained their seed-20260722
#       bb40k head on a rented RTX 5090 (arm5_combab_alignS,
#       arm5_combab_alignT, arm6_v2_nse_alignT — results/machines.txt and
#       the `_broker/<box>/<cell>/` trees are the receipt); t1 runs their
#       replicates on elisa's RTX 4090 because vast is empty. A spread over
#       those three seeds would then measure seed AND hardware together.
#       This tier retrains seed 20260722 itself on the 4090, so the
#       hardware term is a measured number rather than an assumption:
#       4090(s22) minus 5090(s22), same seed, same checkpoint, same
#       everything else. HEAD_TAG=hw4090 gives it its own subtree so the
#       5090 artefacts the ladder was decided from are untouched.
#
#   t2  The two unreplicated cells the review flags as marginal:
#       arm4_combab (teacher margin +0.0318) and arm6_v2_ncpc_alignT
#       (teacher margin +0.0380). Both stops, both heads, both seeds, so
#       each replicate seed gives a delta whose two ends were trained on
#       the same GPU as each other. No hardware control is needed there:
#       seed 20260722's pair is 5090-to-5090 and each replicate's pair is
#       4090-to-4090, so every paired delta is internally matched.
#
# CONCURRENCY. Each job is a GPU phase (the head) then a CPU phase (the
# eval), so N jobs destagger themselves rather than all wanting the same
# resource. Two caps matter and they are different numbers:
#
#   CF393_BB40K_JOBS   jobs in this pool. Bounded by VRAM, because on a
#                      Default-mode GPU gpu_gate does not serialise: every
#                      job in the pool can be in its head phase at once.
#   CF393_EVAL_SLOTS   concurrent GIFT-Evals, exported below. Bounded by
#                      cores: slots x EVAL_SHARDS of elisa's 32, and the
#                      brief keeps eight free for the rest of the box.
#
# Launching this script a second time while the first is still running is
# safe and is how the pool is widened: a job is claimed by `mkdir`, which
# is atomic, so no two workers take the same one.
#
# GPU PLACEMENT. `CF393_BB40K_GPUCYCLE` is a comma-separated cycle of
# device indices, not a count. It defaults to `1,0,1` because elisa's two
# 4090s are not equally free: GPU 0 carries ~7.5 GB of another agent
# session's resident contexts and GPU 1 is empty, so two thirds of the
# heads go to GPU 1.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WT="${WT:-$(dirname "$(dirname "$EXP")")}"
export WT
RES="$EXP/results"
LOG="$RES/seed_replicates_bb40k.log"

PROTOCOL_SEED=20260722
REPLICATE_SEEDS=(20260723 20260724)
HW_TAG=hw4090

# The six cells with bb100k replicates, in the order the card names them,
# and the GPU model each one's seed-20260722 bb40k head was trained on.
T1_CELLS=(arm6_v2_combab_alignS arm6_v2_combab_alignT
          arm5_combab_alignS arm5_combab_alignT
          arm6_v2_nse_alignT arm6_v2_nse_alignS)
declare -A CELL_GPU=(
  [arm6_v2_combab_alignS]=4090 [arm6_v2_combab_alignT]=4090
  [arm6_v2_nse_alignS]=4090
  [arm5_combab_alignS]=5090 [arm5_combab_alignT]=5090
  [arm6_v2_nse_alignT]=5090)
# The two marginal unreplicated cells, and their stops.
T2_CELLS=(arm4_combab arm6_v2_ncpc_alignT)

# Head budget by stop. These are the ladder's own numbers: the delta spans
# a 15,000-step head and a 30,000-step head, and a replicate that changed
# either would not replicate the thing being measured.
head_steps_for_stop(){ case "$1" in 40000) echo 15000;; *) echo 30000;; esac; }

TIERS="${CF393_BB40K_TIERS:-t1,hw,t2}"
JOBS="${CF393_BB40K_JOBS:-4}"
CLAIMS="${CF393_BB40K_CLAIMS:-/tmp/cf393_bb40k_claims}"
IFS=, read -r -a GPUCYCLE <<<"${CF393_BB40K_GPUCYCLE:-1,0,1}"

# Exported so every eval_local.sh under this pool sees them. 6 slots x 4
# shards = 24 of elisa's 32 cores.
export CF393_EVAL_SLOTS="${CF393_EVAL_SLOTS:-6}"
export EVAL_SHARDS="${EVAL_SHARDS:-4}"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [bb40k] $*" | tee -a "$LOG"; }

# Admission control on VRAM, because gpu_gate cannot do it here. gpu_gate
# returns immediately on a Default-mode GPU by design — elisa runs two
# cells per card on purpose — so nothing stops a widened pool from putting
# every one of its jobs into the head phase at the same moment. A measured
# head is 5.5 GB, GPU 1 is empty and GPU 0 carries ~7.5 GB of another
# session's resident contexts, so the two cards hold 4 and 3 heads and the
# fourth on GPU 0 would OOM a job that has already claimed its slot.
#
# Two things are enforced, both per GPU and both under one lock: the card
# has MIN_FREE MiB free, and admissions are at least SETTLE seconds apart.
# The second is not redundant — nvidia-smi does not report a head's memory
# until it allocates, so without it two workers admitted a second apart
# would both read the same free figure and both start.
#
# The lock is released by a background subshell that inherits the fd and
# sleeps: the caller goes on to train its head while the next admission
# waits out the settle window.
CF393_BB40K_MIN_FREE="${CF393_BB40K_MIN_FREE:-7000}"
CF393_BB40K_SETTLE="${CF393_BB40K_SETTLE:-150}"
CF393_BB40K_VRAM_TIMEOUT="${CF393_BB40K_VRAM_TIMEOUT:-86400}"

vram_admit(){  # <gpu>
  local gpu="$1" free waited=0
  exec 7>>"/tmp/cf393_vram_gate_$gpu" || return 0
  flock 7 2>/dev/null || return 0
  while :; do
    free="$(nvidia-smi --id="$gpu" --query-gpu=memory.free \
              --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
    # An unreadable nvidia-smi admits rather than blocks, matching gpu_gate.
    [ -n "$free" ] || { free=999999; }
    [ "$free" -ge "$CF393_BB40K_MIN_FREE" ] && break
    if [ "$waited" -ge "$CF393_BB40K_VRAM_TIMEOUT" ]; then
      say "vram-gate TIMEOUT on gpu $gpu after ${waited}s (${free} MiB free)"
      break
    fi
    [ $(( waited % 600 )) -eq 0 ] && \
      say "vram-gate waiting on gpu $gpu: ${free} MiB free, need $CF393_BB40K_MIN_FREE"
    sleep 30
    waited=$(( waited + 30 ))
  done
  ( sleep "$CF393_BB40K_SETTLE" ) &
  exec 7>&-
}

has_tier(){ case ",$TIERS," in *",$1,"*) return 0;; *) return 1;; esac; }

# One line per job: <cell> <stop> <enc> <seed> <tag> <gpu>. `-` is the empty
# tag, because an empty field cannot survive a whitespace-split job list.
job_list(){
  local i=0 cell enc seed stop
  emit(){ echo "$1 $2 $3 $4 $5 ${GPUCYCLE[$(( i % ${#GPUCYCLE[@]} ))]}"; i=$(( i + 1 )); }
  if has_tier t1; then
    for cell in "${T1_CELLS[@]}"; do
      for seed in "${REPLICATE_SEEDS[@]}"; do
        for enc in student teacher; do emit "$cell" 40000 "$enc" "$seed" -; done
      done
    done
  fi
  if has_tier hw; then
    for cell in "${T1_CELLS[@]}"; do
      [ "${CELL_GPU[$cell]}" = 5090 ] || continue
      for enc in student teacher; do
        emit "$cell" 40000 "$enc" "$PROTOCOL_SEED" "$HW_TAG"
      done
    done
  fi
  if has_tier t2; then
    for cell in "${T2_CELLS[@]}"; do
      for seed in "${REPLICATE_SEEDS[@]}"; do
        for stop in 40000 100000; do
          for enc in student teacher; do emit "$cell" "$stop" "$enc" "$seed" -; done
        done
      done
    done
  fi
}

# Mirrors eval_stop.sh's own suffix rule, so the score file sits next to
# the eval directory that produced it under the same name.
suffix(){  # <seed> <tag>
  local sfx=""
  [ "$1" != "$PROTOCOL_SEED" ] && sfx="_s$1"
  [ "$2" != "-" ] && sfx="${sfx}_$2"
  printf '%s' "$sfx"
}

score_path(){  # <cell> <stop> <enc> <seed> <tag>
  local root="${RUNS:-/home/jupyter/checkpoints_backup/cf-393}"
  printf '%s/%s/eval/score_bb%dk_%s%s.txt\n' \
    "$root" "$1" "$(( $2 / 1000 ))" "$3" "$(suffix "$4" "$5")"
}

run_one(){  # <cell> <stop> <enc> <seed> <tag> <gpu>
  local cell="$1" stop="$2" enc="$3" seed="$4" tag="$5" gpu="$6"
  local out; out="$(score_path "$cell" "$stop" "$enc" "$seed" "$tag")"
  local id="${cell}_bb$(( stop / 1000 ))k_${enc}_s${seed}"
  [ "$tag" != "-" ] && id="${id}_${tag}"

  if [ -s "$out" ]; then
    say "SKIP $id — score $(cat "$out")"
    return 0
  fi
  mkdir -p "$CLAIMS" 2>/dev/null
  if ! mkdir "$CLAIMS/$id" 2>/dev/null; then
    say "claimed already: $id"
    return 0
  fi

  vram_admit "$gpu"
  say "START $id gpu=$gpu heads=$(head_steps_for_stop "$stop")"
  local t0=$SECONDS
  local tagenv=""; [ "$tag" != "-" ] && tagenv="$tag"
  HEAD_SEED="$seed" HEAD_TAG="$tagenv" BB_GPU="$gpu" \
    bash "$HERE/eval_stop.sh" "$cell" "$stop" "$enc" \
      "$(head_steps_for_stop "$stop")" "$out" \
    >>"$RES/bb40k_${id}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -s "$out" ]; then
    say "DONE  $id in $(( (SECONDS - t0) / 60 ))m — $(cat "$out")"
  else
    # Release the claim so a later pass retries. A head that trained but
    # whose eval died is picked up where it stopped: eval_stop.sh skips a
    # head whose _final.pth exists and eval_local.sh resumes its shards.
    rmdir "$CLAIMS/$id" 2>/dev/null
    say "FAIL  $id rc=$rc — claim released, see bb40k_${id}.log"
  fi
  return $rc
}

status(){
  local line cell stop enc seed tag gpu out id n=0 done_=0 run=0
  printf '%-24s %-6s %-8s %-9s %-8s %s\n' cell stop head seed tag score
  while read -r cell stop enc seed tag gpu; do
    [ -n "$cell" ] || continue
    out="$(score_path "$cell" "$stop" "$enc" "$seed" "$tag")"
    id="${cell}_bb$(( stop / 1000 ))k_${enc}_s${seed}"
    [ "$tag" != "-" ] && id="${id}_${tag}"
    n=$(( n + 1 ))
    if [ -s "$out" ]; then
      done_=$(( done_ + 1 ))
      printf '%-24s %-6s %-8s %-9s %-8s %s\n' \
        "$cell" "$(( stop / 1000 ))k" "$enc" "$seed" "$tag" "$(cat "$out")"
    else
      if [ -d "$CLAIMS/$id" ]; then run=$(( run + 1 )); line=running; else line=-; fi
      printf '%-24s %-6s %-8s %-9s %-8s %s\n' \
        "$cell" "$(( stop / 1000 ))k" "$enc" "$seed" "$tag" "$line"
    fi
  done < <(job_list)
  echo
  echo "$done_/$n scored, $run running (tiers: $TIERS)"
}

case "${1:-}" in
  --list)   job_list; exit 0 ;;
  --status) status; exit 0 ;;
  --one)    shift; run_one "$@"; exit $? ;;
esac

[ -d "${RUNS:-/home/jupyter/checkpoints_backup/cf-393}" ] || {
  echo "ABORT: RUNS is not a directory" >&2; exit 2; }
say "pool of $JOBS over $(job_list | wc -l) job(s); tiers=$TIERS; claims in $CLAIMS"
say "eval slots=$CF393_EVAL_SLOTS x $EVAL_SHARDS shards; gpu cycle=${GPUCYCLE[*]}"
job_list | xargs -P "$JOBS" -L 1 bash "$HERE/seed_replicates_bb40k.sh" --one
say "pool drained"
status | tee -a "$LOG"
