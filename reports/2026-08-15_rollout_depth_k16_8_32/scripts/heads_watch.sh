#!/bin/bash
# #401 — on elisa: one head and one 97-config GIFT-Eval per backbone stop, as
# each stop lands.
#
# The backbones train on a rented box and arrive here through the sync loop,
# 15 minutes apart (`sync/launch_sync.sh`). This watcher polls for stops whose
# checkpoint is here and whose score file is not, and fires `head_eval.sh` on
# each. It replaces nothing: same head trainer, same 30,000-step phase-1
# budget, same eval, same score file.
#
# Why a watcher and not phase1.sh. `phase1.sh` trains the backbone itself and
# then its head. Here the backbone trains somewhere else, so the loop that
# waits for it has to be its own process — one that survives a leg that
# arrives six hours after the one before it.
#
# Phase order. Every phase-1 head (30,000 steps) runs before any phase-2 head
# (40k / 100k / 200k steps), because phase 1 is what picks the arms and
# phase 2 costs up to seven times as much per head. Phase 2 starts by itself
# once every phase-1 cell of the pass holds a score.
#
# Concurrency. One head per GPU, through `head_eval_bb.sh`'s own VRAM gate and
# its per-GPU lock. HEAD_GPUS is the card list: `0` while another session
# holds GPU 1, `0 1` once it frees. The GIFT-Eval that follows each head runs
# on the CPU, so several evals overlap without touching a card.
#
# Everything is idempotent. A scored tag is a no-op, a trained head skips
# straight to its eval, and an eval resumes per shard.
#
# Usage:
#   CF401_ROOT=/home/jupyter/checkpoints_backup/cf-401-mean-box/sync \
#     HEAD_GPUS="0" nohup setsid bash scripts/heads_watch.sh &
#
#   ONCE=1 bash scripts/heads_watch.sh          # one pass, then exit
#   CF401_DRY_RUN=1 bash scripts/heads_watch.sh # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

HEAD_GPUS="${HEAD_GPUS:-0}"
POLL="${POLL:-300}"
ONCE="${ONCE:-}"
mkdir -p "$CF401_RESULTS"

LOG="$CF401_RESULTS/heads_watch.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#401 heads] $*" | tee -a "$LOG"; }

read -r -a gpu_list <<<"$HEAD_GPUS"
[ "${#gpu_list[@]}" -ge 1 ] || { echo "ABORT: HEAD_GPUS is empty" >&2; exit 2; }

# The score file one (depth, stop, head budget) writes, through the same tag
# builder head_eval.sh and collect.sh use.
score_file(){  # <k> <stop> <head steps>
  printf '%s/score_%s.txt\n' "$CF401_RESULTS" "$(cf401_tag "$1" "$2" "$3")"
}

# A cell is due when its backbone is here and its score file is not. An empty
# score file counts as missing: an eval killed between opening and writing
# leaves one, and 0.0 would be the best GM-Relative MASE this project ever
# recorded.
cell_is_due(){  # <k> <stop> <head steps>
  local bb
  [ -s "$(score_file "$1" "$2" "$3")" ] && return 1
  bb="$(cf401_bb_ckpt "$1" "$2")"
  [ -n "$bb" ] && [ -f "$bb" ]
}

# Phase 1 covers every (depth, stop) at the fixed budget. Phase 2 repeats the
# two best arms with the head budget set to the backbone stop.
phase1_jobs(){
  local k stop
  for k in $CF401_DEPTHS; do
    for stop in $CF401_STOPS; do
      printf '%s %s %s\n' "$k" "$stop" "$CF401_HEAD_STEPS_P1"
    done
  done
}

phase1_is_complete(){
  local k stop
  for k in $CF401_DEPTHS; do
    for stop in $CF401_STOPS; do
      [ -s "$(score_file "$k" "$stop" "$CF401_HEAD_STEPS_P1")" ] || return 1
    done
  done
  return 0
}

# The two arms phase 2 repeats. `pick_phase2_arms.py` refuses an incomplete
# phase 1, which is the card's own rule. ARMS overrides it.
phase2_arms(){
  if [ -n "${ARMS:-}" ]; then printf '%s\n' "$ARMS"; return 0; fi
  python3 "$HERE/pick_phase2_arms.py" --scores "$CF401_RESULTS/scores.csv" \
    2>>"$LOG"
}

phase2_jobs(){
  local arms k stop
  arms="$(phase2_arms)" || return 1
  for k in $arms; do
    cf401_is_in "$k" "$CF401_DEPTHS" || continue
    for stop in $CF401_STOPS; do
      # The card's rule: head steps = backbone steps.
      printf '%s %s %s\n' "$k" "$stop" "$stop"
    done
  done
}

# Every job of this pass, phase 1 first. Phase 2 is asked for only once every
# phase-1 cell holds a score, so the picker never sees a table with a hole.
due_jobs(){
  local k stop steps
  while read -r k stop steps; do
    [ -n "$k" ] || continue
    cell_is_due "$k" "$stop" "$steps" && printf '%s %s %s\n' "$k" "$stop" "$steps"
  done < <(phase1_jobs)
  phase1_is_complete || return 0
  while read -r k stop steps; do
    [ -n "$k" ] || continue
    cell_is_due "$k" "$stop" "$steps" && printf '%s %s %s\n' "$k" "$stop" "$steps"
  done < <(phase2_jobs)
}

if [ -n "${CF401_DRY_RUN:-}" ]; then
  echo "watch reduce=$CF401_REDUCE root=$CF401_ROOT results=$CF401_RESULTS"
  echo "  gpus='$HEAD_GPUS' poll=${POLL}s"
  bash "$HERE/collect.sh" >/dev/null 2>&1
  due_jobs | while read -r k stop steps; do
    echo "head k=$k stop=$stop steps=$steps enc=$CF401_ENC"
  done
  exit 0
fi

log "START reduce=$CF401_REDUCE gpus='$HEAD_GPUS' poll=${POLL}s root=$CF401_ROOT"

# One slot per GPU. A slot holds the pid of the head running on that card, so
# the next job takes the first card whose job finished. `head_eval_bb.sh` has
# its own VRAM gate and its own per-GPU lock underneath, which is what stops
# a head from starting into a card another session filled.
declare -A slot_pid=() slot_name=()

reap_slots(){  # <block: 1 waits for one to finish>
  local gpu rc freed=0
  for gpu in "${gpu_list[@]}"; do
    [ -n "${slot_pid[$gpu]:-}" ] || { freed=1; continue; }
    if ! kill -0 "${slot_pid[$gpu]}" 2>/dev/null; then
      wait "${slot_pid[$gpu]}"; rc=$?
      log "head ${slot_name[$gpu]} rc=$rc"
      unset 'slot_pid[$gpu]'; freed=1
    fi
  done
  [ "$freed" = "1" ] || [ "${1:-0}" != "1" ] || sleep 20
}

free_gpu(){
  local gpu
  for gpu in "${gpu_list[@]}"; do
    [ -n "${slot_pid[$gpu]:-}" ] || { printf '%s\n' "$gpu"; return 0; }
  done
  return 1
}

while true; do
  reap_slots
  # `collect.sh` rebuilds scores.csv and splits.csv from the score files, and
  # it runs BEFORE the job list, because the phase-2 half of that list is what
  # `pick_phase2_arms.py` reads out of scores.csv.
  bash "$HERE/collect.sh" >>"$LOG" 2>&1
  started=0
  while read -r k stop steps; do
    [ -n "$k" ] || continue
    gpu="$(free_gpu)" || break
    tag="$(cf401_tag "$k" "$stop" "$steps")"
    log "head $tag START on gpu $gpu ($(basename "$(cf401_bb_ckpt "$k" "$stop")"))"
    BB_GPU="$gpu" GPU_GATE_LOCKDIR="${GPU_GATE_LOCKDIR:-/tmp}" \
      nohup bash "$HERE/head_eval.sh" "$k" "$stop" "$steps" \
        >>"$CF401_RESULTS/head_${tag}.out" 2>&1 &
    slot_pid[$gpu]=$!; slot_name[$gpu]="$tag"
    started=$(( started + 1 ))
  done < <(due_jobs)

  if [ -n "$ONCE" ]; then
    for gpu in "${gpu_list[@]}"; do
      [ -n "${slot_pid[$gpu]:-}" ] && { wait "${slot_pid[$gpu]}"
                                        log "head ${slot_name[$gpu]} rc=$?"; }
    done
    log "ONCE pass done — $started head(s) started"
    exit 0
  fi

  # Nothing due and nothing running means every stop that is here is scored.
  # The loop keeps polling, because the box is still climbing to the next one.
  [ "$started" -gt 0 ] || sleep "$POLL"
  reap_slots 1
done
