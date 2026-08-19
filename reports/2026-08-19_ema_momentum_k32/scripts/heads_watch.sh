#!/bin/bash
# #404 — on elisa: one head and one 97-config GIFT-Eval per arm, as each arm's
# backbone lands.
#
# The backbones train on a rented box and arrive here through the sync loop,
# 15 minutes apart (`sync/launch_sync.sh`). This watcher polls for arms whose
# checkpoint is here and whose score file is not, and fires `head_eval.sh` on
# each. It replaces nothing: same head trainer, same 30,000-step budget, same
# eval, same score file.
#
# Why a watcher and not phase1.sh. `phase1.sh` trains the backbone itself and
# then its head. Here the backbone trains somewhere else, so the loop that
# waits for it has to be its own process — one that survives an arm that
# arrives five hours after the one before it.
#
# The root. Unset, it is the box's tree where the sync loop lands it
# (CF404_SYNC_ROOT). It is never a local checkpoints directory: a backbone
# under one root and a watcher on another is an arm that climbs for five hours
# and is never scored.
#
# Concurrency. One head per GPU, through `head_eval_bb.sh`'s own VRAM gate and
# its per-GPU lock. The GIFT-Eval that follows each head runs on the CPU, so
# several evals overlap without touching a card.
#
# Everything is idempotent. A scored arm is a no-op, a trained head skips
# straight to its eval, and an eval resumes per shard.
#
# Usage:
#   HEAD_GPUS="0" nohup setsid bash scripts/heads_watch.sh &
#
#   ONCE=1 bash scripts/heads_watch.sh           # one pass, then exit
#   CF404_DRY_RUN=1 bash scripts/heads_watch.sh  # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
. "$HERE/study.sh"
cf404_use_root "$CF404_SYNC_ROOT"

HEAD_GPUS="${HEAD_GPUS:-0}"
POLL="${POLL:-300}"
ONCE="${ONCE:-}"
mkdir -p "$CF404_RESULTS"

LOG="$CF404_RESULTS/heads_watch.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 heads] $*" | tee -a "$LOG"; }

read -r -a gpu_list <<<"$HEAD_GPUS"
[ "${#gpu_list[@]}" -ge 1 ] || { echo "ABORT: HEAD_GPUS is empty" >&2; exit 2; }

score_file(){  # <arm> <stop>
  printf '%s/score_%s.txt\n' "$CF404_RESULTS" \
    "$(cf404_tag "$1" "$2" "$CF404_HEAD_STEPS")"
}

# One pass: every (arm, stop) whose backbone is here and whose score is not.
pending(){
  local arm stop
  for arm in $CF404_ARMS; do
    for stop in $CF404_STOPS; do
      [ -s "$(score_file "$arm" "$stop")" ] && continue
      [ -n "$(cf404_bb_ckpt "$arm" "$stop")" ] || continue
      printf '%s %s\n' "$arm" "$stop"
    done
  done
}

if [ -n "${CF404_DRY_RUN:-}" ]; then
  echo "heads root=$CF404_ROOT results=$CF404_RESULTS gpus='$HEAD_GPUS'"
  echo "  budget=$CF404_HEAD_STEPS enc=$CF404_ENC poll=${POLL}s"
  pending | sed 's/^/  pending /'
  exit 0
fi

log "START root=$CF404_ROOT arms='$CF404_ARMS' gpus='$HEAD_GPUS'"
lane=0
while :; do
  fired=0
  while read -r arm stop; do
    [ -n "$arm" ] || continue
    gpu="${gpu_list[$(( lane % ${#gpu_list[@]} ))]}"
    lane=$(( lane + 1 ))
    log "head $arm bb$(cf404_steps_label "$stop") on gpu $gpu"
    BB_GPU="$gpu" bash "$HERE/head_eval.sh" "$arm" "$stop"
    rc=$?
    log "head $arm rc=$rc"
    fired=$(( fired + 1 ))
  done < <(pending)

  [ "$fired" -gt 0 ] && bash "$HERE/collect.sh" >>"$LOG" 2>&1

  # Every arm scored: the study's head half is done.
  if [ -z "$(pending)" ]; then
    scored=0
    for arm in $CF404_ARMS; do
      for stop in $CF404_STOPS; do
        [ -s "$(score_file "$arm" "$stop")" ] && scored=$(( scored + 1 ))
      done
    done
    if [ "$scored" -gt 0 ]; then
      log "every arm on this side is scored ($scored) — done"
      exit 0
    fi
  fi
  [ -n "$ONCE" ] && exit 0
  sleep "$POLL"
done
