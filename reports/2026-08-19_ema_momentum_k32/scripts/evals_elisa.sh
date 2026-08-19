#!/bin/bash
# #404 — the four 97-config GIFT-Evals on elisa, at the same time.
#
# The heads train on the box (`heads_box.sh`) and arrive here through the sync
# loop. The eval reads gift-eval-data and the `gift_eval` package, which live
# on elisa only, and it runs on the CPU (`eval_local.sh`, `--device cpu`). So
# this stage needs no GPU at all.
#
# Why not `heads_watch.sh`. That watcher fires one pair at a time, in the
# foreground, because it also TRAINS each head and elisa has one part-free
# GPU lane. Here every head is already on disk, so the four evals are four CPU
# jobs and they run together. #393's counting semaphore (`eval_slot.sh`, 5
# slots x 4 shards) is the cap that keeps elisa's 32 cores shared with the
# other sessions. This script does not raise it.
#
# THE GATE. An arm whose head is NOT here yet is SKIPPED, loudly. Without the
# gate `head_eval_bb.sh` would fall through to head TRAINING, take a GPU lane
# elisa does not have, and wait four hours for VRAM.
#
# Usage:
#   nohup setsid bash scripts/evals_elisa.sh > results/evals_elisa.out 2>&1 &
#
#   ARMS="a08 a09" bash scripts/evals_elisa.sh
#   CF404_DRY_RUN=1 bash scripts/evals_elisa.sh   # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
. "$HERE/study.sh"
cf404_use_root "$CF404_SYNC_ROOT"

ARMS="${ARMS:-$CF404_ARMS}"
STOP="${STOP:-$CF404_STOPS}"
mkdir -p "$CF404_RESULTS"

LOG="$CF404_RESULTS/evals_elisa.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 evals] $*" | tee -a "$LOG"; }

read -r -a arm_list <<<"$ARMS"
cf404_require_stop "$STOP" || exit $?
for arm in "${arm_list[@]}"; do cf404_require_arm "$arm" || exit $?; done

head_ckpt(){  # <arm>
  local tag
  tag="$(cf404_tag "$1" "$STOP" "$CF404_HEAD_STEPS")"
  printf '%s/qhead_%s_s%s_final.pth\n' "$(cf404_eval_dir "$1" "$tag")" \
    "$tag" "${HEAD_SEED:-20260722}"
}

if [ -n "${CF404_DRY_RUN:-}" ]; then
  echo "evals root=$CF404_ROOT results=$CF404_RESULTS wt=$CF404_WT"
  echo "  gift=${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}" \
       "shards=${EVAL_SHARDS:-4} slots=${CF393_EVAL_SLOTS:-5}"
  for arm in "${arm_list[@]}"; do
    ck="$(head_ckpt "$arm")"
    echo "eval $arm head=$([ -f "$ck" ] && wc -c <"$ck" || echo MISSING) $ck"
    echo "  bb=$(cf404_bb_ckpt "$arm" "$STOP" || true)"
    echo "  score=$(cf404_score_file "$arm" "$STOP")"
  done
  exit 0
fi

log "START arms='$ARMS' root=$CF404_ROOT shards=${EVAL_SHARDS:-4}"
pids=(); names=(); skipped=0
for arm in "${arm_list[@]}"; do
  ck="$(head_ckpt "$arm")"
  if [ ! -f "$ck" ]; then
    log "eval $arm SKIP — no head at $ck. The sync loop has not landed it."
    skipped=$(( skipped + 1 ))
    continue
  fi
  if [ -s "$(cf404_score_file "$arm" "$STOP")" ]; then
    log "eval $arm SKIP — already scored $(cat "$(cf404_score_file "$arm" "$STOP")")"
    continue
  fi
  bb="$(cf404_bb_ckpt "$arm" "$STOP")"
  if [ -z "$bb" ]; then
    log "eval $arm SKIP — no bb$(cf404_steps_label "$STOP") checkpoint here"
    skipped=$(( skipped + 1 ))
    continue
  fi
  log "eval $arm head $(wc -c <"$ck") B"
  nohup bash "$HERE/head_eval.sh" "$arm" "$STOP" \
    >>"$CF404_RESULTS/eval_${arm}_bb$(cf404_steps_label "$STOP").out" 2>&1 &
  pids+=($!); names+=("$arm")
done

failed=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; rc=$?
  log "eval ${names[$i]} rc=$rc"
  [ $rc -eq 0 ] || failed=$(( failed + 1 ))
done

for arm in "${arm_list[@]}"; do
  s="$(cf404_score_file "$arm" "$STOP")"
  if [ -s "$s" ]; then log "score $arm $(tr -d ' \t\r\n' <"$s")"
  else log "score $arm MISSING"; fi
done
bash "$HERE/collect.sh" 2>&1 | tee -a "$LOG"
log "EVALS DONE — $failed failed, $skipped skipped"
[ "$failed" -eq 0 ] && [ "$skipped" -eq 0 ]
