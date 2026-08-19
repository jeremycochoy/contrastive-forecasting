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
# Order. The watcher runs `head_eval.sh` in the foreground, and waits for it.
# The pairs of one pass run one after the other, head and GIFT-Eval together.
# HEAD_GPUS gives the GPU that each pair takes, in turn. A second GPU gives no
# overlap, because only one pair runs at a time. Four heads in series, at
# 30,000 steps each, are an acceptable cost for this study.
#
# When it stops. When no pair is left to fire: each one is scored, or each one
# used its whole try budget (`cf404_heads_done`). It does NOT stop at the first
# scored arm — the four backbones arrive about five hours apart, so three arms
# are still on the box when the first one is scored.
#
# The try budget. A head or an eval that fails for a stable reason — a bad
# checkpoint, a missing package, a full disk — would otherwise re-fire on
# every pass, and delay each pair behind it for as long as the session runs.
# Each pair gets CF404_HEAD_TRIES attempts. The counter is a file in results/,
# so a watcher restarted after a reboot does not give a broken head three more
# hours. The log names that file when it drops a pair.
#
# Everything is idempotent. A scored arm is a no-op, a trained head skips
# straight to its eval, and an eval resumes per shard.
#
# Usage:
#   HEAD_GPUS="0" nohup setsid bash scripts/heads_watch.sh &
#
#   ONCE=1 bash scripts/heads_watch.sh           # one pass, then exit
#   CF404_HEAD_TRIES=5 bash scripts/heads_watch.sh
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

# One pass: every (arm, stop) this watcher must fire now. A pair qualifies when
# its backbone is here, its score is not, and it has a try left.
pending(){
  local arm stop
  while read -r arm stop; do
    [ -n "$arm" ] || continue
    [ -s "$(cf404_score_file "$arm" "$stop")" ] && continue
    cf404_exhausted "$arm" "$stop" && continue
    [ -n "$(cf404_bb_ckpt "$arm" "$stop")" ] || continue
    printf '%s %s\n' "$arm" "$stop"
  done < <(cf404_pairs)
}

# The pairs this watcher gave up on, for the plan and for the closing line.
dropped(){
  local arm stop
  while read -r arm stop; do
    [ -n "$arm" ] || continue
    cf404_exhausted "$arm" "$stop" && printf '%s %s\n' "$arm" "$stop"
  done < <(cf404_pairs)
}

if [ -n "${CF404_DRY_RUN:-}" ]; then
  echo "heads root=$CF404_ROOT results=$CF404_RESULTS gpus='$HEAD_GPUS'"
  echo "  budget=$CF404_HEAD_STEPS enc=$CF404_ENC poll=${POLL}s"
  echo "  pairs=$(cf404_pair_count) scored=$(cf404_heads_scored)" \
       "tries=$CF404_HEAD_TRIES per pair"
  pending | sed 's/^/  pending /'
  dropped | sed 's/^/  dropped /'
  exit 0
fi

log "START root=$CF404_ROOT arms='$CF404_ARMS' gpus='$HEAD_GPUS'" \
    "pairs=$(cf404_pair_count) tries=$CF404_HEAD_TRIES"
lane=0
while :; do
  fired=0
  while read -r arm stop; do
    [ -n "$arm" ] || continue
    gpu="${gpu_list[$(( lane % ${#gpu_list[@]} ))]}"
    lane=$(( lane + 1 ))
    # Count the attempt BEFORE the head runs. A head that takes the machine
    # down with it still spent a try.
    try="$(cf404_bump_tries "$arm" "$stop")"
    log "head $arm bb$(cf404_steps_label "$stop") on gpu $gpu" \
        "(try $try of $CF404_HEAD_TRIES)"
    BB_GPU="$gpu" bash "$HERE/head_eval.sh" "$arm" "$stop"
    rc=$?
    log "head $arm rc=$rc"
    if cf404_exhausted "$arm" "$stop"; then
      log "GAVE UP on $arm bb$(cf404_steps_label "$stop"): $try attempt(s)," \
          "no score. Fix the cause, delete" \
          "$(cf404_tries_file "$arm" "$stop"), and start this watcher again."
    fi
    fired=$(( fired + 1 ))
  done < <(pending)

  [ "$fired" -gt 0 ] && bash "$HERE/collect.sh" >>"$LOG" 2>&1

  # No pair is left to fire: every one is scored, or every one used its whole
  # try budget. A pair whose backbone is still on the box is neither, so the
  # loop keeps waiting for it.
  if cf404_heads_done; then
    log "done: $(cf404_heads_scored) of $(cf404_pair_count) pair(s) scored," \
        "$(dropped | wc -l | tr -d ' ') dropped"
    exit 0
  fi
  [ -n "$ONCE" ] && exit 0
  sleep "$POLL"
done
