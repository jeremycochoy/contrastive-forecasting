#!/bin/bash
# #409 — the head half of an arm whose backbone is ALREADY running.
#
# WHY THIS SCRIPT EXISTS. `phase1.sh` is a lane: it trains its arms in turn and
# queues each head. A lane reads its arms once, at the start, so a lane that
# holds two arms trains both on one card.
#
# This run started three lanes of two arms on card 1, because card 0 carried
# another agent's work. Card 0 came free 38 minutes later. The three arms that
# had not started belong on it, and the way to release them is to stop the lane
# managers — which also stops the head queue of the three arms in flight.
#
# So this takes over that half for ONE arm. It waits for the backbone the
# running trainer writes, then calls `head_eval.sh`, which is the same head and
# the same 97-config GIFT-Eval a lane would have called.
#
# It never starts a backbone. An arm whose trainer died leaves no checkpoint,
# and this exits 1 rather than train a second copy of a leg the lane owns.
#
# A collapse is a RESULT. `auc_guard.sh` writes `collapsed_<arm>.txt` and stops
# the leg, and this then exits 4 with no head, which is what the lane does.
#
# Usage:  BB_GPU=1 nohup setsid bash scripts/head_when_ready.sh dec_s20 40000 &
set -uo pipefail

ARM="${1:?usage: head_when_ready.sh <arm> <stop steps>}"
STOP="${2:?usage: head_when_ready.sh <arm> <stop steps>}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"
cf409_require_arm "$ARM" || exit $?
cf409_require_stop "$STOP" || exit $?

BB_GPU="${BB_GPU:-0}"
POLL="${CF409_WAIT_POLL:-60}"
mkdir -p "$CF409_RESULTS"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#409 wait $ARM] $*" \
  | tee -a "$CF409_RESULTS/head_when_ready.log"; }

# The trainer this arm's leg runs under. `run_arm.sh` is the wrapper the lane
# started, and it lives as long as the leg does.
leg_alive(){
  pgrep -f "run_arm.sh $ARM $STOP" >/dev/null 2>&1
}

log "waiting for the bb$(( STOP / 1000 ))k checkpoint of $ARM on gpu $BB_GPU"
while :; do
  ckpt="$(cf409_bb_ckpt "$ARM" "$STOP")"
  [ -n "$ckpt" ] && break
  if [ -f "$(cf409_collapse_file "$ARM")" ]; then
    log "STOPPED by the AUC gate — no head, no score. See" \
        "$(cf409_collapse_file "$ARM")"
    exit "$CF409_RC_COLLAPSED"
  fi
  if ! leg_alive; then
    # The leg can write its checkpoint and exit between two reads, so this
    # looks once more before it gives up.
    sleep "$POLL"
    ckpt="$(cf409_bb_ckpt "$ARM" "$STOP")"
    [ -n "$ckpt" ] && break
    log "the leg of $ARM is gone and it wrote no bb$(( STOP / 1000 ))k" \
        "checkpoint — no head"
    exit 1
  fi
  sleep "$POLL"
done
log "backbone $(basename "$ckpt") — starting the head"

try=1
while :; do
  BB_GPU="$BB_GPU" bash "$HERE/head_eval.sh" "$ARM" "$STOP" \
    >>"$CF409_RESULTS/head_${ARM}_bb$(cf409_steps_label "$STOP").out" 2>&1
  rc=$?
  log "head $ARM rc=$rc (try $try of $CF409_HEAD_TRIES)"
  [ "$rc" -eq 0 ] && break
  [ "$try" -ge "$CF409_HEAD_TRIES" ] && break
  try=$(( try + 1 ))
  sleep 60
done
bash "$HERE/collect.sh" >>"$CF409_RESULTS/head_when_ready.log" 2>&1
exit "$rc"
