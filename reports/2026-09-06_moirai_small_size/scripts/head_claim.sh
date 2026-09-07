#!/bin/bash
# #412 — the head of a checkpoint that NO lane will claim, started on this
# card the moment the checkpoint lands.
#
# WHY IT EXISTS. Two sweeps run on this box, one on each card, and both hold a
# five-minute age gate: a checkpoint younger than that belongs to the lane
# that wrote it. The arms below have no lane. `queue_backbones.sh` calls
# `run_arm.sh` and trains no head, and the two drivers that would have trained
# a head on gpu 0 are stopped. So the age gate protects nothing here, and it
# hands the race to whichever sweep ticks first. The sweep on gpu 0 would win
# a head that cannot fit beside a k = 32 backbone.
#
# This waiter takes those checkpoints as they land, on THIS card. Both sweeps
# then read `head_eval.sh <arm> <stop>` in the process table and skip.
#
# It starts nothing twice: the same two guards as `missing_heads.sh`, plus the
# score file.
#
# Usage:  BB_GPU=1 CF412_CLAIM="k3_r100_09_dec:40000 k3_r100_09:200000" \
#           bash scripts/head_claim.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

BB_GPU="${BB_GPU:-1}"
CLAIM="${CF412_CLAIM:?usage: CF412_CLAIM=\"<arm>:<stop> ...\" head_claim.sh}"
POLL="${CF412_CLAIM_POLL:-60}"
mkdir -p "$CF412_RESULTS"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412 claim] $*" \
  | tee -a "$CF412_RESULTS/phase1.log"; }
running(){ local pat="$1"; pgrep -f "[${pat:0:1}]${pat:1}" >/dev/null 2>&1; }

for pair in $CLAIM; do
  cf412_require_arm "${pair%%:*}" || exit $?
  cf412_require_stop "${pair##*:}" || exit $?
done
log "gpu $BB_GPU  waits on: $CLAIM"

left="$CLAIM"
while [ -n "$left" ]; do
  next=""
  for pair in $left; do
    arm="${pair%%:*}"; stop="${pair##*:}"
    if [ -f "$(cf412_collapse_file "$arm")" ]; then
      log "$arm at $stop dropped — it lost the contrastive task"; continue
    fi
    if [ -s "$(cf412_score_file "$arm" "$stop")" ]; then
      log "$arm at $stop already scored"; continue
    fi
    bb="$(cf412_bb_ckpt "$arm" "$stop")"
    if [ -z "$bb" ] || [ ! -f "$bb" ]; then next="$next $pair"; continue; fi
    if running "head_eval.sh $arm $stop" || running "$(basename "$bb")"; then
      log "$arm at $stop — a head already runs it"; continue
    fi
    log "$arm at $stop — claiming its head on gpu $BB_GPU"
    BB_GPU="$BB_GPU" nohup bash "$HERE/head_eval.sh" "$arm" "$stop" \
      >>"$CF412_RESULTS/sweep.log" 2>&1 &
    sleep 5
  done
  left="${next# }"
  [ -n "$left" ] || break
  sleep "$POLL"
done
log "every claimed head started"
