#!/bin/bash
# #412 — hand one card to the momentum test after the first k = 32 arm.
#
# THE COMMITMENT. Pass 5 owes the orchestrator session ONE card for one
# 40,000-step run of `k3_r100_09_lr56_fix09`, the single-axis momentum test,
# after `k32_r200_08_lr56` lands its checkpoint and before this queue places
# `k32_r100_09_dec_lr56`.
#
# THE RACE, AND WHY THE FILE IS WRITTEN EARLY. The obvious version waits for
# the checkpoint and then writes the defer file. But the queue polls every 120
# s and the checkpoint appears at the END of the leg, so the queue can place
# the next arm on that card before the file lands. There is no lock between
# them.
#
# So the file is written when the leg STARTS, not when it ends. While the leg
# runs, the card is held by that leg anyway and the file changes nothing. The
# moment the leg ends, the file is already in place and the queue never sees
# the card free. No race, and no session has to be awake.
#
# Usage:  nohup bash scripts/pass5_defer_after_first_k32.sh >>results/pass5_defer.out 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

ARM="${CF412_DEFER_ARM:-k32_r200_08_lr56}"
POLL="${CF412_DEFER_POLL:-30}"
TIMEOUT="${CF412_DEFER_TIMEOUT:-172800}"
mkdir -p "$CF412_RESULTS"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412 defer] $*" \
  | tee -a "$CF412_RESULTS/pass5_defer.log"; }

# The card this arm's trainer runs on, or nothing.
arm_card(){
  local name pid dev
  name="$(cf412_run_name "$ARM")"
  for pid in $(ps -eo pid,args --no-headers 2>/dev/null \
       | awk -v n="--run-name $name " '$2 ~ /python/ && index($0 " ", n) { print $1 }'); do
    dev="$(tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null \
           | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)"
    [ -n "$dev" ] && { printf '%s\n' "$dev"; return 0; }
  done
  return 1
}

log "waits for $ARM to start, then hands its card to the momentum test"
waited=0
while [ "$waited" -lt "$TIMEOUT" ]; do
  if card="$(arm_card)"; then
    f="$CF412_RESULTS/pass5_defer_gpu${card}.txt"
    if [ -f "$f" ]; then
      log "gpu $card is already claimed — nothing to do"
      exit 0
    fi
    cat >"$f" <<EOF
GPU $card is claimed for the MOMENTUM TEST, \`k3_r100_09_lr56_fix09\`.

WHY. Pass 5 owes the orchestrator session one card for one 40,000-step run,
in exchange for the card that session handed to pass 5 earlier. That arm is
the single-axis momentum test: it differs from \`k3_r100_09_lr56\` in the EMA
schedule alone, ramped to 1.0 over 100,000 steps against no ramp at all.
Neither pass-5 arm can answer that question.

WHEN THIS FILE APPEARED. When \`$ARM\` STARTED on this card, not when it
finished. The pass-5 queue polls every 120 s and a checkpoint lands at the end
of a leg, so a file written at the end would race the queue for the card. The
leg holds the card while it runs, so an early file costs nothing.

TO RELEASE IT, delete this file:

    rm $f

The pass-5 queue reads it on every poll, so no restart and no message are
needed. The claiming session releases it once its leg holds a checkpoint.

Written $(date '+%Y-%m-%d %H:%M') by scripts/pass5_defer_after_first_k32.sh.
EOF
    log "wrote $f — $ARM runs on gpu $card, and that card goes to the momentum test when it ends"
    exit 0
  fi
  sleep "$POLL"; waited=$(( waited + POLL ))
done
log "TIMEOUT after ${waited}s — $ARM never started, so no card was handed over"
exit 1
