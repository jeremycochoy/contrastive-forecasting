#!/bin/bash
# #412 — a head and a 97-config eval for every backbone that has none.
#
# WHY THIS AND NOT `missing_heads.sh`. That script calls `head_eval.sh` and
# WAITS for it. One call is a 1.7-hour head train plus a 2.9-hour CPU
# evaluation, so a sweep that waits starts one arm every 4.6 hours. Seven arms
# would take 32 hours of that.
#
# This sweep starts each head in the BACKGROUND. The head train still
# serializes, because `head_eval_bb.sh` holds a `flock` over the card while it
# trains, and it drops that lock before the CPU evaluation. So the heads queue
# on the card and the evaluations overlap, which is what the box has spare.
#
# The duplicate guards are `missing_heads.sh`'s, and they read the process
# table, so a head this loop started in an earlier tick is visible to the next
# tick: the driver carries `head_eval.sh <arm> <stop>` and the head and the
# evaluation carry the backbone file name.
#
# Usage:  BB_GPU=1 bash scripts/head_sweep.sh          # loop until stopped
#         CF412_SWEEP_ONCE=1 bash scripts/head_sweep.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

BB_GPU="${BB_GPU:-1}"
EVERY="${CF412_SWEEP_EVERY:-600}"
AGE_MIN="${CF412_SWEEP_AGE_MIN:-5}"
mkdir -p "$CF412_RESULTS"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412 sweep] $*" \
  | tee -a "$CF412_RESULTS/phase1.log"; }

tick(){
  local started=0 arm stop bb
  for arm in $CF412_ARMS; do
    for stop in $CF412_STOPS; do
      bb="$(cf412_bb_ckpt "$arm" "$stop")"
      [ -n "$bb" ] && [ -f "$bb" ] || continue
      [ -s "$(cf412_score_file "$arm" "$stop")" ] && continue
      [ -n "$(find "$bb" -mmin -"$AGE_MIN" 2>/dev/null)" ] && continue
      # The whitelist guard. `pgrep -f` matched a peer session's watcher
      # shell on 2026-09-07 and skipped a head that never ran.
      bash "$HERE/head_busy.sh" "$arm" "$stop" >/dev/null 2>&1 && continue
      log "$arm at $stop — no score and no head. Starting one on gpu $BB_GPU."
      BB_GPU="$BB_GPU" nohup bash "$HERE/head_eval.sh" "$arm" "$stop" \
        >>"$CF412_RESULTS/sweep.log" 2>&1 &
      started=$(( started + 1 ))
      sleep 5   # so the next guard reads this driver in the process table
    done
  done
  [ "$started" -gt 0 ] && log "started $started head(s)"
  return 0
}

if [ -n "${CF412_SWEEP_ONCE:-}" ]; then tick; exit 0; fi
while :; do tick; sleep "$EVERY"; done
