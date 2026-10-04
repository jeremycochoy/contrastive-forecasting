#!/bin/bash
# #412 pass 4 — re-fire a leg whose lane exited without training it.
#
# WHY THIS EXISTS. `phase1.sh` logs a failed leg and goes to the next stop. At
# an arm's LAST stop there is no next one, so the lane exits and nothing
# retries the leg. For `k3_r100_09_lr56_dec` that last stop is 200,000 steps,
# which is 9.2 hours and the answer this card rests on. A detector that only
# tells a session is not enough: the session may not be there.
#
# WHAT IT DOES. Every CF412_REFIRE_EVERY seconds it looks for an arm that has
# an untrained leg and NO lane of its own. It then starts one `phase1.sh` for
# that arm over the stops that still lack a backbone, and it counts the
# attempt. Heads are not its business: the head sweep starts a head for any
# checkpoint that has no score.
#
# WHAT IT WILL NOT DO, because a repeat would waste the card:
#   - It never re-fires an arm that lost the contrastive task. That is a
#     MEASUREMENT, and the same objective loses it again.
#   - It never re-fires past CF412_REFIRE_MAX attempts for one arm.
#   - It never starts a second trainer. It refuses while any trainer of that
#     arm runs, and while any lane names that arm. A lane ORPHAN carries its
#     ARMS in its environment, so `cf412_lane_for_arm` sees one.
#
# Usage:  BB_GPU=0 bash scripts/pass4_refire.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"
. "$HERE/pass4_lib.sh"

BB_GPU="${BB_GPU:-0}"
EVERY="${CF412_REFIRE_EVERY:-300}"
MAX="${CF412_REFIRE_MAX:-3}"
ARMS="${CF412_PASS4_ARMS:-k3_r100_09_lr56_dec k3_r100_09_lr56_dec10k}"
LOG="$CF412_RESULTS/pass4_refire.log"
mkdir -p "$CF412_RESULTS"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412 refire] $*" | tee -a "$LOG"; }

stops_of(){  # <arm>
  case "$1" in
    k3_r100_09_lr56_dec)    echo "40000 100000 200000" ;;
    k3_r100_09_lr56_dec10k) echo "40000 100000" ;;
    *) echo "$CF412_STOPS" ;;
  esac
}

declare -A tries
log "armed — arms: $ARMS, gpu $BB_GPU, every ${EVERY}s, max $MAX per arm"
while :; do
  sleep "$EVERY"
  for arm in $ARMS; do
    # An arm that lost the contrastive task does not climb.
    [ -f "$(cf412_collapse_file "$arm")" ] && continue
    # The stops this arm still has no backbone for.
    missing=""
    for stop in $(stops_of "$arm"); do
      bb="$(cf412_bb_ckpt "$arm" "$stop")"
      [ -n "$bb" ] && [ -f "$bb" ] || missing="$missing $stop"
    done
    [ -n "$missing" ] || continue
    # Something is already working on it.
    busy=0
    for stop in $(stops_of "$arm"); do
      cf412_trainer_pid "$arm" "$stop" >/dev/null && busy=1
    done
    [ "$busy" -eq 1 ] && continue
    cf412_lane_for_arm "$arm" >/dev/null && continue
    n="${tries[$arm]:-0}"
    if [ "$n" -ge "$MAX" ]; then
      log "$arm has no lane and no backbone at$missing, and it used all $MAX attempts. A human must look."
      continue
    fi
    tries[$arm]=$(( n + 1 ))
    log "$arm has no lane and no backbone at$missing — attempt $(( n + 1 )) of $MAX on gpu $BB_GPU"
    BB_GPU="$BB_GPU" ARMS="$arm" STOPS="$missing" \
      setsid nohup bash "$HERE/phase1.sh" \
      >>"$CF412_RESULTS/pass4_refire_${arm}.log" 2>&1 < /dev/null &
    sleep 10   # so the next pass reads the new lane in the process table
  done
done
