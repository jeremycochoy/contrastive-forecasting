#!/bin/bash
# #407 round-3 item 2 — wait for the protocol re-draw, then read it back.
#
# The re-draw is two heads of 30,000 steps and two 97-config evals, behind
# the band in the card-1 flock. That is hours of GPU, and no agent should
# sit in a poll loop for it. This blocks instead, and it does the read-back
# the moment both heads score.
#
# It runs as one background task. The harness re-invokes the agent when this
# exits, so the agent waits for an event rather than for a clock.
#
# On exit it has already run, in order:
#
#   collect_replicates.sh   brings the draws into the checkout. A pair
#                           crosses only when its eval holds all 97 configs.
#   head_band.py            the band, the re-draw delta and the review-gap-6
#                           comparison, to `results/head_band.csv`.
#   teacher_pool.py         the teacher pool, refreshed.
#   plot_full_pass.py       the figure and its caption.
#   mirror_durable.sh       everything off `/tmp`.
#
# Exit codes: 0 scored and read back, 2 timed out, 3 the draw died unscored.
# On 3 the agent does nothing: `band_queue.sh` re-fires a lost re-draw by
# itself, inside its own cap.
#
# AWAIT_TIMEOUT  seconds to wait before giving up (default 25200, 7 h).
# AWAIT_POLL     seconds between checks (default 60).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="${CF407_RESULTS:-$STUDY/results}"

export WT="${WT:-/tmp/contrastive-forecasting-407}"
export RUNS="${RUNS:-/home/jupyter/cf373_r3/sync}"
PARENT_RES="$WT/reports/2026-08-08_rollout_depth/results"

SEED="${AWAIT_SEED:-20260722}"
STOP_K="${AWAIT_STOP_K:-200}"
TIMEOUT="${AWAIT_TIMEOUT:-25200}"
POLL="${AWAIT_POLL:-60}"
LOG="$RES/await_redraw.log"

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [cf407-await] $*" | tee -a "$LOG"; }

scored(){
  local head
  for head in student teacher; do
    [ -s "$PARENT_RES/score_A4_k3_bb${STOP_K}k_${head}_s${SEED}.txt" ] || return 1
  done
  return 0
}

# The re-draw's own chain, by its seed. `replicate_heads.sh` takes the seed
# as `argv[3]`, so the band's chains never match this.
redraw_alive(){
  local p a1 a2 a3
  for p in $(pgrep -f 'replicate_heads\.sh' 2>/dev/null); do
    a1=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n 2p)
    a2=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n 3p)
    a3=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n 4p)
    case "$a1" in */replicate_heads.sh) ;; *) continue;; esac
    [ "$a2" = "$(( STOP_K * 1000 ))" ] || continue
    [ "$a3" = "$SEED" ] && return 0
  done
  return 1
}

log "start seed=$SEED stop=${STOP_K}k timeout=${TIMEOUT}s"
deadline=$(( $(date +%s) + TIMEOUT ))
last=""

while :; do
  if scored; then
    log "both heads scored"
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    log "TIMEOUT after ${TIMEOUT}s. The re-draw has not scored."
    exit 2
  fi
  if ! redraw_alive; then
    log "the re-draw chain is gone and not scored. band_queue.sh owns the retry."
    exit 3
  fi
  # One line per state change, so the log stays readable over hours.
  now=$(tail -1 "$RUNS/eval/A4_k3_bb${STOP_K}k_student_s${SEED}/stop.log" \
        2>/dev/null | cut -c1-110)
  [ "$now" != "$last" ] && [ -n "$now" ] && { log "student: $now"; last="$now"; }
  sleep "$POLL"
done

log "student $(cat "$PARENT_RES/score_A4_k3_bb${STOP_K}k_student_s${SEED}.txt")  teacher $(cat "$PARENT_RES/score_A4_k3_bb${STOP_K}k_teacher_s${SEED}.txt")"

bash "$HERE/collect_replicates.sh" "$STOP_K" >>"$LOG" 2>&1 || \
  log "WARN: collect_replicates rc=$?"
python3 "$HERE/head_band.py" --stop "$(( STOP_K * 1000 ))" \
  --csv "$RES/head_band.csv" 2>&1 | tee -a "$LOG"
python3 "$HERE/teacher_pool.py" --csv "$RES/teacher_pool.csv" \
  >"$RES/teacher_pool.txt" 2>&1 || log "WARN: teacher_pool rc=$?"
python3 "$HERE/plot_full_pass.py" --out "$STUDY/plots/full_pass.png" \
  2>&1 | grep -v Warning | tee -a "$LOG"
bash "$HERE/mirror_durable.sh" >>"$LOG" 2>&1 || log "WARN: mirror rc=$?"
log "read-back done"
