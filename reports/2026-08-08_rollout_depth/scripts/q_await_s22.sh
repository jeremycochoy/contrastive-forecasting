#!/bin/bash
# #373 session 22 — block until the round's last number lands, or until the
# job that makes it dies.
#
# One deliverable is left: ev_A4_200k_teacher, behind hd_A4_200k_teacher.
# Both run on elisa and cost nothing. A plain wait on the score file hides a
# dead head for the whole deadline, so this also exits when the dispatcher
# marks either job failed, or when neither job is running and no score exists.
#
# Usage: q_await_s22.sh [hours] [tick seconds]
set -uo pipefail
HOURS="${1:-4}"; TICK="${2:-120}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"; RES="$STUDY/results"; Q="$RES/queue"
R3="${CF373_R3:-/home/jupyter/cf373_r3/sync}"
SCORE="$RES/score_A4_k3_bb200k_teacher.txt"
LOG="$RES/q_await_s22.log"
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [s22] $*" | tee -a "$LOG"; }

hd_step(){ tail -40 "$R3/eval/A4_k3_bb200k_teacher/head.log" 2>/dev/null \
           | grep -oE '^\[ *[0-9]+\]' | tail -1 | tr -dc 0-9; }
ev_rows(){ local n=0 f
  for f in "$R3/eval/A4_k3_bb200k_teacher"/gift/shard_*/all_results.csv; do
    [ -f "$f" ] || continue; n=$(( n + $(( $(wc -l < "$f") - 1 )) )); done; echo "$n"; }
st(){ cat "$Q/$1.state" 2>/dev/null || echo queued; }

log "start: want $SCORE, deadline $(date -d "@$DEADLINE" '+%H:%M')"
while :; do
  if [ -s "$SCORE" ]; then log "SCORE IN: $(cat "$SCORE")"; exit 0; fi
  h="$(st hd_A4_200k_teacher)"; e="$(st ev_A4_200k_teacher)"
  case "$h$e" in *failed*) log "ABORT: hd=$h ev=$e"; exit 3;; esac
  live=0
  pgrep -f "train_forecasting_head.py.*A4_k3_bb200k_teacher" >/dev/null && live=1
  pgrep -f "A4_k3_bb200k_teacher" >/dev/null && live=1
  [ "$e" = "queued" ] && live=1
  if [ "$live" -eq 0 ]; then log "ABORT: nothing running, hd=$h ev=$e"; exit 4; fi
  log "hd=$h step $(hd_step)/30000  ev=$e rows $(ev_rows)/97"
  [ "$(date +%s)" -lt "$DEADLINE" ] || { log "DEADLINE"; exit 5; }
  sleep "$TICK"
done
