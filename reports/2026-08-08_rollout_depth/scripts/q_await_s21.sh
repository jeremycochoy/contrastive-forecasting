#!/bin/bash
# #373 session 21 — block until the round's last two numbers land.
#
# What is left after the box was released:
#   ev_A4_200k_student   running on elisa cores since 12:32Z
#   hd_A4_200k_teacher   running on elisa GPU 1 since 13:02Z, 30,000 steps
#   ev_A4_200k_teacher   queued behind that head
#
# Both are free: no rented hardware is running, and none will be. This
# waits on the two score files, prints a line every tick so the wait is
# readable after the fact, and exits when both exist.
#
# Usage: q_await_s21.sh [hours] [tick minutes]
set -uo pipefail

HOURS="${1:-6}"
TICK_MIN="${2:-10}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
LOG="$RES/q_await_s21.log"
R3="${CF373_R3:-/home/jupyter/cf373_r3/sync}"

WANT=(score_A4_k3_bb200k_student.txt score_A4_k3_bb200k_teacher.txt)
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
TICK=$(( TICK_MIN * 60 ))

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [s21] $*" | tee -a "$LOG"; }

have(){ local n=0 f; for f in "${WANT[@]}"; do [ -s "$RES/$f" ] && n=$(( n + 1 )); done; echo "$n"; }

# The eval writes one CSV per shard. Rows across the four is the honest
# progress number for a 97-config eval; the head has its own losses CSV.
ev_rows(){ # <tag>
  local t="$1" n=0 f
  for f in "$R3/eval/$t"/gift/shard_*/all_results.csv; do
    [ -f "$f" ] || continue
    n=$(( n + $(( $(wc -l < "$f") - 1 )) ))
  done; echo "$n"
}
hd_step(){ # <tag>
  local f="$R3/eval/$1/qhead_${1}_s20260722_losses.csv"
  [ -f "$f" ] && tail -1 "$f" | cut -d, -f1 || echo -
}

log "start: want ${#WANT[@]} scores, deadline $(date -u -d "@$DEADLINE" '+%H:%MZ')"
while :; do
  n="$(have)"
  log "scores $n/${#WANT[@]}  ev_student $(ev_rows A4_k3_bb200k_student)/97  hd_teacher step $(hd_step A4_k3_bb200k_teacher)/30000  ev_teacher $(ev_rows A4_k3_bb200k_teacher)/97"
  [ "$n" -eq "${#WANT[@]}" ] && { log "BOTH SCORES IN"; exit 0; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "DEADLINE reached with $n/${#WANT[@]}"; exit 2; }
  sleep "$TICK"
done
