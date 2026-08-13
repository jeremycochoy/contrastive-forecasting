#!/usr/bin/env bash
# #373 session 17 — block until the round's last five numbers land.
#
# Left: A4's bb200k student score, and the four A1/B3 student reproductions.
# Exits 0 when all five are in, 2 on timeout. One line per event.
set -u
RES="/home/jupyter/wt-cf-373-run2/reports/2026-08-08_rollout_depth/results"
MAXH="${1:-6}"
WANT="A4_k3_bb200k_student A1rep_k3_bb40k_student B3rep_k3_bb40k_student \
A1rep_k3_bb100k_student B3rep_k3_bb100k_student"

t0=$(date +%s); last_hb=0
declare -A seen
while true; do
  left=0
  for w in $WANT; do
    f="$RES/score_${w}.txt"
    if [ -s "$f" ]; then
      [ -n "${seen[$w]:-}" ] || { seen[$w]=1; echo "[$(date -u +%H:%M:%SZ)] SCORE $w = $(cat "$f")"; }
    else
      left=$((left+1))
    fi
  done
  [ "$left" -eq 0 ] && { echo "[$(date -u +%H:%M:%SZ)] ALL FIVE IN"; exit 0; }
  now=$(date +%s); el=$((now-t0))
  if [ $((el-last_hb)) -ge 1800 ]; then
    last_hb=$el
    cr=$(PATH="$HOME/.local/bin:$PATH" timeout 60 vastrun-balance 2>/dev/null | awk '/Credit/{print $2}')
    bb=$(tail -1 /home/jupyter/cf373_r3/sync/arm6_v2_combab_alignS/leg_200k/cf393_arm6_v2_combab_alignS_cf373k3_r2_losses.csv 2>/dev/null | cut -d, -f1)
    echo "[$(date -u +%H:%M:%SZ)] HEARTBEAT left=$left credit=${cr:-?} A4_bb_step=${bb:-?}"
  fi
  [ "$el" -gt $((MAXH*3600)) ] && { echo "[$(date -u +%H:%M:%SZ)] TIMEOUT left=$left"; exit 2; }
  sleep 60
done
