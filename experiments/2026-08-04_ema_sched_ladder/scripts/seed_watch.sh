#!/bin/bash
# #393 — one line per replicate score as it lands, across elisa and the
# rented boxes, and nothing while nothing changes.
#
# Usage:  WT=<checkout> bash scripts/seed_watch.sh [poll_seconds]
#
# Reads results/seed_boxes.txt for the roster, so a box added or released
# there is picked up without editing this. Pulls each box's score files as
# it finds them, which is what makes the run survive a box going away: the
# score is 7 bytes and the sync loop's 15-minute tick is slower than a
# spot reclaim.
#
# Exits 0 when all 36 (cell, head, seed) cells of the grid are scored —
# the 12 seed-20260722 numbers the study already has, plus the 24 this
# run adds.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
RES="$EXP/results"
RUNS="${RUNS:-/home/jupyter/checkpoints_backup/cf-393}"
POLL="${1:-600}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)
SEEN=/tmp/cf393_seed_seen
mkdir -p "$SEEN"

CELLS=(arm6_v2_combab_alignS arm6_v2_combab_alignT
       arm5_combab_alignS arm5_combab_alignT
       arm6_v2_nse_alignT arm6_v2_nse_alignS)
SEEDS=(20260722 20260723 20260724)

sfx(){ [ "$1" = 20260722 ] && echo "" || echo "_s$1"; }

# Bring every score file a box has produced back here. Scores are tiny and
# idempotent, so this is a plain copy rather than a safe_pull: a truncated
# 7-byte file is caught by the numeric test at read time.
pull_boxes(){
  local lbl contract host port gpu cores rate cell seeds
  while read -r lbl contract host port gpu cores rate cell seeds; do
    case "$lbl" in ''|\#*) continue ;; esac
    # In parallel and on a short leash: a box that is still installing, or
    # that has gone away, must not hold the tick behind it.
    timeout 45 scp -q "${SSH_OPTS[@]}" -P "$port" \
      "root@$host:/root/cf393_runs/$cell/eval/score_bb100k_*.txt" \
      "$RUNS/$cell/eval/" 2>/dev/null &
  done < "$RES/seed_boxes.txt"
  wait
}

n_done(){
  local n=0 cell enc seed f
  for cell in "${CELLS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for enc in student teacher; do
        f="$RUNS/$cell/eval/score_bb100k_${enc}$(sfx "$seed").txt"
        [ -s "$f" ] && n=$(( n + 1 ))
      done
    done
  done
  printf '%d\n' "$n"
}

while :; do
  pull_boxes
  for cell in "${CELLS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for enc in student teacher; do
        f="$RUNS/$cell/eval/score_bb100k_${enc}$(sfx "$seed").txt"
        key="${cell}_${enc}_${seed}"
        [ -s "$f" ] || continue
        [ -e "$SEEN/$key" ] && continue
        touch "$SEEN/$key"
        echo "[$(date -u '+%H:%M:%SZ')] scored $cell $enc s$seed = $(tr -d '[:space:]' <"$f")  ($(n_done)/36)"
      done
    done
  done
  d=$(n_done)
  if [ "$d" -ge 36 ]; then
    echo "[$(date -u '+%H:%M:%SZ')] ALL 36 replicate cells scored"
    exit 0
  fi
  sleep "$POLL"
done
