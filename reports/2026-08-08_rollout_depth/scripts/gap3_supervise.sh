#!/bin/bash
# #373 review gap 3 — one event line per thing worth acting on, and a restart.
#
# `gap_watch.sh` reports. It does not repair. This supervises the two
# processes that carry the run and prints one line per event:
#
#   backbone   run_arm_k.sh -> the 40k checkpoint
#   heads      gap3_heads.sh -> 2 heads, 2 evals, 2 score files
#
# It restarts `gap3_heads.sh` if that driver exits before both scores land.
# The restart is safe: `head_eval_bb.sh` skips a head whose score file
# exists, reuses a head checkpoint that is already final, and resumes an
# eval per shard. A restart therefore costs only what did not finish.
#
# It exits 0 when both item-3 scores exist. It exits 1 when the backbone
# dies with no checkpoint. Every other terminal state prints a line first.
#
# Usage: bash scripts/gap3_supervise.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../results"
WT="${WT:-/home/jupyter/wt-cf-373-train}"
WRES="$WT/reports/2026-08-08_rollout_depth/results"
CF373_ROOT="${CF373_ROOT:-/home/jupyter/checkpoints_backup/cf-373}"
NAME="bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k0_aw4"
BBLOG="$WRES/run_${NAME}.log"
CK="$CF373_ROOT/$NAME/${NAME}_40k.pth"
S3S="$WRES/score_G_B1_k0_aw4_bb40k_student.txt"
S3T="$WRES/score_G_B1_k0_aw4_bb40k_teacher.txt"
SEEN="/tmp/cf373_sup_seen"
mkdir -p "$SEEN"

say(){ echo "[$(date '+%m-%d %H:%M')] $*"; }
once(){ # <key> <line>
  local k="$SEEN/$(echo "$1" | md5sum | cut -c1-16)"
  [ -f "$k" ] && return 0
  : >"$k"; say "$2"
}

step_now(){ grep -oE '^\[ *[0-9]+\]' "$BBLOG" 2>/dev/null | tail -1 | tr -dc '0-9'; }
bb_alive(){ pgrep -f "run-name $NAME" >/dev/null 2>&1; }
heads_alive(){ pgrep -f 'gap3_heads.sh' >/dev/null 2>&1; }

last_bb=20000
restarts=0

while :; do
  # --- both scores in hand: done ------------------------------------------
  if [ -s "$S3S" ] && [ -s "$S3T" ]; then
    say "ITEM3 DONE student=$(cat "$S3S") teacher=$(cat "$S3T")"
    exit 0
  fi

  # --- the backbone -------------------------------------------------------
  if [ -f "$CK" ]; then
    once ck "ITEM3 backbone 40k checkpoint on disk, $(stat -c %s "$CK") bytes"
  else
    s="$(step_now)"; [ -n "$s" ] || s=0
    if [ $(( s - last_bb )) -ge 10000 ]; then
      once "bb$s" "ITEM3 backbone $s/40000  $(grep -oE '[0-9.]+ sps  ETA [0-9.]+h' "$BBLOG" | tail -1)"
      last_bb=$s
    fi
    if ! bb_alive; then
      say "ITEM3 FAILED: the backbone process is gone at step $s with no 40k checkpoint"
      exit 1
    fi
  fi

  # --- the head driver ----------------------------------------------------
  if ! heads_alive; then
    if [ "$restarts" -ge 3 ]; then
      say "ITEM3 FAILED: gap3_heads.sh died $restarts times, giving up"
      exit 1
    fi
    restarts=$(( restarts + 1 ))
    ( cd "$HERE/.." && nohup bash scripts/gap3_heads.sh >>results/gap3_heads.out 2>&1 & )
    sleep 5
    say "ITEM3 restarted gap3_heads.sh (restart $restarts)"
  fi

  # --- one score in, one still out ---------------------------------------
  for enc in student teacher; do
    f="$WRES/score_G_B1_k0_aw4_bb40k_${enc}.txt"
    [ -s "$f" ] && once "sc$enc" "ITEM3 SCORE B1 k=0 L_align x4 bb40k $enc = $(cat "$f")"
  done

  # --- the driver's own terminal lines ------------------------------------
  if [ -f "$RES/gaps_driver.log" ]; then
    while IFS= read -r line; do
      once "$line" "$line"
    done < <(grep -a '\[gap3\]' "$RES/gaps_driver.log" 2>/dev/null \
             | grep -aE 'HEAD END|GIVING UP|ABORT|TIMEOUT|backbone in hand|gap 3 done')
  fi

  # --- a crash inside a head or an eval -----------------------------------
  for f in "$CF373_ROOT/eval/G_B1_k0_aw4_bb40k_student/stop.log" \
           "$CF373_ROOT/eval/G_B1_k0_aw4_bb40k_teacher/stop.log"; do
    [ -f "$f" ] || continue
    while IFS= read -r line; do
      once "$f$line" "FAILURE $(basename "$(dirname "$f")"): $line"
    done < <(grep -aE 'rc=[1-9]|ABORT|Traceback|CUDA out of memory' "$f" 2>/dev/null | tail -2)
  done

  sleep 120
done
