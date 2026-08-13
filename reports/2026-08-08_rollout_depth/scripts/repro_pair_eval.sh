#!/bin/bash
# #373 — the eval half of the A1/B3 reproduction.
#
# Waits for each repro head on the box, pulls it here, and runs the same
# 97-config GIFT-Eval the study runs, against the SAME backbone file the
# repro head was trained from. It takes an eval slot like every other eval,
# so it never crowds out the queue's own work.
#
# The score lands in results/score_<CELL>rep_k3_bb<STOP>k_student.txt. That
# name is deliberately NOT the cell's canonical name: a reproduction must
# not overwrite the number it is checking.
#
# Usage: repro_pair_eval.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
export WT="${WT:-/home/jupyter/wt-cf-373-run2}"
LOG="$RES/repro_pair_eval.log"
SEED=20260722

BOX_HOST="${BOX_HOST:-ssh6.vast.ai}"
BOX_PORT="${BOX_PORT:-37390}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o ServerAliveInterval=30)

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [repro-eval] $*" | tee -a "$LOG"; }

R2=/home/jupyter/cf373_r2
A1_40="$R2/A1/sync/arm5_combab_alignS/leg_40k/cf393_arm5_combab_alignS_cf373k3_40k.pth"
A1_100="$R2/A1/sync/arm5_combab_alignS/leg_100k/cf393_arm5_combab_alignS_cf373k3_100k.pth"
B3D="$R2/B3/sync/bb_small_arm5_combab_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3"
B3_40="$B3D/bb_small_arm5_combab_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3_40k.pth"
B3_100="$B3D/bb_small_arm5_combab_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3_r2_100k.pth"

# tag  stop_k  backbone
JOBS="
A1rep_k3_bb40k_student	40	$A1_40
B3rep_k3_bb40k_student	40	$B3_40
A1rep_k3_bb100k_student	100	$A1_100
B3rep_k3_bb100k_student	100	$B3_100
"

log "start — 4 reproductions, student head only"

while read -r TAG STOPK BB; do
  [ -n "$TAG" ] || continue
  SCORE="$RES/score_${TAG}.txt"
  if [ -s "$SCORE" ]; then log "SKIP $TAG — score already $(cat "$SCORE")"; continue; fi
  [ -f "$BB" ] || { log "ABORT $TAG — no backbone at $BB"; continue; }

  OUT="$RES/eval/$TAG"
  mkdir -p "$OUT"
  HEAD_LOCAL="$OUT/qhead_${TAG}_s${SEED}_final.pth"
  REMOTE="/root/cf373_runs/eval/$TAG/qhead_${TAG}_s${SEED}_final.pth"

  # Wait for the box to finish this head. The driver runs them in order, so
  # a wait here is a wait on everything before it too.
  waited=0
  while [ ! -f "$HEAD_LOCAL" ]; do
    sz="$(ssh "${SSH_OPTS[@]}" -p "$BOX_PORT" "root@$BOX_HOST" \
          "stat -c %s '$REMOTE' 2>/dev/null" 2>/dev/null | tr -d ' ')"
    if [ -n "$sz" ] && [ "$sz" -gt 1000000 ] 2>/dev/null; then
      # Pull to .tmp then move, so a dropped transfer cannot leave a short file.
      if scp "${SSH_OPTS[@]}" -P "$BOX_PORT" "root@$BOX_HOST:$REMOTE" \
             "$HEAD_LOCAL.tmp" >/dev/null 2>&1; then
        got="$(stat -c %s "$HEAD_LOCAL.tmp" 2>/dev/null || echo 0)"
        if [ "$got" -ge 1000000 ]; then
          mv -f "$HEAD_LOCAL.tmp" "$HEAD_LOCAL"
          ssh "${SSH_OPTS[@]}" -p "$BOX_PORT" "root@$BOX_HOST" \
              "cat /root/cf373_runs/eval/$TAG/backbone_md5.txt" 2>/dev/null \
              > "$OUT/backbone_md5.txt"
          log "pulled $TAG head ($got bytes), bb md5 $(cat "$OUT/backbone_md5.txt" 2>/dev/null)"
        else
          rm -f "$HEAD_LOCAL.tmp"; log "$TAG short pull ($got bytes), retry"
        fi
      fi
    fi
    [ -f "$HEAD_LOCAL" ] && break
    if [ "$waited" -ge 43200 ]; then log "ABORT $TAG — head never appeared"; break; fi
    [ $(( waited % 1800 )) -eq 0 ] && log "waiting for $TAG head (${waited}s)"
    sleep 120; waited=$(( waited + 120 ))
  done
  [ -f "$HEAD_LOCAL" ] || continue

  # The md5 the head recorded must be the backbone this eval pairs it with.
  want="$(md5sum "$BB" | cut -d' ' -f1)"
  got="$(cat "$OUT/backbone_md5.txt" 2>/dev/null | tr -d ' \n')"
  if [ -n "$got" ] && [ "$got" != "$want" ]; then
    log "ABORT $TAG — head trained on $got, evaluating against $want"; continue
  fi

  log "eval $TAG  bb=$(basename "$BB")  md5=$want"
  EVAL_SHARDS="${EVAL_SHARDS:-4}" WT="$WT" \
    bash "$HERE/eval_local.sh" "$TAG" "$STOPK" student \
         "$BB" "$HEAD_LOCAL" "$OUT" "$SCORE" >>"$LOG" 2>&1
  rc=$?
  log "$TAG rc=$rc score=$(cat "$SCORE" 2>/dev/null || echo NONE)"
done <<< "$JOBS"

log "done"
