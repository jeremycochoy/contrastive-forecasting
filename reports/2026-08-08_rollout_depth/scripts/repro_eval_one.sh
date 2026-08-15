#!/bin/bash
# #373 — one A1/B3 reproduction eval, end to end.
#
# Waits for its own repro head on the box, pulls it, and runs the study's
# 97-config GIFT-Eval against the SAME backbone file that head was trained
# from. One tag per process, so the four reproductions overlap instead of
# queueing behind each other.
#
# This replaces the serial loop in repro_pair_eval.sh. That loop also gated
# the pull on 1,000,000 bytes; a quantile head final is 440 KB, so it could
# never pull one. The gate here is 300,000 bytes, the same floor q_finish.sh
# uses for a head.
#
# The score lands in results/score_<TAG>.txt. The tag carries `rep`, so a
# reproduction never overwrites the number it checks.
#
# Usage: repro_eval_one.sh <tag> <stop_k> <backbone.pth>
set -uo pipefail

TAG="${1:?usage: repro_eval_one.sh <tag> <stop_k> <backbone.pth>}"
STOPK="${2:?stop in k}"
BB="${3:?backbone path}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
export WT="${WT:-/home/jupyter/wt-cf-373-run2}"
LOG="$RES/repro_eval_$TAG.log"
SEED=20260722
MIN_HEAD=300000

BOX_HOST="${BOX_HOST:-ssh6.vast.ai}"
BOX_PORT="${BOX_PORT:-37390}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o ServerAliveInterval=30)

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [$TAG] $*" | tee -a "$LOG"; }

SCORE="$RES/score_${TAG}.txt"
[ -s "$SCORE" ] && { log "SKIP — score already $(cat "$SCORE")"; exit 0; }
[ -f "$BB" ] || { log "ABORT — no backbone at $BB"; exit 3; }

OUT="$RES/eval/$TAG"
mkdir -p "$OUT"
HEAD_LOCAL="$OUT/qhead_${TAG}_s${SEED}_final.pth"
REMOTE="/root/cf373_runs/eval/$TAG/qhead_${TAG}_s${SEED}_final.pth"

log "start — waiting for head, bb=$(basename "$BB")"

waited=0
while [ ! -f "$HEAD_LOCAL" ]; do
  sz="$(ssh "${SSH_OPTS[@]}" -p "$BOX_PORT" "root@$BOX_HOST" \
        "stat -c %s '$REMOTE' 2>/dev/null" 2>/dev/null | tr -d ' ')"
  if [ -n "$sz" ] && [ "$sz" -ge "$MIN_HEAD" ] 2>/dev/null; then
    # Pull to .tmp then move, so a dropped transfer cannot leave a short file.
    if scp "${SSH_OPTS[@]}" -P "$BOX_PORT" "root@$BOX_HOST:$REMOTE" \
           "$HEAD_LOCAL.tmp" >/dev/null 2>&1; then
      got="$(stat -c %s "$HEAD_LOCAL.tmp" 2>/dev/null || echo 0)"
      if [ "$got" -ge "$MIN_HEAD" ]; then
        mv -f "$HEAD_LOCAL.tmp" "$HEAD_LOCAL"
        ssh "${SSH_OPTS[@]}" -p "$BOX_PORT" "root@$BOX_HOST" \
            "cat /root/cf373_runs/eval/$TAG/backbone_md5.txt" 2>/dev/null \
            > "$OUT/backbone_md5.txt"
        log "pulled head ($got bytes), bb md5 $(tr -d ' \n' < "$OUT/backbone_md5.txt")"
      else
        rm -f "$HEAD_LOCAL.tmp"; log "short pull ($got bytes), retry"
      fi
    fi
  fi
  [ -f "$HEAD_LOCAL" ] && break
  if [ "$waited" -ge 43200 ]; then log "ABORT — head never appeared"; exit 4; fi
  [ $(( waited % 1800 )) -eq 0 ] && [ "$waited" -gt 0 ] && log "waiting (${waited}s, remote size ${sz:-none})"
  sleep 60; waited=$(( waited + 60 ))
done

# The md5 the head recorded must be the backbone this eval pairs it with.
want="$(md5sum "$BB" | cut -d' ' -f1)"
got="$(tr -d ' \n' < "$OUT/backbone_md5.txt" 2>/dev/null)"
if [ -n "$got" ] && [ "$got" != "$want" ]; then
  log "ABORT — head trained on $got, evaluating against $want"; exit 5
fi

log "eval start  bb=$(basename "$BB")  md5=$want"
EVAL_SHARDS="${EVAL_SHARDS:-4}" WT="$WT" \
  bash "$HERE/eval_local.sh" "$TAG" "$STOPK" student \
       "$BB" "$HEAD_LOCAL" "$OUT" "$SCORE" >>"$LOG" 2>&1
rc=$?
log "rc=$rc score=$(cat "$SCORE" 2>/dev/null || echo NONE)"
exit $rc
