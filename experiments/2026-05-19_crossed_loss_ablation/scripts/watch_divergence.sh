#!/bin/bash
# Divergence / completion watcher. Polls the losses CSV + checkpoint dir +
# training process group. Prints a HEARTBEAT line periodically (so silence
# is never mistaken for success) and exactly one TERMINAL line on a
# terminal state, then exits — which re-invokes the agent.
#
# Terminal states:
#   DONE     — <run>_final.pth / _FINAL.pth written, or last step >= total
#   DIVERGED — NaN/Inf loss, OR _EMERGENCY_*.pth, OR post-warmup loss
#              blow-up (> FACTOR x post-warmup min AND still rising)
#   CRASH    — training process group gone before any DONE/DIVERGED
# On DIVERGED/CRASH the training process group is killed (frees the GPUs).
set -uo pipefail

SAVE_DIR="$1"; NAME="$2"; TOTAL_STEPS="$3"; STATUS_FILE="$4"; TRAIN_PGID="$5"; TLOG="$6"
CSV="${SAVE_DIR}/${NAME}_losses.csv"
# Divergence = loss climbs back to FACTOR x the GLOBAL best-so-far while
# still rising (the global min naturally sits at the healthy trough, so a
# collapse that completes early can no longer hide the baseline — the bug
# in the first version, where a post-warmup-min window anchored on the
# already-collapsed floor). ABS_MAX is a coarse NaN-ish safety ceiling.
# ARM_STEP just skips the first few noisy steps before arming the test.
FACTOR=2.5
ABS_MAX=50
ARM_STEP=500
POLL=60
HEARTBEAT=900        # ~15 min
start=$(date +%s); last_hb=0

emit(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
pg_alive(){ kill -0 "-${TRAIN_PGID}" 2>/dev/null; }
kill_pg(){ kill -TERM "-${TRAIN_PGID}" 2>/dev/null; sleep 8; kill -KILL "-${TRAIN_PGID}" 2>/dev/null; }
finish(){ # $1=STATUS $2=detail
  echo "STATUS=$1" > "$STATUS_FILE"; echo "DETAIL=$2" >> "$STATUS_FILE"
  echo "RUN=$NAME" >> "$STATUS_FILE"; echo "AT=$(date '+%F %T')" >> "$STATUS_FILE"
  [ "$1" != "DONE" ] && kill_pg
  emit "TERMINAL $1 | $2"
  exit 0
}

# CSV scan via awk: prints "step last_loss gmin rising naninf nrows" or "NA".
# gmin = GLOBAL min loss over all logged steps (the healthy trough), so a
# collapse cannot hide the baseline regardless of when it happens.
scan(){
  [ -s "$CSV" ] || { echo "NA"; return; }
  awk -F, '
    NR==1{ for(i=1;i<=NF;i++){h=$i; gsub(/^[ \t]+|[ \t\r]+$/,"",h); if(h=="step")si=i; if(h=="loss")li=i} next }
    si==0||li==0{ next }
    { st=$si+0; ls=$li+0; lsraw=$li
      if(lsraw ~ /[nN][aA][nN]|[iI][nN][fF]/){ni=1}
      n++; step=st; last=ls; if(n>=4)prev3=p3; p3=p2;p2=p1;p1=ls
      if(gm==""||ls<gm)gm=ls }
    END{ if(n==0){print "NA"; exit}
         rising=(prev3!=""&&last>prev3)?1:0
         printf "%d %.6f %s %d %d %d\n", step, last, (gm==""?"NA":sprintf("%.6f",gm)), rising, (ni?1:0), n }
  ' "$CSV"
}

emit "watcher up: run=$NAME pgid=$TRAIN_PGID total=$TOTAL_STEPS csv=$CSV"
while :; do
  # --- terminal: clean finish ---
  if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ] || [ -f "${SAVE_DIR}/${NAME}_final.pth" ]; then
    finish DONE "final checkpoint present"
  fi
  # --- terminal: emergency checkpoint (train.py writes on NaN/Inf) ---
  if ls "${SAVE_DIR}/${NAME}"_EMERGENCY_*.pth >/dev/null 2>&1; then
    finish DIVERGED "EMERGENCY checkpoint written ($(ls "${SAVE_DIR}/${NAME}"_EMERGENCY_*.pth 2>/dev/null|tr '\n' ' '))"
  fi
  r=$(scan)
  if [ "$r" != "NA" ]; then
    set -- $r; step="$1"; last="$2"; gmin="$3"; rising="$4"; naninf="$5"; nrows="$6"
    if [ "$naninf" = "1" ]; then finish DIVERGED "NaN/Inf in loss at step ${step}"; fi
    if [ "${step:-0}" -ge "$TOTAL_STEPS" ]; then finish DONE "reached step ${step} (>= ${TOTAL_STEPS})"; fi
    absblow=$(awk -v l="$last" -v c="$ABS_MAX" 'BEGIN{print (l>c)?1:0}')
    if [ "$absblow" = "1" ]; then finish DIVERGED "loss ${last} > absolute ceiling ${ABS_MAX} at step ${step}"; fi
    if [ "$gmin" != "NA" ] && [ "${step:-0}" -ge "$ARM_STEP" ] && [ "${nrows:-0}" -ge 20 ]; then
      blow=$(awk -v l="$last" -v m="$gmin" -v f="$FACTOR" 'BEGIN{print (m>0 && l>f*m)?1:0}')
      if [ "$blow" = "1" ] && [ "$rising" = "1" ]; then
        finish DIVERGED "loss blow-up: ${last} > ${FACTOR}x global-best ${gmin} at step ${step}, still rising"
      fi
    fi
  fi
  # --- terminal: process group died ---
  if ! pg_alive; then
    sleep 20  # grace: let final/emergency land
    if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ] || [ -f "${SAVE_DIR}/${NAME}_final.pth" ]; then finish DONE "final present (pg exited)"; fi
    if ls "${SAVE_DIR}/${NAME}"_EMERGENCY_*.pth >/dev/null 2>&1; then finish DIVERGED "EMERGENCY + pg exited"; fi
    tail_err="$(grep -aE 'Traceback|Error|RuntimeError|CUDA|OutOfMemory|OOM|Killed|assert' "$TLOG" 2>/dev/null | tail -3 | tr '\n' ' ')"
    finish CRASH "training pgid ${TRAIN_PGID} gone, no final/emergency. log tail: ${tail_err:-<none>}"
  fi
  # --- heartbeat ---
  now=$(date +%s)
  if [ $((now-last_hb)) -ge "$HEARTBEAT" ]; then
    last_hb=$now
    emit "HEARTBEAT step=${step:-?} loss=${last:-?} best=${gmin:-?} elapsed=$(( (now-start)/60 ))min"
  fi
  sleep "$POLL"
done
