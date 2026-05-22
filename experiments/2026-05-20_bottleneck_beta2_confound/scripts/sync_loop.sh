#!/bin/bash
# #309 sync loop — 15min ticks pulling alpha/beta/gamma artifacts from
# the vast box into MAIN checkout. Detached (setsid + pidfile). Exits
# automatically when all three arms are bb-done (FINAL + optimizer + ≥
# 49k losses rows local).
#
#   sync_loop.sh start   — fork detached loop
#   sync_loop.sh stop    — TERM the loop
#   sync_loop.sh tick    — one-shot manual sync (all arms)
#   sync_loop.sh active  — list per-arm done/active state
set -uo pipefail
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
EXP="$WT/experiments/2026-05-20_bottleneck_beta2_confound"
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-20_bottleneck_beta2_confound
LOG="$MAIN/sync.log"
PIDF="$EXP/scripts/state/sync.pid"
mkdir -p "$MAIN" "$EXP/scripts/state"

active(){
  for A in alpha beta gamma; do
    local name="bb_${A}_50k"
    local final="$MAIN/runs/${name}_FINAL.pth"
    local opt="$MAIN/runs/${name}_FINAL_optimizer.pth"
    local lossfile="$MAIN/runs/${name}_losses.csv"
    local rows=0
    [ -f "$lossfile" ] && rows=$(wc -l < "$lossfile" 2>/dev/null || echo 0)
    if [ -f "$final" ] && [ -f "$opt" ] && [ "$rows" -gt 49000 ]; then
      echo "DONE $A"
    else
      echo "ACTIVE $A"
    fi
  done
}

tick(){
  echo "=== sync_loop tick $(date '+%m-%d %H:%M:%S') ==="
  for A in alpha beta gamma; do
    local name="bb_${A}_50k"
    local final="$MAIN/runs/${name}_FINAL.pth"
    local opt="$MAIN/runs/${name}_FINAL_optimizer.pth"
    local lossfile="$MAIN/runs/${name}_losses.csv"
    local rows=0
    [ -f "$lossfile" ] && rows=$(wc -l < "$lossfile" 2>/dev/null || echo 0)
    if [ -f "$final" ] && [ -f "$opt" ] && [ "$rows" -gt 49000 ]; then
      echo "  ✓ $A DONE — no sync needed"
    else
      bash "$EXP/scripts/sync.sh" "$A" || true
    fi
  done
}

case "${1:-start}" in
  start)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      echo "already running pid=$(cat "$PIDF")"; exit 0
    fi
    setsid bash -c "
      while true; do
        bash $0 tick >> $LOG 2>&1
        # exit when no ACTIVE remain
        active=\$(bash $0 active 2>/dev/null | grep -c '^ACTIVE' || echo 0)
        if [ \"\$active\" -le 0 ]; then
          echo \"sync_loop: no ACTIVE arms left, exiting\" >> $LOG
          break
        fi
        sleep 900   # 15 min
      done
      rm -f $PIDF
    " < /dev/null > /dev/null 2>&1 &
    echo $! > "$PIDF"
    echo "sync_loop started pid=$!"
    ;;
  stop)
    if [ -f "$PIDF" ]; then
      kill "$(cat "$PIDF")" 2>/dev/null || true
      rm -f "$PIDF"
      echo "sync_loop stopped"
    else
      echo "not running"
    fi
    ;;
  tick) tick ;;
  active) active ;;
  *) echo "usage: $0 {start|stop|tick|active}"; exit 2 ;;
esac
