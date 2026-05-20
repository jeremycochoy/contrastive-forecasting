#!/bin/bash
# Variance sync loop — 15min ticks for each running (arm,seed). Detached
# (setsid + pid file). Auto-discovers active variance state files; stops
# polling a (arm,seed) once its FINAL.pth + optimizer + ≥ 49k losses rows
# are local (true bb-done).
#   sync_loop_variance.sh start    — fork detached loop, write pid
#   sync_loop_variance.sh stop     — TERM the loop
#   sync_loop_variance.sh tick     — one-shot manual sync for all active
set -uo pipefail
WT=/home/jupyter/cf-wt-crossed-loss
EXP="$WT/experiments/2026-05-19_crossed_loss_xbranch_ablation"
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_xbranch_ablation
LOG="$EXP/sync_variance.log"
PIDF="$EXP/scripts/state/sync_variance.pid"

active(){
  for env in "$EXP/scripts/state"/variance_*.env; do
    [ -f "$env" ] || continue
    local arm seed ss name done_marker
    arm=$(grep '^ARM=' "$env"|cut -d= -f2)
    seed=$(grep '^SEED=' "$env"|cut -d= -f2)
    ss="s${seed:(-2)}"; name="cl_${arm}_50k_${ss}"
    local loc="$MAIN/variance/${arm}_seed${seed}"
    local final="$loc/runs/${name}_FINAL.pth"
    local opt="$loc/runs/${name}_FINAL_optimizer.pth"
    local lossfile="$loc/runs/${name}_losses.csv"
    local rows=0
    [ -f "$lossfile" ] && rows=$(wc -l < "$lossfile" 2>/dev/null || echo 0)
    # done when FINAL + optimizer present AND losses >=49k rows
    if [ -f "$final" ] && [ -f "$opt" ] && [ "$rows" -gt 49000 ]; then
      echo "DONE $arm $seed"
    else
      echo "ACTIVE $arm $seed"
    fi
  done
}

tick(){
  echo "=== sync_loop_variance tick $(date '+%m-%d %H:%M:%S') ==="
  active|grep '^ACTIVE'|while read _ arm seed; do
    bash "$EXP/scripts/sync_variance.sh" "$arm" "$seed" || true
  done
  active|grep '^DONE'|while read _ arm seed; do
    echo "  ✓ $arm/$seed DONE — no sync needed"
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
          echo \"sync_loop_variance: no ACTIVE arms left, exiting\" >> $LOG
          break
        fi
        sleep 900   # 15 min
      done
      rm -f $PIDF
    " < /dev/null > /dev/null 2>&1 &
    echo $! > "$PIDF"
    echo "sync_loop_variance started pid=$!"
    ;;
  stop)
    if [ -f "$PIDF" ]; then
      kill "$(cat "$PIDF")" 2>/dev/null || true
      rm -f "$PIDF"
      echo "sync_loop_variance stopped"
    else
      echo "not running"
    fi
    ;;
  tick) tick ;;
  active) active ;;
  *) echo "usage: $0 {start|stop|tick|active}"; exit 2 ;;
esac
