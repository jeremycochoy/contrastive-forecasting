#!/bin/bash
# #341 — training watchdog: every 10 min append one status line per arm to
# results/watchdog.log (losses.csv progress, process aliveness, GPU state) and
# scan the tail of each run log for NaN. Writes results/ALERT_<arm> when a
# chain process has died without its .done marker. Exits when both chains are
# done. Run with nohup.
set -uo pipefail
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity}"
RES="$OUT/results"; RUNS="$OUT/runs"; mkdir -p "$RES"
WLOG="$RES/watchdog.log"
ARMS="nobn_enc6 bn_enc6"
while :; do
  alldone=1
  for ARM in $ARMS; do
    NAME="bb_allt08_xftrip_${ARM}_sgpos_qk_aon_b1024"
    csv="$RUNS/${NAME}_losses.csv"
    last=$(tail -1 "$csv" 2>/dev/null | cut -d, -f1-2)
    alive=$(pgrep -fc "chain_sgcap.sh $ARM" || true)
    pyalive=$(pgrep -fc "run-name.*${ARM}_sgpos|${ARM}.*sgpos" || true)
    done_f=""; [ -f "$RES/chain_${ARM}.done" ] && done_f="DONE"
    [ -z "$done_f" ] && alldone=0
    if tail -200 "$RES/run_${NAME}.log" 2>/dev/null | grep -qiE '(^|[^a-z])nan([^a-z]|$)'; then
      echo "[$(date '+%m-%d %H:%M:%S')] $ARM NAN DETECTED" >>"$WLOG"; touch "$RES/ALERT_${ARM}_nan"
    fi
    if [ -z "$done_f" ] && [ "$alive" = "0" ]; then
      echo "[$(date '+%m-%d %H:%M:%S')] $ARM chain dead without done marker" >>"$WLOG"
      touch "$RES/ALERT_${ARM}_dead"
    fi
    echo "[$(date '+%m-%d %H:%M:%S')] $ARM ${done_f:-running} chain_procs=$alive py=$pyalive last_csv=[${last:-none}]" >>"$WLOG"
  done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
    | sed "s/^/[$(date '+%m-%d %H:%M:%S')] gpu /" >>"$WLOG"
  [ "$alldone" = 1 ] && { echo "[$(date '+%m-%d %H:%M:%S')] all chains done — watchdog exit" >>"$WLOG"; exit 0; }
  sleep 600
done
