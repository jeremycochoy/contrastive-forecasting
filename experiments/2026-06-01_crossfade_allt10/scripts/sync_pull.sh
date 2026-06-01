#!/bin/bash
# #325 — LAPTOP-SIDE sync loop. Pulls the crossfade run's artifacts from elisa to the
# main checkout every 15 min (offsite backup + local analysis). Atomic (.tmp + per-class
# size gate + 1-deep rotation). Runs ON THE LAPTOP; elisa's own --save-every is the
# resume net if the box reboots. Exits once the FINAL backbone + both q-head FINALs +
# the full-eval summaries have all landed.
#
# Usage: sync_pull.sh   (env HOST defaults to the `elisa` ssh alias)
set -uo pipefail
HOST="${HOST:-jupyter@elisa}"
REMOTE="/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-01_crossfade_allt10"
LOCAL="${LOCAL:-/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/sync_crossfade_allt10}"
NAME="bb_xshh_allt_forked10pct_crossfade10pct_qk_aon_6Lf_b1024"
mkdir -p "$LOCAL/runs" "$LOCAL/results"
LOG="$LOCAL/sync.log"
SSHO="-o StrictHostKeyChecking=no -o ConnectTimeout=30"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
# pull <remote_relpath> <local_relpath> <min_bytes>
pull(){ local r="$1" l="$2" minb="$3"
  scp -q $SSHO "$HOST:$REMOTE/$r" "$LOCAL/$l.tmp" 2>/dev/null || return 1
  local sz; sz=$(wc -c < "$LOCAL/$l.tmp" 2>/dev/null || echo 0)
  if [ "$sz" -ge "$minb" ]; then
    [ -f "$LOCAL/$l" ] && cp -f "$LOCAL/$l" "$LOCAL/$l.prev"
    mv "$LOCAL/$l.tmp" "$LOCAL/$l"; log "✓ $r ($sz)"
  else rm -f "$LOCAL/$l.tmp"; log "✗ $r ($sz<$minb) — kept prior"; fi; }
rls(){ ssh $SSHO "$HOST" "ls -t $REMOTE/$1 2>/dev/null | head -1 | xargs -r basename" 2>/dev/null; }

log "SYNC start $HOST:$REMOTE -> $LOCAL"
while true; do
  # backbone: FINAL (+optimizer) if present, else latest periodic (+optimizer) resume net
  if ssh $SSHO "$HOST" "test -f $REMOTE/runs/${NAME}_FINAL.pth" 2>/dev/null; then
    pull "runs/${NAME}_FINAL.pth" "runs/${NAME}_FINAL.pth" 40000000
  fi
  latest=$(rls "runs/${NAME}_*k.pth")
  [ -n "$latest" ] && { pull "runs/$latest" "runs/$latest" 40000000; pull "runs/${latest%.pth}_optimizer.pth" "runs/${latest%.pth}_optimizer.pth" 70000000; }
  pull "runs/${NAME}_losses.csv" "runs/${NAME}_losses.csv" 100
  pull "results/run_${NAME}.log" "results/run_${NAME}.log" 100
  # q-heads + eval summaries (small) when they appear
  for hl in 2 6; do
    qn="qhead_${hl}L_xshh_allt_forked10pct_crossfade10pct_qk_aon_b1024"
    pull "runs/${qn}_FINAL.pth" "runs/${qn}_FINAL.pth" 1000000
    for s in triage full; do
      pull "results/gift_eval_${s}_xshh_allt_forked10pct_crossfade10pct_qk_aon_b1024_${hl}L/summary.txt" \
           "results/gift_eval_${s}_${hl}L_summary.txt" 50
    done
  done
  pull "results/orchestrate.log" "results/orchestrate.log" 10
  # done? FINAL backbone + both q-heads + both full summaries present locally
  if [ -f "$LOCAL/runs/${NAME}_FINAL.pth" ] \
     && [ -f "$LOCAL/runs/qhead_2L_xshh_allt_forked10pct_crossfade10pct_qk_aon_b1024_FINAL.pth" ] \
     && [ -f "$LOCAL/runs/qhead_6L_xshh_allt_forked10pct_crossfade10pct_qk_aon_b1024_FINAL.pth" ] \
     && [ -f "$LOCAL/results/gift_eval_full_2L_summary.txt" ] \
     && [ -f "$LOCAL/results/gift_eval_full_6L_summary.txt" ]; then
    log "ALL artifacts synced — sync exiting"; exit 0
  fi
  sleep 900
done
