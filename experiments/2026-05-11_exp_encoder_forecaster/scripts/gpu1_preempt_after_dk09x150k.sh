#!/bin/bash
# dk0.85 moved to the free GPU0 (dk0.92 finished early). So GPU1's job is
# now just: once dk0.9-fp32@150k GM-eval lands (the bet reference), stop the
# gpu1_reference pipeline's remaining work — dk0.9-fp32 x100k eval + v17
# (both dropped per user) — so GPU1 doesn't waste ~18h on unwanted runs.
set -uo pipefail
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
RES="$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results"
LOG="$RES/fine_dk_sweep.log"
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null|grep -oE '[0-9]+\.[0-9]+'|head -1; }
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
SIG="$RES/gift_eval_full_dk09fp32x150k/summary.txt"
log "=== gpu1 preempt-watcher START — waiting for dk0.9-fp32@150k GM-eval ==="
while :; do
  [ -f "$SIG" ] && { log "dk0.9-fp32@150k GM=$(gm "$SIG") — preempting GPU1 (drop x100k + v17)"; break; }
  if ! pgrep -f gpu1_reference_pipeline.sh >/dev/null 2>&1; then
    log "gpu1_reference_pipeline ended; dk09fp32x150k=$( [ -f "$SIG" ] && gm "$SIG" || echo MISSING) — nothing to preempt"; exit 0
  fi
  sleep 120
done
pkill -9 -f gpu1_reference_pipeline.sh 2>/dev/null || true
pkill -9 -f 'train\.py .*enc_fcst_v17_dk095_150k' 2>/dev/null || true
pkill -9 -f 'train_forecasting_head.*dk09fp32x100k' 2>/dev/null || true
pkill -9 -f 'train_forecasting_head.*v17x' 2>/dev/null || true
pkill -9 -f 'eval_gift_eval_official.*dk09fp32x100k' 2>/dev/null || true
pkill -9 -f 'eval_gift_eval_official.*v17x' 2>/dev/null || true
log "=== gpu1 preempt-watcher DONE — GPU1 freed; v17 & x100k dropped per user ==="