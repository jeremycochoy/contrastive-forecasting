#!/bin/bash
# #327 — FINAL-checkpoint head lane (serial, adaptive GPU). Trains a 6L then a 2L q-head on the
# fully-trained step-6250 backbone (FINAL.pth) and scores each on GIFT-Eval triage+full.
# SHARED BOX: GPU 1 may be held by a foreign training job (it was, /tmp/cf-328); GPU 0 by foreign
# tenants + this card's plateau-lane eval. A final cell starts only when a GPU has room AND, for
# GPU 0, the plateau-6L FULL eval has actually finished (its summary.txt exists) — gating on the
# real artifact, not a marker file, so a final q-head never co-runs with / OOMs the plateau eval.
# Serial (one q-head per card at a time). Idempotent (downstream_b2048.sh skips finished cells).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-03_forked_allt08_b2048
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RES"
BB="$RUNS/bb_xshh_allt_forked2_qk_aon_6Lf_b2048_FINAL.pth"
PLATEAU_DONE="$RES/gift_eval_full_platpeak_6L/summary.txt"   # the real "plateau lane finished" signal
GATE_MIB=12000
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [final-lane] $*" | tee -a "$RES/downstream.log"; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$(($1+1))p"; }
[ -f "$BB" ] || { log "ABORT: FINAL backbone missing"; exit 1; }

pick_gpu(){ # GPU 1 if genuinely free; else GPU 0 once the plateau-6L full eval is DONE. echo idx|empty.
  [ "$(free_mib 1)" -ge "$GATE_MIB" ] && { echo 1; return; }
  [ -f "$PLATEAU_DONE" ] && [ "$(free_mib 0)" -ge "$GATE_MIB" ] && { echo 0; return; }
  echo ""
}

for hl in 6 2; do
  g=""; w=0
  until g=$(pick_gpu); [ -n "$g" ]; do
    [ $((w%600)) -eq 0 ] && log "WAIT final ${hl}L (gpu1_free=$(free_mib 1) gpu0_free=$(free_mib 0) plateau6L_done=$([ -f "$PLATEAU_DONE" ] && echo y || echo n))"
    sleep 60; w=$((w+60)); done
  log ">>> final ${hl}L start on gpu$g"
  bash "$HERE/downstream_b2048.sh" "$BB" "$hl" forked2_qk_aon "$g" "triage full" >>"$RES/cell_final_${hl}L.log" 2>&1
  log "<<< final ${hl}L rc=$?"
done
log "FINAL LANE DONE"; touch "$RES/.downstream_done"
