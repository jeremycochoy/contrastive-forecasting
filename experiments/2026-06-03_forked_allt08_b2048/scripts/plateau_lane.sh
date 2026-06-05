#!/bin/bash
# #327 — plateau-peak head lane (GPU 0). Trains a fresh 2L then 6L q-head on the
# step-~2500 plateau-PEAK backbone checkpoint (bb_..._platpeak.pth, the local max of the
# floor-subtracted loss before its final descent) and scores each on GIFT-Eval triage+full.
# Mirrors #322's plateau test: does the training tail past the plateau buy forecasting skill?
# Runs while the backbone still owns GPU 1, so GPU 0's idle headroom is used. Sequential
# (one q-head ~10.5 GB; two would crowd GPU 0's foreign tenants), gated on GPU 0 free.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-03_forked_allt08_b2048
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RES"
BB="$RUNS/bb_xshh_allt_forked2_qk_aon_6Lf_b2048_platpeak.pth"
GPU=0; GATE_MIB=12000
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [plateau-lane] $*" | tee -a "$RES/plateau_lane.log"; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((GPU+1))p"; }
[ -f "$BB" ] || { log "ABORT plateau checkpoint missing: $BB"; exit 1; }
for hl in 2 6; do
  w=0; until [ "$(free_mib)" -ge "$GATE_MIB" ]; do
    [ $((w%600)) -eq 0 ] && log "WAIT gpu$GPU free=$(free_mib) MiB (need $GATE_MIB)"; sleep 120; w=$((w+120)); done
  log ">>> platpeak ${hl}L start (gpu$GPU free=$(free_mib) MiB)"
  bash "$HERE/downstream_b2048.sh" "$BB" "$hl" platpeak "$GPU" "triage full" >>"$RES/cell_platpeak_${hl}L.log" 2>&1
  log "<<< platpeak ${hl}L rc=$?"
done
log "PLATEAU LANE DONE"; touch "$RES/.plateau_done"
