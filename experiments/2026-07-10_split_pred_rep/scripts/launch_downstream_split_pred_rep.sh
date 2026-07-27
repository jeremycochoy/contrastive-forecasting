#!/bin/bash
# #374 — drive the split_pred_rep downstream on both head_layers cells:
# 2L on GPU_2L + 6L on GPU_6L in parallel. Same shape as the #366
# launch_downstream.sh but for a single arm.
#
#   launch_downstream_split_pred_rep.sh
#   GPU_2L=0 GPU_6L=1 launch_downstream_split_pred_rep.sh   override GPU pinning
set -uo pipefail
GPU_2L="${GPU_2L:-0}"
GPU_6L="${GPU_6L:-1}"
: "${WT:?WT must be set}"
: "${OUT:?OUT must be set}"
export WT OUT
[ -d "$WT" ] || { echo "[dl-launch split] ABORT: WT does not exist: $WT" >&2; exit 2; }
SCRIPT="$WT/experiments/2026-07-10_split_pred_rep/scripts/downstream_split_pred_rep.sh"
[ -f "$SCRIPT" ] || { echo "[dl-launch split] ABORT: downstream script not found: $SCRIPT" >&2; exit 2; }
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [dl-launch split] $*"; }

log "starting 2L on GPU $GPU_2L + 6L on GPU $GPU_6L (parallel)"
bash "$SCRIPT" 2 "$GPU_2L" >>"$RES/dl_2L.log" 2>&1 &
pid2=$!
bash "$SCRIPT" 6 "$GPU_6L" >>"$RES/dl_6L.log" 2>&1 &
pid6=$!
log "PIDs: 2L=$pid2  6L=$pid6"
wait $pid2; rc2=$?
wait $pid6; rc6=$?
log "2L finished rc=$rc2  6L finished rc=$rc6"
exit $(( rc2 != 0 || rc6 != 0 ))
