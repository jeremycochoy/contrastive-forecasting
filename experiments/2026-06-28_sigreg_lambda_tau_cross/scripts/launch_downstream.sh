#!/bin/bash
# #366 — drive downstream q-head + GIFT-Eval for one cross arm: 2L and 6L
# in parallel on the two local GPUs. Same shape as #363 launch_downstream.sh.
#
#   launch_downstream.sh <suffix>           e.g. launch_downstream.sh lA_...
#   GPU_2L=0 GPU_6L=1 launch_downstream.sh <suffix>   override GPU pinning
#
# Default GPU pinning (2L→0, 6L→1) is set for the elisa 2× RTX 4090 box; under
# the default sequential launch_arms.sh orchestration no sibling BB shares
# either GPU, so the two heads run cleanly in parallel.
set -uo pipefail
SUFFIX="${1:?suffix}"
GPU_2L="${GPU_2L:-0}"
GPU_6L="${GPU_6L:-1}"
: "${WT:?WT (worktree root containing experiments/<this-dir>/scripts/...) must be set; e.g. WT=/home/jupyter/workspaces/contrastive-forecasting}"
: "${OUT:?OUT (per-experiment output dir for runs/ and results/) must be set; e.g. OUT=\$WT/experiments/2026-06-28_sigreg_lambda_tau_cross}"
export WT OUT
[ -d "$WT" ] || { echo "[dl-launch ${SUFFIX}] ABORT: WT does not exist: $WT" >&2; exit 2; }
SCRIPT="$WT/experiments/2026-06-28_sigreg_lambda_tau_cross/scripts/downstream_sigreg.sh"
[ -f "$SCRIPT" ] || { echo "[dl-launch ${SUFFIX}] ABORT: downstream script not found: $SCRIPT" >&2; exit 2; }
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [dl-launch ${SUFFIX}] $*"; }

log "starting 2L on GPU $GPU_2L + 6L on GPU $GPU_6L (parallel) for arm=${SUFFIX}"
bash "$SCRIPT" 2 "$GPU_2L" "$SUFFIX" >>"$RES/dl_2L_${SUFFIX}.log" 2>&1 &
pid2=$!
bash "$SCRIPT" 6 "$GPU_6L" "$SUFFIX" >>"$RES/dl_6L_${SUFFIX}.log" 2>&1 &
pid6=$!
log "PIDs: 2L=$pid2  6L=$pid6"
wait $pid2; rc2=$?
wait $pid6; rc6=$?
log "2L finished rc=$rc2  6L finished rc=$rc6"
exit $(( rc2 != 0 || rc6 != 0 ))
