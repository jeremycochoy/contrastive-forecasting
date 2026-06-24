#!/bin/bash
# #363 — drive downstream q-head + GIFT-Eval for one λ-sweep arm: 2L on GPU 0
# and 6L on GPU 1 in parallel. Same shape as #355 / #359.
#
#   launch_downstream.sh <suffix>      e.g. launch_downstream.sh emb100_enc01
set -uo pipefail
SUFFIX="${1:?suffix}"
WT="${WT:-/tmp/contrastive-forecasting-363}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-24_sigreg_lambda_sweep}"
export WT OUT
SCRIPT="$WT/experiments/2026-06-24_sigreg_lambda_sweep/scripts/downstream_sigreg.sh"
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [dl-launch ${SUFFIX}] $*"; }

log "starting 2L on GPU 0 + 6L on GPU 1 (parallel) for arm=${SUFFIX}"
bash "$SCRIPT" 2 0 "$SUFFIX" >>"$RES/dl_2L_${SUFFIX}.log" 2>&1 &
pid2=$!
bash "$SCRIPT" 6 1 "$SUFFIX" >>"$RES/dl_6L_${SUFFIX}.log" 2>&1 &
pid6=$!
log "PIDs: 2L=$pid2  6L=$pid6"
wait $pid2; rc2=$?
wait $pid6; rc6=$?
log "2L finished rc=$rc2  6L finished rc=$rc6"
exit $(( rc2 != 0 || rc6 != 0 ))
