#!/bin/bash
# #344 — one watcher per arm: wait for that arm's backbone FINAL (best-loss
# copy) to land, then run its full downstream chain (2L/6L heads × best/last +
# full-97 evals) on the same GPU the backbone just freed. Polls every 5 min.
# Fully autonomous: backbones (supervised) -> downstream -> chain_${arm}.done.
#   watch_and_downstream.sh <arm: enc3|enc6> <gpu>
set -uo pipefail
ARM="${1:?arm}"; GPU="${2:?gpu}"
SD="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-13_cpc_infonce_aux}"
TAG="allt08_xftrip_nobn_${ARM}_sgpos_qk_aon_b1024_cpc"
BB="$OUT/runs/bb_${TAG}_FINAL.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [watch-$ARM g$GPU] $*"; }
log "waiting for backbone FINAL: $BB"
until [ -f "$BB" ]; do sleep 300; done
log "backbone FINAL present — starting downstream chain"
DO_EVAL=1 bash "$SD/chain_cpc.sh" "$ARM" "$GPU"
log "downstream chain returned"
