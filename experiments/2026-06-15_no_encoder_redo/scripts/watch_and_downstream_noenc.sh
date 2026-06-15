#!/bin/bash
# #348 — one watcher per arm: wait for that arm's backbone FINAL (best-loss
# copy) to land, then run its full downstream chain on the given GPU. Polls
# every 5 min. Fully autonomous: backbones (supervised) -> downstream ->
# chain_${arm}.done.
#   watch_and_downstream_noenc.sh <arm: base|cpc> <gpu>
set -uo pipefail
ARM="${1:?arm}"; GPU="${2:?gpu}"
SD="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-15_no_encoder_redo}"
TAG="allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_${ARM}"
BB="$OUT/runs/bb_${TAG}_FINAL.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [watch-$ARM g$GPU] $*"; }
log "waiting for backbone FINAL: $BB"
until [ -f "$BB" ]; do sleep 300; done
log "backbone FINAL present — starting downstream chain"
DO_EVAL=1 bash "$SD/chain_noenc.sh" "$ARM" "$GPU"
log "downstream chain returned"
