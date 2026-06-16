#!/bin/bash
set -uo pipefail
ARM="$1"; GPU="$2"
SD="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-13_cpc_infonce_aux}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [supervise-$ARM g$GPU] $*"; }
for i in $(seq 1 8); do
  log "launch attempt $i"
  bash "$SD/train_backbone_cpc.sh" "$ARM" "$GPU" && { log "DONE"; exit 0; }
  rc=$?
  log "exited rc=$rc; retry in 30s"
  sleep 30
done
log "GAVE UP after 8 attempts"; exit 1
