#!/bin/bash
# #320 — full pipeline for ONE forked-6Lf arm: train the 6L-forecaster backbone,
# then evaluate it with a 2L and a 6L q-head, each on triage-11 + full-97.
# Idempotent (skips finished cells). The downstream <tag> mirrors the backbone
# name so q-heads / eval dirs are unambiguous.
#
#   pipeline_arm.sh <arm> <mix> <tag> <gpu> [chunk]
set -uo pipefail
ARM="${1:?arm}"; MIX="${2:?mix}"; TAG="${3:?tag}"; GPU="${4:?gpu}"; CHUNK="${5:-8}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-26_forked_6Lf
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [arm $ARM·$TAG g$GPU] $*"; }
case "$ARM" in
  beta)    NAME="bb_beta_${TAG}_6Lf_50k";      DTAG="beta_${TAG}_6Lf_50k" ;;
  alltime) NAME="bb_xshh_allt_${TAG}_6Lf_50k"; DTAG="xshh_allt_${TAG}_6Lf_50k" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
BB="$OUT/runs/${NAME}_FINAL.pth"
log "=== TRAIN backbone ==="
bash "$HERE/train_backbone_forked_6Lf.sh" "$ARM" "$MIX" "$TAG" "$GPU" "$CHUNK"
[ -f "$BB" ] || { log "backbone missing — abort arm"; exit 1; }
log "=== DOWNSTREAM 2L ==="; bash "$HERE/downstream_6Lf.sh" "$BB" 2 "$DTAG" "$GPU" "triage full"
log "=== DOWNSTREAM 6L ==="; bash "$HERE/downstream_6Lf.sh" "$BB" 6 "$DTAG" "$GPU" "triage full"
log "=== ARM DONE ==="
