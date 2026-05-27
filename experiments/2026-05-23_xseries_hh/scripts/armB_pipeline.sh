#!/bin/bash
# #318 ablation — train the all-time cross-series arm (arm B) then eval it
# (2L + 6L heads, triage + full-97). Self-contained so it can run on whichever
# GPU is free (GPU 1 once the concurrent CPC run finishes, in parallel with
# arm A's eval on GPU 0; or GPU 0 after arm A's eval). Idempotent.
#
#   armB_pipeline.sh <gpu> [chunk]
set -uo pipefail
GPU="${1:?gpu}"; CHUNK="${2:-8}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-23_xseries_hh
BB="$OUT/runs/bb_xshh_allt_50k_FINAL.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [armB g$GPU] $*"; }

log "=== train arm B (all-time, chunk=$CHUNK) on GPU$GPU ==="
bash "$HERE/train_backbone_allt.sh" "$GPU" 50000 "$CHUNK"
[ -f "$BB" ] || { log "ABORT: arm B backbone not produced"; exit 1; }

log "=== eval arm B (2L + 6L, triage + full) on GPU$GPU ==="
bash "$HERE/downstream.sh" "$BB" 2 xshh_allt_50k "$GPU" "triage full"
bash "$HERE/downstream.sh" "$BB" 6 xshh_allt_50k "$GPU" "triage full"
log "=== ARMB_PIPELINE_DONE ==="
