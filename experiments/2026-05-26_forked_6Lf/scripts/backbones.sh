#!/bin/bash
# #320 — backbone-only lane. Trains the forked-6Lf backbones sequentially on ONE
# GPU. The 6L forecaster needs ~18 GB (β) / more (all-time), so only one fits per
# 24 GB card — backbones get the free GPU; downstream (lighter) runs elsewhere.
# Idempotent. Arm spec = "arm:mix:tag".
#
#   backbones.sh <gpu> <chunk> <arm:mix:tag> [<arm:mix:tag> ...]
set -uo pipefail
GPU="${1:?gpu}"; CHUNK="${2:-8}"; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [BB-LANE g$GPU] $*"; }
log "start: $* (chunk=$CHUNK)"
for spec in "$@"; do
  IFS=':' read -r arm mix tag <<<"$spec"
  log ">>> backbone arm=$arm mix=$mix tag=$tag"
  bash "$HERE/train_backbone_forked_6Lf.sh" "$arm" "$mix" "$tag" "$GPU" "$CHUNK" \
    || log "backbone $spec returned nonzero (continuing)"
done
log "=== ALL BACKBONES (g$GPU) DONE ==="
