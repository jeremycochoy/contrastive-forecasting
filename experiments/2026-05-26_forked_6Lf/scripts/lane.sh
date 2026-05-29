#!/bin/bash
# #320 — one GPU lane: run a sequence of forked-6Lf arm pipelines on a fixed GPU.
# Each arm spec is "arm:mix:tag". Sequential within a lane; launch two lanes
# (one per GPU) for parallelism. Idempotent — safe to re-launch after a crash.
#
#   lane.sh <gpu> [chunk] <arm:mix:tag> [<arm:mix:tag> ...]
set -uo pipefail
GPU="${1:?gpu}"; CHUNK="${2:-8}"; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [LANE g$GPU] $*"; }
log "lane start: $* (chunk=$CHUNK)"
for spec in "$@"; do
  IFS=':' read -r arm mix tag <<<"$spec"
  log ">>> arm=$arm mix=$mix tag=$tag"
  bash "$HERE/pipeline_arm.sh" "$arm" "$mix" "$tag" "$GPU" "$CHUNK" || log "arm $spec returned nonzero (continuing)"
done
log "=== LANE g$GPU COMPLETE ==="
