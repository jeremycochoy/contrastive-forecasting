#!/bin/bash
# #348 — supervise one no-encoder backbone arm (base|cpc): auto-resume from the
# latest periodic checkpoint on crash, up to 8 attempts.
#   supervise_noenc.sh <arm: base|cpc> <gpu>
set -uo pipefail
ARM="${1:?arm (base|cpc)}"; GPU="${2:?gpu}"
SD="$(cd "$(dirname "$0")" && pwd)"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [supervise-$ARM g$GPU] $*"; }
for i in $(seq 1 8); do
  log "launch attempt $i"
  bash "$SD/train_backbone_noenc.sh" "$ARM" "$GPU" && { log "DONE"; exit 0; }
  rc=$?
  log "exited rc=$rc; retry in 30s"
  sleep 30
done
log "GAVE UP after 8 attempts"; exit 1
