#!/bin/bash
# #348 — one arm's full pipeline pinned to one GPU: supervised backbone
# (auto-resume) then its downstream chain (2L/6L × best/last q-heads +
# full-97 GIFT-Eval). Idempotent — safe to re-launch after a crash. Used to
# run the two arms in parallel, one per GPU.
#   orchestrate_arm.sh <arm: base|cpc> <gpu>
set -uo pipefail
ARM="${1:?arm (base|cpc)}"; GPU="${2:?gpu}"
SD="$(cd "$(dirname "$0")" && pwd)"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [orch-$ARM g$GPU] $*"; }
log "=== START arm=$ARM on GPU $GPU ==="
log "--- backbone ---";   bash "$SD/supervise_noenc.sh" "$ARM" "$GPU"
log "--- downstream ---"; DO_EVAL=1 bash "$SD/chain_noenc.sh" "$ARM" "$GPU"
log "=== DONE arm=$ARM ==="
