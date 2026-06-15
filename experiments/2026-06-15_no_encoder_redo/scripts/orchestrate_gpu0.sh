#!/bin/bash
# #348 — full autonomous pipeline on a single GPU (default 0): bank both
# backbones first (the irreplaceable long compute), then both downstream chains
# (2L/6L × best/last q-heads + full-97 GIFT-Eval). Every step is idempotent
# (skips if its FINAL/summary/.done exists), so this is safe to re-launch after
# any crash. Backbones are each supervised (auto-resume, 8 attempts).
#   orchestrate_gpu0.sh [gpu]
set -uo pipefail
GPU="${1:-0}"
SD="$(cd "$(dirname "$0")" && pwd)"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [orch g$GPU] $*"; }
log "=== START orchestrator on GPU $GPU ==="
log "--- backbone: base ---";  bash "$SD/supervise_noenc.sh" base "$GPU"
log "--- backbone: cpc  ---";  bash "$SD/supervise_noenc.sh" cpc  "$GPU"
log "--- downstream: base ---"; DO_EVAL=1 bash "$SD/chain_noenc.sh" base "$GPU"
log "--- downstream: cpc  ---"; DO_EVAL=1 bash "$SD/chain_noenc.sh" cpc  "$GPU"
log "=== orchestrator DONE ==="
