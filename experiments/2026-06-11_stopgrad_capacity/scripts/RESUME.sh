#!/bin/bash
# #341 — resume the pipeline after a machine restart. Fully idempotent: every
# stage skips work already on disk (done backbones, done heads, merged eval
# cells). Relaunches, detached:
#   - both per-arm chains (finish any unfinished heads)   one GPU each
#   - the training watchdog
#   - phase 2 (fresh last-checkpoint heads + all GIFT-Eval cells)
# Checkpoints/results live in the persistent home dir (~/workspaces/...), so only
# the in-flight processes were lost; nothing on disk is redone.
#   bash RESUME.sh
set -uo pipefail
SD="$(cd "$(dirname "$0")" && pwd)"
RES=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity/results
mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [resume] $*"; }
# clear stale chain-done markers ONLY if their heads aren't actually all present —
# leave them; the chains/phase2 are idempotent and re-check FINAL files themselves.
log "relaunching chains (finish unfinished heads), watchdog, phase2"
setsid nohup bash "$SD/chain_sgcap.sh"     nobn_enc6 0 > "$RES/chain_nobn_enc6.out" 2>&1 </dev/null &
setsid nohup bash "$SD/chain_sgcap.sh"     bn_enc6   1 > "$RES/chain_bn_enc6.out"   2>&1 </dev/null &
setsid nohup bash "$SD/watchdog_sgcap.sh"               > "$RES/watchdog_driver.out" 2>&1 </dev/null &
# per-arm phase 2 (each waits its own chain; bn on GPU1, nobn on GPU0 — matches
# the chains' GPU assignment so an arm's fresh-last work lands on its freed card)
setsid nohup bash "$SD/run_phase2_arm.sh" bn_enc6   1 > "$RES/run_phase2_bn.out"   2>&1 </dev/null &
setsid nohup bash "$SD/run_phase2_arm.sh" nobn_enc6 0 > "$RES/run_phase2_nobn.out" 2>&1 </dev/null &
log "launched. monitor: tail $RES/run_phase2_*.out and $RES/watchdog.log"
echo "resume OK"
