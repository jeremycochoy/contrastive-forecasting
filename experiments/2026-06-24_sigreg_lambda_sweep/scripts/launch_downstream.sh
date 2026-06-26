#!/bin/bash
# #363 — drive downstream q-head + GIFT-Eval for one λ-sweep arm: 2L on GPU 0
# and 6L on GPU 1 in parallel. Same shape as #355 / #359.
#
#   launch_downstream.sh <suffix>      e.g. launch_downstream.sh emb100_enc01
set -uo pipefail
SUFFIX="${1:?suffix}"
: "${WT:?WT (worktree root containing experiments/<this-dir>/scripts/...) must be set; e.g. WT=/home/jupyter/workspaces/contrastive-forecasting}"
: "${OUT:?OUT (per-experiment output dir for runs/ and results/) must be set; e.g. OUT=\$WT/experiments/2026-06-24_sigreg_lambda_sweep}"
export WT OUT
[ -d "$WT" ] || { echo "[dl-launch ${SUFFIX}] ABORT: WT does not exist: $WT" >&2; exit 2; }
SCRIPT="$WT/experiments/2026-06-24_sigreg_lambda_sweep/scripts/downstream_sigreg.sh"
[ -f "$SCRIPT" ] || { echo "[dl-launch ${SUFFIX}] ABORT: downstream script not found: $SCRIPT" >&2; exit 2; }
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [dl-launch ${SUFFIX}] $*"; }

# Avoid GPU contention if a sibling-arm BB is still running (only relevant when
# arm 2 BB was launched in parallel with arm 1 BB on the other GPU): block on
# any train.py whose --run-name is a sweep backbone, except ours.
while true; do
  others=$(pgrep -af "train.py.*--run-name *bb_.*_sigreg_qk_aon_b512_cpc_(emb100_enc01|emb100_enc10|emb100_enc100|emb10_enc10)" 2>/dev/null | grep -vE -- "--run-name *bb_.*_${SUFFIX}\b" | wc -l)
  [ "$others" = "0" ] && break
  log "waiting for $others sibling-arm BB process(es) to finish before starting dl"
  sleep 60
done

log "starting 2L on GPU 0 + 6L on GPU 1 (parallel) for arm=${SUFFIX}"
bash "$SCRIPT" 2 0 "$SUFFIX" >>"$RES/dl_2L_${SUFFIX}.log" 2>&1 &
pid2=$!
bash "$SCRIPT" 6 1 "$SUFFIX" >>"$RES/dl_6L_${SUFFIX}.log" 2>&1 &
pid6=$!
log "PIDs: 2L=$pid2  6L=$pid6"
wait $pid2; rc2=$?
wait $pid6; rc6=$?
log "2L finished rc=$rc2  6L finished rc=$rc6"
exit $(( rc2 != 0 || rc6 != 0 ))
