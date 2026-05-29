#!/bin/bash
# #322 — downstream driver. For each batch-1024 backbone FINAL, train a 2L and a 6L
# q-head and run GIFT-Eval triage-11 + full-97. Two parallel GPU lanes (one per card)
# so the 10 (arm×head) cells split across both GPUs. Q-heads are light (~10.5 GB) so
# they fit even beside a foreign tenant. Idempotent (per-cell script skips finished
# q-heads/evals). Waits for a backbone FINAL before its cells.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RES"
DLOG="$RES/downstream.log"
GATE_MIB="${GATE_MIB:-12000}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [dl] $*" | tee -a "$DLOG"; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$(($1+1))p"; }

# arm tag -> backbone FINAL basename
declare -A BBNAME=(
  [beta_forked2_b1024]=bb_beta_forked2_6Lf_b1024_FINAL.pth
  [beta_forked10pct_b1024]=bb_beta_forked10pct_6Lf_b1024_FINAL.pth
  [xshh_allt_forked_b1024]=bb_xshh_allt_forked_6Lf_b1024_FINAL.pth
  [xshh_allt_forked10pct_b1024]=bb_xshh_allt_forked10pct_6Lf_b1024_FINAL.pth
  [xshh_allt_forked2_b1024]=bb_xshh_allt_forked2_6Lf_b1024_FINAL.pth
)
# Lane assignment (tag:head). Lane 0 -> GPU0, lane 1 -> GPU1.
LANE0=( "beta_forked2_b1024:2" "beta_forked2_b1024:6" "beta_forked10pct_b1024:2" "beta_forked10pct_b1024:6" "xshh_allt_forked_b1024:2" )
LANE1=( "xshh_allt_forked_b1024:6" "xshh_allt_forked10pct_b1024:2" "xshh_allt_forked10pct_b1024:6" "xshh_allt_forked2_b1024:2" "xshh_allt_forked2_b1024:6" )

run_lane(){
  local gpu="$1"; shift; local cells=("$@")
  for cell in "${cells[@]}"; do
    local tag="${cell%%:*}" hl="${cell##*:}"
    local bb="$RUNS/${BBNAME[$tag]}"
    # wait for the backbone FINAL (produced by the orchestrator)
    while [ ! -f "$bb" ]; do log "lane$gpu waiting for $(basename "$bb")"; sleep 120; done
    # wait for a GPU slot
    until [ "$(free_mib "$gpu")" -ge "$GATE_MIB" ]; do sleep 60; done
    log "lane$gpu START $tag ${hl}L"
    bash "$HERE/downstream_b1024.sh" "$bb" "$hl" "$tag" "$gpu" "triage full" \
      >>"$RES/lane${gpu}.log" 2>&1
    log "lane$gpu DONE $tag ${hl}L rc=$?"
  done
  log "lane$gpu complete"
}

run_lane 0 "${LANE0[@]}" &
P0=$!
run_lane 1 "${LANE1[@]}" &
P1=$!
wait $P0; wait $P1
log "DOWNSTREAM DRIVER DONE"
