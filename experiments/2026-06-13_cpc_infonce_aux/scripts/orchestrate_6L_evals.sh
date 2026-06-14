#!/bin/bash
# #344 — run the four 6L GIFT-Eval cells concurrently (enc3 best+last on GPU0,
# enc6 best+last on GPU1; the eval is CPU-bound so two fit per GPU). Each cell
# waits for its 6L head FINAL (produced by the DO_EVAL=0 pretrain), then runs
# the byte-identical do_eval via eval_cell.sh. Idempotent. Used after chain_cpc
# is stopped so nothing else evals the 6L cells (no race).
set -uo pipefail
SD="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-13_cpc_infonce_aux}"
RUNS="$OUT/runs"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [orch-6L] $*"; }
CELLS=( "enc3:best:0" "enc3:last:0" "enc6:best:1" "enc6:last:1" )
pids=()
for c in "${CELLS[@]}"; do
  IFS=: read -r arm ck gpu <<<"$c"
  TAG="allt08_xftrip_nobn_${arm}_sgpos_qk_aon_b1024_cpc"
  if [ "$ck" = best ]; then hf="$RUNS/qhead_6L_${TAG}_FINAL.pth"; else hf="$RUNS/qhead_6L_${TAG}_last_FINAL.pth"; fi
  ( log "wait head $(basename "$hf")"
    until [ -f "$hf" ]; do sleep 60; done
    log "head ready -> eval $arm 6L $ck (gpu $gpu)"
    bash "$SD/eval_cell.sh" "$arm" 6 "$ck" "$gpu" ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
log "all 6L evals returned (rc=$rc)"
touch "$OUT/results/orch_6L.done"
