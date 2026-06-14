#!/bin/bash
# #344 follow-up — autonomous downstream for the enc6 cpc+align/no-main arm.
# Waits for the backbone FINAL, trains the four q-heads in parallel across both
# GPUs (2L on GPU0, 6L on GPU1, heads-only), then evals the four cells 4-wide
# (2 per GPU). All idempotent. Single arm, so both GPUs are used for downstream.
set -uo pipefail
SD="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-13_cpc_infonce_aux}"
TAG="allt08_xftrip_nobn_enc6_cpcalign_qk_aon_b1024_cpc"
BB="$OUT/runs/bb_${TAG}_FINAL.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [watch-cpcalign] $*"; }
log "waiting for backbone FINAL: $BB"
until [ -f "$BB" ]; do sleep 300; done
log "backbone FINAL present — training heads (2L g0, 6L g1)"
TAG_OVERRIDE="$TAG" DO_EVAL=0 bash "$SD/downstream_cpc.sh" enc6 2 0 &
p2=$!
TAG_OVERRIDE="$TAG" DO_EVAL=0 bash "$SD/downstream_cpc.sh" enc6 6 1 &
p6=$!
wait "$p2"; wait "$p6"
log "heads done — evaluating 4-wide (best/last × 2L/6L)"
pids=()
TAG_OVERRIDE="$TAG" bash "$SD/eval_cell.sh" enc6 2 best 0 & pids+=($!)
TAG_OVERRIDE="$TAG" bash "$SD/eval_cell.sh" enc6 2 last 0 & pids+=($!)
TAG_OVERRIDE="$TAG" bash "$SD/eval_cell.sh" enc6 6 best 1 & pids+=($!)
TAG_OVERRIDE="$TAG" bash "$SD/eval_cell.sh" enc6 6 last 1 & pids+=($!)
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
log "cpcalign downstream complete (rc=$rc)"
touch "$OUT/results/cpcalign.done"
