#!/bin/bash
# #363 — arm 6 queue (emb10000_enc10, λ_e=1000.0, λ_h=1.0)
# 1) launches backbone on GPU 0
# 2) waits for FINAL.pth (skip-safe — idempotent)
# 3) launches downstream (2L on GPU 0 + 6L on GPU 1 in parallel)
# All sub-steps no-op if their FINAL.pth / summary.txt already exists.
set -uo pipefail
export WT=/tmp/cf-revert-363
export OUT="$WT/experiments/2026-06-24_sigreg_lambda_sweep"
SUFFIX=emb10000_enc10
GPU_BB=0
RES="$OUT/results"
RUNS="$OUT/runs"
TAG="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_${SUFFIX}"
BB="$RUNS/bb_${TAG}_FINAL.pth"
QLOG="$RES/queue_arm6.log"
mkdir -p "$RES" "$RUNS"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [queue-arm6] $*" | tee -a "$QLOG" >&2; }

log "queue start  WT=$WT  OUT=$OUT  SUFFIX=$SUFFIX  GPU_BB=$GPU_BB"

if [ -f "$BB" ]; then
  log "BB FINAL exists — skipping backbone phase"
else
  log "BB phase: bash $OUT/scripts/train_backbone_sigreg.sh $GPU_BB 1000.0 1.0 $SUFFIX"
  bash "$OUT/scripts/train_backbone_sigreg.sh" "$GPU_BB" 1000.0 1.0 "$SUFFIX" \
    >>"$RES/sweep_bb_${SUFFIX}.log" 2>&1
  rc=$?
  log "BB phase finished rc=$rc"
  if [ "$rc" -ne 0 ] || [ ! -f "$BB" ]; then
    log "BB phase failed (rc=$rc) — aborting before downstream"
    exit 1
  fi
fi

log "DL phase: bash $OUT/scripts/launch_downstream.sh $SUFFIX"
bash "$OUT/scripts/launch_downstream.sh" "$SUFFIX" >>"$RES/sweep_dl_${SUFFIX}.log" 2>&1
rc=$?
log "DL phase finished rc=$rc"

# Confirm all four downstream cells (2L/6L × best/last) produced summaries
need=()
for hl in 2L 6L; do
  for kind in '' '_last'; do
    f="$RES/gift_eval_full_${TAG}${kind}_${hl}/summary.txt"
    [ -f "$f" ] || need+=("$f")
  done
done
if [ "${#need[@]}" -gt 0 ]; then
  log "DL phase incomplete — missing: ${need[*]}"
  exit 2
fi
log "DL phase complete — all 4 summary.txt files present"
log "queue done"
