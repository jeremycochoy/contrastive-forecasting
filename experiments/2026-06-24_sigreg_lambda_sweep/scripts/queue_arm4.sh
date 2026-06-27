#!/bin/bash
# #363 — arm 4 queue (emb10_enc10, λ_e=1.0, λ_h=1.0)
# 1) launches backbone on GPU 1 (arm 6 is on GPU 0)
# 2) waits for FINAL.pth
# 3) waits for arm-6 queue (PID arg) to exit, because launch_downstream.sh
#    uses BOTH GPUs (2L on GPU 0, 6L on GPU 1) — must not overlap with arm-6 DL.
# 4) launches downstream (2L on GPU 0 + 6L on GPU 1 in parallel)
# All sub-steps no-op if their FINAL.pth / summary.txt already exists.
set -uo pipefail
export WT=/tmp/cf-revert-363
export OUT="$WT/experiments/2026-06-24_sigreg_lambda_sweep"
SUFFIX=emb10_enc10
LAMBDA_E=1.0
LAMBDA_H=1.0
GPU_BB=1
ARM6_QUEUE_PID="${ARM6_QUEUE_PID:-1774205}"
RES="$OUT/results"
RUNS="$OUT/runs"
TAG="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_${SUFFIX}"
BB="$RUNS/bb_${TAG}_FINAL.pth"
QLOG="$RES/queue_arm4.log"
mkdir -p "$RES" "$RUNS"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [queue-arm4] $*" | tee -a "$QLOG" >&2; }

log "queue start  WT=$WT  OUT=$OUT  SUFFIX=$SUFFIX  GPU_BB=$GPU_BB  arm6_queue_pid=$ARM6_QUEUE_PID"

if [ -f "$BB" ]; then
  log "BB FINAL exists — skipping backbone phase"
else
  log "BB phase: bash $OUT/scripts/train_backbone_sigreg.sh $GPU_BB $LAMBDA_E $LAMBDA_H $SUFFIX"
  bash "$OUT/scripts/train_backbone_sigreg.sh" "$GPU_BB" "$LAMBDA_E" "$LAMBDA_H" "$SUFFIX" \
    >>"$RES/sweep_bb_${SUFFIX}.log" 2>&1
  rc=$?
  log "BB phase finished rc=$rc"
  if [ "$rc" -ne 0 ] || [ ! -f "$BB" ]; then
    log "BB phase failed (rc=$rc) — aborting before downstream"
    exit 1
  fi
fi

# Wait for arm-6 queue (BB + DL) to fully finish before launching arm-4 DL,
# because launch_downstream.sh uses BOTH GPUs and arm-6 DL also uses both.
if [ -n "$ARM6_QUEUE_PID" ] && [ "$ARM6_QUEUE_PID" != "0" ]; then
  log "waiting for arm-6 queue (PID $ARM6_QUEUE_PID) to exit before launching DL"
  while kill -0 "$ARM6_QUEUE_PID" 2>/dev/null; do
    sleep 60
  done
  log "arm-6 queue PID $ARM6_QUEUE_PID has exited"
fi

# Belt-and-braces: also wait for any leftover arm-6 train.py / eval processes
# (covers the case where arm-6 queue exited but a child eval is still wrapping
# up, or where ARM6_QUEUE_PID was unset/0).
while true; do
  busy=$(pgrep -af "(train\.py|eval_gift_eval_official).*emb10000_enc10" 2>/dev/null | wc -l)
  [ "$busy" = "0" ] && break
  log "  $busy arm-6 train/eval process(es) still running — sleeping 60s"
  sleep 60
done
log "no arm-6 processes — proceeding to DL"

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
