#!/bin/bash
# #373 review item 3, fourth corner — one status line every INTERVAL seconds.
#
# The worker's own log writes milestone lines only, so a silent death between
# milestones looks exactly like a healthy run. This prints the step count, the
# phase and the liveness of the worker on a fixed clock, and it exits when the
# chain finishes or when the worker is gone.
#
# It reads state and starts nothing.
#
# Usage: gap4_heartbeat.sh [interval_seconds]
set -uo pipefail

INTERVAL="${1:-1800}"
RUN=bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3_aw025
CKPT="/home/jupyter/checkpoints_backup/cf-373/$RUN"
WT=/home/jupyter/wt-cf-373-train/reports/2026-08-08_rollout_depth
GIT=/tmp/contrastive-forecasting-373/reports/2026-08-08_rollout_depth
TARGET=40000

for slot in student teacher; do :; done

while true; do
  now=$(date '+%H:%M')
  step=$(tail -1 "$CKPT/${RUN}_losses.csv" 2>/dev/null | cut -d, -f1)
  [[ "$step" =~ ^[0-9]+$ ]] || step=0
  alive=$(pgrep -f "gap_worker.sh" >/dev/null && echo yes || echo NO)
  train=$(pgrep -f "cf373k3_aw025" >/dev/null && echo yes || echo no)

  # Which of the four deliverables are on disk.
  done_n=0
  for h in student teacher; do
    for root in "$WT" "$GIT"; do
      [ -s "$root/results/score_G_B1_k3_aw025_bb40k_${h}.txt" ] && { done_n=$((done_n+1)); break; }
    done
  done

  if [ "$done_n" -ge 2 ]; then
    echo "[$now] gap4 COMPLETE: both scores on disk"
    exit 0
  fi
  if [ "$alive" = "NO" ]; then
    echo "[$now] gap4 WORKER GONE: step=$step/$TARGET scores=$done_n/2 — chain stopped short"
    exit 1
  fi

  if [ "$step" -ge "$TARGET" ]; then
    phase="backbone done, heads/evals running"
  else
    phase="backbone $step/$TARGET ($((step * 100 / TARGET))%)"
  fi
  echo "[$now] gap4 $phase, worker=$alive train=$train scores=$done_n/2"
  sleep "$INTERVAL"
done
