#!/bin/bash
# #322 — rerun the RED LINE (b1024, LR 1e-3, β·0.8% — the FIRST config that diverged)
# WITH QK-norm, to test whether QK-norm stops the activation-amplitude divergence.
# Waits for GPU1 to free (does NOT stop the running τ=0.20 probe) then launches.
# Amplitude logging is on (--log-attn-amplitude in train_backbone_b1024_1gpu.sh), now
# reporting the POST-norm QK logits. Distinct run-name (..._forked2_qknorm_...).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RES=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024/results
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [qknorm-redline] $*" | tee -a "$RES/qknorm_redline.log"; }
free1(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '2p'; }
log "waiting for GPU1 free >=21500 MiB (NOT disturbing the running τ probe)"
w=0; until [ "$(free1)" -ge 21500 ]; do
  [ $((w % 600)) -eq 0 ] && log "WAIT gpu1 free=$(free1) MiB"
  sleep 120; w=$((w + 120))
done
log "GPU1 free ($(free1) MiB) — launching RED LINE + QK-norm: b1024 lr 1e-3 β·0.8%, 12.5k steps, amplitudes on"
LR=1e-3 QK_NORM=1 bash "$HERE/train_backbone_b1024_1gpu.sh" beta 0.0078125 forked2_qknorm 12500 2500 2 1
log "RED LINE + QK-norm rc=$?"
