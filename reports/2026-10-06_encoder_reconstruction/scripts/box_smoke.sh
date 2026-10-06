#!/bin/bash
# #425 — the end-to-end test of the reconstruction head and its score on the
# box: real checkpoints, a short head, and a few configs.
#
# One checkpoint of each kind the queue meets:
#   bank    a run with patch sizes 8 to 128, mean/std, zero padding (OAF 40k)
#   old     one patch size, EWMA, the old data from Hugging Face (LOW 665k)
#   blk     one patch size, EWMA, zero padding (BLK 200k)
#   bms     one patch size, mean/std, zero padding (BMS 40k)
#
# Each one runs head_eval_bb.sh in the reconstruction mode, with the queue's
# settings, into a test directory that the queue never reads.
#
# Usage, on the box:  bash box_smoke.sh <kind> [head steps] [config regex] [n configs]
set -uo pipefail

KIND="${1:?usage: box_smoke.sh <bank|old|blk|bms> [steps] [regex] [n]}"
STEPS="${2:-1000}"
FILTER="${3:-^(m4_hourly/short|electricity/H/short|ett1/15T/long|us_births/D/short)$}"
N_CONFIGS="${4:-4}"
CODE="${CF425_CODE:-/workspace/cf-425}"
CK=/workspace/ckpt
P=k3_r100_09_lr56_fix09_dec10k
N=cf393_arm6_v2_combab_alignT_cf373k3_cf412_${P}
case "$KIND" in
  bank) BB=$CK/cf-412oa2/leg_40k/cf412oa2_k3_40k.pth ;;
  old)  BB=$CK/${P}_lr30x/arm6_v2_combab_alignT/leg_665k/${N}_lr30x_665k.pth ;;
  blk)  BB=$CK/cf-419c/cos200k/leg_665k/cf419_cos200k_r2_200k.pth ;;
  bms)  BB=$CK/cf-419ms/leg_40k/cf419ms_k3_40k.pth ;;
  *) echo "ABORT: unknown kind $KIND" >&2; exit 2 ;;
esac
TAG="smoke_${KIND}_s${STEPS}_recon"
mkdir -p "/tmp/cf425_smoke_$KIND" /workspace/results/cf-425/smoke

WT="$CODE" CF373_ROOT="$CK/cf-425/smoke" CF_RESULTS=/workspace/results/cf-425/smoke \
  CF_STOP_K=0 CF_BB_SHAPE="--d-model 384 --n-heads 8 --num-layers 3" \
  GIFT_EVAL=/workspace/gift-eval-data EVAL_SHARDS=1 \
  EVAL_CONFIG_FILTER="$FILTER" EVAL_EXPECT_CONFIGS="$N_CONFIGS" \
  CF393_EVAL_SLOTDIR=/tmp/cf425_smoke_slots \
  BB_GPU=0 HEAD_VRAM_MIB="${HEAD_VRAM_MIB:-4000}" \
  GPU_GATE_LOCKDIR="/tmp/cf425_smoke_$KIND" \
  CF_RECONSTRUCTION=encoder HEAD_SAVE_EVERY=1000000 \
  bash "$CODE/reports/2026-08-08_rollout_depth/scripts/head_eval_bb.sh" \
  "$TAG" "$BB" student "$STEPS"
echo "SMOKE $KIND rc=$?"
