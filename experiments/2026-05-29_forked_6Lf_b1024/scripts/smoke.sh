#!/bin/bash
# #322 — batch-1024 feasibility smoke. Measures peak GPU mem + sps for a few
# steps at a given batch, BYTE-IDENTICAL to #320's backbone recipe except
# --batch-size and --total-steps. Single-process (world_size=1): the loss
# naturally pools ALL negatives of the (single) batch — so single-GPU@1024
# directly measures the full-1024 "all-together" loss memory + the full-1024
# forward. If it fits, no DDP is needed; if the forward OOMs, DDP must split it.
#
# Usage: smoke.sh <arm> <mix> <batch> <steps> <gpu> [chunk]
set -uo pipefail
ARM="${1:?arm=beta|alltime}"; MIX="${2:?mix}"; BS="${3:?batch}"; STEPS="${4:?steps}"
GPU="${5:?gpu}"; export XSHH_ALLT_CHUNK="${6:-8}"
SEED=20260520
case "$ARM" in
  beta)    SHAPE=cosine_similarity_batch_full_hh_negs ;;
  alltime) SHAPE=cosine_similarity_batch_full_hh_negs_xshh_allt ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
WT=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
export PYTHONPATH="${MEMPROBE:+$MEMPROBE:}$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
NAME="smoke_${ARM}_b${BS}"
tlog="$RES/${NAME}.log"; memf="$RES/${NAME}.mem"; : >"$memf"
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[smoke] arm=$ARM shape=$SHAPE batch=$BS steps=$STEPS gpu=$GPU chunk=$XSHH_ALLT_CHUNK -> $tlog"

# Background memory sampler for the target GPU (physical index = $GPU).
( while true; do
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
      | awk -v g="$GPU" -F', *' '$1==g{print $2}' >>"$memf"
    sleep 2
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

python3 -u "$TRAIN" \
  --batch-size "$BS" --device cuda --total-steps "$STEPS" --lr "${LR:-1e-3}" --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every 1000000 --save-dir "$RUNS" --run-name "$NAME" --log-every 10 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 6 \
  --forecaster-d-model 128 --forecaster-n-heads 4 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape "$SHAPE" --pos-in-denominator \
  --tau "${TAU:-0.10}" --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio "$MIX" \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
kill $SAMPLER 2>/dev/null
peak=$(sort -n "$memf" 2>/dev/null | tail -1)
echo "[smoke] rc=$rc  PEAK_GPU${GPU}_MiB=${peak:-NA}"
echo "[smoke] sps tail:"; grep -oE '[0-9.]+ sps' "$tlog" | tail -5
echo "[smoke] last log lines:"; tail -6 "$tlog"
exit $rc
