#!/bin/bash
# v13 — same as v11c (6L encoder + 1L forecaster, NEW conv, dropkey 0.9,
# pure fp32) BUT with a JEPA Linear bottleneck wrapped around the
# forecaster: encoder stays at d=384 (untouched), the forecaster runs
# at d=128 with 4 heads, and Linear projects 384→128 in / 128→384 out.
# Contrastive loss still operates in 384-dim space (forecaster output
# vs encoder(future)), so x_original is unchanged.
#
# Hypothesis: tighter information bottleneck on the encoder forces it
# to emit more semantic features → better downstream MASE on top of
# v11c (current best, GM-MASE 1.388).
#
# v11c = 6L enc d=384 + 1L fcst d=384, NEW conv, fp32     → headline
# v13  = 6L enc d=384 + 1L fcst d=128 (Linear up/down), fp32 → THIS RUN
#
# Forecaster bottleneck is no-op on legacy checkpoints (down/up are
# nn.Identity when forecaster_d_model == d_model), so this only affects
# fresh-init runs that pass --forecaster-d-model.

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
# v15 (12L) is on GPU 0 until ~3-4h from launch; v13 lands on GPU 1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_v13_jepa_fcstbottleneck128_newconv_fp32_50k"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

[ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ] && { echo "=== SKIP — exists ==="; exit 0; }
TOTAL_STEPS="${TOTAL_STEPS:-50000}"

echo "=== START ${NAME} (fresh, JEPA 6L+1L, new conv, PURE fp32, fcst-bottleneck d=128/4h) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps "${TOTAL_STEPS}" --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 1 \
    --forecaster-d-model 128 --forecaster-n-heads 4 \
    --num-encoder-layers 6 --encoder-dropkey 0.9 \
    --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 \
    --patch-emb-dtype fp32 --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --tau 0.10 \
    --loss-shape "cosine_similarity_batch" \
    --encoder-type gru \
    2>&1 | tee -a "${LOG_DIR}/run_${NAME}.log"

cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
echo "=== DONE ${NAME} ===" && date
