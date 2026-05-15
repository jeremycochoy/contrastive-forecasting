#!/bin/bash
# v17 — branched from v11c (fresh-init JEPA, NEW conv placement, PURE fp32,
# encoder 6L, forecaster 1L). ONLY difference: --encoder-dropkey 0.95 (v11c
# was 0.9 — tighter dropkey to test if heavier key-masking improves the
# encoder's downstream representations).
#
# v11c (dk=0.9): contrastive loss ~2.10 → qhead ema_loss 0.221 → GM-MASE 1.388
# v17  (dk=0.95): ??? → ??? → ???  (running on GPU 1; v16 dk=0.7 on GPU 0)
#
# Mirrors v15 launcher; the ONLY changes vs v15 are:
#   * --encoder-dropkey 0.95 (v15 had 0.9)
#   * --num-layers 1 (v15 had 4 — v17 forecaster back to 1L like v11c)
#   * run name → v17
#
# All other knobs identical to v11c/v15: enc 6L, dk share heads+layers,
# depthwise-conv=3 deprecated=0, all-fp32 body, GRU, RevEWMNorm 128,
# B=256, lr=1e-3, tau=0.10, mixup 0.3, 50k steps, save every 5000.

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_v17_jepa_enc6_fcst1_dk095_newconv_fp32_50k"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

[ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ] && { echo "=== SKIP — exists ==="; exit 0; }
TOTAL_STEPS="${TOTAL_STEPS:-50000}"

echo "=== START ${NAME} (fresh, JEPA 6L+1L, dk=0.95, new conv, PURE fp32) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps "${TOTAL_STEPS}" --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 1 \
    --num-encoder-layers 6 --encoder-dropkey 0.95 \
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
