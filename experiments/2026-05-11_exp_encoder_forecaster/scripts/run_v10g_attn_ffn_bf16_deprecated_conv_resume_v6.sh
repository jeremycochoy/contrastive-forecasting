#!/bin/bash
# v10g — the hypothesis test: resume v6's _best_loss (pre-divergence weights
# from the recipe that originally diverged in bf16) with the bf16 footprint
# pushed as wide as possible WITHOUT touching the residual stream:
#   --attn-bf16 true              (SA Q/K/V/scores/out → bf16, output cast back fp32)
#   --ffn-bf16 true               (FFN x4 expansion+projection → bf16, output cast back fp32)
#   --deprecated-depthwise-conv 3 (match v6's training graph: legacy in-place conv on residual)
#   The legacy conv path is force-wrapped in autocast(enabled=False) so it
#   stays fp32 even with the new bf16 flags — that conv mutates the residual,
#   so its precision matters.
#   --amp-dtype none              (outer fp32; no global autocast — only the
#                                  surgical bf16 islands inside attn/ffn)
#
# Test: must clear v6's historical step-4500 wall, matching v10d (pure fp32)
# and v10f (FFN-only bf16). If yes, residual-clean bf16 islands are safe
# across the WHOLE block (attn + ffn together) — biggest bf16 surface yet,
# strongest speedup case.
#
# Run name: enc_fcst_v10g_attnffnbf16_deprconv_resume_v6_15k. GPU 0.

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_v10g_attnffnbf16_deprconv_resume_v6_15k"
RESUME="/home/jupyter/contrastive-forecasting/checkpoints/enc_fcst_dk09_hsl_b256_50k_best_loss.pth"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

[ -f "$RESUME" ] || { echo "ERROR: v6 best_loss missing at $RESUME" >&2; exit 1; }
[ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ] && { echo "=== SKIP — exists ==="; exit 0; }

echo "=== START ${NAME} (resume v6 best_loss, attn+ffn bf16, deprecated conv fp32) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps 15000 --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 2000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --resume "${RESUME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --num-encoder-layers 6 --encoder-dropkey 0.9 \
    --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 0 --deprecated-depthwise-conv 3 \
    --amp-dtype none --attn-bf16 true --ffn-bf16 true \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --tau 0.10 \
    --loss-shape "cosine_similarity_batch" \
    --encoder-type gru \
    2>&1 | tee -a "${LOG_DIR}/run_${NAME}.log"

cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
echo "=== DONE ${NAME} ===" && date
