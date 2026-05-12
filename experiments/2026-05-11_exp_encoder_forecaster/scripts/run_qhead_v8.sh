#!/bin/bash
# v8 q-head — same R9_E13 + bf16 recipe as v7, but on v8's backbone.
#
# v8 backbone: enc_fcst_dk07_pb_fp32_resume_50k (dropkey=0.7
# per-(B,head) full-indep, fp32, resumed from a2's best_loss → step 50k).
# v7 backbone: enc_fcst_dk09_hsl_b256_fp32_50k (dropkey=0.9
# heads+layers-shared, fp32, fresh).
#
# Compare downstream MASE for v7 vs v8 → which mask sharing axis
# helps the q-head most.
#
# Run name: enc_fcst_v8_qhead_xfmr12L_quant_30k. Runs on GPU 0
# (v7 q-head ran on GPU 1; this can run on either now both are free).

set -e

ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

BB_PATH="$MAIN/checkpoints/enc_fcst_dk07_pb_fp32_resume_50k_FINAL.pth"
SAVE_DIR="$MAIN/checkpoints"
RUN_NAME="enc_fcst_v8_qhead_xfmr12L_quant_30k"
LOG_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$LOG_DIR"

if [ ! -f "$BB_PATH" ]; then
    echo "ERROR: backbone missing at $BB_PATH" >&2
    exit 1
fi
if [ -f "$SAVE_DIR/${RUN_NAME}_FINAL.pth" ]; then
    echo "=== SKIP — $SAVE_DIR/${RUN_NAME}_FINAL.pth exists ==="
    exit 0
fi

echo "=== START $RUN_NAME ===" && date
python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py" \
    --backbone-path "$BB_PATH" --forecast-len 16 \
    --quantile-head --head-arch transformer --head-causal true \
    --head-num-layers 12 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 \
    --head-train-input e_then_f \
    --total-steps 30000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 \
    --save-dir "$SAVE_DIR" --run-name "$RUN_NAME" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster \
    --amp-dtype bf16 \
    2>&1 | tee -a "$LOG_DIR/run_${RUN_NAME}.log"
cp -f "$SAVE_DIR/${RUN_NAME}_best.pth" "$SAVE_DIR/${RUN_NAME}_FINAL.pth"
echo "=== DONE $RUN_NAME — saved ${RUN_NAME}_FINAL.pth ===" && date
