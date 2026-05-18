#!/bin/bash
# v12 JEPA q-head — 2L causal transformer q-head on v12's JEPA backbone
# (encoder=6L residual_silu, forecaster=1L). Same q-head recipe as v10 chain
# (R9_E13 + e_then_f + Moirai HP + cosine + 2k warmup, bf16, 30k steps).
# Only the backbone-arch flag that changes vs v10 is --encoder-type residual_silu.
# freq_emb_dim, seasonality_emb_dim, num_encoder_layers auto-detected from
# state dict (see train_forecasting_head.py lines 341-385).
#
# Run name: enc_fcst_v12_qhead_xfmr2L_quant_30k.
set -e
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

BB_PATH="$MAIN/checkpoints/enc_fcst_v12_jepa_newconv_residualsilu_fp32_50k_FINAL.pth"
SAVE_DIR="$MAIN/checkpoints"
RUN_NAME="enc_fcst_v12_qhead_xfmr2L_quant_30k"
LOG_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$LOG_DIR"

[ -f "$BB_PATH" ] || { echo "ERROR: backbone missing at $BB_PATH" >&2; exit 1; }
[ -f "$SAVE_DIR/${RUN_NAME}_FINAL.pth" ] && { echo "=== SKIP — exists ==="; exit 0; }
# safety: don't collide with any existing checkpoint (project rule)
for suf in _best.pth _final.pth _losses.csv _5k.pth; do
    if [ -e "$SAVE_DIR/${RUN_NAME}${suf}" ]; then
        echo "ERROR: ${RUN_NAME}${suf} already exists — refusing to clobber" >&2
        exit 1
    fi
done

echo "=== START $RUN_NAME (2L qhead on v12 JEPA backbone, residual_silu enc) ===" && date
python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py" \
    --backbone-path "$BB_PATH" --forecast-len 16 \
    --quantile-head --head-arch transformer --head-causal true \
    --head-num-layers 2 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f \
    --total-steps 30000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 \
    --save-dir "$SAVE_DIR" --run-name "$RUN_NAME" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 1 \
    --encoder-type residual_silu \
    --mix-ratio 0.0 --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster --amp-dtype bf16 \
    2>&1 | tee -a "$LOG_DIR/run_${RUN_NAME}.log"
cp -f "$SAVE_DIR/${RUN_NAME}_best.pth" "$SAVE_DIR/${RUN_NAME}_FINAL.pth"
echo "=== DONE $RUN_NAME ===" && date
