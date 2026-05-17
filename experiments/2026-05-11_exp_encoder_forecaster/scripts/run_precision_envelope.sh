#!/bin/bash
# Precision-envelope test: resume v11c at step 5k (just past where v11/v11b
# diverged historically) with ONE precision axis flipped to fp16 or bf16.
# Train 5-10k more steps and observe whether loss stays on v11c's trajectory
# (~2.10) or diverges. Cheaper than full re-trains.
#
# Usage:  CUDA_VISIBLE_DEVICES=N bash run_precision_envelope.sh AXIS
#   where AXIS ∈ {ffn_bf16, ffn_fp16, attnffn_fp16, allbody_fp16, pemb_fp16}
#
# Each axis builds incrementally on the previous: ffn_bf16 → ffn_fp16 →
# attnffn_fp16 → allbody_fp16 → pemb_fp16 (= all fp16, the v11 setup that broke).

set -euo pipefail
AXIS="${1:?usage: bash run_precision_envelope.sh AXIS (ffn_bf16|ffn_fp16|attnffn_fp16|allbody_fp16|pemb_fp16)}"

ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"

case "$AXIS" in
    ffn_bf16)
        FLAGS="--patch-emb-dtype fp32 --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype bf16" ;;
    ffn_fp16)
        FLAGS="--patch-emb-dtype fp32 --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp16" ;;
    attnffn_fp16)
        FLAGS="--patch-emb-dtype fp32 --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16" ;;
    allbody_fp16)
        FLAGS="--patch-emb-dtype fp32 --residual-dtype fp16 --attn-dtype fp16 --ffn-dtype fp16" ;;
    pemb_fp16)
        FLAGS="--patch-emb-dtype fp16 --residual-dtype fp16 --attn-dtype fp16 --ffn-dtype fp16" ;;
    *)
        echo "ERROR: unknown AXIS '$AXIS'" >&2; exit 1 ;;
esac

NAME="enc_fcst_precenv_${AXIS}_v11c_5k_15k"
RESUME="$MAIN/checkpoints/enc_fcst_v11c_jepa_newconv_fp32_50k_5k.pth"
SAVE_DIR="$MAIN/checkpoints"
LOG_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$LOG_DIR"

[ -f "$RESUME" ] || { echo "ERROR: v11c 5k resume missing at $RESUME" >&2; exit 1; }
[ -f "$SAVE_DIR/${NAME}_FINAL.pth" ] && { echo "=== SKIP — exists ==="; exit 0; }

cd "$ROOT"
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

echo "=== START $NAME (resume v11c _5k.pth → step 15k, axis=$AXIS, flags=$FLAGS) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps 15000 --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 2500 --save-dir "$SAVE_DIR" --run-name "$NAME" \
    --resume "$RESUME" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 1 \
    --num-encoder-layers 6 --encoder-dropkey 0.9 \
    --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 \
    $FLAGS \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --tau 0.10 \
    --loss-shape "cosine_similarity_batch" \
    --encoder-type gru \
    2>&1 | tee -a "$LOG_DIR/run_${NAME}.log"
echo "=== DONE $NAME ===" && date
