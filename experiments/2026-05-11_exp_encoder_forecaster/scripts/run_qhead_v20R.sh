#!/bin/bash
# v20R = RESUME of the v20 2L q-head training (it crashed at step ~17k).
# Resumes from enc_fcst_v20_qhead_xfmr2L_quant_30k_best.pth (+ its
# _best_optimizer.pth → restores optimizer state, step counter, best_loss,
# and fast-forwards the HF stream). Continues to total-steps 30000.
#
# New run-name suffix `qheadR` → distinct --save-path (project rule:
# never reuse --save-path on resume). Same backbone + identical q-head
# recipe as run_qhead_v20.sh; only --resume + run-name differ.
#
# Run name: enc_fcst_v20_qheadR_xfmr2L_quant_30k.
set -e
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

BB_PATH="$MAIN/checkpoints/enc_fcst_v20_v11c_freshwarmup_fp16_50k_FINAL.pth"
RESUME_FROM="$MAIN/checkpoints/enc_fcst_v20_qhead_xfmr2L_quant_30k_best.pth"
SAVE_DIR="$MAIN/checkpoints"
RUN_NAME="enc_fcst_v20_qheadR_xfmr2L_quant_30k"
LOG_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$LOG_DIR"

[ -f "$BB_PATH" ]      || { echo "ERROR: backbone missing at $BB_PATH" >&2; exit 1; }
[ -f "$RESUME_FROM" ]  || { echo "ERROR: resume source missing at $RESUME_FROM" >&2; exit 1; }
[ -f "${RESUME_FROM%.pth}_optimizer.pth" ] || { echo "ERROR: resume optimizer missing at ${RESUME_FROM%.pth}_optimizer.pth" >&2; exit 1; }
[ -f "$SAVE_DIR/${RUN_NAME}_FINAL.pth" ] && { echo "=== SKIP — exists ==="; exit 0; }
# safety: don't collide with any existing checkpoint (project rule)
for suf in _best.pth _final.pth _losses.csv _5k.pth; do
    if [ -e "$SAVE_DIR/${RUN_NAME}${suf}" ]; then
        echo "ERROR: ${RUN_NAME}${suf} already exists — refusing to clobber" >&2
        exit 1
    fi
done

echo "=== START $RUN_NAME (RESUME v20 qhead from $(basename "$RESUME_FROM")) ===" && date
python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py" \
    --backbone-path "$BB_PATH" --forecast-len 16 \
    --resume "$RESUME_FROM" \
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
    --encoder-type gru \
    --mix-ratio 0.0 --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster --amp-dtype bf16 \
    2>&1 | tee -a "$LOG_DIR/run_${RUN_NAME}.log"

# Robust finalize: resuming FROM the best checkpoint means the run may
# never beat the restored best_loss, so ${RUN_NAME}_best.pth may not be
# written. Fall back to the guaranteed _final.pth, then newest periodic.
FIN_SRC=""
if [ -f "$SAVE_DIR/${RUN_NAME}_best.pth" ]; then
    FIN_SRC="$SAVE_DIR/${RUN_NAME}_best.pth"
elif [ -f "$SAVE_DIR/${RUN_NAME}_final.pth" ]; then
    FIN_SRC="$SAVE_DIR/${RUN_NAME}_final.pth"
else
    FIN_SRC="$(ls -t "$SAVE_DIR/${RUN_NAME}"_*k.pth 2>/dev/null | head -1)"
fi
[ -n "$FIN_SRC" ] && [ -f "$FIN_SRC" ] || { echo "ERROR: no v20R checkpoint to finalize" >&2; exit 1; }
echo "finalize: $FIN_SRC -> ${RUN_NAME}_FINAL.pth"
cp -f "$FIN_SRC" "$SAVE_DIR/${RUN_NAME}_FINAL.pth"
echo "=== DONE $RUN_NAME ===" && date
