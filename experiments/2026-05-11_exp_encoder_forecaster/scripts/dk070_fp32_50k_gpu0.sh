#!/bin/bash
# Finer dk sweep — GPU0 leg: ONLY dk=0.92 pure-fp32 50k (dk=0.85 moved to
# GPU1 per user, after dk0.9-fp32@150k eval, to parallelize & avoid collision).
# v11c/v16/v17-family recipe, seed 20260516, from-scratch. backbone 0->50k ->
# standard 2L q-head 30k -> 97-cfg full-eval. GPU0.
set -uo pipefail
ROOT="/home/jupyter/cf-encoder-forecaster-v2"; MAIN="/home/jupyter/contrastive-forecasting"
CK="$MAIN/checkpoints"; RES="$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results"
LOG="$RES/fine_dk_sweep.log"; SEED=20260516
cd "$ROOT"; export PYTHONPATH="$ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt"); export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export CUDA_VISIBLE_DEVICES=0
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null|grep -oE '[0-9]+\.[0-9]+'|head -1; }
COMMON=(--device cuda --batch-size 256 --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
  --save-every 5000 --save-dir "$CK" --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1"
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 --num-encoder-layers 6
  --encoder-dropkey-share-heads --encoder-dropkey-share-layers --depthwise-conv 3 --deprecated-depthwise-conv 0
  --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3
  --rev-norm-kind ewma --rev-norm-span 128 --tau 0.10 --loss-shape cosine_similarity_batch --encoder-type gru)
B=enc_fcst_dk070_fp32_50k; TAG=dk070fp32_50k
QN=${TAG}_qhead_xfmr2L_quant_30k; QF="$CK/${QN}_FINAL.pth"; FOUT="$RES/gift_eval_full_${TAG}"
log "=== dk070_fp32_50k START (GPU0) ==="
if [ ! -f "$CK/${B}_FINAL.pth" ]; then
  log "$B from-scratch pure-fp32 dk0.70 seed$SEED 0->50k"
  python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py "${COMMON[@]}" \
    --seed $SEED --encoder-dropkey 0.70 --total-steps 50000 --run-name "$B" \
    --patch-emb-dtype fp32 --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 \
    >>"$RES/run_${B}.log" 2>&1 || { log "$B TRAIN FAILED"; exit 1; }
  if [ -f "$CK/${B}_best_loss.pth" ]; then cp -f "$CK/${B}_best_loss.pth" "$CK/${B}_FINAL.pth"
  elif [ -f "$CK/${B}_final.pth" ]; then cp -f "$CK/${B}_final.pth" "$CK/${B}_FINAL.pth"
  else cp -f "$(ls -t "$CK/${B}"_*k.pth 2>/dev/null|head -1)" "$CK/${B}_FINAL.pth"; fi
  log "$B DONE"
fi
if [ ! -f "$FOUT/summary.txt" ]; then
  if [ ! -f "$QF" ]; then
    log "$TAG q-head train 30k"
    python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py" \
      --backbone-path "$CK/${B}_FINAL.pth" --forecast-len 16 --quantile-head --head-arch transformer --head-causal true \
      --head-num-layers 2 --head-nhead 6 --head-ffn-mult 4.0 --head-dropout 0.1 --head-train-input e_then_f \
      --total-steps 30000 --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
      --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 --save-every 5000 --log-every 200 \
      --save-dir "$CK" --run-name "$QN" --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
      --device cuda --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 \
      --encoder-type gru --mix-ratio 0.0 --rev-norm-kind ewma --rev-norm-span 128 \
      --reconstruction forecaster --amp-dtype bf16 >>"$RES/run_${QN}.log" 2>&1 || { log "$TAG qhead FAILED"; exit 1; }
    if [ -f "$CK/${QN}_best.pth" ]; then cp -f "$CK/${QN}_best.pth" "$QF"
    elif [ -f "$CK/${QN}_final.pth" ]; then cp -f "$CK/${QN}_final.pth" "$QF"
    else cp -f "$(ls -t "$CK/${QN}"_*k.pth 2>/dev/null|head -1)" "$QF"; fi
  fi
  log "$TAG full-eval (97 cfg)"; mkdir -p "$FOUT"
  GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
  python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py" \
    --backbone-path "$CK/${B}_FINAL.pth" --head-path "$QF" --output-dir "$FOUT" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 1 --encoder-type gru \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
    >>"$RES/run_full_eval_${TAG}.log" 2>&1
fi
log "=== dk070_fp32_50k DONE GM=$(gm "$FOUT/summary.txt") (vs 0.90/v11c=1.2920) ==="