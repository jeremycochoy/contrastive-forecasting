#!/bin/bash
# Per user: once dk0.9-fp32 lands @150k AND its q-head is GM-evaluated, use
# GPU1 for the dk=0.85 sweep @50k. This watcher waits for that signal
# (gift_eval_full_dk09fp32x150k/summary.txt), preempts the gpu1_reference
# pipeline's later steps (dk0.9 _100k eval + v17), frees GPU1, then runs
# dk0.85 pure-fp32 50k + q-head + 97-cfg full-eval on GPU1.
set -uo pipefail
ROOT="/home/jupyter/cf-encoder-forecaster-v2"; MAIN="/home/jupyter/contrastive-forecasting"
CK="$MAIN/checkpoints"; RES="$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results"
LOG="$RES/fine_dk_sweep.log"; SEED=20260516
cd "$ROOT"; export PYTHONPATH="$ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt"); export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export CUDA_VISIBLE_DEVICES=1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null|grep -oE '[0-9]+\.[0-9]+'|head -1; }
SIG="$RES/gift_eval_full_dk09fp32x150k/summary.txt"

log "=== gpu1_dk085 watcher START — waiting for dk0.9-fp32@150k GM-eval ($SIG) ==="
while :; do
  [ -f "$SIG" ] && { log "signal: dk0.9-fp32@150k GM=$(gm "$SIG") — preempting GPU1 for dk0.85"; break; }
  if ! pgrep -f gpu1_reference_pipeline.sh >/dev/null 2>&1; then
    log "gpu1_reference_pipeline ended without dk09fp32x150k summary — proceeding to dk0.85 anyway"; break
  fi
  sleep 120
done

# preempt: stop the dk0.9 _100k eval / v17 work + free GPU1 (dk0.9 backbone+150k-eval already captured)
pkill -9 -f gpu1_reference_pipeline.sh 2>/dev/null || true
pkill -9 -f 'train\.py .*enc_fcst_v17_dk095_150k' 2>/dev/null || true
pkill -9 -f 'train_forecasting_head.*dk09fp32x100k' 2>/dev/null || true
pkill -9 -f 'train_forecasting_head.*v17x' 2>/dev/null || true
pkill -9 -f 'eval_gift_eval_official.*dk09fp32x100k' 2>/dev/null || true
pkill -9 -f 'eval_gift_eval_official.*v17x' 2>/dev/null || true
sleep 5
while :; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
  case "$u" in ''|*[!0-9]*) u=999999;; esac
  [ "$u" -lt 3000 ] && break
  log "waiting for GPU1 to free (used=${u}MiB)"; sleep 30
done
log "GPU1 free — starting dk0.85 pure-fp32 50k"

COMMON=(--device cuda --batch-size 256 --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
  --save-every 5000 --save-dir "$CK" --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1"
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 --num-encoder-layers 6
  --encoder-dropkey-share-heads --encoder-dropkey-share-layers --depthwise-conv 3 --deprecated-depthwise-conv 0
  --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3
  --rev-norm-kind ewma --rev-norm-span 128 --tau 0.10 --loss-shape cosine_similarity_batch --encoder-type gru)
B=enc_fcst_dk085_fp32_50k; TAG=dk085fp32_50k
QN=${TAG}_qhead_xfmr2L_quant_30k; QF="$CK/${QN}_FINAL.pth"; FOUT="$RES/gift_eval_full_${TAG}"
if [ ! -f "$CK/${B}_FINAL.pth" ]; then
  log "$B from-scratch pure-fp32 dk0.85 seed$SEED 0->50k GPU1"
  python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py "${COMMON[@]}" \
    --seed $SEED --encoder-dropkey 0.85 --total-steps 50000 --run-name "$B" \
    --patch-emb-dtype fp32 --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 \
    >>"$RES/run_${B}.log" 2>&1 || { log "$B TRAIN FAILED"; exit 1; }
  if [ -f "$CK/${B}_best_loss.pth" ]; then cp -f "$CK/${B}_best_loss.pth" "$CK/${B}_FINAL.pth"
  elif [ -f "$CK/${B}_final.pth" ]; then cp -f "$CK/${B}_final.pth" "$CK/${B}_FINAL.pth"
  else cp -f "$(ls -t "$CK/${B}"_*k.pth 2>/dev/null|head -1)" "$CK/${B}_FINAL.pth"; fi
  log "$B DONE"
fi
if [ ! -f "$FOUT/summary.txt" ]; then
  if [ ! -f "$QF" ]; then
    log "$TAG q-head train 30k GPU1"
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
  log "$TAG full-eval (97 cfg) GPU1"; mkdir -p "$FOUT"
  GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
  python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py" \
    --backbone-path "$CK/${B}_FINAL.pth" --head-path "$QF" --output-dir "$FOUT" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 1 --encoder-type gru \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
    >>"$RES/run_full_eval_${TAG}.log" 2>&1
fi
log "=== dk0.85 DONE GM=$(gm "$FOUT/summary.txt") (sweep@50k: 0.7=1.3349 0.85=$(gm "$FOUT/summary.txt") 0.90/v11c=1.2920 0.92=$(gm "$RES/gift_eval_full_dk092fp32_50k/summary.txt") 0.95=1.4093) ==="