#!/bin/bash
# Finer dk sweep (user request): pure-fp32, 50k, bracketing the 0.9 optimum
# with dk=0.92 then dk=0.85. Same v11c-family recipe as v16/v17 (enc6+fcst1,
# newconv, all-fp32). From-scratch => continuous optimizer. GPU0. seed 20260516.
# Each: backbone 0->50k -> standard 2L q-head 30k -> 97-cfg full-eval.
set -uo pipefail
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
CK="$MAIN/checkpoints"
RES="$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results"
LOG="$RES/fine_dk_sweep.log"
SEED=20260516
cd "$ROOT"
export PYTHONPATH="$ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

train_fp32_50k(){ # $1=dk $2=run-name
  local dk="$1" B="$2"
  [ -f "$CK/${B}_FINAL.pth" ] && { log "$B FINAL exists — skip"; return 0; }
  log "$B from-scratch pure-fp32 dk$dk seed$SEED 0->50k GPU0"
  python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py "${COMMON[@]}" \
    --seed $SEED --encoder-dropkey "$dk" --total-steps 50000 --run-name "$B" \
    --patch-emb-dtype fp32 --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 \
    >>"$RES/run_${B}.log" 2>&1 || { log "$B TRAIN FAILED"; return 1; }
  if [ -f "$CK/${B}_best_loss.pth" ]; then cp -f "$CK/${B}_best_loss.pth" "$CK/${B}_FINAL.pth"
  elif [ -f "$CK/${B}_final.pth" ]; then cp -f "$CK/${B}_final.pth" "$CK/${B}_FINAL.pth"
  else cp -f "$(ls -t "$CK/${B}"_*k.pth 2>/dev/null|head -1)" "$CK/${B}_FINAL.pth"; fi
  log "$B DONE"
}

qhead_eval(){ # $1=backbone_path  $2=tag
  local bb="$1" tag="$2"
  local qn="${2}_qhead_xfmr2L_quant_30k"
  local qf="$CK/${2}_qhead_xfmr2L_quant_30k_FINAL.pth"
  local fout="$RES/gift_eval_full_${2}"
  [ -f "$fout/summary.txt" ] && { log "$tag full-eval exists GM=$(gm "$fout/summary.txt")"; return 0; }
  [ -f "$bb" ] || { log "$tag backbone missing — skip"; return 1; }
  if [ ! -f "$qf" ]; then
    log "$tag q-head train 30k GPU0"
    python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py" \
      --backbone-path "$bb" --forecast-len 16 --quantile-head --head-arch transformer --head-causal true \
      --head-num-layers 2 --head-nhead 6 --head-ffn-mult 4.0 --head-dropout 0.1 --head-train-input e_then_f \
      --total-steps 30000 --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
      --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 --save-every 5000 --log-every 200 \
      --save-dir "$CK" --run-name "$qn" --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
      --device cuda --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 \
      --encoder-type gru --mix-ratio 0.0 --rev-norm-kind ewma --rev-norm-span 128 \
      --reconstruction forecaster --amp-dtype bf16 >>"$RES/run_${qn}.log" 2>&1 || { log "$tag qhead FAILED"; return 1; }
    if [ -f "$CK/${qn}_best.pth" ]; then cp -f "$CK/${qn}_best.pth" "$qf"
    elif [ -f "$CK/${qn}_final.pth" ]; then cp -f "$CK/${qn}_final.pth" "$qf"
    else cp -f "$(ls -t "$CK/${qn}"_*k.pth 2>/dev/null|head -1)" "$qf"; fi
  fi
  [ -f "$qf" ] || { log "$tag qhead FINAL missing"; return 1; }
  log "$tag full-eval (97 cfg) GPU0"; mkdir -p "$fout"
  GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
  python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py" \
    --backbone-path "$bb" --head-path "$qf" --output-dir "$fout" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 1 --encoder-type gru \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
    >>"$RES/run_full_eval_${2}.log" 2>&1
  [ -f "$fout/summary.txt" ] && log "$tag full-eval DONE GM=$(gm "$fout/summary.txt")" || { log "$tag full-eval NO SUMMARY"; return 1; }
}

arm(){ # $1=dk $2=run-name $3=tag
  train_fp32_50k "$1" "$2" || { log "$3 backbone failed — skip eval"; return 1; }
  qhead_eval "$CK/${2}_FINAL.pth" "$3"
  log "arm $3 COMPLETE"
}

log "=== fine_dk_sweep START (GPU0): dk0.92 then dk0.85, fp32, 50k ==="
arm 0.92 enc_fcst_dk092_fp32_50k dk092fp32_50k
arm 0.85 enc_fcst_dk085_fp32_50k dk085fp32_50k
log "=== fine_dk_sweep DONE: 0.85=$(gm "$RES/gift_eval_full_dk085fp32_50k/summary.txt") 0.90(v11c)=1.2920 0.92=$(gm "$RES/gift_eval_full_dk092fp32_50k/summary.txt") ==="