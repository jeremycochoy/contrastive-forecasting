#!/bin/bash
# GPU1 reference pipeline (corrective): v20-fp16 diverged at ~45k, so the
# dk0.9 @150k reference MUST be pure-fp32 (v11c's own recipe extended).
# Order: dk0.9-fp32 0->150k (the bet reference) FIRST, then v17 dk0.95-fp32.
# Each: backbone -> q-head+full-eval at 50k, 150k, 100k. From-scratch =>
# continuous optimizer (never reset). seed 20260516. GPU1 only.
set -uo pipefail
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
CK="$MAIN/checkpoints"
RES="$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results"
LOG="$RES/gpu1_reference_pipeline.log"
SEED=20260516
cd "$ROOT"
export PYTHONPATH="$ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt"); export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export CUDA_VISIBLE_DEVICES=1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null|head -1|grep -oE '[0-9]+\.[0-9]+'|head -1; }
dgate(){ local f; f=$(df -BG / | awk 'NR==2{gsub("G","",$4);print $4}'); [ "$f" -ge 35 ]; }

COMMON=(--device cuda --batch-size 256 --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
  --save-every 5000 --save-dir "$CK" --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1"
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 --num-encoder-layers 6
  --encoder-dropkey-share-heads --encoder-dropkey-share-layers --depthwise-conv 3 --deprecated-depthwise-conv 0
  --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3
  --rev-norm-kind ewma --rev-norm-span 128 --tau 0.10 --loss-shape cosine_similarity_batch --encoder-type gru)

train_fp32_150k(){ # $1=dk  $2=run-name
  local dk="$1" B="$2"
  [ -f "$CK/${B}_FINAL.pth" ] && { log "$B FINAL exists — skip"; return 0; }
  dgate || { log "disk<35G — abort $B"; return 1; }
  log "$B from-scratch pure-fp32 dk$dk seed$SEED 0->150k GPU1"
  python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py "${COMMON[@]}" \
    --seed $SEED --encoder-dropkey "$dk" --total-steps 150000 --run-name "$B" \
    --patch-emb-dtype fp32 --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 \
    >>"$RES/run_${B}.log" 2>&1 || { log "$B TRAIN FAILED"; return 1; }
  if [ -f "$CK/${B}_best_loss.pth" ]; then cp -f "$CK/${B}_best_loss.pth" "$CK/${B}_FINAL.pth"
  elif [ -f "$CK/${B}_final.pth" ]; then cp -f "$CK/${B}_final.pth" "$CK/${B}_FINAL.pth"
  else cp -f "$(ls -t "$CK/${B}"_*k.pth 2>/dev/null|head -1)" "$CK/${B}_FINAL.pth"; fi
  log "$B DONE (FINAL set)"
}

qhead_eval(){ # $1=backbone_path  $2=tag
  local bb="$1" tag="$2" qn="${2}_qhead_xfmr2L_quant_30k" qf fout
  qf="$CK/${2}_qhead_xfmr2L_quant_30k_FINAL.pth"; fout="$RES/gift_eval_full_${tag}"
  [ -f "$fout/summary.txt" ] && { log "$tag full-eval exists GM=$(gm "$fout/summary.txt")"; return 0; }
  [ -f "$bb" ] || { log "$tag backbone missing ($bb) — skip"; return 1; }
  dgate || { log "disk<35G — abort $tag"; return 1; }
  if [ ! -f "$qf" ]; then
    log "$tag q-head train 30k"
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
  log "$tag full-eval (97 cfg)"; mkdir -p "$fout"
  GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
  python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py" \
    --backbone-path "$bb" --head-path "$qf" --output-dir "$fout" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 1 --encoder-type gru \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
    >>"$RES/run_full_eval_${tag}.log" 2>&1
  [ -f "$fout/summary.txt" ] && log "$tag full-eval DONE GM=$(gm "$fout/summary.txt")" || { log "$tag full-eval NO SUMMARY"; return 1; }
}

arm(){ # $1=dk $2=run-name $3=tagbase
  local dk="$1" B="$2" tb="$3"
  train_fp32_150k "$dk" "$B" || { log "arm $tb backbone failed — skip evals"; return 1; }
  qhead_eval "$CK/${B}_50k.pth"  "${tb}x50k"
  qhead_eval "$CK/${B}_FINAL.pth" "${tb}x150k"
  qhead_eval "$CK/${B}_100k.pth" "${tb}x100k"
  log "arm $tb COMPLETE"
}

# wait for GPU1 to actually be free (the killed v17 python released it)
log "=== gpu1_reference_pipeline START ==="
while :; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
  case "$u" in ''|*[!0-9]*) u=999999;; esac
  [ "$u" -lt 3000 ] && break
  log "waiting for GPU1 to free (used=${u}MiB)"; sleep 60
done
log "GPU1 free — starting dk0.9 pure-fp32 (the bet reference) FIRST"
arm 0.9  enc_fcst_dk09_fp32_150k    dk09fp32
log "dk0.9-fp32 arm done — now v17 dk0.95 pure-fp32 (challenger / spare)"
arm 0.95 enc_fcst_v17_dk095_150k    v17
log "=== gpu1_reference_pipeline DONE ==="