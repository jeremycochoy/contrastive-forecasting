#!/bin/bash
# Autonomous extra dk-sweep @50k (user 2026-05-16, corrected: NO 150k extends —
# 50k is the sweep). Priority order, NO gates, run as GPUs free:
#   1. dk0.50 @50k   -> enc_fcst_dk050_fp32_50k ; tag dk050fp32_50k
#   2. dk0.70 @50k   -> enc_fcst_dk070_fp32_50k ; tag dk070fp32_50k
#   3. dk0.30 @50k   -> enc_fcst_dk030_fp32_50k ; tag dk030fp32_50k   ("if still compute")
# All from-scratch pure-fp32, seed 20260516, same recipe as dk0.85/0.92. Each:
# backbone 0->50k -> standard 2L qhead 30k -> 97-cfg full-eval.
# Never kills anything. Claims a GPU only when its mem<3000 (genuinely idle) so
# it cannot stomp the in-flight dk0.85@50k (GPU0) / dk0.9@150k (GPU1).
# Idempotent (full-eval summary present => done). set -u-safe.
set -uo pipefail
ROOT="/home/jupyter/cf-encoder-forecaster-v2"; MAIN="/home/jupyter/contrastive-forecasting"
CK="$MAIN/checkpoints"; RES="$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results"
LOG="$RES/finesweep_150k_queue.log"; STATE="$RES/.q50k_state"; SEED=20260516
mkdir -p "$STATE"
cd "$ROOT"; export PYTHONPATH="$ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt"); export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null|grep -oE '[0-9]+\.[0-9]+'|head -1; }
gpu_free(){ local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null|tr -d ' '); case "$u" in ''|*[!0-9]*) return 1;; esac; [ "$u" -lt 5000 ]; }
# A GPU is "owned" by its in-flight pipeline (incl. its internal qhead->eval
# gaps where mem briefly drops). Never claim a GPU whose owner script is alive.
gpu_owner_running(){ case "$1" in
  0) pgrep -f 'dk085_fp32_50k_gpu0\.sh' >/dev/null 2>&1 ;;
  1) pgrep -f 'gpu1_reference_pipeline\.sh' >/dev/null 2>&1 ;;
  *) return 1 ;; esac; }
gpu_claimable(){ gpu_free "$1" && ! gpu_owner_running "$1"; }
COMMON=(--device cuda --batch-size 256 --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
  --save-every 5000 --save-dir "$CK" --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1"
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 --num-encoder-layers 6
  --encoder-dropkey-share-heads --encoder-dropkey-share-layers --depthwise-conv 3 --deprecated-depthwise-conv 0
  --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3
  --rev-norm-kind ewma --rev-norm-span 128 --tau 0.10 --loss-shape cosine_similarity_batch --encoder-type gru)

train_50k(){ # $1=dk $2=run-name $3=gpu
  local dk B gpu
  dk="$1"; B="$2"; gpu="$3"
  [ -f "$CK/${B}_FINAL.pth" ] && { log "$B FINAL exists — skip backbone"; return 0; }
  log "$B from-scratch pure-fp32 dk$dk seed$SEED 0->50k (GPU$gpu)"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py "${COMMON[@]}" \
    --seed $SEED --encoder-dropkey "$dk" --total-steps 50000 --run-name "$B" \
    --patch-emb-dtype fp32 --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 \
    >>"$RES/run_${B}.log" 2>&1 || { log "$B TRAIN FAILED"; return 1; }
  if [ -f "$CK/${B}_best_loss.pth" ]; then cp -f "$CK/${B}_best_loss.pth" "$CK/${B}_FINAL.pth"
  elif [ -f "$CK/${B}_final.pth" ]; then cp -f "$CK/${B}_final.pth" "$CK/${B}_FINAL.pth"
  else cp -f "$(ls -t "$CK/${B}"_*k.pth 2>/dev/null|head -1)" "$CK/${B}_FINAL.pth"; fi
  log "$B backbone DONE"
}
qhead_eval(){ # $1=backbone $2=tag $3=gpu
  local bb tag gpu qn qf fout
  bb="$1"; tag="$2"; gpu="$3"
  qn="${tag}_qhead_xfmr2L_quant_30k"; qf="$CK/${qn}_FINAL.pth"; fout="$RES/gift_eval_full_${tag}"
  [ -f "$fout/summary.txt" ] && { log "$tag eval exists GM=$(gm "$fout/summary.txt")"; return 0; }
  [ -f "$bb" ] || { log "$tag backbone missing — skip"; return 1; }
  if [ ! -f "$qf" ]; then
    log "$tag qhead 30k (GPU$gpu)"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py" \
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
  log "$tag full-eval 97cfg (GPU$gpu)"; mkdir -p "$fout"
  CUDA_VISIBLE_DEVICES=$gpu GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
  python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py" \
    --backbone-path "$bb" --head-path "$qf" --output-dir "$fout" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 1 --encoder-type gru \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
    >>"$RES/run_full_eval_${tag}.log" 2>&1
  [ -f "$fout/summary.txt" ] && log "$tag full-eval DONE GM=$(gm "$fout/summary.txt")" || { log "$tag NO SUMMARY"; return 1; }
}
run_item(){ # $1=item $2=gpu
  case "$1" in
    dk050) train_50k 0.50 enc_fcst_dk050_fp32_50k "$2" && qhead_eval "$CK/enc_fcst_dk050_fp32_50k_FINAL.pth" dk050fp32_50k "$2" ;;
    dk070) train_50k 0.70 enc_fcst_dk070_fp32_50k "$2" && qhead_eval "$CK/enc_fcst_dk070_fp32_50k_FINAL.pth" dk070fp32_50k "$2" ;;
    dk030) train_50k 0.30 enc_fcst_dk030_fp32_50k "$2" && qhead_eval "$CK/enc_fcst_dk030_fp32_50k_FINAL.pth" dk030fp32_50k "$2" ;;
  esac
}
done_item(){ case "$1" in
  dk050) [ -f "$RES/gift_eval_full_dk050fp32_50k/summary.txt" ];;
  dk070) [ -f "$RES/gift_eval_full_dk070fp32_50k/summary.txt" ];;
  dk030) [ -f "$RES/gift_eval_full_dk030fp32_50k/summary.txt" ];; esac; }
PRIO=(dk050 dk070 dk030)
log "=== q50k START — prio: ${PRIO[*]} (dk0.50, dk0.70, then dk0.30 if compute) ==="
while :; do
  if done_item dk050 && done_item dk070 && done_item dk030; then log "=== ALL done — exit ==="; break; fi
  for G in 0 1; do
    LK="$STATE/gpu${G}.lock"
    if [ -d "$LK" ]; then
      pid=$(cat "$LK/pid" 2>/dev/null || echo "")
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then continue; fi
      rm -rf "$LK"
    fi
    gpu_claimable "$G" || continue
    sleep 20; gpu_claimable "$G" || continue
    for it in "${PRIO[@]}"; do
      done_item "$it" && continue
      [ -d "$STATE/run_${it}" ] && continue
      mkdir "$STATE/run_${it}" 2>/dev/null || continue
      mkdir -p "$LK"
      ( run_item "$it" "$G"; rm -rf "$STATE/run_${it}"; rm -rf "$LK" ) &
      echo $! > "$LK/pid"
      log "launched [$it] on GPU$G (pid $(cat "$LK/pid"))"
      break
    done
  done
  sleep 160
done