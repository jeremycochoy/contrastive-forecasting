#!/bin/bash
# Recovery: the overnight_controller's qhead_eval crashed (set -u: unbound
# `qn`), so v27c's 150k & 100k q-head+full-evals never ran. v27c-150k
# backbone is intact. This runs the missing v27c evals on the free GPU0
# with the CORRECT qhead_eval (positional-arg form, like gpu1 pipeline).
# v27c 50k = already done (gift_eval_full_v27c = 1.3313). Idempotent.
set -uo pipefail
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
CK="$MAIN/checkpoints"
RES="$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results"
LOG="$RES/v27c_150k_evals.log"
cd "$ROOT"
export PYTHONPATH="$ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt"); export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export CUDA_VISIBLE_DEVICES=0
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null|head -1|grep -oE '[0-9]+\.[0-9]+'|head -1; }

qhead_eval(){ # $1=backbone_path  $2=tag
  local bb="$1" tag="$2"
  local qn="${2}_qhead_xfmr2L_quant_30k"
  local qf="$CK/${2}_qhead_xfmr2L_quant_30k_FINAL.pth"
  local fout="$RES/gift_eval_full_${2}"
  [ -f "$fout/summary.txt" ] && { log "$tag full-eval exists GM=$(gm "$fout/summary.txt")"; return 0; }
  [ -f "$bb" ] || { log "$tag backbone missing ($bb) — skip"; return 1; }
  if [ ! -f "$qf" ]; then
    log "$tag q-head train 30k (GPU0)"
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

log "=== v27c_150k_evals START (GPU0) ==="
qhead_eval "$CK/enc_fcst_v27c_dk08_ffnfp16_150k_FINAL.pth" v27cx150k
qhead_eval "$CK/enc_fcst_v27c_dk08_ffnfp16_150k_100k.pth"  v27cx100k
log "=== v27c_150k_evals DONE — v27c 50k=1.3313 100k=$(gm "$RES/gift_eval_full_v27cx100k/summary.txt") 150k=$(gm "$RES/gift_eval_full_v27cx150k/summary.txt") ==="