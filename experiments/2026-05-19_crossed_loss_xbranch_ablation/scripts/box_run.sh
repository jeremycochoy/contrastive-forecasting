#!/bin/bash
# #307 cross-branch ablation — box-side orchestrator (runs ON a vast.ai
# box at /workspace/app). Idempotent: every phase skips if its output
# exists, so a re-exec resumes after any interruption.
#
# Recipe = the #303 crossed-loss-ablation run_all.sh, which is the #296
# (A) bottleneck-fullfh backbone-OF-RECORD recipe. ONLY --loss-shape
# changes per arm. Backbone: 50k DDP (2 GPU), seed 20260517,
# --pos-in-denominator. Downstream: 30k 2L-causal q-head + official
# GIFT-Eval (triage 11 + full 97), single GPU.
#
# Usage (on box):  bash box_run.sh <PHASE> <ARM_SHORT> [GPU]
#   PHASE      = backbone | qhead | eval_triage | eval_full | downstream
#   ARM_SHORT  = hhff | fhhhff | hhxbf
#   GPU        = CUDA device for single-GPU phases (default 0)
set -uo pipefail
APP=/workspace/app
cd "$APP"
PHASE="${1:?PHASE}"; ARM="${2:?ARM_SHORT}"; GPU="${3:-0}"
SEED=20260517; TOTAL=50000
RUNS="$APP/runs"; RES="$APP/results"; mkdir -p "$RUNS" "$RES"
NAME="cl_${ARM}_50k"
BB="$RUNS/${NAME}_FINAL.pth"
QN="${NAME}_qhead_xfmr2L_quant_30k"; QF="$RUNS/${NAME}_qhead_FINAL.pth"
TRIAGE='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'

declare -A LS=(
  [hhff]=cosine_similarity_batch_full_hh_ff_negs
  [fhhhff]=cosine_similarity_batch_full_fh_hh_ff_negs
  [hhxbf]=cosine_similarity_batch_full_hh_negs_xbfree
)
[ -n "${LS[$ARM]:-}" ] || { echo "unknown ARM $ARM"; exit 2; }

export PYTHONPATH="$APP" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$APP/experiments/hf_token.txt")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export GIFT_EVAL=/workspace/gift-eval-data
TRAIN="$APP/experiments/2026-04-27_freq-embedding/scripts/train.py"
QTRAIN="$APP/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$APP/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1; }
freeport(){ python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

setup(){
  local M="$APP/.setup_done"
  [ -f "$M" ] && { log "setup: already done"; return 0; }
  log "SETUP begin"
  apt-get update -qq 2>/dev/null || true
  apt-get install -y -qq python3-pip rsync 2>/dev/null || true
  pip install --break-system-packages "torch>=2.8,<2.9" \
    --index-url https://download.pytorch.org/whl/cu128 >/dev/null 2>&1 || \
    pip install --break-system-packages "torch>=2.8,<2.9" >/dev/null 2>&1 || true
  pip install --break-system-packages 'numpy<2' pandas pyarrow statsmodels \
    matplotlib datasets huggingface_hub tqdm gluonts >/dev/null 2>&1
  pip install --break-system-packages \
    "salesforce-gift-eval @ git+https://github.com/SalesforceAIResearch/gift-eval.git" \
    >/dev/null 2>&1 || true
  python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count(),'x',(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))"
  if [ ! -d /workspace/gift-eval-data ] || [ -z "$(ls -A /workspace/gift-eval-data 2>/dev/null)" ]; then
    log "SETUP: downloading GIFT-Eval data"
    python3 -c "
from huggingface_hub import snapshot_download
import os, shutil
p=snapshot_download('jeremycochoy/gift-pretrain-small-4096',repo_type='dataset',allow_patterns='eval/**',local_dir='/workspace/gift-eval-download')
src=os.path.join(p,'eval'); dst='/workspace/gift-eval-data'
shutil.rmtree(dst,ignore_errors=True); shutil.copytree(src,dst); print('GIFT-Eval ready',dst)
"
  fi
  touch "$M"; log "SETUP done"
}

backbone(){
  [ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; return 0; }
  setup || { log "SETUP FAILED"; return 1; }
  local tlog="$RES/run_${NAME}.log"
  local ng; ng=$(python3 -c 'import torch;print(torch.cuda.device_count())')
  [ "${ng:-0}" -ge 2 ] || { log "BB ERROR: need 2 GPUs, have $ng"; return 1; }
  log "BB START $ARM loss=${LS[$ARM]} DDP nproc=2 bs128 ${TOTAL}"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port="$(freeport)" "$TRAIN" \
    --batch-size 128 --device cuda --total-steps "$TOTAL" --lr 1e-3 --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.95 --seed "$SEED" \
    --save-every 5000 --save-dir "$RUNS" --run-name "$NAME" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
    --num-encoder-layers 6 --num-layers 1 \
    --forecaster-d-model 128 --forecaster-n-heads 4 \
    --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 \
    --loss-shape "${LS[$ARM]}" --pos-in-denominator \
    --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
    --mixup-p 0.3 --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 \
    --log-attn-amplitude --log-attn-amplitude-every 200 \
    --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 \
    --patch-emb-dtype fp32 >>"$tlog" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "BB train exited rc=$rc (tail: $(tail -3 "$tlog"|tr '\n' ' '))"; fi
  if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
  elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
  else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
  cp -f "${BB%.pth}".pth 2>/dev/null
  # also pin the optimizer companion of whatever became FINAL
  for c in "${NAME}_best_loss" "${NAME}_final"; do
    [ -f "$RUNS/${c}_optimizer.pth" ] && cp -f "$RUNS/${c}_optimizer.pth" "$RUNS/${NAME}_FINAL_optimizer.pth" && break
  done
  [ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; return 0; }
  log "BB FAILED no checkpoint"; return 1
}

qhead(){
  [ -f "$BB" ] || { log "QH SKIP (no backbone FINAL)"; return 1; }
  [ -f "$QF" ] && { log "QH SKIP (FINAL exists)"; return 0; }
  setup || return 1
  local arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 \
              --forecaster-d-model 128 --forecaster-n-heads 4 --encoder-type gru \
              --rev-norm-kind ewma --rev-norm-span 128)
  log "QH START $ARM on GPU$GPU ($QN)"
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u "$QTRAIN" \
    --backbone-path "$BB" --forecast-len 16 --quantile-head --head-arch transformer \
    --head-causal true --head-num-layers 2 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f \
    --total-steps 30000 --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 \
    --weight-decay 0.1 --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 --save-dir "$RUNS" --run-name "$QN" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --device cuda "${arch[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype bf16 \
    >>"$RES/run_${QN}.log" 2>&1 || { log "QH FAILED (tail: $(tail -3 "$RES/run_${QN}.log"|tr '\n' ' '))"; return 1; }
  if   [ -f "$RUNS/${QN}_best.pth" ];  then cp -f "$RUNS/${QN}_best.pth"  "$QF"
  elif [ -f "$RUNS/${QN}_final.pth" ]; then cp -f "$RUNS/${QN}_final.pth" "$QF"
  else cp -f "$(ls -t "$RUNS/${QN}"_*k.pth 2>/dev/null|head -1)" "$QF"; fi
  [ -f "$QF" ] && { log "QH DONE -> ${NAME}_qhead_FINAL.pth"; return 0; }
  log "QH FAILED no checkpoint"; return 1
}

do_eval(){ # $1=tag $2=outdir $3=filter
  local tag="$1" out="$2" filt="$3"
  [ -f "$out/summary.txt" ] && { log "EVAL $tag exists GM=$(gm "$out/summary.txt")"; return 0; }
  [ -f "$BB" ] && [ -f "$QF" ] || { log "EVAL $tag SKIP (missing bb/qf)"; return 1; }
  setup || return 1
  mkdir -p "$out"
  local ff=(); [ -n "$filt" ] && ff=(--config-filter "$filt")
  log "EVAL $tag START $ARM GPU$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u "$QEVAL" \
    --backbone-path "$BB" --head-path "$QF" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 \
    --num-layers 1 --forecaster-d-model 128 --forecaster-n-heads 4 \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 \
    --device cuda --head-causal true "${ff[@]}" \
    >>"$RES/run_eval_${tag}.log" 2>&1 || { log "EVAL $tag FAILED (tail: $(tail -3 "$RES/run_eval_${tag}.log"|tr '\n' ' '))"; return 1; }
  log "EVAL $tag DONE $ARM GM=$(gm "$out/summary.txt")"
}

case "$PHASE" in
  setup)        setup ;;
  backbone)     backbone ;;
  qhead)        qhead ;;
  eval_triage)  do_eval "triage_${ARM}" "$RES/gift_eval_triage_${NAME}" "$TRIAGE" ;;
  eval_full)    do_eval "full_${ARM}"   "$RES/gift_eval_full_${NAME}"   "" ;;
  downstream)   qhead && do_eval "triage_${ARM}" "$RES/gift_eval_triage_${NAME}" "$TRIAGE"; \
                do_eval "full_${ARM}" "$RES/gift_eval_full_${NAME}" "" ; \
                log "DOWNSTREAM COMPLETE $ARM triageGM=$(gm "$RES/gift_eval_triage_${NAME}/summary.txt") fullGM=$(gm "$RES/gift_eval_full_${NAME}/summary.txt")" ;;
  *) echo "bad PHASE $PHASE"; exit 2 ;;
esac
