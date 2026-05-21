#!/bin/bash
# #309 fp32 continuation — resume a diverged arm from its pre-divergence
# best_loss checkpoint and continue to 50k in ALL-fp32 (fp16/bf16 body
# diverges for the bottleneck-removed arms, so the continuation runs the
# whole transformer body in fp32). Only the dtype flags change vs the
# original arm recipe; everything else (no bottleneck, β2, dropkey, loss,
# 1-GPU bs256) is identical.
#
# The resume checkpoint + its optimizer companion must be staged at
#   /workspace/app/resume/<NAME>_best_loss.pth (+ _optimizer.pth)
#
# Usage (ON BOX):  bash box_continue_fp32.sh <arm>
#   arm = alpha | gamma   (the two diverged arms)
set -uo pipefail
APP=/workspace/app
cd "$APP"
ARM="${1:?arm = alpha|gamma}"
case "$ARM" in
  alpha) BETA2=0.98 ;;
  gamma) BETA2=0.95 ;;
  *) echo "unknown arm $ARM (alpha|gamma)"; exit 2 ;;
esac
SEED=20260520
SRC="bb_${ARM}_50k"                 # original diverged run name
NAME="bb_${ARM}_fp32cont_50k"       # continuation run name (distinct save path)
TOTAL=50000
RUNS="$APP/runs"; RES="$APP/results"; mkdir -p "$RUNS" "$RES"
RESUME="$APP/resume/${SRC}_best_loss.pth"
BB="$RUNS/${NAME}_FINAL.pth"

export PYTHONPATH="$APP" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$APP/experiments/hf_token.txt")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$APP/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [${ARM}-fp32cont] $*"; }
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
  python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())"
  touch "$M"; log "SETUP done"
}

backbone(){
  [ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; return 0; }
  [ -f "$RESUME" ] || { log "BB ERROR: resume checkpoint $RESUME missing"; return 1; }
  [ -f "${RESUME%.pth}_optimizer.pth" ] || { log "BB ERROR: optimizer companion missing"; return 1; }
  setup || { log "SETUP FAILED"; return 1; }
  local tlog="$RES/run_${NAME}.log"
  local ng; ng=$(python3 -c 'import torch;print(torch.cuda.device_count())')
  [ "${ng:-0}" -ge 1 ] || { log "BB ERROR: need ≥1 GPU, have $ng"; return 1; }
  local LAUNCHER PER_RANK_BS
  if [ "$ng" -ge 2 ]; then
    LAUNCHER=(torchrun --nproc_per_node=2 --master_port="$(freeport)"); PER_RANK_BS=128
    export CUDA_VISIBLE_DEVICES=0,1
    log "BB START $ARM fp32cont resume=$RESUME β2=$BETA2 ALL-fp32 DDP bs128 -> $TOTAL"
  else
    LAUNCHER=(python3 -u); PER_RANK_BS=256
    export CUDA_VISIBLE_DEVICES=0
    log "BB START $ARM fp32cont resume=$RESUME β2=$BETA2 ALL-fp32 1-GPU bs256 -> $TOTAL"
  fi
  "${LAUNCHER[@]}" "$TRAIN" \
    --resume "$RESUME" \
    --batch-size "$PER_RANK_BS" --device cuda --total-steps "$TOTAL" --lr 1e-3 --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 "$BETA2" --seed "$SEED" \
    --save-every 5000 --save-dir "$RUNS" --run-name "$NAME" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
    --num-encoder-layers 6 --num-layers 1 \
    --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 \
    --loss-shape cosine_similarity_batch_full_hh_negs --pos-in-denominator \
    --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
    --mixup-p 0.3 --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 \
    --log-attn-amplitude --log-attn-amplitude-every 200 \
    --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 --conv-dtype fp32 \
    --patch-emb-dtype fp32 >>"$tlog" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "BB train exited rc=$rc (tail: $(tail -3 "$tlog"|tr '\n' ' '))"; fi
  if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
  elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
  else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
  for c in "${NAME}_best_loss" "${NAME}_final"; do
    [ -f "$RUNS/${c}_optimizer.pth" ] && cp -f "$RUNS/${c}_optimizer.pth" "$RUNS/${NAME}_FINAL_optimizer.pth" && break
  done
  [ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; return 0; }
  log "BB FAILED no checkpoint"; return 1
}

backbone
