#!/bin/bash
# #309 bottleneck × β2 confound on (B) — box-side runner for one arm.
# Recipe = byte-identical to #303/#307 cl_hh_50k box runner (loss kept
# at (B) = cosine_similarity_batch_full_hh_negs, 2-GPU DDP, fp16 body,
# residual+pemb fp32, dropkey 0.70 shared, 50k @ bs128/GPU).
#
# Per arm only two knobs change vs (B):
#   alpha  : --adam-beta2 0.98 AND forecaster bottleneck removed (default = encoder width)
#   beta   : --adam-beta2 0.98 (bottleneck kept)
#   gamma  : forecaster bottleneck removed (β2 0.95 kept)
#
# Usage (ON BOX):  bash box_run.sh <ARM>
#   ARM = alpha | beta | gamma
set -uo pipefail
APP=/workspace/app
cd "$APP"
ARM="${1:?ARM = alpha|beta|gamma}"
case "$ARM" in
  alpha|beta|gamma) :;;
  *) echo "unknown ARM $ARM"; exit 2 ;;
esac
SEED=20260520
NAME="bb_${ARM}_50k"
TOTAL=50000
RUNS="$APP/runs"; RES="$APP/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"

export PYTHONPATH="$APP" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$APP/experiments/hf_token.txt")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$APP/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [$ARM] $*"; }
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
  python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count(),'x',(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))"
  touch "$M"; log "SETUP done"
}

# Arm-specific knob overrides applied on top of the (B) base recipe.
# alpha = β2=0.98 + no bottleneck; beta = β2=0.98 + bottleneck; gamma = β2=0.95 + no bottleneck.
case "$ARM" in
  alpha) BETA2=0.98; FCST=();;
  beta)  BETA2=0.98; FCST=(--forecaster-d-model 128 --forecaster-n-heads 4);;
  gamma) BETA2=0.95; FCST=();;
esac

backbone(){
  [ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; return 0; }
  setup || { log "SETUP FAILED"; return 1; }
  local tlog="$RES/run_${NAME}.log"
  local ng; ng=$(python3 -c 'import torch;print(torch.cuda.device_count())')
  [ "${ng:-0}" -ge 1 ] || { log "BB ERROR: need ≥1 GPU, have $ng"; return 1; }
  # 2-GPU DDP at bs=128/GPU is the (B) recipe; 1-GPU at bs=256 is
  # mathematically identical (train.py:739-740: gathered-loss runs are
  # "identical to single-GPU runs" — only --shard-loss-on-batch breaks
  # equivalence, and we never use it).
  local LAUNCHER PER_RANK_BS
  if [ "$ng" -ge 2 ]; then
    LAUNCHER=(torchrun --nproc_per_node=2 --master_port="$(freeport)")
    PER_RANK_BS=128
    log "BB START $ARM seed=$SEED β2=$BETA2 fcst-bneck=${FCST[*]:-none} loss=cosine_similarity_batch_full_hh_negs DDP nproc=2 bs128 ${TOTAL}"
    export CUDA_VISIBLE_DEVICES=0,1
  else
    LAUNCHER=(python3 -u)
    PER_RANK_BS=256
    log "BB START $ARM seed=$SEED β2=$BETA2 fcst-bneck=${FCST[*]:-none} loss=cosine_similarity_batch_full_hh_negs 1-GPU bs256 ${TOTAL}"
    export CUDA_VISIBLE_DEVICES=0
  fi
  "${LAUNCHER[@]}" "$TRAIN" \
    --batch-size "$PER_RANK_BS" --device cuda --total-steps "$TOTAL" --lr 1e-3 --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 "$BETA2" --seed "$SEED" \
    --save-every 5000 --save-dir "$RUNS" --run-name "$NAME" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
    --num-encoder-layers 6 --num-layers 1 \
    "${FCST[@]}" \
    --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 \
    --loss-shape cosine_similarity_batch_full_hh_negs --pos-in-denominator \
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
  for c in "${NAME}_best_loss" "${NAME}_final"; do
    [ -f "$RUNS/${c}_optimizer.pth" ] && cp -f "$RUNS/${c}_optimizer.pth" "$RUNS/${NAME}_FINAL_optimizer.pth" && break
  done
  [ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; return 0; }
  log "BB FAILED no checkpoint"; return 1
}

backbone
