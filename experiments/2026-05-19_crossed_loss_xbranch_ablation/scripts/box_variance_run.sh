#!/bin/bash
# Vast.ai box-side variance runner — backbone ONLY for one (arm, seed)
# pair. Recipe byte-identical to box_run.sh (#307 of-record); only --seed
# differs. Downstream (q-head + GIFT-Eval) runs on free elisa GPU from
# the synced backbone, NOT on vast.
#
# Usage (ON BOX):  bash box_variance_run.sh <ARM_SHORT> <SEED>
#   ARM_SHORT = hh | hhxbf
#   SEED      = 20260518 | 20260519 | ...
set -uo pipefail
APP=/workspace/app
cd "$APP"
ARM="${1:?ARM_SHORT}"; SEED="${2:?SEED}"
SHORT_SEED="s${SEED:(-2)}"
NAME="cl_${ARM}_50k_${SHORT_SEED}"
TOTAL=50000
RUNS="$APP/runs"; RES="$APP/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"

declare -A LS=(
  [hh]=cosine_similarity_batch_full_hh_negs
  [hhxbf]=cosine_similarity_batch_full_hh_negs_xbfree
)
[ -n "${LS[$ARM]:-}" ] || { echo "unknown ARM $ARM"; exit 2; }

export PYTHONPATH="$APP" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$APP/experiments/hf_token.txt")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$APP/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [$ARM seed=$SEED] $*"; }
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

backbone(){
  [ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; return 0; }
  setup || { log "SETUP FAILED"; return 1; }
  local tlog="$RES/run_${NAME}.log"
  local ng; ng=$(python3 -c 'import torch;print(torch.cuda.device_count())')
  [ "${ng:-0}" -ge 2 ] || { log "BB ERROR: need 2 GPUs, have $ng"; return 1; }
  log "BB START $ARM seed=$SEED loss=${LS[$ARM]} DDP nproc=2 bs128 ${TOTAL}"
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
  for c in "${NAME}_best_loss" "${NAME}_final"; do
    [ -f "$RUNS/${c}_optimizer.pth" ] && cp -f "$RUNS/${c}_optimizer.pth" "$RUNS/${NAME}_FINAL_optimizer.pth" && break
  done
  [ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; return 0; }
  log "BB FAILED no checkpoint"; return 1
}

backbone
