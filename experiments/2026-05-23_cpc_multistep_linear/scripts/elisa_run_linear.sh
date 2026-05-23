#!/bin/bash
# #316 study arms #2/#3: LINEAR-head CPC multi-step (forecaster_kind=linear_cpc,
# K linear heads W_k: H->H on the encoder, no bottleneck). β recipe otherwise,
# fp32, seed-controlled, 50k, bs256.
#   elisa_run_linear.sh <seed> <gpu> <K> <negs>
#     negs = beta  -> loss cpc_multistep            (#2: β's negatives)
#            cpcneg -> loss cpc_multistep_cpcnegs   (#3: original CPC-canonical negs)
set -uo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; K="${3:?K}"; NEGS="${4:?beta|cpcneg}"
case "$NEGS" in
  beta)   LOSS=cpc_multistep;          TAG=bn ;;
  cpcneg) LOSS=cpc_multistep_cpcnegs;  TAG=cn ;;
  *) echo "unknown negs $NEGS"; exit 2 ;;
esac
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-23_cpc_multistep_linear
NAME="bb_lin${TAG}_k${K}_s${SEED}_fp32_50k"
TOTAL=50000
RUNS="$MAIN/runs"; RES="$MAIN/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null || cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [lin-$TAG-k$K s$SEED g$GPU] $*"; }
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
export CUDA_VISIBLE_DEVICES="$GPU"
log "BB START linear_cpc K=$K negs=$NEGS ($LOSS) β2=0.98 τ=0.10 fp32 bs256 GPU=$GPU -> $TOTAL"
python3 -u "$TRAIN" \
  --batch-size 256 --device cuda --total-steps "$TOTAL" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every 5000 --save-dir "$RUNS" --run-name "$NAME" \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 1 \
  --forecaster-kind linear_cpc --cpc-k-steps "$K" \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape "$LOSS" --pos-in-denominator \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --mixup-p 0.3 --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 --conv-dtype fp32 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "BB train exited rc=$rc (tail: $(tail -3 "$tlog"|tr '\n' ' '))"; fi
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
[ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth"; exit 0; }
log "BB FAILED no checkpoint"; exit 1
