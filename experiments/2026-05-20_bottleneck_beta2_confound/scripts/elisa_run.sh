#!/bin/bash
# #309 follow-up — fresh α/β/γ training on elisa at a configurable τ.
# Same recipe as the original arms (loss hh_negs, dropkey 0.70 shared,
# fp16 body / fp32 residual+pemb, seed 20260520, 50k); only --tau and
# the per-arm β2 / forecaster-bottleneck flags vary. Auto-uses DDP when
# ≥2 GPUs are visible (global batch 256), else 1-GPU bs256 (identical
# loss per train.py:739-740).
#
# Code from the WORKTREE; outputs to the MAIN checkout (CLAUDE.md).
#
# Usage:  elisa_run.sh <arm> <tau> <runtag> [gpus]
#   arm    = alpha | beta | gamma
#   tau    = e.g. 0.80
#   runtag = name tag, e.g. tau08  ->  bb_<arm>_<runtag>_50k
#   gpus   = CUDA_VISIBLE_DEVICES (default "0,1")
set -uo pipefail
ARM="${1:?arm = alpha|beta|gamma}"; TAU="${2:?tau}"; RUNTAG="${3:?runtag}"
GPUS="${4:-0,1}"; PREC="${5:-fp16}"   # fp16 = (B) recipe body; fp32 = all-fp32 (no-bneck arms diverge in fp16)
case "$ARM" in
  alpha) BETA2=0.98; FCST=() ;;
  beta)  BETA2=0.98; FCST=(--forecaster-d-model 128 --forecaster-n-heads 4) ;;
  gamma) BETA2=0.95; FCST=() ;;
  bbase) BETA2=0.95; FCST=(--forecaster-d-model 128 --forecaster-n-heads 4) ;;  # (B) baseline: bneck + β2=0.95
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
case "$PREC" in
  fp16) DT=(--residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32) ;;
  fp32) DT=(--residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 --conv-dtype fp32 --patch-emb-dtype fp32) ;;
  *) echo "unknown prec $PREC (fp16|fp32)"; exit 2 ;;
esac
SEED=20260520
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-20_bottleneck_beta2_confound
NAME="bb_${ARM}_${RUNTAG}_50k"
TOTAL=50000
RUNS="$MAIN/runs"; RES="$MAIN/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null || cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [${ARM}-${RUNTAG}] $*"; }
freeport(){ python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
export CUDA_VISIBLE_DEVICES="$GPUS"
ng=$(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null)
if [ "${ng:-0}" -ge 2 ]; then
  LAUNCHER=(torchrun --nproc_per_node=2 --master_port="$(freeport)"); PER_RANK_BS=128
  log "BB START τ=$TAU β2=$BETA2 bneck=${FCST[*]:-none} prec=$PREC DDP nproc=2 bs128 GPUs=$GPUS -> $TOTAL"
else
  LAUNCHER=(python3 -u); PER_RANK_BS=256
  log "BB START τ=$TAU β2=$BETA2 bneck=${FCST[*]:-none} prec=$PREC 1-GPU bs256 GPUs=$GPUS -> $TOTAL"
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
  --tau "$TAU" --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --mixup-p 0.3 --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  "${DT[@]}" >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "BB train exited rc=$rc (tail: $(tail -3 "$tlog"|tr '\n' ' '))"; fi
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
for c in "${NAME}_best_loss" "${NAME}_final"; do
  [ -f "$RUNS/${c}_optimizer.pth" ] && cp -f "$RUNS/${c}_optimizer.pth" "$RUNS/${NAME}_FINAL_optimizer.pth" && break
done
[ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; exit 0; }
log "BB FAILED no checkpoint"; exit 1
