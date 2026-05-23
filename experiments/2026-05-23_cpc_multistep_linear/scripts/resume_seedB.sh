#!/bin/bash
# Recover seed B (20260523) after it was stopped at ~step 32.7k: resume from the
# last periodic checkpoint (step 30k, with optimizer + RNG) to 50k on GPU1.
# train.py's safe_run_name branches the resumed run to <NAME>_r2; on completion
# we promote its best_loss to the standard <NAME>_FINAL.pth so downstream finds it.
set -uo pipefail
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-23_cpc_multistep_linear
R="$MAIN/runs"; RES="$MAIN/results"
NAME=bb_cpc_k12_s20260523_fp32_50k
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1
export HF_TOKEN="$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [resume-sB] $*"; }
log "resume from ${NAME}_30k.pth -> 50k on GPU1"
python3 -u "$TRAIN" --resume "$R/${NAME}_30k.pth" \
  --batch-size 256 --device cuda --total-steps 50000 --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed 20260523 \
  --save-every 5000 --save-dir "$R" --run-name "$NAME" \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 1 --forecaster-kind linear_cpc --cpc-k-steps 12 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 --loss-shape cpc_multistep --pos-in-denominator \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --mixup-p 0.3 --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 --conv-dtype fp32 --patch-emb-dtype fp32 \
  >> "$RES/run_${NAME}_resume.log" 2>&1
rc=$?
log "resume train exited rc=$rc"
RUN2="${NAME}_r2"
if   [ -f "$R/${RUN2}_best_loss.pth" ]; then cp -f "$R/${RUN2}_best_loss.pth" "$R/${NAME}_FINAL.pth"
elif [ -f "$R/${RUN2}_final.pth" ];     then cp -f "$R/${RUN2}_final.pth"     "$R/${NAME}_FINAL.pth"; fi
# Stitch a combined losses CSV (orig 1..~32.7k, then _r2 from 30k) for the curve.
if [ -f "$R/${NAME}_losses.csv" ]; then
  awk -F, 'NR==1{print;next} $1<30000{print}' "$R/${NAME}_losses.csv" > "$R/${NAME}_combined_losses.csv"
  tail -n +2 "$R/${RUN2}_losses.csv" 2>/dev/null >> "$R/${NAME}_combined_losses.csv"
fi
[ -f "$R/${NAME}_FINAL.pth" ] && log "seed B FINAL promoted ($(du -h "$R/${NAME}_FINAL.pth"|cut -f1))" || log "seed B FINAL MISSING"
