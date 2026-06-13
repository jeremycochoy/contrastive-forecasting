#!/bin/bash
# #328 follow-up: continue the best arm (#328: L3 + no-bottleneck + triplet) from
# step 12500 to a longer total (default 25000 = +1 base-length), to test whether
# the downstream forecast keeps improving with more training/data. Resumes the FULL
# training state (step, optimizer momentum, RNG, and data position hf_rows_consumed)
# so it streams NEW data for the extra steps — equivalent to a single 25000-step run.
#   extend_triplet.sh <gpu> [total_steps] [save_every]
set -uo pipefail
GPU="${1:?gpu}"; TOTAL="${2:-25000}"; SAVE_EVERY="${3:-2500}"
WT="${WT:-/tmp/cf-328}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-03_crossfade_triplet}"
SRCNAME="bb_allt08_xftrip_nobn_enc3_qk_aon_b1024"           # original #328 (step 12500)
NAME="${SRCNAME}_to${TOTAL}"                                # extension run
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export XSHH_ALLT_CHUNK="${XSHH_ALLT_CHUNK:-2}" CUDA_VISIBLE_DEVICES="$GPU"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK="${PATCH_ENC_CHUNK:-4}"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [ext g$GPU] $*"; }
[ -f "$BB" ] && { log "SKIP ($NAME FINAL exists)"; exit 0; }
# Resume from the extension's own latest periodic (crash recovery), else the original #328 step-12500.
latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | head -1)
if [ -n "$latest" ]; then RES_FROM="$latest"; else RES_FROM="$RUNS/${SRCNAME}_12k.pth"; fi
[ -f "$RES_FROM" ] || { log "ABORT no resume checkpoint: $RES_FROM"; exit 1; }
log "EXT START resume=$(basename "$RES_FROM") -> total=$TOTAL"
python3 -u "$TRAIN" --resume "$RES_FROM" --qk-norm --attn-out-norm \
  --batch-size 1024 --device cuda --total-steps "$TOTAL" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed 20260520 \
  --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 3 --num-layers 6 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt --pos-in-denominator --subtract-contrastive-floor \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.0078125 --crossfade-triplets 1 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$RES/run_${NAME}.log" 2>&1
rc=$?
[ $rc -ne 0 ] && { log "EXT exited rc=$rc — NOT creating FINAL (--resume next launch). tail: $(tail -3 "$RES/run_${NAME}.log"|tr '\n' ' ')"; exit 1; }
# Keep best-loss as FINAL (primary) and final.pth as the step-$TOTAL full-training checkpoint.
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
log "EXT DONE -> ${NAME}_FINAL.pth"
