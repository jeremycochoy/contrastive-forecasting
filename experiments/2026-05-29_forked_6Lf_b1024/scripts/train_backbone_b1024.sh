#!/bin/bash
# #322 — forked-continuation arms × 6L forecaster, retrained at GLOBAL BATCH 1024.
# Byte-identical to #320's backbone recipe EXCEPT global batch 256 → 1024, run as
# 2-GPU DDP: --batch-size is PER RANK (512) and torchrun --nproc_per_node=2 makes
# the global batch 2×512 = 1024. --shard-loss-on-batch is LEFT OFF (default), so
# train.py all-gathers the per-rank latents and the contrastive loss pools its
# negatives over the FULL 1024-batch (2-GPU @ 512 == 1-GPU @ 1024) — i.e. "all
# terms together in the negatives of the batch" (#322). Gradient equivalence to a
# single-process @1024 run is pinned in tests/test_dist_gather.py.
#
# Usage: train_backbone_b1024.sh <arm> <mix> <tag> <steps> [save_every] [chunk] [port]
#   arm        beta | alltime
#   mix        mix-ratio (0.0078125=2/256, 0.10, 0.5)  — fraction of the batch, scale-invariant
#   tag        run-name fragment (forked2 | forked10pct | forked)
#   steps      --total-steps
#   save_every periodic checkpoint cadence (default 2500)
#   chunk      XSHH_ALLT_CHUNK for the all-time Gram (default 4; alltime only)
#   port       torchrun master port (default 29501)
set -uo pipefail
ARM="${1:?arm=beta|alltime}"; MIX="${2:?mix}"; TAG="${3:?tag}"; STEPS="${4:?steps}"
SAVE_EVERY="${5:-2500}"; export XSHH_ALLT_CHUNK="${6:-4}"; PORT="${7:-29501}"
SEED=20260520
case "$ARM" in
  beta)    SHAPE=cosine_similarity_batch_full_hh_negs;            NAME="bb_beta_${TAG}_6Lf_b1024" ;;
  alltime) SHAPE=cosine_similarity_batch_full_hh_negs_xshh_allt;  NAME="bb_xshh_allt_${TAG}_6Lf_b1024" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
WT=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export CUDA_VISIBLE_DEVICES="0,1"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [b1024 $ARM $TAG] $*"; }
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
# Crash resilience: resume from latest periodic checkpoint (model+opt+step+RNG+data).
RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | head -1)
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
log "BB START shape=$SHAPE per-rank-bs=512 global=1024 (DDP x2) forked-arma mix=$MIX chunk=$XSHH_ALLT_CHUNK steps=$STEPS ${RESUME}"
torchrun --nproc_per_node=2 --master_port="$PORT" "$TRAIN" $RESUME \
  --batch-size 512 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 6 \
  --forecaster-d-model 128 --forecaster-n-heads 4 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape "$SHAPE" --pos-in-denominator \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio "$MIX" \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "BB train exited rc=$rc — NOT creating FINAL (incomplete; --resume next launch). tail: $(tail -3 "$tlog"|tr '\n' ' ')"
  log "BB FAILED rc=$rc"; exit 1
fi
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
if [ -f "$BB" ]; then
  find "$RUNS" -maxdepth 1 -name "${NAME}_*.pth" ! -name "${NAME}_FINAL.pth" -delete 2>/dev/null
  log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1)); pruned intermediates"
  exit 0
fi
log "BB FAILED no checkpoint"; exit 1
