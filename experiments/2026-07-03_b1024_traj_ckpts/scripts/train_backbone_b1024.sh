#!/bin/bash
# #369 — retrain the #366 τ=0.90 last-ckpt winner at B=1024 with a
# fine-grained trajectory-checkpoint cadence.
#
# Byte-identical to #366's train_backbone_sigreg.sh except for two flags:
#   --batch-size 1024  (was 512)
#   --traj-save-every 500  (new; writes `_step<N>.pth` every 500 steps)
# Seed, τ, λ pair, dataset, dtypes, encoder/decoder shapes stay fixed.
#
#   train_backbone_b1024.sh <gpu> <lambda_e> <lambda_h> <tau> <suffix> [steps] [save_every] [traj_save_every]
set -uo pipefail
GPU="${1:?gpu}"; LAMBDA_E="${2:?lambda_e}"; LAMBDA_H="${3:?lambda_h}"
TAU="${4:?tau}"; SUFFIX="${5:?suffix}"
STEPS="${6:-12500}"; SAVE_EVERY="${7:-2500}"; TRAJ_SAVE_EVERY="${8:-500}"
SEED=20260520
: "${WT:?WT (worktree root containing train.py + experiments/hf_token.txt) must be set; e.g. WT=/home/jupyter/workspaces/contrastive-forecasting}"
: "${OUT:?OUT (per-experiment output dir for runs/ and results/) must be set}"
[ -d "$WT" ] || { echo "[bb-${SUFFIX}] ABORT: WT does not exist: $WT" >&2; exit 2; }
ENC_LAYERS=6; NENC=3
export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
NAME="bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b1024_cpc_${SUFFIX}"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="$GPU"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
# Teacher BiGRU embed under no_grad still allocates; chunk it to fit
# alongside the B=1024 SIGReg pass. Value halved from #366 (B=512, chunk=16)
# since per-step batch doubled — leaves the OP unchanged in absolute rows.
export TEACHER_EMBED_CHUNK="${TEACHER_EMBED_CHUNK:-8}"
HF_TOKEN_PATH="$WT/experiments/hf_token.txt"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [bb-${SUFFIX} g$GPU] $*"; }
[ -f "$TRAIN" ] || { log "ABORT: TRAIN script not found at $TRAIN (check WT=$WT)"; exit 2; }
[ -f "$HF_TOKEN_PATH" ] || { log "ABORT: HF token not found at $HF_TOKEN_PATH — HF stream throttles GPU, refusing to launch"; exit 2; }
export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { log "ABORT: empty HF_TOKEN — HF stream throttles GPU, refusing to launch"; exit 2; }
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
# Resume from the newest coarse `_Nk.pth` (traj `_step<N>.pth` files
# accumulate quickly, so we deliberately prefer the coarse cadence for
# auto-resume — matches #366's ls-based resume idiom).
RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
log "BB START arm=${SUFFIX} λ_e=${LAMBDA_E} λ_h=${LAMBDA_H} τ=${TAU} bs=1024 steps=$STEPS traj_save_every=$TRAJ_SAVE_EVERY ${RESUME}"
python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
  --batch-size 1024 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --traj-save-every "$TRAJ_SAVE_EVERY" \
  --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers "$NENC" --num-layers "$ENC_LAYERS" \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt --pos-in-denominator --subtract-contrastive-floor \
  --ema-embedding --ema-encoder --ema-tau "$TAU" --cpc-infonce-weight 1.0 \
  --sigreg-embedding --sigreg-encoding --sigreg-n-chunk 2048 \
  --sigreg-embedding-weight "$LAMBDA_E" --sigreg-encoding-weight "$LAMBDA_H" \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.0078125 --crossfade-triplets 1 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "BB train exited rc=$rc — NOT creating FINAL. tail: $(tail -3 "$tlog"|tr '\n' ' ')"
  exit 1
fi
# Prefer the deterministic last-step trajectory checkpoint so `_FINAL.pth`
# is always the model at STEPS. Falling back to `_best_loss.pth` (as an
# earlier revision did) silently substituted the best-training-loss step
# — for #369's arm that step was 700, not 12500, which would rewind any
# resume tooling picking up `_FINAL.pth`.
if   [ -f "$RUNS/${NAME}_step${STEPS}.pth" ]; then cp -f "$RUNS/${NAME}_step${STEPS}.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];        then cp -f "$RUNS/${NAME}_final.pth"        "$BB"
elif [ -f "$RUNS/${NAME}_best_loss.pth" ];    then cp -f "$RUNS/${NAME}_best_loss.pth"    "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
[ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; exit 0; }
log "BB FAILED no checkpoint"; exit 1
