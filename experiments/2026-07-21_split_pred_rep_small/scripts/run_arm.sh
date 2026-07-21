#!/bin/bash
# #379 arm launcher — small backbone × 200k steps on elisa, one script per arm.
# Backbone-only: trains the backbone, saves checkpoints, records training
# dynamics (loss / ff / u_batchtime / u_batchtime_e). No q-head training,
# no GIFT-Eval — the deliverable is the training-dynamics trajectories.
#
# Usage:
#   WT=/absolute/path/to/checkout BB_GPU=1 bash run_arm.sh <arm>
#
# where <arm> ∈ {arm1, arm3, arm4, arm5, arm6_v2, bimoco}. Each arm shares
# the same backbone architecture and training schedule (d_model=64,
# n_heads=8, num_encoder_layers=3, num_layers=3, bs=64, 200k steps,
# save-every=25000, extra snapshot at 2500); only the arm-specific bits
# (run name, human label, extra CLI flags after --loss-shape) come from
# the per-arm case block below. See `README.md` for the parent-experiment
# mapping to #374.
set -uo pipefail

ARM="${1:?usage: run_arm.sh <arm1|arm3|arm4|arm5|arm6_v2|bimoco>}"

# WT MUST be an absolute path under a persistent checkout — never /tmp.
# /tmp gets wiped by reboots and by `git worktree remove --force`
# (CLAUDE.md § Checkpoint Safety Rule 4, Apr-2026 incident). Fall back
# to the documented elisa layout when unset.
WT="${WT:-$HOME/workspaces/contrastive-forecasting}"
case "$WT" in
  /tmp/*|/tmp)
    echo "ABORT: WT=$WT is under /tmp. Set WT to an absolute path under a" >&2
    echo "  persistent checkout (e.g. \$HOME/workspaces/contrastive-forecasting)." >&2
    exit 2
    ;;
esac

OUT="$WT/experiments/2026-07-21_split_pred_rep_small"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"

# ---- Per-arm dispatch --------------------------------------------------------
# NAME:      base run name (all backbone artefacts prefix on it).
# ARM_DESC:  human-readable one-line description used in the BB START log line.
# LOSS_ARGS: extra CLI flags added AFTER --loss-shape (loss-shape itself included).
#
# When adding a 7th arm, add one case here and one entry in the launcher
# shape test (tests/test_small_long_launcher_shape.py). Nothing else in
# this script needs to change.
case "$ARM" in
  arm1)
    NAME="bb_small_arm1_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 1: split_pred_rep"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep)
    ;;
  arm3)
    NAME="bb_small_arm3_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 3: split_pred_rep + moco-negatives"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep --moco-negatives)
    ;;
  arm4)
    NAME="bb_small_arm4_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 4: xshh_allt + moco-negatives"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt \
               --pos-in-denominator --subtract-contrastive-floor --moco-negatives)
    ;;
  arm5)
    NAME="bb_small_arm5_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 5: rep_only + align-loss-weight 1.0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only --align-loss-weight 1.0)
    ;;
  arm6_v2)
    NAME="bb_small_arm6_v2_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 6 v2: rep_only + align-loss-weight 1.0 + moco-rep-keys"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys)
    ;;
  bimoco)
    NAME="bb_small_bimoco_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="bimoco: split_pred_rep + moco-negatives + moco-rep-keys"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep \
               --moco-negatives --moco-rep-keys)
    ;;
  *)
    echo "ABORT: unknown arm '$ARM'" >&2
    echo "  valid: arm1 arm3 arm4 arm5 arm6_v2 bimoco" >&2
    exit 2
    ;;
esac

SEED=20260520
STEPS="${STEPS:-200000}"; SAVE_EVERY="${SAVE_EVERY:-25000}"
EXTRA_SAVES="${EXTRA_SAVES:-2500}"
NENC=3; NLAY=3
BB_GPU="${BB_GPU:-0}"

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
export TEACHER_EMBED_CHUNK="${TEACHER_EMBED_CHUNK:-16}"
HF_TOKEN_PATH="$WT/experiments/hf_token.txt"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
DL_LOG="$RES/dl_${ARM}.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [elisa-$ARM] $*" | tee -a "$DL_LOG"; }
[ -f "$TRAIN" ]  || { log "ABORT: TRAIN not at $TRAIN"; exit 2; }
[ -f "$HF_TOKEN_PATH" ] || { log "ABORT: HF token missing at $HF_TOKEN_PATH"; exit 2; }
export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { log "ABORT: empty HF_TOKEN"; exit 2; }

BB="$RUNS/${NAME}_FINAL.pth"
tlog="$RES/run_${NAME}.log"

if [ -f "$BB" ]; then
  log "BB SKIP ($NAME FINAL exists)"
  exit 0
fi

RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
log "BB START ($ARM_DESC) gpu=$BB_GPU steps=$STEPS bs=64 ${RESUME}"
CUDA_VISIBLE_DEVICES="$BB_GPU" python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
  --batch-size 64 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --extra-save-steps "$EXTRA_SAVES" \
  --save-dir "$RUNS" --run-name "$NAME" --log-every 200 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8 \
  --num-encoder-layers "$NENC" --num-layers "$NLAY" \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  "${LOSS_ARGS[@]}" \
  --ema-embedding --ema-encoder --ema-tau 0.9 --cpc-infonce-weight 1.0 \
  --sigreg-embedding --sigreg-encoding --sigreg-n-chunk 2048 \
  --sigreg-embedding-weight 1.0 --sigreg-encoding-weight 1.0 \
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
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
[ -f "$BB" ] || { log "BB FAILED no checkpoint"; exit 1; }
log "$ARM complete: BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"
