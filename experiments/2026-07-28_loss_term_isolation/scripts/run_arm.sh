#!/bin/bash
# #382 — single-term isolation on the small #379 backbone.
#
# Usage: run_arm.sh <arm_slug> <gpu> [total_steps] [save_every]
#
# One CLI flag per arm keeps the launcher trivial. Every arm shares the
# small-backbone architecture from #379 (d_model=64, n_heads=8, 3 encoder /
# 3 forecaster layers, batch=64, seed=20260520, dataset small_v1). What
# differs is exactly one active loss term, wired up via #382's new
# per-term weights (pred/rep) or via --no-main-contrastive-loss plus one
# of --align/--cpc/--sigreg-*.
#
# WT is the checked-out worktree root — same convention as
# 2026-07-10_split_pred_rep/scripts/train_backbone_split_pred_rep.sh.
set -uo pipefail

ARM="${1:?arm slug (pred|rep|align|pred_moco|rep_moco|sigreg_e|sigreg_h|cpc)}"
GPU="${2:?gpu index}"
STEPS="${3:-100000}"
SAVE_EVERY="${4:-5000}"
SEED=20260520

WT="${WT:?WT (worktree root) must be set; e.g. WT=/workspace/cf-382}"
OUT="${OUT:-$WT/experiments/2026-07-28_loss_term_isolation}"
RUNS="$OUT/artifacts/$ARM"
RES="$OUT/results"
mkdir -p "$RUNS" "$RES"

# Small backbone (same as #379) — every arm shares this arch.
# --t-raw 1024: the experiments-branch HFStreamingLoader crops rows at the
# module constant T_RAW=1024 regardless of the CLI value, so --t-raw 4096
# would silently downgrade AND size-mismatch against the synth mix. When
# the #379 dataloader fix (respect args.t_raw on HF too) lands on
# experiments, bump this to 4096 to match the issue's T spec.
ARCH=(--t-raw 1024 --n-channels 1 --d-model 64 --n-heads 8
      --num-encoder-layers 3 --num-layers 3
      --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru
      --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1)

TRAIN_HYP=(--batch-size 64 --lr 1e-3 --weight-decay 0.1
           --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED"
           --total-steps "$STEPS" --save-every "$SAVE_EVERY"
           --log-every 100 --tau 0.10)

# Per-arm loss wiring. Only exactly one term contributes to backprop.
case "$ARM" in
  pred)
    ARM_FLAGS=(--loss-shape cosine_similarity_batch_split_pred_rep
               --pred-loss-weight 1.0 --rep-loss-weight 0.0)
    ;;
  rep)
    ARM_FLAGS=(--loss-shape cosine_similarity_batch_split_pred_rep
               --pred-loss-weight 0.0 --rep-loss-weight 1.0)
    ;;
  align)
    ARM_FLAGS=(--loss-shape cosine_similarity_batch_no_time_neg
               --no-main-contrastive-loss --align-loss-weight 1.0)
    ;;
  pred_moco)
    ARM_FLAGS=(--loss-shape cosine_similarity_batch_split_pred_rep
               --pred-loss-weight 1.0 --rep-loss-weight 0.0
               --ema-embedding --ema-encoder --ema-tau 0.9 --moco-negatives)
    ;;
  rep_moco)
    ARM_FLAGS=(--loss-shape cosine_similarity_batch_split_pred_rep
               --pred-loss-weight 0.0 --rep-loss-weight 1.0
               --ema-embedding --ema-encoder --ema-tau 0.9 --moco-rep-keys)
    ;;
  sigreg_e)
    ARM_FLAGS=(--loss-shape cosine_similarity_batch_no_time_neg
               --no-main-contrastive-loss
               --sigreg-embedding --sigreg-embedding-weight 1.0
               --sigreg-n-chunk 2048)
    ;;
  sigreg_h)
    ARM_FLAGS=(--loss-shape cosine_similarity_batch_no_time_neg
               --no-main-contrastive-loss
               --sigreg-encoding --sigreg-encoding-weight 1.0
               --sigreg-n-chunk 2048)
    ;;
  cpc)
    ARM_FLAGS=(--loss-shape cosine_similarity_batch_no_time_neg
               --no-main-contrastive-loss --cpc-infonce-weight 1.0)
    ;;
  *)
    echo "unknown arm slug: $ARM" >&2; exit 2 ;;
esac

NAME="lti_${ARM}"
BB="$RUNS/${NAME}_FINAL.pth"
[ -f "$BB" ] && { echo "[$(date +%m-%d-%H:%M)] SKIP $ARM ($NAME FINAL exists)"; exit 0; }

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="$GPU"
export XSHH_ALLT_CHUNK="${XSHH_ALLT_CHUNK:-1}"
export CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { echo "ABORT: empty HF_TOKEN (create $WT/experiments/hf_token.txt)"; exit 1; }

TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
tlog="$RES/run_${NAME}.log"

# Auto-resume from the most recent checkpoint of this arm, if any.
latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
RESUME_FLAG=()
[ -n "$latest" ] && { RESUME_FLAG=(--resume "$latest"); echo "[$(date +%m-%d-%H:%M)] RESUME from $(basename "$latest")"; }

echo "[$(date +%m-%d-%H:%M)] START arm=$ARM steps=$STEPS save_every=$SAVE_EVERY gpu=$GPU"
python3 -u "$TRAIN" "${RESUME_FLAG[@]}" \
    "${ARCH[@]}" "${TRAIN_HYP[@]}" "${ARM_FLAGS[@]}" \
    --save-dir "$RUNS" --run-name "$NAME" \
    --device cuda \
    >>"$tlog" 2>&1
rc=$?

if [ $rc -ne 0 ]; then
  echo "[$(date +%m-%d-%H:%M)] arm=$ARM exited rc=$rc — tail: $(tail -3 "$tlog"|tr '\n' ' ')"
  exit 1
fi

# Materialise a FINAL symlink target so downstream tooling can find each
# arm's checkpoint by a stable name.
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null || true; fi

[ -f "$BB" ] && { echo "[$(date +%m-%d-%H:%M)] DONE arm=$ARM ($(du -h "$BB"|cut -f1))"; exit 0; }
echo "[$(date +%m-%d-%H:%M)] FAIL arm=$ARM: no checkpoint produced"; exit 1
