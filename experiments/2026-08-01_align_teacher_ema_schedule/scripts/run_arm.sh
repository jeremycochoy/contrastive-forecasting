#!/bin/bash
# #388 — teacher-target L_align and the 0.9 -> 1.0 EMA-momentum schedule.
#
# Usage: run_arm.sh <arm_slug> <gpu> [total_steps] [save_every]
#
# Same backbone, seed, dataset and 100k-step budget as #382
# (experiments/2026-07-28_loss_term_isolation/scripts/run_arm.sh), so the
# curves overlay directly. What changes per arm is the loss wiring and the
# EMA momentum α.
#
#   align_teacher_a09    L_align vs the EMA teacher's h_{t+1}, α = 0.9
#   align_teacher_sched  same, α linear 0.9 -> 1.0 over the budget
#   pred_moco_sched      #382's pred_moco, α linear 0.9 -> 1.0
#   rep_moco_sched       #382's rep_moco,  α linear 0.9 -> 1.0
#
# The α = 0.9 halves of pred_moco / rep_moco already exist in #382 and are
# not re-run. Neither are pred, rep, sigreg_e, sigreg_h, cpc — none has a
# teacher.
#
# Checkpoints land OUTSIDE the worktree (checkpoint-safety rule 4: `git
# worktree remove --force` deletes untracked files). WT is the checked-out
# worktree root; RUNS defaults to the durable backup tree.
set -uo pipefail

ARM="${1:?arm slug (align_teacher_a09|align_teacher_sched|pred_moco_sched|rep_moco_sched)}"
GPU="${2:?gpu index}"
STEPS="${3:-100000}"
SAVE_EVERY="${4:-5000}"
PROBE_EVERY="${PROBE_EVERY:-500}"
SEED=20260520

WT="${WT:?WT (worktree root) must be set; e.g. WT=/tmp/contrastive-forecasting-388}"
OUT="${OUT:-$WT/experiments/2026-08-01_align_teacher_ema_schedule}"
RUNS="${RUNS:-/home/jupyter/checkpoints_backup/cf-388}/$ARM"
RES="$OUT/results"
mkdir -p "$RUNS" "$RES"

# Small #379 backbone — identical to #382's ARCH block, including the
# --t-raw 1024 note (the experiments-branch HFStreamingLoader crops at the
# module constant T_RAW=1024 regardless of the CLI value).
ARCH=(--t-raw 1024 --n-channels 1 --d-model 64 --n-heads 8
      --num-encoder-layers 3 --num-layers 3
      --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru
      --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1)

TRAIN_HYP=(--batch-size 64 --lr 1e-3 --weight-decay 0.1
           --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED"
           --total-steps "$STEPS" --save-every "$SAVE_EVERY"
           --log-every 100 --tau 0.10
           --latent-drift-probe-every "$PROBE_EVERY")

# Every arm here has an EMA teacher; α starts at 0.9 in all four.
EMA=(--ema-embedding --ema-encoder --ema-tau 0.9)

case "$ARM" in
  align_teacher_a09)
    ARM_FLAGS=(--loss-shape cosine_similarity_batch_no_time_neg
               --no-main-contrastive-loss --align-loss-weight 1.0
               --align-target teacher "${EMA[@]}")
    ;;
  align_teacher_sched)
    ARM_FLAGS=(--loss-shape cosine_similarity_batch_no_time_neg
               --no-main-contrastive-loss --align-loss-weight 1.0
               --align-target teacher "${EMA[@]}" --ema-tau-end 1.0)
    ;;
  pred_moco_sched)
    ARM_FLAGS=(--loss-shape cosine_similarity_batch_split_pred_rep
               --pred-loss-weight 1.0 --rep-loss-weight 0.0
               "${EMA[@]}" --ema-tau-end 1.0 --moco-negatives)
    ;;
  rep_moco_sched)
    ARM_FLAGS=(--loss-shape cosine_similarity_batch_split_pred_rep
               --pred-loss-weight 0.0 --rep-loss-weight 1.0
               "${EMA[@]}" --ema-tau-end 1.0 --moco-rep-keys)
    ;;
  *)
    echo "unknown arm slug: $ARM" >&2; exit 2 ;;
esac

NAME="ats_${ARM}"
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

echo "[$(date +%m-%d-%H:%M)] START arm=$ARM steps=$STEPS save_every=$SAVE_EVERY probe_every=$PROBE_EVERY gpu=$GPU"
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

# Stable name for the last checkpoint, so downstream tooling never has to
# guess (checkpoint-safety rule 1).
if   [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
elif [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null || true; fi

[ -f "$BB" ] && { echo "[$(date +%m-%d-%H:%M)] DONE arm=$ARM ($(du -h "$BB"|cut -f1))"; exit 0; }
echo "[$(date +%m-%d-%H:%M)] FAIL arm=$ARM: no checkpoint produced"; exit 1
