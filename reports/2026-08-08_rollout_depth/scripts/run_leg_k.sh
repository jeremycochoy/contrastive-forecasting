#!/bin/bash
# #393 — one leg of one cell's backbone ladder.
#
# Usage: run_leg.sh <cell_slug> <target_steps>
#        WT=<checkout> [RUNS=<checkpoint dir>] [BB_GPU=0] bash run_leg.sh ...
#
# Each cell is ONE continuous run. A leg trains from the cell's newest
# checkpoint (with its saved optimizer state) up to <target_steps>, then
# exits. The ladder driver calls it once per stop: 40k, 100k, 200k, ...
#
# The recipe is #379's (experiments/2026-07-21_split_pred_rep_small/
# scripts/run_arm.sh) so the numbers compare to both parent reports. The
# one change is the EMA momentum: α rises linearly 0.9 -> 1.0 over steps
# 0..100k and holds at 1.0 after, anchored to the step count rather than
# to the leg's budget. Without the anchor each leg would ramp α across its
# own --total-steps and every stop would sit on a different curve.
#
# Cells never resume a #379 or #388 checkpoint. α matches those runs at
# step 0 only; it rises from step 1, so their trajectories are not on this
# schedule. Every cell starts fresh at step 0, once.
set -uo pipefail

CELL="${1:?usage: run_leg.sh <cell_slug> <target_steps>}"
TARGET_STEPS="${2:?usage: run_leg.sh <cell_slug> <target_steps>}"

WT="${WT:-$HOME/workspaces/contrastive-forecasting}"
OUT="$WT/reports/2026-08-08_rollout_depth"
RES="$OUT/results"
# Durable root, per-leg save dirs and step-ordered checkpoint lookup all
# come from one place — see the header of leg_paths.sh for why each of the
# three is not the obvious thing.
. "$(dirname "${BASH_SOURCE[0]}")/leg_paths.sh"
# Waits for a free device on an Exclusive_Process GPU instead of dying at
# step 0 with "CUDA-capable device(s) is/are busy". No-op on elisa, which
# shares each 4090 between two cells deliberately.
. "$(dirname "${BASH_SOURCE[0]}")/gpu_gate.sh"
ROOT="$(runs_root)" || exit 2
RUNS="$ROOT/$CELL"
LEG="$(leg_dir "$RUNS" "$TARGET_STEPS")"
mkdir -p "$LEG" "$RES"

# ---- Per-cell dispatch -------------------------------------------------------
# LOSS_ARGS: the #379 arm's loss flags, verbatim.
# ALIGN_ARGS: L_align's target for this cell (empty for the two cells whose
#             recipe carries no L_align term).
# EXTRA_ARGS: appended LAST so a repeated flag overrides the shared default
#             (argparse keeps the last value on repeat).
ALIGN_ARGS=()
case "$CELL" in
  arm6_v2_combab_alignS)
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys --tau-rep 1.0)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ALIGN_ARGS=(--align-target student)
    ;;
  arm6_v2_combab_alignT)
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys --tau-rep 1.0)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ALIGN_ARGS=(--align-target teacher)
    ;;
  arm5_combab_alignS)
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --tau-rep 1.0)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ALIGN_ARGS=(--align-target student)
    ;;
  arm5_combab_alignT)
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --tau-rep 1.0)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ALIGN_ARGS=(--align-target teacher)
    ;;
  arm6_v2_ncpc_alignS)
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ALIGN_ARGS=(--align-target student)
    ;;
  arm6_v2_ncpc_alignT)
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ALIGN_ARGS=(--align-target teacher)
    ;;
  arm6_v2_nse_alignS)
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys)
    EXTRA_ARGS=(--sigreg-embedding-weight 0.0)
    ALIGN_ARGS=(--align-target student)
    ;;
  arm6_v2_nse_alignT)
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys)
    EXTRA_ARGS=(--sigreg-embedding-weight 0.0)
    ALIGN_ARGS=(--align-target teacher)
    ;;
  arm4_combab)
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt \
               --pos-in-denominator --subtract-contrastive-floor \
               --moco-negatives --tau 1.0)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0 --sigreg-embedding-weight 0.0)
    ;;
  arm1_nse)
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep)
    EXTRA_ARGS=(--sigreg-embedding-weight 0.0)
    ;;
  *)
    echo "ABORT: unknown cell '$CELL'" >&2
    exit 2 ;;
esac

SEED="${SEED:-20260520}"
# #373: rollout depth. Every cell of this study takes it from the SHARED
# flag block below, never from EXTRA_ARGS.
K="${K:-3}"
LOG_EVERY="${LOG_EVERY:-200}"
# #373 review gaps: extra trainer flags for the control runs. Empty for
# every cell of the card.
read -r -a GAP_ARGS_ARR <<<"${GAP_ARGS:-}"
# Stops are 40k / 100k / 200k / 300k ..., all multiples of 20000, so the
# periodic save always lands one. EXTRA_SAVES covers an off-cadence target.
SAVE_EVERY="${SAVE_EVERY:-20000}"
EXTRA_SAVES="${EXTRA_SAVES:-$TARGET_STEPS}"
BB_GPU="${BB_GPU:-0}"

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
export TEACHER_EMBED_CHUNK="${TEACHER_EMBED_CHUNK:-16}"

TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
HF_TOKEN_PATH="$WT/experiments/hf_token.txt"
NAME="cf393_${CELL}_cf373k${K}${RUN_SUFFIX:-}"
tlog="$RES/run_${NAME}.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [$CELL] $*" | tee -a "$RES/leg_${CELL}.log"; }

[ -f "$TRAIN" ] || { log "ABORT: TRAIN not at $TRAIN"; exit 2; }
[ -f "$HF_TOKEN_PATH" ] || { log "ABORT: HF token missing at $HF_TOKEN_PATH"; exit 2; }
export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { log "ABORT: empty HF_TOKEN"; exit 2; }

# Idempotent: a leg whose target checkpoint already exists is a no-op, so a
# re-fired stop after a crash costs nothing.
target_k=$(( TARGET_STEPS / 1000 ))
done_ckpt="$(ckpt_at_step "$RUNS" "$NAME" "$target_k")"
[ -n "$done_ckpt" ] && {
  log "SKIP: $(basename "$done_ckpt") already on disk"; exit 0; }

# Resume from the cell's FURTHEST checkpoint, chosen by the step in its
# name. `ls -t` would pick by mtime, which a checkpoint set copied between
# machines reorders.
RESUME=()
latest="$(newest_ckpt "$RUNS" "$NAME")"
if [ -n "$latest" ]; then
  RESUME=(--resume "$latest")
  log "RESUME from $(basename "$latest") (step $(ckpt_step_k "$latest")k)"
else
  log "FRESH start at step 0"
fi

# Which machine owns this cell. Ten cells run on seven machines and a cell
# must exist on exactly one of them: two copies write the same
# `<cell>/leg_<N>k/` names into two roots that the sync loops then merge,
# and the second copy's checkpoints are indistinguishable from the first's.
#
# This is not hypothetical. `ladder.py` reaches HOLD_ABOVE by RETURNING
# from climb() — deliberately, so the cells behind it in a `--cells a,b`
# invocation still run — so the elisa driver started as
# `--cells arm6_v2_nse_alignS,arm4_combab` would begin arm4_combab the
# moment nse_alignS stopped at 100k, while arm4_combab was already six
# hours into the same cell on vast F.
#
# The check lives here rather than in ladder.py because run_leg.sh is a new
# process on every leg, so it reaches a driver that has been running for
# hours; ladder.py in memory is whatever was on disk when it started. A
# non-zero exit is what stops the driver: ladder.py raises SystemExit on a
# failed leg, so the duplicate cell takes the driver down with it instead
# of moving on to the next cell in the list.
# `results/MACHINE` names this box, because a vast container's `hostname`
# is a per-boot ID that no committed file can be written against.
CLAIMS="$RES/cell_claims.txt"
if [ -f "$CLAIMS" ]; then
  owner="$(awk -v c="$CELL" '$1==c {print $2}' "$CLAIMS" | head -1)"
  me="$(cat "$RES/MACHINE" 2>/dev/null | tr -dc 'a-zA-Z0-9_-')"
  [ -n "$me" ] || me="${CF393_MACHINE:-$(hostname)}"
  if [ -n "$owner" ] && [ "$owner" != "$me" ]; then
    log "CLAIM: $CELL belongs to '$owner', this machine is '$me' — refusing"
    exit 10
  fi
  [ -n "$owner" ] || log "CLAIM: $CELL unlisted in $CLAIMS — proceeding"
fi

# A session-wide ceiling on how far any cell may climb, read fresh on every
# leg so it reaches the ladder drivers that are already running. The spend
# order is "all ten cells to bb100k first, extensions with what is left";
# without this, one fast cell reaches bb300k while another has not reached
# bb40k. Refusing (rather than blocking) frees the GPU and lets the queue
# behind this cell start: the cell stops with both its stops recorded, and
# `ladder.py` replays it from disk when the hold is lifted.
HOLD_FILE="$RES/HOLD_ABOVE"
if [ -f "$HOLD_FILE" ]; then
  hold_at="$(tr -dc '0-9' <"$HOLD_FILE")"
  if [ -n "$hold_at" ] && [ "$TARGET_STEPS" -gt "$hold_at" ]; then
    log "HOLD: session capped at ${hold_at}; not starting the ${TARGET_STEPS} leg"
    exit 9
  fi
fi

# Blocks here, not inside torch, when another cell already owns the device.
gpu_gate "$BB_GPU" || { log "ABORT: GPU $BB_GPU never came free"; exit 1; }

log "START target=$TARGET_STEPS gpu=$BB_GPU save_every=$SAVE_EVERY leg=$LEG"
CUDA_VISIBLE_DEVICES="$BB_GPU" python3 -u "$TRAIN" "${RESUME[@]}" \
  --qk-norm --attn-out-norm \
  --batch-size 64 --device cuda --total-steps "$TARGET_STEPS" \
  --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  --seed "$SEED" \
  --save-every "$SAVE_EVERY" --extra-save-steps "$EXTRA_SAVES" \
  --save-dir "$LEG" --run-name "$NAME" --log-every "$LOG_EVERY" \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8 \
  --num-encoder-layers 3 --num-layers 3 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  "${LOSS_ARGS[@]}" "${ALIGN_ARGS[@]}" \
  --ema-embedding --ema-encoder \
  --ema-tau 0.9 --ema-tau-end 1.0 --ema-tau-ramp-steps 100000 \
  --cpc-infonce-weight 1.0 \
  --sigreg-embedding --sigreg-encoding --sigreg-n-chunk 2048 \
  --sigreg-embedding-weight 1.0 --sigreg-encoding-weight 1.0 \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.0078125 --crossfade-triplets 1 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 \
  --patch-emb-dtype fp32 \
  --train-rollout-depth "$K" \
  "${EXTRA_ARGS[@]}" \
  "${GAP_ARGS_ARR[@]}" \
  >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "train exited rc=$rc — tail: $(tail -3 "$tlog" | tr '\n' ' ')"
  exit 1
fi

produced="$(ckpt_at_step "$RUNS" "$NAME" "$target_k")"
[ -n "$produced" ] || {
  log "FAIL: no ${NAME}_${target_k}k.pth under $LEG after a clean exit"; exit 1; }
log "DONE target=$TARGET_STEPS ($(du -h "$produced" | cut -f1)) -> $produced"
