#!/bin/bash
# #415 — one leg of the value-space reference model.
#
# Usage:  run_leg_value.sh <target steps>
#         BB_GPU=0 bash run_leg_value.sh 40000
#         CF415_DRY_RUN=1 bash run_leg_value.sh 40000   # print, do not run
#
# The model is #414's cell `arm6_v2_combab_alignT` at `d_model` 384. The BODY
# and the INPUT HEAD are that cell's, flag for flag: the same width, the same
# two stacks, the same patching, the same reversible normalisation, the same
# dropkey, the same numeric policy, the same data mix and the same seed.
#
# The OBJECTIVE is the whole difference, and it is one flag.
# `--value-space-objective` makes the forecaster latent decode into the
# quantiles of the next patch's values, reads the loss off the ACTUAL values,
# and rolls the forecast out in VALUE space one step at a time. The teacher,
# the EMA, `L_rep`, `L_align`, the CPC auxiliary and SIGReg are gone. The
# trainer refuses each of them under that flag, so a copied line cannot bring
# one back.
#
# ---- Why this is a second runner --------------------------------------------
#
# #373's `run_leg_k.sh` dispatches on a CELL NAME and hardcodes the teacher,
# the EMA, the CPC weight and both SIGReg terms for every one of its ten
# cells. This card removes exactly those. A tenth cell there would either
# carry them (wrong) or need the dispatch to grow an exception for a card that
# shares none of its objective.
#
# Everything downstream IS #373's and is reused, not copied: `leg_paths.sh`
# for the durable root and the step-ordered resume, `gpu_gate.sh` for the
# device, `hub_gate.sh` for a Hub outage, and — in `head_eval_value.sh` —
# `head_eval_bb.sh` and `eval_local.sh` for the head and the 97-config
# GIFT-Eval. That is the path #414 scores on, so the two numbers compare.
#
# ---- The rate ----------------------------------------------------------------
#
# 1e-3 is the Moirai recipe, which `exp_realonly_full4096_moirai_hp_FINAL`
# ran against this corpus at H = 384. #412's pass 1 found 1e-3 too high for
# the CONTRASTIVE objective at this width and settled at 5.6e-4; that verdict
# is about a different loss and does not carry here. Watch the first 5,000
# steps. If the loss diverges, the rate is the first thing to move, and
# `LR=5.6e-4 bash run_leg_value.sh ...` moves it without editing this file.
set -uo pipefail

TARGET_STEPS="${1:?usage: run_leg_value.sh <target steps>}"

. "$(dirname "${BASH_SOURCE[0]}")/paths.sh"
. "$CF415_PARENT/scripts/leg_paths.sh"
. "$CF415_PARENT/scripts/gpu_gate.sh"
. "$CF415_WT/scripts/hub_gate.sh"

BB_GPU="${BB_GPU:-0}"
SAVE_EVERY="${SAVE_EVERY:-20000}"
# 665,000 is not a multiple of 20,000, so the periodic save never lands it.
EXTRA_SAVES="${EXTRA_SAVES:-$TARGET_STEPS}"

RUNS="$CF415_RUNS"
ROOT="$(runs_root)" || exit 2
CELL_RUNS="$ROOT/value_space"
LEG="$(leg_dir "$CELL_RUNS" "$TARGET_STEPS")"
NAME="$CF415_RUN_NAME"

TRAIN="$CF415_WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
HF_TOKEN_PATH="$CF415_WT/experiments/hf_token.txt"
tlog="$CF415_RESULTS/run_${NAME}.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#415] $*" \
  | tee -a "$CF415_RESULTS/leg.log"; }

# The trainer command line, in one array, so the dry run prints exactly what
# the leg runs. Every flag above `--value-space-objective` is the cell's.
TRAIN_ARGS=(
  --qk-norm --attn-out-norm
  --batch-size "$CF415_BATCH_SIZE" --device cuda
  --total-steps "$TARGET_STEPS"
  --lr "$CF415_LR" --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
  --seed "$CF415_SEED"
  --save-every "$SAVE_EVERY" --extra-save-steps "$EXTRA_SAVES"
  --save-dir "$LEG" --run-name "$NAME" --log-every "${LOG_EVERY:-200}"
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1
  --t-raw 4096 --n-channels 1
  --d-model "$CF415_D_MODEL" --n-heads "$CF415_N_HEADS"
  --num-encoder-layers "$CF415_NUM_ENCODER_LAYERS"
  --num-layers "$CF415_NUM_LAYERS"
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads
  --encoder-dropkey-share-layers
  --depthwise-conv 3 --deprecated-depthwise-conv 0
  --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru
  --synth-kind forked-arma --mix-ratio 0.0078125 --crossfade-triplets 1
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3
  --log-attn-amplitude --log-attn-amplitude-every 200
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16
  --patch-emb-dtype fp32
  --value-space-objective
  --train-rollout-depth "$CF415_K" --train-rollout-reduce sum
)

# The dry run prints the command line the leg WOULD run, and creates
# nothing. Every guard of this card reads that line: the objective, the
# shape, the recipe and the data all reach the trainer through variables, so
# the resolved line is the only place they can be checked.
if [ -n "${CF415_DRY_RUN:-}" ]; then
  echo "leg target=$TARGET_STEPS gpu=$BB_GPU"
  echo "  runs=$CELL_RUNS"
  echo "  ckpt=$LEG/${NAME}_$(( TARGET_STEPS / 1000 ))k.pth"
  printf 'Command line: %s\n' "python3 -u $TRAIN ${TRAIN_ARGS[*]}"
  exit 0
fi

mkdir -p "$LEG" "$CF415_RESULTS"
[ -f "$TRAIN" ] || { log "ABORT: no trainer at $TRAIN"; exit 2; }
[ -f "$HF_TOKEN_PATH" ] || { log "ABORT: HF token missing at $HF_TOKEN_PATH"; exit 2; }
export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { log "ABORT: empty HF_TOKEN"; exit 2; }

export PYTHONPATH="$CF415_WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export FCST_GRAD_CKPT=1 PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4

# Idempotent: a stop already on disk is a no-op, so a re-fired leg after a
# crash costs nothing.
target_k=$(( TARGET_STEPS / 1000 ))
done_ckpt="$(ckpt_at_step "$CELL_RUNS" "$NAME" "$target_k")"
[ -n "$done_ckpt" ] && {
  log "SKIP: $(basename "$done_ckpt") already on disk"; exit 0; }

# Resume from the FURTHEST checkpoint, chosen by the step in its name, with
# its optimizer state. A fresh start throws away every step the run holds, so
# it is correct only when the run holds nothing.
RESUME=()
latest="$(newest_ckpt "$CELL_RUNS" "$NAME")"
if [ -n "$latest" ]; then
  RESUME=(--resume "$latest")
  log "RESUME from $(basename "$latest") (step $(ckpt_step_k "$latest")k)"
else
  stray="$(step_ckpts "$CELL_RUNS")"
  if [ -n "$stray" ]; then
    log "ABORT: no checkpoint named '$NAME', but $CELL_RUNS holds step checkpoints:"
    while read -r f; do log "  $(basename "$f")"; done <<<"$stray"
    log "  Point CF415_RUN_NAME at the run that wrote them, or move them aside."
    exit 2
  fi
  log "FRESH start at step 0"
fi

gpu_gate "$BB_GPU" || { log "ABORT: GPU $BB_GPU never came free"; exit 1; }

log "START target=$TARGET_STEPS gpu=$BB_GPU lr=$CF415_LR k=$CF415_K leg=$LEG"
CUDA_VISIBLE_DEVICES="$BB_GPU" python3 -u "$TRAIN" "${RESUME[@]}" \
  "${TRAIN_ARGS[@]}" >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "train exited rc=$rc — tail: $(tail -3 "$tlog" | tr '\n' ' ')"
  # A leg the Hub killed is not a failed run. It carries its own code so a
  # lane can wait for the network and re-fire without spending a try.
  if hub_outage_in_log "$tlog"; then
    log "the Hub was unreachable — a network failure, not a bad run"
    exit "$HUB_GATE_RC"
  fi
  exit 1
fi

produced="$(ckpt_at_step "$CELL_RUNS" "$NAME" "$target_k")"
[ -n "$produced" ] || {
  log "FAIL: no ${NAME}_${target_k}k.pth under $LEG after a clean exit"; exit 1; }
log "DONE target=$TARGET_STEPS ($(du -h "$produced" | cut -f1)) -> $produced"
