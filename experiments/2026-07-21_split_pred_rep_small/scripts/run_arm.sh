#!/bin/bash
# #379 arm launcher — small backbone × 200k steps on elisa, one script per arm.
#
# Usage:
#   WT=/absolute/path/to/checkout \
#   BB_GPU=1 GPU_2L=0 GPU_6L=1 \
#     bash run_arm.sh <arm>
#
# where <arm> ∈ {arm1, arm3, arm4, arm5, arm6_v2, bimoco}. Each arm shares
# the same backbone architecture and training schedule (d_model=128,
# n_heads=16, num_encoder_layers=3, num_layers=3, bs=128, 200k steps,
# save-every=10000, extras at 2500,25000); the arm-specific bits (run
# name, human label, extra CLI flags after --loss-shape) come from the
# per-arm case block below. See `README.md` for the parent-experiment
# mapping to #374.
#
# Backbone on gpu $BB_GPU (default 1). Downstream: 5 backbone-step cells
# (2k, 25k, 50k, 100k, 200k) × 2 head-layer sizes (2L, 6L) = 10 cells
# pipelined 2L on gpu $GPU_2L and 6L on gpu $GPU_6L in parallel. Each
# cell trains a fresh 40k-step transformer q-head then runs full-97
# GIFT-Eval B4.
set -uo pipefail

ARM="${1:?usage: run_arm.sh <arm1|arm3|arm4|arm5|arm6_v2|bimoco>}"

# WT MUST be an absolute path under the elisa checkout (or another
# checkout the user manages) — never /tmp. /tmp gets wiped by reboots
# and by `git worktree remove --force` (CLAUDE.md § Checkpoint Safety
# Rule 4, Apr-2026 incident). Fall back to the documented elisa layout
# when unset.
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
# NAME:      base run name (all backbone / q-head / eval artefacts prefix on it).
# ARM_DESC:  human-readable one-line description used in the BB START log line.
# LOSS_ARGS: extra CLI flags added AFTER --loss-shape (loss-shape itself included).
#
# When adding a 7th arm, add one case here and one entry in the launcher
# shape test (tests/test_small_long_launcher_shape.py). Nothing else in
# this script needs to change.
case "$ARM" in
  arm1)
    NAME="bb_small_arm1_split_pred_rep_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 1: split_pred_rep"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep)
    ;;
  arm3)
    NAME="bb_small_arm3_split_pred_rep_moco_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 3: split_pred_rep + moco-negatives"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep --moco-negatives)
    ;;
  arm4)
    NAME="bb_small_arm4_xshh_allt_moco_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 4: xshh_allt + moco-negatives"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt \
               --pos-in-denominator --subtract-contrastive-floor --moco-negatives)
    ;;
  arm5)
    NAME="bb_small_arm5_lalign_lrep_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 5: rep_only + align-loss-weight 1.0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only --align-loss-weight 1.0)
    ;;
  arm6_v2)
    NAME="bb_small_arm6_v2_lalign_lrepmoco_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 6 v2: rep_only + align-loss-weight 1.0 + moco-rep-keys"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys)
    ;;
  bimoco)
    NAME="bb_small_bimoco_split_pred_rep_moco_bothsides_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090"
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

TAG="${NAME#bb_}"
SEED=20260520
STEPS="${STEPS:-200000}"; SAVE_EVERY="${SAVE_EVERY:-10000}"
EXTRA_SAVES="${EXTRA_SAVES:-2500,25000}"
NENC=3; NLAY=3
HEAD_STEPS="${HEAD_STEPS:-40000}"; HEAD_WARMUP="${HEAD_WARMUP:-3000}"
BB_GPU="${BB_GPU:-1}"; GPU_2L="${GPU_2L:-0}"; GPU_6L="${GPU_6L:-1}"
# Optional restrict list — set BB_STEPS_K="2 25 50 100 200" to shrink the
# downstream fan-out (used by the smoke script). Defaults to all five cells.
BB_STEPS_K="${BB_STEPS_K:-2 25 50 100 200}"
# Optional extra args passed through to the GIFT-Eval invocation — used
# by the smoke script to add `--config-filter <regex>` for a fast eval.
QEVAL_EXTRA_ARGS="${QEVAL_EXTRA_ARGS:-}"

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
export TEACHER_EMBED_CHUNK="${TEACHER_EMBED_CHUNK:-16}"
HF_TOKEN_PATH="$WT/experiments/hf_token.txt"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
DL_LOG="$RES/dl_${ARM}.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [elisa-$ARM] $*" | tee -a "$DL_LOG"; }
gm(){ grep 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+$' | tail -1; }
[ -f "$TRAIN" ]  || { log "ABORT: TRAIN not at $TRAIN"; exit 2; }
[ -f "$QTRAIN" ] || { log "ABORT: QTRAIN not at $QTRAIN"; exit 2; }
[ -f "$QEVAL" ]  || { log "ABORT: QEVAL not at $QEVAL"; exit 2; }
[ -f "$HF_TOKEN_PATH" ] || { log "ABORT: HF token missing at $HF_TOKEN_PATH"; exit 2; }
export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { log "ABORT: empty HF_TOKEN"; exit 2; }

BB="$RUNS/${NAME}_FINAL.pth"
tlog="$RES/run_${NAME}.log"

if [ -f "$BB" ]; then
  log "BB SKIP ($NAME FINAL exists)"
else
  RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
  [ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
  log "BB START ($ARM_DESC) gpu=$BB_GPU steps=$STEPS bs=128 ${RESUME}"
  CUDA_VISIBLE_DEVICES="$BB_GPU" python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
    --batch-size 128 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
    --save-every "$SAVE_EVERY" --extra-save-steps "$EXTRA_SAVES" \
    --save-dir "$RUNS" --run-name "$NAME" --log-every 200 \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --t-raw 4096 --n-channels 1 --d-model 128 --n-heads 16 \
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
  log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"
fi

# ---- DOWNSTREAM: N backbone-step cells × 2 head-layer sizes -----------------
# Downstream arch args pin the values that BACKBONE_CONFIG does NOT
# auto-detect from the checkpoint (T-raw / C / H / nhead / num_layers /
# encoder / rev-norm). freq_emb_dim, seasonality_emb_dim, num_encoder_layers,
# qk_norm, attn_out_norm, depthwise_conv are auto-detected from the state_dict
# by train_forecasting_head.py + eval_gift_eval_official.py (auto-detect code
# lives in the auto-detected keys block of both scripts).
#
# Note: q-head's --head-nhead default is 6 (chosen for #374 backbone with
# d_model=384; 384/6=64). For the #379 small backbone d_model=128 is NOT
# divisible by 6 (the smoke run surfaced the assertion `embed_dim must be
# divisible by num_heads`), so we override --head-nhead 8 (128/8=16) in
# both the q-head trainer and the evaluator below.
arch=(--t-raw 4096 --n-channels 1 --d-model 128 --n-heads 16 --num-layers 3 \
      --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128)

# Resolve the backbone checkpoint for a given step_k, tolerant of restart
# suffixes. `safe_run_name` appends `_r2`, `_r3`, … to `--run-name` when a
# resume-launch finds existing checkpoints (see docs/restart_protocol.md);
# post-restart snapshots therefore land as `${NAME}_r2_50k.pth`, not
# `${NAME}_50k.pth`. Glob newest-first so the restart-safe variant wins
# when both exist. Prints nothing (returns non-zero) when no ckpt matches.
ckpt_path(){ # step_k → prints newest matching path, or empty on miss.
  local sk="$1" cand=""
  if [ "$sk" -eq 200 ]; then
    for pat in "$RUNS/${NAME}"*_final.pth "$RUNS/${NAME}"*_200k.pth; do
      cand=$(ls -t $pat 2>/dev/null | grep -v optimizer | head -1)
      [ -n "$cand" ] && { echo "$cand"; return 0; }
    done
  fi
  cand=$(ls -t "$RUNS/${NAME}"*_${sk}k.pth 2>/dev/null | grep -v optimizer | head -1)
  [ -n "$cand" ] && { echo "$cand"; return 0; }
  return 1
}

train_head_cell(){ # HL gpu qn bb tot wu
  local HL="$1" gpu="$2" qn="$3" bb="$4" tot="${5:-$HEAD_STEPS}" wu="${6:-$HEAD_WARMUP}"
  local qf="$RUNS/${qn}_FINAL.pth"
  [ -f "$qf" ] && { log "QH $qn skip (FINAL exists)"; return 0; }
  [ -f "$bb" ] || { log "QH $qn SKIP: backbone $bb missing"; return 1; }
  log "QH $qn train $tot on $(basename "$bb") gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python3 -u "$QTRAIN" \
    --backbone-path "$bb" --forecast-len 16 --quantile-head \
    --head-arch transformer --head-causal true \
    --head-num-layers "$HL" --head-nhead 8 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f \
    --total-steps "$tot" --batch-size 256 --lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps "$wu" --final-lr-ratio 0.1 \
    --save-every 100000 --log-every 200 --save-dir "$RUNS" --run-name "$qn" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --device cuda "${arch[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype none \
    >>"$RES/run_${qn}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "QH $qn FAILED rc=$rc (tail: $(tail -3 "$RES/run_${qn}.log"|tr '\n' ' '))"; return 1
  fi
  if   [ -f "$RUNS/${qn}_best.pth" ];  then cp -f "$RUNS/${qn}_best.pth"  "$qf"
  elif [ -f "$RUNS/${qn}_final.pth" ]; then cp -f "$RUNS/${qn}_final.pth" "$qf"
  else cp -f "$(ls -t "$RUNS/${qn}"_*k.pth 2>/dev/null|head -1)" "$qf"; fi
  [ -f "$qf" ] || { log "QH $qn no checkpoint"; return 1; }
  log "QH $qn done"
}

eval_cell(){ # HL gpu qn bb tag
  local HL="$1" gpu="$2" qn="$3" bb="$4" tag="$5"
  local qf="$RUNS/${qn}_FINAL.pth"
  local out="$RES/gift_eval_full_${tag}_${HL}L"
  [ -f "$out/summary.txt" ] && { log "EVAL $tag ${HL}L skip GM=$(gm "$out/summary.txt")"; return 0; }
  mkdir -p "$out"; log "EVAL $tag ${HL}L full-97 start gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python3 -u "$QEVAL" \
    --backbone-path "$bb" --head-path "$qf" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 128 --n-heads 16 \
    --num-layers 3 --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 \
    --device cuda --head-causal true --head-nhead 8 \
    $QEVAL_EXTRA_ARGS \
    >>"$RES/run_eval_full_${tag}_${HL}L.log" 2>&1 \
    || { log "EVAL $tag ${HL}L FAILED"; return 1; }
  log "EVAL $tag ${HL}L done GM=$(gm "$out/summary.txt")"
}

downstream_hl(){ # HL gpu
  local HL="$1" gpu="$2" fail=0
  local sk bbc tag qn
  for sk in $BB_STEPS_K; do
    bbc="$(ckpt_path "$sk")" || {
      # Loud abort — a missing backbone-step checkpoint is a real problem
      # (a restart landed with a suffix the glob didn't catch, or the
      # snapshot never happened). Counting this cell as a "failed head
      # cell" would hide the real cause; abort the whole downstream_hl
      # so the operator sees it in dl_${arm}.log immediately.
      log "ABORT downstream ${HL}L: no backbone ckpt for step ${sk}k under $RUNS/${NAME}*_${sk}k.pth"
      return 99
    }
    tag="${TAG}_${sk}k"
    qn="qhead_${HL}L_${tag}"
    if train_head_cell "$HL" "$gpu" "$qn" "$bbc"; then
      eval_cell "$HL" "$gpu" "$qn" "$bbc" "$tag" || fail=$((fail+1))
    else fail=$((fail+1)); fi
  done
  return "$fail"
}

log "downstream start: 2L on gpu $GPU_2L + 6L on gpu $GPU_6L (parallel)"
downstream_hl 2 "$GPU_2L" >>"$RES/dl_2L_${ARM}.log" 2>&1 & pid2=$!
downstream_hl 6 "$GPU_6L" >>"$RES/dl_6L_${ARM}.log" 2>&1 & pid6=$!
log "downstream PIDs: 2L=$pid2 6L=$pid6"
wait $pid2; rc2=$?
wait $pid6; rc6=$?
log "$ARM complete: 2L failed-cells=$rc2 6L failed-cells=$rc6"
exit $(( rc2 != 0 || rc6 != 0 ))
