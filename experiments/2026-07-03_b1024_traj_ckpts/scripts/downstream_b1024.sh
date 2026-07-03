#!/bin/bash
# #369 — downstream q-head + GIFT-Eval for the retrained B=1024 arm at
# TWO backbone loci (matching #369's two-ckpt requirement):
#   1. `_step<PARENT_BEST_LOSS_STEP>.pth` — the retrained backbone at the
#      SAME step number the parent's `best-loss` head landed on.
#   2. `_step<TOTAL_STEPS>.pth` — the retrained `last` at 12,500 steps
#      (same wall-time budget as the parent, before any 2× extension).
#
#   downstream_b1024.sh <head_layers: 2|6> <gpu> <suffix> <parent_best_loss_step> [total_steps]
set -uo pipefail
HL="${1:?head_layers}"; GPU="${2:?gpu}"; SUFFIX="${3:?suffix}"
PARENT_STEP="${4:?parent_best_loss_step}"; TOTAL_STEPS="${5:-12500}"
TAG="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b1024_cpc_${SUFFIX}"
: "${WT:?WT (worktree root containing train.py + experiments/hf_token.txt) must be set; e.g. WT=/home/jupyter/workspaces/contrastive-forecasting}"
: "${OUT:?OUT (per-experiment output dir for runs/ and results/) must be set}"
[ -d "$WT" ] || { echo "[dl-${SUFFIX} ${HL}L] ABORT: WT does not exist: $WT" >&2; exit 2; }
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB_PARENT="$RUNS/bb_${TAG}_step${PARENT_STEP}.pth"
BB_LAST="$RUNS/bb_${TAG}_step${TOTAL_STEPS}.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$GPU"
HF_TOKEN_PATH="$WT/experiments/hf_token.txt"
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [dl-${SUFFIX} ${HL}L g$GPU] $*"; }
gm(){ grep 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+$' | tail -1; }
[ -f "$QTRAIN" ] || { log "ABORT: QTRAIN script not found at $QTRAIN (check WT=$WT)"; exit 2; }
[ -f "$QEVAL" ]  || { log "ABORT: QEVAL script not found at $QEVAL (check WT=$WT)"; exit 2; }
[ -f "$HF_TOKEN_PATH" ] || { log "ABORT: HF token not found at $HF_TOKEN_PATH"; exit 2; }
export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { log "ABORT: empty HF_TOKEN — HF stream throttles GPU"; exit 2; }

# Both loci are load-bearing (parent-best AND retrained-last are half the
# comparison each). A missing trajectory checkpoint means the backbone
# run stopped before that step — refuse to silently downgrade.
if [ ! -f "$BB_PARENT" ]; then
  log "ABORT: parent-best-step backbone missing at $BB_PARENT."
  log "    Backbone did not reach step ${PARENT_STEP} or the trajectory"
  log "    checkpoint at that step was lost. This locus is required."
  exit 1
fi
if [ ! -f "$BB_LAST" ]; then
  log "ABORT: last-step backbone missing at $BB_LAST."
  log "    Backbone did not reach step ${TOTAL_STEPS} — the retrained-last"
  log "    cell cannot be scored without it."
  exit 1
fi

arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 6 \
      --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128)

train_head(){ # run_name backbone resume_src total warmup
  local qn="$1" bb="$2" src="$3" tot="$4" wu="$5" qf="$RUNS/$1_FINAL.pth"
  [ -f "$qf" ] && { log "QH $qn skip (FINAL exists)"; return 0; }
  local rflag=(); [ -n "$src" ] && rflag=(--resume "$src")
  log "QH $qn train $tot on $(basename "$bb")"
  python3 -u "$QTRAIN" "${rflag[@]}" --backbone-path "$bb" --forecast-len 16 --quantile-head \
    --head-arch transformer --head-causal true --head-num-layers "$HL" --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f --total-steps "$tot" --batch-size 256 --lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --weight-decay 0.1 --schedule cosine --warmup-steps "$wu" --final-lr-ratio 0.1 \
    --save-every 100000 --log-every 200 --save-dir "$RUNS" --run-name "$qn" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --device cuda "${arch[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype none \
    >>"$RES/run_${qn}.log" 2>&1 || { log "QH $qn FAILED (tail: $(tail -3 "$RES/run_${qn}.log"|tr '\n' ' '))"; return 1; }
  if   [ -f "$RUNS/${qn}_best.pth" ];  then cp -f "$RUNS/${qn}_best.pth"  "$qf"
  elif [ -f "$RUNS/${qn}_final.pth" ]; then cp -f "$RUNS/${qn}_final.pth" "$qf"
  else cp -f "$(ls -t "$RUNS/${qn}"_*k.pth 2>/dev/null|head -1)" "$qf"; fi
  [ -f "$qf" ] || { log "QH $qn no checkpoint"; return 1; }; log "QH $qn done"; }

do_eval(){ # run_name backbone out_tag
  local qf="$RUNS/$1_FINAL.pth" out="$RES/gift_eval_full_$3_${HL}L"
  [ -f "$out/summary.txt" ] && { log "EVAL $3 skip GM=$(gm "$out/summary.txt")"; return 0; }
  mkdir -p "$out"; log "EVAL $3 full-97 start"
  python3 -u "$QEVAL" --backbone-path "$2" --head-path "$qf" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 6 \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
    >>"$RES/run_eval_full_$3_${HL}L.log" 2>&1 || { log "EVAL $3 FAILED"; return 1; }
  log "EVAL $3 done GM=$(gm "$out/summary.txt")"; }

# Per-cell rc capture (identical shape to #366's downstream_sigreg.sh).
fail=0
run_cell(){ # qn bb resume_src total warmup eval_tag
  local qn="$1" bb="$2" src="$3" tot="$4" wu="$5" tag="$6"
  if ! train_head "$qn" "$bb" "$src" "$tot" "$wu"; then
    fail=$((fail+1)); return
  fi
  if [ "${DO_EVAL:-1}" = 1 ]; then
    do_eval "$qn" "$bb" "$tag" || fail=$((fail+1))
  fi
}
# Cell 1 — retrained backbone at parent's best-loss step (fresh 30k head).
run_cell "qhead_${HL}L_${TAG}_parentstep" "$BB_PARENT" "" \
         "30000" "2000" "${TAG}_parentstep"
# Cell 2 — retrained backbone at last step (resume from cell-1 head, 10k re-adapt).
run_cell "qhead_${HL}L_${TAG}_last"       "$BB_LAST" \
         "$RUNS/qhead_${HL}L_${TAG}_parentstep_FINAL.pth" "10000" "1000" "${TAG}_last"
log "b1024 downstream complete (${HL}L for ${SUFFIX}): failed cells=$fail"
exit "$fail"
