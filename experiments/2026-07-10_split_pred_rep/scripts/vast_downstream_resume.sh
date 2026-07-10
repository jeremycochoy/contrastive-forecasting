#!/bin/bash
# #374 — resume the split_pred_rep downstream on a single vast.ai GPU.
# Sequentially: 2L best-eval (with --resume) → 2L last-train (10k) → 2L last-eval
#              → 6L best-eval (with --resume) → 6L last-train (10k) → 6L last-eval.
# Uses the head best.pth checkpoints (renamed to FINAL) already shipped from
# elisa, plus the partial all_results.csv files for the two best-evals.
set -uo pipefail
WT=/workspace/cf-374
OUT="$WT/experiments/2026-07-10_split_pred_rep"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
TAG="split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"
BB="$RUNS/bb_${TAG}_FINAL.pth"
BBLAST="$RUNS/bb_${TAG}_final.pth"
export PYTHONPATH="$WT:/workspace/gift-eval/src" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
export GIFT_EVAL="${GIFT_EVAL:-/workspace/gift-eval-data}"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [vast-dl] $*"; }

[ -n "$HF_TOKEN" ] || { log "ABORT: empty HF_TOKEN"; exit 1; }
[ -f "$BB" ] || { log "ABORT: backbone FINAL missing: $BB"; exit 1; }
[ -f "$BBLAST" ] || { log "ABORT: backbone last missing: $BBLAST"; exit 1; }

arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 6 \
      --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128)

train_head(){ # HL run_name backbone resume_src total warmup
  local HL="$1" qn="$2" bb="$3" src="$4" tot="$5" wu="$6"
  local qf="$RUNS/${qn}_FINAL.pth"
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
    >>"$RES/run_${qn}.log" 2>&1 || { log "QH $qn FAILED"; return 1; }
  if   [ -f "$RUNS/${qn}_best.pth" ];  then cp -f "$RUNS/${qn}_best.pth"  "$qf"
  elif [ -f "$RUNS/${qn}_final.pth" ]; then cp -f "$RUNS/${qn}_final.pth" "$qf"
  fi
  [ -f "$qf" ] || { log "QH $qn no checkpoint"; return 1; }; log "QH $qn done"; }

do_eval(){ # HL run_name backbone out_tag
  local HL="$1" qn="$2" bb="$3" tag="$4"
  local qf="$RUNS/${qn}_FINAL.pth"
  local out="$RES/gift_eval_full_${tag}_${HL}L"
  [ -f "$out/summary.txt" ] && { log "EVAL ${tag}_${HL}L skip (summary exists)"; return 0; }
  mkdir -p "$out"; local resume_flag=""
  [ -f "$out/all_results.csv" ] && resume_flag="--resume"
  log "EVAL ${tag}_${HL}L full-97 start ${resume_flag:+(resuming)}"
  python3 -u "$QEVAL" $resume_flag --backbone-path "$bb" --head-path "$qf" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 6 \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
    >>"$RES/run_eval_full_${tag}_${HL}L.log" 2>&1 || { log "EVAL ${tag}_${HL}L FAILED"; return 1; }
  log "EVAL ${tag}_${HL}L done"; }

# Promote best.pth → FINAL.pth so train_head skips (best-loss head already trained on elisa).
for HL in 2 6; do
  qn="qhead_${HL}L_${TAG}"
  [ -f "$RUNS/${qn}_best.pth" ] && [ ! -f "$RUNS/${qn}_FINAL.pth" ] && cp -f "$RUNS/${qn}_best.pth" "$RUNS/${qn}_FINAL.pth" && log "promoted ${qn}_best.pth → FINAL"
done

# Sequential per cell: 2L first (small remaining work), then 6L.
for HL in 2 6; do
  qn_best="qhead_${HL}L_${TAG}"
  qn_last="qhead_${HL}L_${TAG}_last"
  # cell 1: best-loss backbone, best-loss head (resumes eval CSV where left off).
  do_eval "$HL" "$qn_best" "$BB"     "$TAG"      || exit 1
  # cell 2: last backbone, resume from best head for 10k re-adapt, then eval.
  train_head "$HL" "$qn_last" "$BBLAST" "$RUNS/${qn_best}_FINAL.pth" "10000" "1000" || exit 1
  do_eval    "$HL" "$qn_last" "$BBLAST" "${TAG}_last" || exit 1
done
log "downstream complete for both HL=2 and HL=6"
