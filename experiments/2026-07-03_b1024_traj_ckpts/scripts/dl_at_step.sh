#!/bin/bash
# #369 follow-up — train + eval ONE head cell (head_depth × single BB
# checkpoint) at an arbitrary trajectory step. Used to score the extended
# BB (step 12500 → 25000) at intermediate points {15000, 20000, 25000}.
# Structurally identical to `downstream_b1024.sh`'s cell-2 pattern:
# resume the head from the arm's `parentstep_FINAL` and retrain 10k
# steps, then evaluate on GIFT-Eval full-97.
#
#   dl_at_step.sh <head_layers: 2|6> <gpu> <suffix> <bb_step>
set -uo pipefail
HL="${1:?head_layers}"; GPU="${2:?gpu}"; SUFFIX="${3:?suffix}"; STEP="${4:?bb_step}"
TAG="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b1024_cpc_${SUFFIX}"
: "${WT:?}"; : "${OUT:?}"
[ -d "$WT" ] || { echo "[dl-step-${STEP} ${HL}L] ABORT: WT does not exist: $WT" >&2; exit 2; }
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"

# Locate BB checkpoint at STEP — original trajectory used `_step<N>.pth`,
# the resumed extend run auto-suffixes to `_r2_step<N>.pth` (or higher).
BB=""
for cand in "$RUNS/bb_${TAG}_step${STEP}.pth" \
            "$RUNS/bb_${TAG}"_r*_step"${STEP}".pth; do
  [ -f "$cand" ] && { BB="$cand"; break; }
done
[ -n "$BB" ] || { echo "[dl-step-${STEP} ${HL}L] ABORT: no bb ..._step${STEP}.pth" >&2; exit 1; }

RESUME_HEAD="$RUNS/qhead_${HL}L_${TAG}_parentstep_FINAL.pth"
[ -f "$RESUME_HEAD" ] || { echo "ABORT: parentstep head missing at $RESUME_HEAD" >&2; exit 1; }

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$GPU"
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
HF_TOKEN_PATH="$WT/experiments/hf_token.txt"
export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [dl-step${STEP} ${HL}L g$GPU] $*"; }
gm(){ grep 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+$' | tail -1; }

qn="qhead_${HL}L_${TAG}_step${STEP}"
qf="$RUNS/${qn}_FINAL.pth"
out="$RES/gift_eval_full_${TAG}_step${STEP}_${HL}L"

if [ -f "$qf" ]; then
  log "QH ${qn} skip (FINAL exists)"
else
  log "QH ${qn} train 10000 on $(basename "$BB") resume=$(basename "$RESUME_HEAD")"
  python3 -u "$QTRAIN" --resume "$RESUME_HEAD" --backbone-path "$BB" --forecast-len 16 --quantile-head \
    --head-arch transformer --head-causal true --head-num-layers "$HL" --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f --total-steps 10000 --batch-size 256 --lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --weight-decay 0.1 --schedule cosine --warmup-steps 1000 --final-lr-ratio 0.1 \
    --save-every 100000 --log-every 200 --save-dir "$RUNS" --run-name "$qn" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 6 \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 \
    --device cuda --mix-ratio 0.0 --reconstruction forecaster --amp-dtype none \
    >>"$RES/run_${qn}.log" 2>&1 || { log "QH ${qn} FAILED"; exit 1; }
  # Prefer deterministic _final over _best to stay consistent with the
  # main train_backbone_b1024.sh fix.
  if   [ -f "$RUNS/${qn}_final.pth" ]; then cp -f "$RUNS/${qn}_final.pth" "$qf"
  elif [ -f "$RUNS/${qn}_best.pth" ];  then cp -f "$RUNS/${qn}_best.pth"  "$qf"
  else cp -f "$(ls -t "$RUNS/${qn}"_*k.pth 2>/dev/null|head -1)" "$qf"; fi
  [ -f "$qf" ] || { log "QH ${qn} no checkpoint"; exit 1; }
  log "QH ${qn} done"
fi

if [ -f "$out/summary.txt" ]; then
  log "EVAL skip GM=$(gm "$out/summary.txt")"
else
  mkdir -p "$out"
  # Best-effort lock so a second chain (e.g. GPU 1 backfill) that reaches
  # the same cell before the first completes backs out cleanly instead of
  # co-writing to the same output_dir (which truncates all_results.csv).
  # A stale lock from a crashed run is auto-cleared after 24h.
  LOCK="$out/.eval.lock"
  if [ -f "$LOCK" ]; then
    lock_age_h=$(( ( $(date +%s) - $(stat -c %Y "$LOCK") ) / 3600 ))
    if [ "$lock_age_h" -lt 24 ]; then
      log "EVAL step${STEP} skip (concurrent eval in flight — lock $lock_age_h h old)"
      exit 0
    fi
    log "EVAL step${STEP} stale lock (${lock_age_h}h old) — clearing"
  fi
  echo "$$ @ $(date -Iseconds) on gpu $GPU" > "$LOCK"
  trap 'rm -f "$LOCK"' EXIT
  log "EVAL step${STEP} full-97 start"
  python3 -u "$QEVAL" --backbone-path "$BB" --head-path "$qf" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 6 \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
    >>"$RES/run_eval_full_${TAG}_step${STEP}_${HL}L.log" 2>&1 || { log "EVAL FAILED"; exit 1; }
  log "EVAL step${STEP} done GM=$(gm "$out/summary.txt")"
fi
