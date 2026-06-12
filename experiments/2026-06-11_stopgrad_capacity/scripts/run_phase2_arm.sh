#!/bin/bash
# #341 phase 2, PER-ARM (overlaps one arm's fresh-last work with the other arm's
# still-running chain). For its arm: wait chain_<arm>.done, train fresh 30k last
# heads (2L,6L) directly on the last backbone (no 10k re-adapt — best-loss is at
# step ~1k for these arms, which makes the re-adapt underfit and blow up), then
# eval this arm's cells on its GPU: lastfresh (PRIMARY) -> best -> re-adapt-last
# (secondary, for the comparison). Idempotent; detached.
#   run_phase2_arm.sh <arm: nobn_enc6|bn_enc6> <gpu>
set -uo pipefail
ARM="${1:?arm}"; GPU="${2:?gpu}"
SD="$(cd "$(dirname "$0")" && pwd)"
WT="${WT:-/tmp/cf-341}"
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
TAG="allt08_xftrip_${ARM}_sgpos_qk_aon_b1024"
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [phase2-$ARM g$GPU] $*"; }
log "waiting for chain_${ARM}.done"
while [ ! -f "$RES/chain_${ARM}.done" ]; do sleep 60; done
FCST=(); case "$ARM" in nobn_enc6) FCST=() ;; bn_enc6) FCST=(--forecaster-d-model 128 --forecaster-n-heads 4) ;; esac
train_fresh(){ # <hl>
  local hl="$1" qn="qhead_${hl}L_${TAG}_lastfresh" qf="$RUNS/qhead_${hl}L_${TAG}_lastfresh_FINAL.pth"
  [ -f "$qf" ] && { log "head $qn skip (FINAL exists)"; return 0; }
  log "train $qn (30k fresh) on bb_${TAG}_final.pth"
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u "$QTRAIN" --backbone-path "$RUNS/bb_${TAG}_final.pth" \
    --forecast-len 16 --quantile-head --head-arch transformer --head-causal true --head-num-layers "$hl" \
    --head-nhead 6 --head-ffn-mult 4.0 --head-dropout 0.1 --head-train-input e_then_f --total-steps 30000 \
    --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 --schedule cosine --warmup-steps 2000 \
    --final-lr-ratio 0.1 --save-every 100000 --log-every 200 --save-dir "$RUNS" --run-name "$qn" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 --device cuda --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 "${FCST[@]}" --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 \
    --mix-ratio 0.0 --reconstruction forecaster --amp-dtype none >>"$RES/run_${qn}.log" 2>&1 \
    || { log "$qn FAILED (tail: $(tail -2 "$RES/run_${qn}.log"|tr '\n' ' '))"; return 1; }
  if [ -f "$RUNS/${qn}_best.pth" ]; then cp -f "$RUNS/${qn}_best.pth" "$qf"; elif [ -f "$RUNS/${qn}_final.pth" ]; then cp -f "$RUNS/${qn}_final.pth" "$qf"; fi
  [ -f "$qf" ] && log "$qn done" || { log "$qn no checkpoint"; return 1; }
}
log "chain done — training fresh last heads on gpu$GPU"
train_fresh 2; train_fresh 6
log "fresh heads done — evaluating this arm's cells on gpu$GPU (12 shards/cell)"
for cell in "lastfresh 2" "lastfresh 6" "best 2" "best 6" "last 2" "last 6"; do
  set -- $cell
  bash "$SD/run_eval_cell.sh" "$TAG" "$1" "$2" 12 "$GPU"
done
log "PHASE2 $ARM DONE"; touch "$RES/phase2_${ARM}.done"
