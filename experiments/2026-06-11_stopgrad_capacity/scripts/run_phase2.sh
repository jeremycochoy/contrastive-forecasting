#!/bin/bash
# #341 phase 2 (post-chains, resumable): the LAST-checkpoint cells use a FRESH
# head trained 30k directly on the last backbone (NOT the 10k re-adapt from the
# best head) — these arms' best-loss lands at step ~1k, so a 10k re-adapt from a
# barely-trained head underfits and blows up (arm4 2L re-adapt last = 2.27).
# Trains 4 fresh last heads (arm3/arm4 x 2L/6L), then evals, priority order:
#   fresh-last (primary)  ->  best  ->  re-adapt-last (secondary, for comparison).
# Idempotent everywhere (skips heads/cells already done). Detached; survives logout.
set -uo pipefail
SD="$(cd "$(dirname "$0")" && pwd)"
WT="${WT:-/tmp/cf-341}"
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [phase2] $*"; }

log "waiting for both chains (chain_nobn_enc6.done + chain_bn_enc6.done)"
while [ ! -f "$RES/chain_nobn_enc6.done" ] || [ ! -f "$RES/chain_bn_enc6.done" ]; do sleep 60; done
log "both chains done — training fresh last heads (30k on last backbone)"

train_fresh_last(){ # <tag> <hl> <gpu>
  local tag="$1" hl="$2" gpu="$3" qn="qhead_${hl}L_${1}_lastfresh" qf
  qf="$RUNS/qhead_${hl}L_${1}_lastfresh_FINAL.pth"
  [ -f "$qf" ] && { log "head $qn skip (FINAL exists)"; return 0; }
  local FCST=(); case "$tag" in *nobn_enc6*) FCST=() ;; *bn_enc6*) FCST=(--forecaster-d-model 128 --forecaster-n-heads 4) ;; esac
  log "train $qn (30k, fresh, gpu$gpu) on bb_${tag}_final.pth"
  CUDA_VISIBLE_DEVICES="$gpu" python3 -u "$QTRAIN" --backbone-path "$RUNS/bb_${tag}_final.pth" \
    --forecast-len 16 --quantile-head --head-arch transformer --head-causal true --head-num-layers "$hl" \
    --head-nhead 6 --head-ffn-mult 4.0 --head-dropout 0.1 --head-train-input e_then_f --total-steps 30000 \
    --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 --schedule cosine --warmup-steps 2000 \
    --final-lr-ratio 0.1 --save-every 100000 --log-every 200 --save-dir "$RUNS" --run-name "$qn" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 --device cuda --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 "${FCST[@]}" --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 \
    --mix-ratio 0.0 --reconstruction forecaster --amp-dtype none >>"$RES/run_${qn}.log" 2>&1 \
    || { log "head $qn FAILED (tail: $(tail -2 "$RES/run_${qn}.log"|tr '\n' ' '))"; return 1; }
  if   [ -f "$RUNS/${qn}_best.pth" ];  then cp -f "$RUNS/${qn}_best.pth"  "$qf"
  elif [ -f "$RUNS/${qn}_final.pth" ]; then cp -f "$RUNS/${qn}_final.pth" "$qf"; fi
  [ -f "$qf" ] && log "head $qn done" || { log "head $qn no checkpoint"; return 1; }
}

BN=allt08_xftrip_bn_enc6_sgpos_qk_aon_b1024
NOBN=allt08_xftrip_nobn_enc6_sgpos_qk_aon_b1024
# Train fresh last heads, paired across the two GPUs (bn->0, nobn->1). 2L first (faster).
( train_fresh_last "$BN" 2 0 ) & ( train_fresh_last "$NOBN" 2 1 ) & wait
( train_fresh_last "$BN" 6 0 ) & ( train_fresh_last "$NOBN" 6 1 ) & wait
log "fresh last heads done — evaluating"

pair(){ # <ck> <hl>   bn on GPU0, nobn on GPU1, 8 shards each
  bash "$SD/run_eval_cell.sh" "$BN"   "$1" "$2" 8 0 &
  bash "$SD/run_eval_cell.sh" "$NOBN" "$1" "$2" 8 1 &
  wait
}
pair lastfresh 2 ; pair lastfresh 6    # PRIMARY last-checkpoint cells (fresh heads)
pair best 2      ; pair best 6         # best-loss cells
pair last 2      ; pair last 6         # re-adapt last (secondary; for the comparison)
log "ALL PHASE2 EVALS DONE"; touch "$RES/all_evals.done"
