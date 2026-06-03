#!/bin/bash
# #325 — recover the 6L q-head after a transient HF CDN connect-timeout crashed it at
# step ~20k. Resumes from the latest periodic checkpoint (model+optimizer+step+data
# position all restored) and RETRIES on further HF flakiness (each crash loses <=5000
# steps). Once the q-head reaches 30k -> FINAL, hands off to downstream_crossfade.sh for
# the (idempotent) triage+full GIFT-Eval. Runs on GPU1 by default.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU="${1:-1}"
WT="${WT:-/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/crossfade-allt10}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-01_crossfade_allt10}"
RUNS="$OUT/runs"; RES="$OUT/results"
QN="qhead_6L_xshh_allt_forked10pct_crossfade10pct_qk_aon_b1024"
QF="$RUNS/${QN}_FINAL.pth"
BB="$RUNS/bb_xshh_allt_forked10pct_crossfade10pct_qk_aon_6Lf_b1024_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_DOWNLOAD_TIMEOUT=60          # the crash was a 10s connect-timeout to xethub
export CUDA_VISIBLE_DEVICES="$GPU"
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
tlog="$RES/run_${QN}.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [recover-6L] $*" | tee -a "$RES/recover_6L.log"; }
arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 6 \
      --forecaster-d-model 128 --forecaster-n-heads 4 --encoder-type gru \
      --rev-norm-kind ewma --rev-norm-span 128)

for attempt in $(seq 1 6); do
  [ -f "$QF" ] && { log "FINAL exists — training done"; break; }
  latest=$(ls -t "$RUNS/${QN}"_*k.pth 2>/dev/null | head -1)
  RES_ARG=""; [ -n "$latest" ] && RES_ARG="--resume $latest"
  log "attempt $attempt: train->30k, resume from ${latest:-scratch}"
  python3 -u "$QTRAIN" $RES_ARG \
    --backbone-path "$BB" --forecast-len 16 --quantile-head --head-arch transformer \
    --head-causal true --head-num-layers 6 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f \
    --total-steps 30000 --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 \
    --weight-decay 0.1 --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 --save-dir "$RUNS" --run-name "$QN" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --device cuda "${arch[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype none \
    >>"$tlog" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    if   [ -f "$RUNS/${QN}_best.pth" ];  then cp -f "$RUNS/${QN}_best.pth"  "$QF"
    elif [ -f "$RUNS/${QN}_final.pth" ]; then cp -f "$RUNS/${QN}_final.pth" "$QF"
    else cp -f "$(ls -t "$RUNS/${QN}"_*k.pth | head -1)" "$QF"; fi
    log "train rc=0 -> FINAL created"; break
  fi
  log "attempt $attempt crashed rc=$rc (tail: $(tail -2 "$tlog"|tr '\n' ' ')); retrying"
  sleep 20
done
[ -f "$QF" ] || { log "GAVE UP — no FINAL after retries"; exit 1; }
log "running 6L GIFT-Eval (triage+full) via downstream_crossfade.sh"
bash "$HERE/downstream_crossfade.sh" 6 "$GPU" "triage full"
log "6L recovery complete rc=$?"
# if 2L already done, mark all-done so the monitor detects completion
[ -f "$RUNS/qhead_2L_xshh_allt_forked10pct_crossfade10pct_qk_aon_b1024_FINAL.pth" ] \
  && [ -f "$RES/gift_eval_full_xshh_allt_forked10pct_crossfade10pct_qk_aon_b1024_2L/summary.txt" ] \
  && [ -f "$RES/gift_eval_full_xshh_allt_forked10pct_crossfade10pct_qk_aon_b1024_6L/summary.txt" ] \
  && touch "$RES/.all_done" && log "touched .all_done"
