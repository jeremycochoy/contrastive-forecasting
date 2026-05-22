#!/bin/bash
# #313 parallel downstream — same protocol as downstream.sh, but the
# full-97 GIFT-Eval is SHARDED across both elisa GPUs (≈halves wall time)
# and the triage(11) runs separately with the exact #309 filter. The
# q-head (single-GPU, unavoidable) is reused if already trained/running.
#
# Shard regexes (results/shard{A,B}.regex) are anchored alternations over
# the eval's FILTER strings f"{ds_name}/{term}" (NOT the freq-bearing
# output names) — see scripts notes.
#
#   parallel_downstream.sh        (no args; uses GPU0 + GPU1)
set -uo pipefail
GPU_A=0; GPU_B=1
FCST=(--forecaster-d-model 128 --forecaster-n-heads 4)   # (B) bottleneck

WT=/home/jupyter/cf-wt-align-floor
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-22_align_floor_loss_B
NAME=bb_alignfloor_50k
BB="$MAIN/runs/${NAME}_FINAL.pth"
QN="${NAME}_qhead_xfmr2L_quant_30k"
QF="$MAIN/runs/${NAME}_qhead_FINAL.pth"
RUNS="$MAIN/runs"; RES="$MAIN/results"; mkdir -p "$RES"
TRIAGE='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null || cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
LG="$RES/finish.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [pdown] $*" | tee -a "$LG"; }
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1; }

arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 "${FCST[@]}" \
      --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128)

# ---- 1. ensure backbone + q-head ----
[ -f "$BB" ] || { log "ABORT: backbone $BB missing"; exit 1; }
ensure_qhead(){
  [ -f "$QF" ] && { log "QF exists -> skip q-head"; return 0; }
  if pgrep -f "train_forecasting_head.py.*${QN}" >/dev/null 2>&1; then
    log "q-head already running; waiting for it to finish 30k..."
    while pgrep -f "train_forecasting_head.py.*${QN}" >/dev/null 2>&1; do sleep 30; done
    log "q-head process ended"
  fi
  if   [ -f "$RUNS/${QN}_final.pth" ]; then cp -f "$RUNS/${QN}_final.pth" "$QF"; log "promoted from _final.pth"
  elif [ -f "$RUNS/${QN}_best.pth" ];  then cp -f "$RUNS/${QN}_best.pth"  "$QF"; log "promoted from _best.pth"
  elif ls "$RUNS/${QN}"_*k.pth >/dev/null 2>&1; then cp -f "$(ls -t "$RUNS/${QN}"_*k.pth|head -1)" "$QF"; log "promoted from latest _Nk.pth"
  fi
  [ -f "$QF" ] && return 0
  log "no q-head checkpoint -> training fresh 30k on GPU$GPU_B"
  CUDA_VISIBLE_DEVICES="$GPU_B" python3 -u "$QTRAIN" \
    --backbone-path "$BB" --forecast-len 16 --quantile-head --head-arch transformer \
    --head-causal true --head-num-layers 2 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f \
    --total-steps 30000 --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 \
    --weight-decay 0.1 --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 --save-dir "$RUNS" --run-name "$QN" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --device cuda "${arch[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype bf16 \
    >>"$RES/run_${QN}.log" 2>&1 || { log "QH train FAILED"; return 1; }
  if   [ -f "$RUNS/${QN}_best.pth" ];  then cp -f "$RUNS/${QN}_best.pth"  "$QF"
  elif [ -f "$RUNS/${QN}_final.pth" ]; then cp -f "$RUNS/${QN}_final.pth" "$QF"; fi
  [ -f "$QF" ]
}
ensure_qhead || { log "ABORT: q-head unavailable"; exit 1; }
log "q-head ready: $(du -h "$QF"|cut -f1)"

# ---- 2. eval helper ----
run_eval(){ # $1=gpu $2=outdir $3=filter
  local g="$1" out="$2" filt="$3"
  [ -f "$out/summary.txt" ] && { log "EVAL $out exists GM=$(gm "$out/summary.txt")"; return 0; }
  mkdir -p "$out"
  local ff=(); [ -n "$filt" ] && ff=(--config-filter "$filt")
  CUDA_VISIBLE_DEVICES="$g" python3 -u "$QEVAL" \
    --backbone-path "$BB" --head-path "$QF" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 \
    --num-layers 1 "${FCST[@]}" \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 \
    --device cuda --head-causal true "${ff[@]}" \
    >>"$RES/run_eval_$(basename "$out").log" 2>&1
}

# ---- 3. full-97 + triage(11). Shard the full eval across GPU0∥GPU1 only
#         if GPU0 has room (good neighbour on the shared box); else run the
#         full eval single-GPU on GPU_B. Either way the result is canonical. ----
RXA="$(cat "$RES/shardA.regex")"; RXB="$(cat "$RES/shardB.regex")"
FULL="$RES/gift_eval_full_${NAME}"
g0_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_A" 2>/dev/null | tr -d ' ')
g0_free=$(( 24564 - ${g0_used:-24564} ))
if [ ! -f "$FULL/summary.txt" ]; then
  if [ "${g0_free:-0}" -ge 10000 ]; then
    log "full-97 SHARD START: A(gpu$GPU_A,free=${g0_free}MiB) ∥ B(gpu$GPU_B)"
    run_eval "$GPU_A" "$RES/shardA_${NAME}" "$RXA" & pa=$!
    run_eval "$GPU_B" "$RES/shardB_${NAME}" "$RXB" & pb=$!
    wait $pa; ra=$?; wait $pb; rb=$?
    log "shards done rcA=$ra rcB=$rb"
    python3 "$WT/experiments/2026-05-22_align_floor_loss_B/scripts/merge_shards.py" \
      "$FULL" "$RES/shardA_${NAME}" "$RES/shardB_${NAME}" 2>&1 | tee -a "$LG"
    nfull=$(grep -oE '\(([0-9]+) configs\)' "$FULL/summary.txt" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    if [ "${nfull:-0}" -ne 97 ]; then
      log "MERGE gave ${nfull:-0}≠97 configs — FALLBACK to single-GPU full eval on GPU$GPU_B"
      rm -f "$FULL/summary.txt"
      run_eval "$GPU_B" "$FULL" ""
    fi
  else
    log "GPU$GPU_A busy (free=${g0_free}MiB<10GB) — full eval single-GPU on GPU$GPU_B (no shard)"
    run_eval "$GPU_B" "$FULL" ""
  fi
fi
log "full-97 GM=$(gm "$FULL/summary.txt")"

TRIG="$RES/gift_eval_triage_${NAME}"
run_eval "$GPU_A" "$TRIG" "$TRIAGE" || log "triage eval rc=$?"
log "triage-11 GM=$(gm "$TRIG/summary.txt")"

# ---- 4. plots ----
log "--- plots ---"
python3 "$WT/experiments/2026-05-22_align_floor_loss_B/scripts/plot_loss.py"    2>&1 | tee -a "$LG" || true
python3 "$WT/experiments/2026-05-22_align_floor_loss_B/scripts/plot_radar.py"   2>&1 | tee -a "$LG" || true
python3 "$WT/experiments/2026-05-22_align_floor_loss_B/scripts/plot_summary.py" 2>&1 | tee -a "$LG" || true

echo "PARALLEL_DOWNSTREAM_DONE $(date -u +%FT%TZ) fullGM=$(gm "$FULL/summary.txt") triageGM=$(gm "$TRIG/summary.txt")" > "$RES/.chain_done"
log "=== COMPLETE fullGM=$(gm "$FULL/summary.txt") triageGM=$(gm "$TRIG/summary.txt") ==="
