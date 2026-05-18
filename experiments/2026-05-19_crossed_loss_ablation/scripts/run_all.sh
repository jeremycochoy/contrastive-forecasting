#!/bin/bash
# #303 crossed-loss ablation orchestrator. Self-contained, idempotent
# (every step skips if its output exists, so a re-run resumes after any
# interruption). Recipe = the (A) bottleneck-fullfh backbone-OF-RECORD
# run (#296 orchestrator a1: 1L forecaster, fp16 group residual-fp32 /
# attn-ffn-conv-fp16, 2-GPU DDP global-batch 256, 50k, seed 20260517,
# --pos-in-denominator) — the run #296/#300 evaluate. ONLY --loss-shape
# changes. (run_ddp.sh's literal 2L/bf16 diverged at ~1.1k per
# 2026-05-17 RESULTS.md, so it is NOT the comparable (A) recipe.)
#
# Phase 1: 50k DDP backbone per variant, serial (DDP holds both GPUs),
#          order B, C, A+B (issue #303). (A) already has a run.
# Phase 2: per good backbone, 30k 2L-causal q-head + official GIFT-Eval
#          (triage 11 + full 97). Two variants run concurrently, one GPU
#          each (q-head/eval are single-GPU).
set -uo pipefail

CODE=/home/jupyter/cf-wt-crossed-loss
ART=/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_ablation
RUNS="$ART/runs"; RES="$ART/results"
TRAIN="$CODE/experiments/2026-04-27_freq-embedding/scripts/train.py"
QTRAIN="$CODE/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$CODE/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
WATCH="$CODE/experiments/2026-05-19_crossed_loss_ablation/scripts/watch_divergence.sh"
GE=/home/jupyter/workspaces/gift-eval-data
HFT=/home/jupyter/contrastive-forecasting/experiments/hf_token.txt
OLOG="$RES/orchestrate.log"
TRIAGE='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'
SEED=20260517; TOTAL=50000

mkdir -p "$RUNS" "$RES"
export PYTHONPATH="$CODE" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$HFT")" HUGGING_FACE_HUB_TOKEN="$(cat "$HFT")"
export GIFT_EVAL="$GE"
cd "$CODE"
exec 9>"$RES/.run_all.lock"; flock -n 9 || { echo "[lock] run_all already running"; exit 0; }
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$OLOG"; }
freeport(){ python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1; }

# variant short -> loss_shape (issue order: B, C, A+B)
VARS=(hh ff fhhh)
declare -A LS=(
  [hh]=cosine_similarity_batch_full_hh_negs
  [ff]=cosine_similarity_batch_full_ff_negs
  [fhhh]=cosine_similarity_batch_full_fh_hh_negs
)

DT=(--residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32)
common(){ # $1=NAME $2=loss_shape
  echo --device cuda --total-steps "$TOTAL" --lr 1e-3 --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.95 --seed "$SEED" \
    --save-every 5000 --save-dir "$RUNS" --run-name "$1" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
    --num-encoder-layers 6 --num-layers 1 \
    --forecaster-d-model 128 --forecaster-n-heads 4 \
    --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 \
    --loss-shape "$2" --pos-in-denominator \
    --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
    --mixup-p 0.3 --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 \
    --log-attn-amplitude --log-attn-amplitude-every 200
}

train_backbone(){ # $1=short
  local v="$1" name="cl_${1}_50k"
  local tlog="$RES/run_${name}.log" st="$RES/status_${name}.txt"
  if [ -f "$RUNS/${name}_FINAL.pth" ]; then log "BB SKIP $v ($name FINAL exists)"; return 0; fi
  : > "$tlog"
  export CUDA_VISIBLE_DEVICES=0,1
  log "BB START $v | $name | loss=${LS[$v]} | DDP 0,1 bs128 50k"
  setsid bash -c 'exec "$@" >>"'"$tlog"'" 2>&1' _ \
    torchrun --nproc_per_node=2 --master_port="$(freeport)" "$TRAIN" \
    --batch-size 128 $(common "$name" "${LS[$v]}") "${DT[@]}" < /dev/null &
  local tp="" pgid="" k=0
  while [ "$k" -lt 60 ]; do
    tp=$(pgrep -f -- "--run-name $name" 2>/dev/null | head -1)
    [ -n "$tp" ] && { pgid=$(ps -o pgid= -p "$tp" 2>/dev/null | tr -d ' '); [ -n "$pgid" ] && break; }
    k=$((k+1)); sleep 1
  done
  [ -z "$pgid" ] && { log "BB ERROR $v never started (tail: $(tail -2 "$tlog"|tr '\n' ' '))"; return 1; }
  log "BB $v launched pgid=$pgid; watcher attached"
  bash "$WATCH" "$RUNS" "$name" "$TOTAL" "$st" "$pgid" "$tlog" >>"$OLOG" 2>&1
  local s; s=$(grep -m1 '^STATUS=' "$st" 2>/dev/null | cut -d= -f2)
  if [ "$s" = DONE ]; then
    if   [ -f "$RUNS/${name}_best_loss.pth" ]; then cp -f "$RUNS/${name}_best_loss.pth" "$RUNS/${name}_FINAL.pth"
    elif [ -f "$RUNS/${name}_final.pth" ];     then cp -f "$RUNS/${name}_final.pth"     "$RUNS/${name}_FINAL.pth"
    else cp -f "$(ls -t "$RUNS/${name}"_*k.pth 2>/dev/null|head -1)" "$RUNS/${name}_FINAL.pth" 2>/dev/null; fi
    [ -f "$RUNS/${name}_FINAL.pth" ] && { log "BB DONE $v -> ${name}_FINAL.pth"; return 0; }
    log "BB $v DONE-but-no-checkpoint"; return 1
  fi
  log "BB $s $v ($(grep -m1 '^DETAIL=' "$st" 2>/dev/null|cut -d= -f2-)) — variant excluded from downstream"
  return 1
}

downstream(){ # $1=short $2=gpu  (q-head 30k + triage + full eval, single GPU)
  local v="$1" g="$2" name="cl_${1}_50k"
  local bb="$RUNS/${name}_FINAL.pth"
  [ -f "$bb" ] || { log "DOWN SKIP $v (no backbone FINAL)"; return 1; }
  local qn="${name}_qhead_xfmr2L_quant_30k" qf="$RUNS/${name}_qhead_FINAL.pth"
  local arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 \
              --forecaster-d-model 128 --forecaster-n-heads 4 --encoder-type gru \
              --rev-norm-kind ewma --rev-norm-span 128)
  if [ ! -f "$qf" ]; then
    log "DOWN q-head $v on GPU$g ($qn)"
    CUDA_VISIBLE_DEVICES="$g" python3 -u "$QTRAIN" \
      --backbone-path "$bb" --forecast-len 16 --quantile-head --head-arch transformer \
      --head-causal true --head-num-layers 2 --head-nhead 6 --head-ffn-mult 4.0 \
      --head-dropout 0.1 --head-train-input e_then_f \
      --total-steps 30000 --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 \
      --weight-decay 0.1 --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
      --save-every 5000 --log-every 200 --save-dir "$RUNS" --run-name "$qn" \
      --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
      --device cuda "${arch[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype bf16 \
      >>"$RES/run_${qn}.log" 2>&1 || { log "DOWN q-head FAILED $v"; return 1; }
    if   [ -f "$RUNS/${qn}_best.pth" ];  then cp -f "$RUNS/${qn}_best.pth"  "$qf"
    elif [ -f "$RUNS/${qn}_final.pth" ]; then cp -f "$RUNS/${qn}_final.pth" "$qf"
    else cp -f "$(ls -t "$RUNS/${qn}"_*k.pth 2>/dev/null|head -1)" "$qf"; fi
  fi
  [ -f "$qf" ] || { log "DOWN q-head no-checkpoint $v"; return 1; }
  log "DOWN q-head ready $v"
  do_eval(){ # $1=tag $2=outdir $3=filter
    local tag="$1" out="$2" filt="$3"
    [ -f "$out/summary.txt" ] && { log "DOWN $tag eval exists (GM=$(gm "$out/summary.txt"))"; return 0; }
    mkdir -p "$out"
    local ff=(); [ -n "$filt" ] && ff=(--config-filter "$filt")
    log "DOWN $tag eval $v on GPU$g"
    CUDA_VISIBLE_DEVICES="$g" python3 -u "$QEVAL" \
      --backbone-path "$bb" --head-path "$qf" --output-dir "$out" --strategy B4 \
      --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 \
      --num-layers 1 --forecaster-d-model 128 --forecaster-n-heads 4 \
      --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 \
      --device cuda --head-causal true "${ff[@]}" \
      >>"$RES/run_eval_${tag}.log" 2>&1 || { log "DOWN $tag eval FAILED $v"; return 1; }
    log "DOWN $tag eval DONE $v GM=$(gm "$out/summary.txt")"
  }
  do_eval "triage_${v}" "$RES/gift_eval_triage_${name}" "$TRIAGE" || true
  do_eval "full_${v}"   "$RES/gift_eval_full_${name}"   ""        || true
  log "DOWN COMPLETE $v | triageGM=$(gm "$RES/gift_eval_triage_${name}/summary.txt") fullGM=$(gm "$RES/gift_eval_full_${name}/summary.txt")"
}

log "=== RUN_ALL START $(date) | code @ $(git -C "$CODE" rev-parse --short HEAD) ==="
# ---- Phase 1: backbones, serial (DDP uses both GPUs) ----
for v in "${VARS[@]}"; do train_backbone "$v" || true; done
log "=== PHASE1 BACKBONES COMPLETE $(date) ==="
# ---- Phase 2: downstream, two variants concurrent (one GPU each) ----
GPUS=(0 1); i=0; pids=()
for v in "${VARS[@]}"; do
  [ -f "$RUNS/cl_${v}_50k_FINAL.pth" ] || { log "PHASE2 skip $v (no backbone)"; continue; }
  g=${GPUS[$((i % 2))]}
  ( downstream "$v" "$g" ) >>"$OLOG" 2>&1 &
  pids+=($!); i=$((i+1))
  [ $((i % 2)) -eq 0 ] && { wait "${pids[@]}"; pids=(); }
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
log "=== RUN_ALL COMPLETE $(date) ==="
for v in "${VARS[@]}"; do
  n="cl_${v}_50k"
  log "RESULT $v: triageGM=$(gm "$RES/gift_eval_triage_${n}/summary.txt" ) fullGM=$(gm "$RES/gift_eval_full_${n}/summary.txt")"
done
