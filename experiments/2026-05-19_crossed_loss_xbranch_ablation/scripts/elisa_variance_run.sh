#!/bin/bash
# elisa variance run — train one (arm, seed) pair fully on elisa (2× 4090).
# Recipe = byte-identical to #303/#307 box_run.sh (--loss-shape + --seed
# the only knobs). Backbone: 50k DDP, batch 128/GPU = 256 global. q-head:
# 30k 2L-causal. Eval: triage 11 + full 97.
#
# Usage: elisa_variance_run.sh <arm> <seed>
#   arm  = hh | hhxbf         (only the two top arms — B and B-xbfree)
#   seed = 20260518 | 20260519 | ...
#
# Writes to:
#   <MAIN>/variance/<arm>_seed<seed>/{runs,results}/
# Idempotent: skips backbone / q-head / eval phases when their FINAL/
# summary.txt already exists.
set -uo pipefail
ARM="${1:?arm = hh|hhxbf}"; SEED="${2:?seed}"; GPU="${3:-0}"
case "$ARM" in
  hh)    LS=cosine_similarity_batch_full_hh_negs ;;
  hhxbf) LS=cosine_similarity_batch_full_hh_negs_xbfree ;;
  *) echo "unknown arm $ARM (expected: hh | hhxbf)" >&2; exit 2 ;;
esac

# MAIN checkout holds the run outputs (CLAUDE.md: sync dirs / valuable
# untracked state in main checkout). WORKTREE holds the new #303/#307
# code (loss shapes + tests + scripts) — main checkout's train.py on
# branch experiments-synced doesn't yet know about the new loss-shape
# choices, so we load source from the worktree explicitly.
APP=/home/jupyter/contrastive-forecasting
WT=/home/jupyter/cf-wt-crossed-loss
cd "$APP" || { echo "cannot cd $APP" >&2; exit 3; }
EXP="$APP/experiments/2026-05-19_crossed_loss_xbranch_ablation"
ROOT="$EXP/variance/${ARM}_seed${SEED}"
RUNS="$ROOT/runs"; RES="$ROOT/results"; mkdir -p "$RUNS" "$RES"
TOTAL=50000
NAME="cl_${ARM}_50k_s${SEED:(-2)}"   # short suffix e.g. s18, s19
BB="$RUNS/${NAME}_FINAL.pth"
QN="${NAME}_qhead_xfmr2L_quant_30k"; QF="$RUNS/${NAME}_qhead_FINAL.pth"
TRIAGE='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null || cat "$APP/experiments/hf_token.txt")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data

TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [$ARM seed=$SEED] $*"; }
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1; }
freeport(){ python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

backbone(){
  [ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; return 0; }
  local tlog="$RES/run_${NAME}.log"
  local ng; ng=$(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null)
  [ "${ng:-0}" -ge 2 ] || { log "BB ERROR: need 2 GPUs, have $ng"; return 1; }
  log "BB START loss=$LS DDP nproc=2 bs128 ${TOTAL} -> $RUNS"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port="$(freeport)" "$TRAIN" \
    --batch-size 128 --device cuda --total-steps "$TOTAL" --lr 1e-3 --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.95 --seed "$SEED" \
    --save-every 5000 --save-dir "$RUNS" --run-name "$NAME" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
    --num-encoder-layers 6 --num-layers 1 \
    --forecaster-d-model 128 --forecaster-n-heads 4 \
    --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 \
    --loss-shape "$LS" --pos-in-denominator \
    --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
    --mixup-p 0.3 --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 \
    --log-attn-amplitude --log-attn-amplitude-every 200 \
    --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 \
    --patch-emb-dtype fp32 >>"$tlog" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "BB train exited rc=$rc (tail: $(tail -3 "$tlog"|tr '\n' ' '))"; fi
  if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
  elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
  else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
  for c in "${NAME}_best_loss" "${NAME}_final"; do
    [ -f "$RUNS/${c}_optimizer.pth" ] && cp -f "$RUNS/${c}_optimizer.pth" "$RUNS/${NAME}_FINAL_optimizer.pth" && break
  done
  [ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; return 0; }
  log "BB FAILED no checkpoint"; return 1
}

qhead(){
  [ -f "$BB" ] || { log "QH SKIP (no backbone FINAL)"; return 1; }
  [ -f "$QF" ] && { log "QH SKIP (FINAL exists)"; return 0; }
  local GPU="${1:-0}"
  local arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 \
              --forecaster-d-model 128 --forecaster-n-heads 4 --encoder-type gru \
              --rev-norm-kind ewma --rev-norm-span 128)
  log "QH START on GPU$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u "$QTRAIN" \
    --backbone-path "$BB" --forecast-len 16 --quantile-head --head-arch transformer \
    --head-causal true --head-num-layers 2 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f \
    --total-steps 30000 --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 \
    --weight-decay 0.1 --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 --save-dir "$RUNS" --run-name "$QN" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --device cuda "${arch[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype bf16 \
    >>"$RES/run_${QN}.log" 2>&1 || { log "QH FAILED (tail: $(tail -3 "$RES/run_${QN}.log"|tr '\n' ' '))"; return 1; }
  if   [ -f "$RUNS/${QN}_best.pth" ];  then cp -f "$RUNS/${QN}_best.pth"  "$QF"
  elif [ -f "$RUNS/${QN}_final.pth" ]; then cp -f "$RUNS/${QN}_final.pth" "$QF"
  else cp -f "$(ls -t "$RUNS/${QN}"_*k.pth 2>/dev/null|head -1)" "$QF"; fi
  [ -f "$QF" ] && { log "QH DONE -> ${QN}_FINAL.pth"; return 0; }
  log "QH FAILED no checkpoint"; return 1
}

do_eval(){ # $1=tag $2=outdir $3=filter $4=gpu
  local tag="$1" out="$2" filt="$3" GPU="${4:-0}"
  [ -f "$out/summary.txt" ] && { log "EVAL $tag exists GM=$(gm "$out/summary.txt")"; return 0; }
  [ -f "$BB" ] && [ -f "$QF" ] || { log "EVAL $tag SKIP (missing bb/qf)"; return 1; }
  mkdir -p "$out"
  local ff=(); [ -n "$filt" ] && ff=(--config-filter "$filt")
  log "EVAL $tag START GPU$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u "$QEVAL" \
    --backbone-path "$BB" --head-path "$QF" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 \
    --num-layers 1 --forecaster-d-model 128 --forecaster-n-heads 4 \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 \
    --device cuda --head-causal true "${ff[@]}" \
    >>"$RES/run_eval_${tag}.log" 2>&1 || { log "EVAL $tag FAILED (tail: $(tail -3 "$RES/run_eval_${tag}.log"|tr '\n' ' '))"; return 1; }
  log "EVAL $tag DONE GM=$(gm "$out/summary.txt")"
}

# Full pipeline (backbone + downstream)
log "=== START arm=$ARM seed=$SEED gpu=$GPU -> $ROOT ==="
backbone || { log "ABORT: backbone failed"; exit 1; }
qhead "$GPU"  || { log "ABORT: qhead failed"; exit 1; }
do_eval "triage_${NAME}" "$RES/gift_eval_triage_${NAME}" "$TRIAGE" "$GPU" || true
do_eval "full_${NAME}"   "$RES/gift_eval_full_${NAME}"   ""        "$GPU" || true
log "=== COMPLETE arm=$ARM seed=$SEED triageGM=$(gm "$RES/gift_eval_triage_${NAME}/summary.txt") fullGM=$(gm "$RES/gift_eval_full_${NAME}/summary.txt") ==="
