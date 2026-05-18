#!/bin/bash
# Overnight downstream pipeline. Waits for the orchestrator to land a clean
# 50k backbone, then (proven v13 bottleneck recipe, paths/names adapted):
#   1. train the standard 2L causal-transformer q-head (30k)
#   2. GIFT-Eval triage (11 cfg) then full (97 cfg) -> GM-Relative MASE
#   3. PRIMARY DONE marker (the must-have for the morning)
#   4. best-effort, time-gated, CONTINUOUS-optimizer backbone extensions
#      50k->100k then 100k->150k (resume from the run's own _Nk.pth +
#      _Nk_optimizer.pth — never a cold optimizer; OVERNIGHT_PLAN rule),
#      each guarded by the fixed divergence watcher.
# Idempotent: every step skips if its output exists. Single-GPU work runs
# on GPU1 (GPU0 holds other sessions' notebooks).
set -uo pipefail

EXP=/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-17_bottleneck_fullfh_ddp
CODE=/home/jupyter/cf-wt-bottleneck-fullfh
GE=/home/jupyter/workspaces/gift-eval-data
HFT=/home/jupyter/contrastive-forecasting/experiments/hf_token.txt
RUNS="$EXP/runs"; RES="$EXP/results"
DLOG="$RES/downstream.log"; DSTAT="$RES/downstream_status.txt"
OSTAT="$RES/orchestrator_status.txt"; WATCH="$EXP/scripts/watch_divergence.sh"
TRAIN="$CODE/experiments/2026-04-27_freq-embedding/scripts/train.py"
QTRAIN="$CODE/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$CODE/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
# Stop *starting* a new ~2h backbone-extension leg after this clock time
# (HH:MM, 24h) so the morning deliverables are stable. q-head/eval still
# allowed after (they are the must-have).
EXT_CUTOFF="05:30"
TRIAGE='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'
export PYTHONPATH="$CODE" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="$(cat "$HFT")" HUGGING_FACE_HUB_TOKEN="$(cat "$HFT")"
export GIFT_EVAL="$GE"
mkdir -p "$RUNS" "$RES"; cd "$CODE"
# Single-instance guard — prevents the duplicate-process race that broke
# the orchestrator (a detached copy surviving a cancelled launch).
exec 9>"$RES/.downstream.lock"
flock -n 9 || { echo "[lock] another downstream_pipeline.sh already holds $RES/.downstream.lock — exiting"; exit 0; }
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$DLOG"; }
dstat(){ echo "$*" >> "$DSTAT"; }
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1; }
before_cutoff(){ [ "$(date +%H%M)" -lt "$(echo "$EXT_CUTOFF"|tr -d :)" ] && return 0 || return 1; }

: > "$DSTAT"; log "=== DOWNSTREAM START $(date) — waiting for backbone SUCCESS ==="
# 1) Wait for orchestrator SUCCESS; parse the winning run name + FINAL path.
BB="" NAME=""
while :; do
  s=$(head -1 "$OSTAT" 2>/dev/null || true)
  case "$s" in
    SUCCESS\ *) NAME=$(echo "$s" | awk '{print $3}'); BB=$(echo "$s" | sed -n 's/.*FINAL=\([^ ]*\).*/\1/p'); break ;;
    FAILED*)    log "orchestrator FAILED ('$s') — no backbone; downstream cannot run"; dstat "ABORTED no-backbone $(date '+%F %T')"; exit 1 ;;
  esac
  sleep 120
done
[ -f "$BB" ] || BB="$RUNS/${NAME}_FINAL.pth"
[ -f "$BB" ] || { log "FINAL not found for $NAME ($BB)"; dstat "ABORTED missing-FINAL $(date '+%F %T')"; exit 1; }
log "backbone SUCCESS: $NAME | FINAL=$BB"
dstat "BACKBONE $NAME at $(date '+%F %T')"
# winning precision/mode are irrelevant to q-head (arch identical); needed
# only to make the extensions a faithful continuation.
GRP=fp32; echo "$NAME" | grep -q fp16 && GRP=fp16
DT_FP16="--residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32"
DT_FP32="--residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 --conv-dtype fp32 --patch-emb-dtype fp32"
DT=$DT_FP32; [ "$GRP" = fp16 ] && DT=$DT_FP16

ARCH=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 \
      --forecaster-d-model 128 --forecaster-n-heads 4 --encoder-type gru \
      --rev-norm-kind ewma --rev-norm-span 128)

# 2) q-head (2L causal transformer, 30k) — v13 recipe verbatim, our paths.
QN="${NAME}_qhead_xfmr2L_quant_30k"; QF="$RUNS/${QN}_FINAL.pth"
if [ ! -f "$QF" ]; then
  log "q-head train: $QN (GPU1)"
  CUDA_VISIBLE_DEVICES=1 python3 -u "$QTRAIN" \
    --backbone-path "$BB" --forecast-len 16 --quantile-head --head-arch transformer --head-causal true \
    --head-num-layers 2 --head-nhead 6 --head-ffn-mult 4.0 --head-dropout 0.1 --head-train-input e_then_f \
    --total-steps 30000 --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 --save-every 5000 --log-every 200 \
    --save-dir "$RUNS" --run-name "$QN" --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --device cuda "${ARCH[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype bf16 \
    >>"$RES/run_${QN}.log" 2>&1 || { log "q-head FAILED"; dstat "QHEAD-FAILED $(date '+%F %T')"; exit 1; }
  if   [ -f "$RUNS/${QN}_best.pth" ];  then cp -f "$RUNS/${QN}_best.pth"  "$QF"
  elif [ -f "$RUNS/${QN}_final.pth" ]; then cp -f "$RUNS/${QN}_final.pth" "$QF"
  else cp -f "$(ls -t "$RUNS/${QN}"_*k.pth 2>/dev/null|head -1)" "$QF"; fi
fi
log "q-head ready: $QF"; dstat "QHEAD $QN at $(date '+%F %T')"

# 3) GIFT-Eval triage (11) then full (97).
do_eval(){ # $1=tag $2=outdir $3=filter|''
  local tag="$1" out="$2" filt="$3"
  [ -f "$out/summary.txt" ] && { log "$tag eval exists (GM=$(gm "$out/summary.txt"))"; return 0; }
  mkdir -p "$out"
  local ff=(); [ -n "$filt" ] && ff=(--config-filter "$filt")
  log "$tag eval starting (GPU1)"
  CUDA_VISIBLE_DEVICES=1 python3 -u "$QEVAL" \
    --backbone-path "$BB" --head-path "$QF" --output-dir "$out" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 1 \
    --forecaster-d-model 128 --forecaster-n-heads 4 --encoder-type gru \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true "${ff[@]}" \
    >>"$RES/run_eval_${tag}.log" 2>&1 || { log "$tag eval FAILED"; return 1; }
  log "$tag eval DONE GM=$(gm "$out/summary.txt")"
}
do_eval "triage_${NAME}" "$RES/gift_eval_triage_${NAME}" "$TRIAGE" || true
do_eval "full_${NAME}"   "$RES/gift_eval_full_${NAME}"   ""        || true
TGM=$(gm "$RES/gift_eval_triage_${NAME}/summary.txt"); FGM=$(gm "$RES/gift_eval_full_${NAME}/summary.txt")
log "=== PRIMARY DONE: backbone+qhead+eval | triage GM=${TGM:-?} full GM=${FGM:-?} ==="
dstat "PRIMARY-DONE triageGM=${TGM:-?} fullGM=${FGM:-?} at $(date '+%F %T')"

# 4) Best-effort continuous-optimizer extensions 50k->100k->150k.
#    Resume from the winning run's own _Nk.pth (+ _Nk_optimizer.pth) so the
#    optimizer trajectory is unbroken; never reuse a save-path.
extend(){ # $1=from_k $2=to_steps $3=mode $4=bs $5=shard  -> sets EXT_NAME on success
  local fk="$1" to="$2" mode="$3" bs="$4" shard="$5"
  local src="$RUNS/${NAME}_${fk}k.pth"
  [ -f "$src" ] && [ -f "$RUNS/${NAME}_${fk}k_optimizer.pth" ] || { log "ext: missing ${fk}k(+opt) for continuous resume — skip"; return 1; }
  local en="${NAME%_50k}_resume${fk}k_${to}"; local ef="$RUNS/${en}_FINAL.pth"
  [ -f "$ef" ] && { log "ext $en already FINAL"; EXT_NAME="$en"; return 0; }
  before_cutoff || { log "ext: past ${EXT_CUTOFF} cutoff — not starting ${en}"; return 1; }
  log "ext: $en  resume $src  -> $to steps (continuous optimizer, $mode bs$bs shard$shard $GRP)"
  local sh=(); [ "$shard" = 1 ] && sh=(--shard-loss-on-batch)
  local launch
  if [ "$mode" = ddp ]; then export CUDA_VISIBLE_DEVICES=0,1
    launch=(torchrun --nproc_per_node=2 --master_port="$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')" "$TRAIN" --batch-size "$bs" "${sh[@]}")
  else export CUDA_VISIBLE_DEVICES=1; launch=(python3 -u "$TRAIN" --batch-size "$bs"); fi
  setsid bash -c 'exec "$@" >>"'"$RES/run_${en}.log"'" 2>&1' _ "${launch[@]}" \
    --device cuda --total-steps "$to" --resume "$src" \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --seed 20260517 \
    --save-every 5000 --save-dir "$RUNS" --run-name "$en" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    "${ARCH[@]}" --num-encoder-layers 6 \
    --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 $DT \
    --loss-shape cosine_similarity_batch_full_fh_negs --pos-in-denominator \
    --tau 0.10 --mixup-p 0.3 --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 \
    --log-attn-amplitude --log-attn-amplitude-every 200 < /dev/null &
  sleep 3; local tp pgid k=0
  while [ "$k" -lt 40 ]; do tp=$(pgrep -f -- "--run-name $en" 2>/dev/null|head -1); [ -n "$tp" ] && { pgid=$(ps -o pgid= -p "$tp"|tr -d ' '); [ -n "$pgid" ] && break; }; k=$((k+1)); sleep 1; done
  [ -z "${pgid:-}" ] && { log "ext $en never started"; return 1; }
  bash "$WATCH" "$RUNS" "$en" "$to" "$RES/status_${en}.txt" "$pgid" "$RES/run_${en}.log" >>"$DLOG" 2>&1
  local st; st=$(grep -m1 '^STATUS=' "$RES/status_${en}.txt" 2>/dev/null|cut -d= -f2)
  if [ "$st" = DONE ]; then
    [ -f "$RUNS/${en}_best_loss.pth" ] && cp -f "$RUNS/${en}_best_loss.pth" "$ef"
    [ ! -f "$ef" ] && [ -f "$RUNS/${en}_final.pth" ] && cp -f "$RUNS/${en}_final.pth" "$ef"
    log "ext $en DONE -> $ef"; dstat "EXT $en DONE at $(date '+%F %T')"; EXT_NAME="$en"; return 0
  fi
  log "ext $en NOT clean (status=$st) — stopping extensions, prior checkpoints preserved"
  dstat "EXT $en $st at $(date '+%F %T')"; return 1
}
# infer winning mode/bs/shard from the run name for a faithful continuation
MODE=ddp; echo "$NAME" | grep -q single && MODE=single
BS=$(echo "$NAME" | grep -oE '(ddp|single)[0-9]+' | grep -oE '[0-9]+'); BS=${BS:-128}
SHARD=0; echo "$NAME" | grep -q shard && SHARD=1
if extend 50 100000 "$MODE" "$BS" "$SHARD"; then
  NAME="$EXT_NAME"   # chain 100k->150k from the just-finished 100k run
  extend 100 150000 "$MODE" "$BS" "$SHARD" || true
fi
log "=== DOWNSTREAM COMPLETE $(date) ==="
dstat "COMPLETE at $(date '+%F %T')"
