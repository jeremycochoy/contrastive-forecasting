#!/bin/bash
# Unattended overnight orchestrator. Goal: a trained 1L-forecaster backbone
# by morning. Encodes the user's fallback chain (2026-05-17, user leaving,
# no questions):
#   forecaster = 1L. Try fp16 body first. If it DIVERGES -> pure fp32.
#   If it OOMs -> --shard-loss-on-batch. If still OOM -> shrink per-rank
#   batch, then single-GPU. First attempt that finishes 50k clean wins.
#
# Precision (residual stays fp32 — the standing safety anchor / 2026-05-16
# rule; only the attn/ffn/conv body precision changes):
#   fp16 group: residual fp32 + attn/ffn/conv fp16
#   fp32 group: everything fp32
# Divergence -> jump to the next precision group (not a memory problem).
# OOM        -> next memory config within the same precision group.
# Each attempt is monitored by the fixed scripts/watch_divergence.sh.
set -uo pipefail

EXP=/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-17_bottleneck_fullfh_ddp
CODE=/home/jupyter/cf-wt-bottleneck-fullfh
TRAIN="$CODE/experiments/2026-04-27_freq-embedding/scripts/train.py"
SAVE_DIR="$EXP/runs"; RES="$EXP/results"
OLOG="$RES/orchestrator.log"; OSTAT="$RES/orchestrator_status.txt"
WATCH="$EXP/scripts/watch_divergence.sh"
TOTAL_STEPS=50000; SEED=20260517
HFT=/home/jupyter/contrastive-forecasting/experiments/hf_token.txt
mkdir -p "$SAVE_DIR" "$RES"
export PYTHONPATH="$CODE" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$HFT")" HUGGING_FACE_HUB_TOKEN="$(cat "$HFT")"
cd "$CODE"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$OLOG"; }
stat(){ echo "$*" > "$OSTAT"; }
freeport(){ python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

# Attempt table: ID GROUP MODE BS SHARD LR  (order encodes the policy)
ATTEMPTS=(
  "a1 fp16 ddp 128 0 1e-3"
  "a2 fp16 ddp 128 1 1e-3"
  "a3 fp16 ddp 96 1 1e-3"
  "a4 fp16 ddp 64 1 1e-3"
  "a5 fp16 single 128 0 1e-3"
  "a6 fp16 single 64 0 1e-3"
  "b1 fp32 ddp 128 0 1e-3"
  "b2 fp32 ddp 128 1 1e-3"
  "b3 fp32 ddp 96 1 1e-3"
  "b4 fp32 ddp 64 1 1e-3"
  "b5 fp32 single 96 0 1e-3"
  "b6 fp32 single 64 0 1e-3"
  "c1 fp32 ddp 128 0 5e-4"
)
dtype_args(){ # $1=group
  if [ "$1" = fp16 ]; then echo "--residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32"
  else echo "--residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 --conv-dtype fp32 --patch-emb-dtype fp32"; fi; }

run_attempt(){ # $1=ID $2=GROUP $3=MODE $4=BS $5=SHARD $6=LR  -> echoes DONE|DIVERGED|OOM|CRASHED
  local id="$1" grp="$2" mode="$3" bs="$4" shard="$5" lr="$6"
  local tag="${grp}_${mode}${bs}"; [ "$shard" = 1 ] && tag="${tag}_shard"; [ "$lr" != "1e-3" ] && tag="${tag}_lr${lr}"
  local NAME="enc_fcst_bneck128_dk07_fullfh_norminfonce_1L_${tag}_50k"
  local TLOG="$RES/run_${NAME}.log"; local ST="$RES/status_${NAME}.txt"
  if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then log "SKIP $id ($NAME) FINAL exists"; echo DONE; return; fi
  : > "$TLOG"
  local common=(--device cuda --total-steps "$TOTAL_STEPS" --lr "$lr" \
    --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --seed "$SEED" \
    --save-every 5000 --save-dir "$SAVE_DIR" --run-name "$NAME" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
    --num-encoder-layers 6 --num-layers 1 \
    --forecaster-d-model 128 --forecaster-n-heads 4 \
    --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 \
    --loss-shape cosine_similarity_batch_full_fh_negs --pos-in-denominator \
    --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
    --mixup-p 0.3 --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 \
    --log-attn-amplitude --log-attn-amplitude-every 200)
  read -r -a dt <<< "$(dtype_args "$grp")"
  local launch
  if [ "$mode" = ddp ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    local sh=(); [ "$shard" = 1 ] && sh=(--shard-loss-on-batch)
    launch=(torchrun --nproc_per_node=2 --master_port="$(freeport)" "$TRAIN" --batch-size "$bs" "${sh[@]}" "${common[@]}" "${dt[@]}")
  else
    export CUDA_VISIBLE_DEVICES=1   # GPU1 is the free card (GPU0 has notebooks)
    launch=(python3 -u "$TRAIN" --batch-size "$bs" "${common[@]}" "${dt[@]}")
  fi
  log "ATTEMPT $id | $NAME | mode=$mode bs=$bs shard=$shard lr=$lr grp=$grp"
  setsid bash -c 'exec "$@" >>"'"$TLOG"'" 2>&1' _ "${launch[@]}" < /dev/null &
  # Resolve the real training process group via the UNIQUE run-name
  # (setsid's $! is unreliable). The cmdline carries --run-name NAME from
  # the first instant the process exists.
  local tp="" pgid="" k=0
  while [ "$k" -lt 40 ]; do
    tp=$(pgrep -f -- "--run-name $NAME" 2>/dev/null | head -1)
    [ -n "$tp" ] && { pgid=$(ps -o pgid= -p "$tp" 2>/dev/null | tr -d ' '); [ -n "$pgid" ] && break; }
    k=$((k+1)); sleep 1
  done
  if [ -z "$pgid" ]; then
    log "  ERROR: training process for $NAME never appeared (tail: $(tail -2 "$TLOG"|tr '\n' ' '))"
    echo CRASHED; return
  fi
  log "  launched tp=$tp pgid=$pgid; watcher attached"
  bash "$WATCH" "$SAVE_DIR" "$NAME" "$TOTAL_STEPS" "$ST" "$pgid" "$TLOG" >>"$OLOG" 2>&1
  local s; s=$(grep -m1 '^STATUS=' "$ST" 2>/dev/null | cut -d= -f2)
  if [ "$s" = DONE ]; then
    [ -f "${SAVE_DIR}/${NAME}_best_loss.pth" ] && cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
    [ ! -f "${SAVE_DIR}/${NAME}_FINAL.pth" ] && [ -f "${SAVE_DIR}/${NAME}_final.pth" ] && cp -f "${SAVE_DIR}/${NAME}_final.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
    echo DONE; return
  fi
  if [ "$s" = DIVERGED ]; then
    log "  -> DIVERGED ($(grep -m1 '^DETAIL=' "$ST"|cut -d= -f2-))"
    rm -f "${SAVE_DIR}/${NAME}"_*.pth   # prune useless weights, keep CSVs+log
    echo DIVERGED; return
  fi
  # CRASH: classify OOM vs other
  if grep -qaiE 'out of memory|outofmemory|CUDA error: out of memory' "$TLOG"; then
    log "  -> OOM ($(grep -aiE 'out of memory' "$TLOG"|tail -1|cut -c1-160))"
    rm -f "${SAVE_DIR}/${NAME}"_*.pth; echo OOM; return
  fi
  log "  -> CRASHED non-OOM ($(grep -m1 '^DETAIL=' "$ST"|cut -d= -f2-))"
  echo CRASHED; return
}

log "=== ORCHESTRATOR START $(date) | code @ $(git -C "$CODE" rev-parse --short HEAD) ==="
stat "RUNNING since $(date '+%F %T')"
i=0; n=${#ATTEMPTS[@]}; launches=0; crash_retried=""
while [ "$i" -lt "$n" ]; do
  set -- ${ATTEMPTS[$i]}; ID="$1"; GRP="$2"; MODE="$3"; BS="$4"; SHARD="$5"; LR="$6"
  launches=$((launches+1))
  if [ "$launches" -gt 16 ]; then log "ABORT: >16 launches, giving up"; stat "FAILED exhausted $(date '+%F %T')"; exit 1; fi
  res=$(run_attempt "$ID" "$GRP" "$MODE" "$BS" "$SHARD" "$LR")
  case "$res" in
    DONE)
      NAME="enc_fcst_bneck128_dk07_fullfh_norminfonce_1L_${GRP}_${MODE}${BS}"
      [ "$SHARD" = 1 ] && NAME="${NAME}_shard"; [ "$LR" != "1e-3" ] && NAME="${NAME}_lr${LR}"; NAME="${NAME}_50k"
      log "=== SUCCESS: $ID $NAME finished 50k clean. FINAL=${SAVE_DIR}/${NAME}_FINAL.pth ==="
      stat "SUCCESS $ID $NAME FINAL=${SAVE_DIR}/${NAME}_FINAL.pth at $(date '+%F %T')"
      exit 0 ;;
    DIVERGED)
      # precision problem -> jump to the next precision group's first attempt
      if [ "$GRP" = fp16 ]; then nx=fp32; else nx=safety; fi
      if [ "$nx" = safety ]; then
        # only the c1 last-resort remains
        j=$i; while [ "$j" -lt "$n" ]; do set -- ${ATTEMPTS[$j]}; [ "$1" = c1 ] && break; j=$((j+1)); done
        if [ "$j" -ge "$n" ]; then log "FAILED: fp32 diverged and no safety left"; stat "FAILED fp32-diverged $(date '+%F %T')"; exit 1; fi
        i=$j; continue
      fi
      j=0; while [ "$j" -lt "$n" ]; do set -- ${ATTEMPTS[$j]}; [ "$2" = "$nx" ] && break; j=$((j+1)); done
      log "  jump: divergence in $GRP -> group $nx (attempt index $j)"; i=$j; continue ;;
    OOM)
      i=$((i+1)); continue ;;            # next memory config (same group order)
    CRASHED)
      if [ "$crash_retried" != "$ID" ]; then
        crash_retried="$ID"; log "  retry $ID once (transient non-OOM crash)"; continue
      fi
      log "  $ID crashed twice non-OOM; advancing memory config"; i=$((i+1)); continue ;;
  esac
done
log "FAILED: attempt table exhausted with no clean 50k"
stat "FAILED exhausted $(date '+%F %T')"; exit 1
