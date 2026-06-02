#!/bin/bash
# #322 — GPU-gated orchestrator. Trains the 5 batch-1024 backbones sequentially via
# 2-GPU DDP. Each launch waits until BOTH physical GPUs have enough free memory
# (~23.6 GB/rank measured peak) so we never OOM ourselves OR a foreign tenant on the
# shared box. Idempotent (train script skips FINAL-complete arms) and resumable.
#
# A one-off DDP smoke (40 steps, throwaway) validates the real per-rank fit/sps before
# committing to the full runs; if it OOMs the orchestrator stops with a clear marker
# (→ switch to chunked-GRU-checkpointing contingency).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024
RES="$OUT/results"; mkdir -p "$RES"
OLOG="$RES/orchestrate.log"
STEPS="${STEPS:-12500}"; SAVE_EVERY="${SAVE_EVERY:-2500}"; ALLT_CHUNK="${ALLT_CHUNK:-2}"
GATE_MIB="${GATE_MIB:-23600}"   # require this many MiB free on EACH gpu before a DDP launch
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$OLOG"; }

free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$(($1+1))p"; }
gpus_free(){ local f0 f1; f0=$(free_mib 0); f1=$(free_mib 1); [ "${f0:-0}" -ge "$GATE_MIB" ] && [ "${f1:-0}" -ge "$GATE_MIB" ]; }
wait_for_gpus(){
  local waited=0
  until gpus_free; do
    [ $((waited % 600)) -eq 0 ] && log "WAIT gpus free0=$(free_mib 0) free1=$(free_mib 1) MiB (need ${GATE_MIB} each)"
    sleep 120; waited=$((waited+120))
  done
  log "GPUs free (free0=$(free_mib 0) free1=$(free_mib 1)) — launching"
}

# ---- one-off DDP smoke (validate real per-rank fit + sps) -------------------------
smoke_ok="$RES/.ddp_smoke_ok"
if [ ! -f "$smoke_ok" ]; then
  wait_for_gpus
  log "DDP SMOKE: β 40 steps (throwaway) to verify forward512+loss1024 fit"
  slog="$RES/ddp_smoke.log"; : >"$slog"
  WT=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024
  export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="0,1"
  export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  torchrun --nproc_per_node=2 --master_port=29500 "$WT/experiments/2026-04-27_freq-embedding/scripts/train.py" \
    --batch-size 512 --device cuda --total-steps 40 --lr 1e-3 --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.98 --seed 20260520 --save-every 1000000 \
    --save-dir "$OUT/runs" --run-name ddp_smoke --log-every 10 \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-encoder-layers 6 --num-layers 6 \
    --forecaster-d-model 128 --forecaster-n-heads 4 \
    --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 \
    --loss-shape cosine_similarity_batch_full_hh_negs --pos-in-denominator \
    --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
    --synth-kind forked-arma --mix-ratio 0.10 --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
    --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
    >>"$slog" 2>&1
  rc=$?
  rm -f "$OUT"/runs/ddp_smoke_*.pth "$OUT"/runs/ddp_smoke_losses.csv 2>/dev/null
  if [ $rc -ne 0 ] || grep -qiE 'out of memory|CUDA error' "$slog"; then
    log "DDP SMOKE FAILED rc=$rc (OOM? $(grep -ciE 'out of memory' "$slog")). tail: $(tail -2 "$slog"|tr '\n' ' ')"
    log "STOP — need chunked-GRU-checkpointing contingency. Not launching full runs."
    exit 3
  fi
  log "DDP SMOKE OK: $(grep -oE '[0-9.]+ sps' "$slog" | tail -1) ; $(grep -i 'timing' "$slog" | tail -1)"
  touch "$smoke_ok"
fi

# ---- full runs: 5 arms, β first (faster) then all-time -----------------------------
#         arm      mix         tag           chunk  port
ARMS=(
  "beta     0.10        forked10pct   8   29501"
  "beta     0.0078125   forked2       8   29502"
  "alltime  0.5         forked        $ALLT_CHUNK 29503"
  "alltime  0.10        forked10pct   $ALLT_CHUNK 29504"
  "alltime  0.0078125   forked2       $ALLT_CHUNK 29505"
)
for spec in "${ARMS[@]}"; do
  set -- $spec; arm="$1"; mix="$2"; tag="$3"; chunk="$4"; port="$5"
  log ">>> ARM $arm $tag (mix=$mix chunk=$chunk steps=$STEPS)"
  wait_for_gpus
  bash "$HERE/train_backbone_b1024.sh" "$arm" "$mix" "$tag" "$STEPS" "$SAVE_EVERY" "$chunk" "$port"
  log "<<< ARM $arm $tag rc=$?"
done
log "ORCHESTRATOR DONE (all 5 arms attempted)"
