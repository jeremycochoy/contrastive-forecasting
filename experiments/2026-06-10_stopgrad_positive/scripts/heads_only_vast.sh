#!/bin/bash
# #339 budget restructure: remaining q-head TRAININGS only on the vast 4090
# (evals moved to elisa). Mirrors downstream_sgpos.sh train_head exactly:
#   1. 2L last-head (resume from 2L best FINAL, 10k re-adapt on final.pth)
#   2. 6L best-head (30k on FINAL.pth)
#   3. 6L last-head (resume from 6L best FINAL, 10k on final.pth)
# Then prints ALL HEADS COMPLETE (sync watches run_all.log).
# PROVENANCE NOTE: this is the verbatim script that ran the three heads on
# instance 40410773 (launched 2026-06-11 09:40 UTC as /workspace/heads_only.sh);
# committed for protocol provenance. --save-every 100000 (no periodic head
# checkpoints) is an accepted deviation from the reference ops — heads are
# ~1-3h and the streaming _best.pth was synced every 15 min.
set -uo pipefail
WT=/workspace/app
OUT=/workspace/out
TAG="allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024"
RUNS="$OUT/runs"; RES="$OUT/results"
exec > >(tee -a "$OUT/run_all.log") 2>&1
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
BB="$RUNS/bb_${TAG}_FINAL.pth"; BBLAST="$RUNS/bb_${TAG}_final.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [heads-only] $*"; }
arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 6 \
      --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128)

train_head(){ # run_name head_layers backbone resume_src total warmup
  local qn="$1" hl="$2" bb="$3" src="$4" tot="$5" wu="$6" qf="$RUNS/$1_FINAL.pth"
  [ -f "$qf" ] && { log "QH $qn skip (FINAL exists)"; return 0; }
  local rflag=(); [ -n "$src" ] && rflag=(--resume "$src")
  log "QH $qn train $tot on $(basename "$bb")"
  python3 -u "$QTRAIN" "${rflag[@]}" --backbone-path "$bb" --forecast-len 16 --quantile-head \
    --head-arch transformer --head-causal true --head-num-layers "$hl" --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f --total-steps "$tot" --batch-size 256 --lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --weight-decay 0.1 --schedule cosine --warmup-steps "$wu" --final-lr-ratio 0.1 \
    --save-every 100000 --log-every 200 --save-dir "$RUNS" --run-name "$qn" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --device cuda "${arch[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype none \
    >>"$RES/run_${qn}.log" 2>&1 || { log "QH $qn FAILED (tail: $(tail -3 "$RES/run_${qn}.log"|tr '\n' ' '))"; return 1; }
  if   [ -f "$RUNS/${qn}_best.pth" ];  then cp -f "$RUNS/${qn}_best.pth"  "$qf"
  elif [ -f "$RUNS/${qn}_final.pth" ]; then cp -f "$RUNS/${qn}_final.pth" "$qf"
  else cp -f "$(ls -t "$RUNS/${qn}"_*k.pth 2>/dev/null|head -1)" "$qf"; fi
  [ -f "$qf" ] || { log "QH $qn no checkpoint"; return 1; }; log "QH $qn done"; }

log "=== HEADS-ONLY CHAIN (evals moved to elisa) ==="
train_head "qhead_2L_${TAG}_last" 2 "$BBLAST" "$RUNS/qhead_2L_${TAG}_FINAL.pth" 10000 1000 || exit 1
train_head "qhead_6L_${TAG}"      6 "$BB"     ""                                30000 2000 || exit 1
train_head "qhead_6L_${TAG}_last" 6 "$BBLAST" "$RUNS/qhead_6L_${TAG}_FINAL.pth" 10000 1000 || exit 1
log "=== ALL HEADS COMPLETE ==="
