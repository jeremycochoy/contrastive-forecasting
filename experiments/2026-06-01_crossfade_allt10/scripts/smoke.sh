#!/bin/bash
# #325 — fast end-to-end smoke (on elisa GPU). Runs the EXACT crossfade recipe for a
# handful of steps to validate the 3-way data pipeline, forward, loss, and step before
# committing ~12 h of GPU. Asserts the mix print and finite loss; leaves no FINAL.
set -uo pipefail
GPU="${1:-1}"; STEPS="${2:-30}"
WT="${WT:-/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/crossfade-allt10}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-01_crossfade_allt10}"
RES="$OUT/results"; mkdir -p "$RES" "$OUT/smoke_runs"
NAME="smoke_xfade"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4 CUDA_VISIBLE_DEVICES="$GPU"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
tlog="$RES/smoke.log"; : > "$tlog"
echo "[smoke] start GPU=$GPU steps=$STEPS"
python3 -u "$TRAIN" --qk-norm --attn-out-norm \
  --batch-size 1024 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed 20260520 \
  --save-every 100000 --save-dir "$OUT/smoke_runs" --run-name "$NAME" --log-every 5 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 6 \
  --forecaster-d-model 128 --forecaster-n-heads 4 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt --pos-in-denominator --subtract-contrastive-floor \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.10 --crossfade-ratio 0.10 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
echo "[smoke] python rc=$rc"
echo "--- mix line ---"; grep -E "Data: MIX" "$tlog" || echo "MISSING MIX LINE"
echo "--- last steps ---"; grep -iE "step|loss" "$tlog" | tail -6
if grep -qiE "nan|inf(inity)?\b|Traceback|Error" "$tlog"; then echo "[smoke] FAIL: nan/error in log"; exit 1; fi
[ $rc -eq 0 ] && grep -q "Data: MIX 80% HF + 10% synth (forked-arma) + 10% crossfade" "$tlog" \
  && echo "[smoke] PASS" || { echo "[smoke] FAIL rc=$rc or mix line mismatch"; exit 1; }
