#!/bin/bash
# #341 — fast end-to-end smoke for one arm at the real batch 1024 for a handful
# of steps: validates data pipeline, forward, loss, step, AND that the model is
# the intended one (param count must match the #336 no-stop-grad twin exactly —
# the stop-grad adds no parameters). Leaves no FINAL.
#   smoke_sgcap.sh <arm: nobn_enc6|bn_enc6> <gpu> [steps]
set -uo pipefail
ARM="${1:?arm}"; GPU="${2:?gpu}"; STEPS="${3:-20}"
WT="${WT:-/tmp/cf-341}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity}"
case "$ARM" in
  nobn_enc6) EXTRA=(); WANT_PARAMS="22,063,164" ;;   # = #336 xftrip_nobn_enc6
  bn_enc6)   EXTRA=(--forecaster-d-model 128 --forecaster-n-heads 4); WANT_PARAMS="12,714,684" ;;  # = #336 xftrip_bn_enc6
  *) echo "unknown arm: $ARM"; exit 2 ;;
esac
RES="$OUT/results"; mkdir -p "$RES" "$OUT/smoke_runs"
NAME="smoke_sgcap_${ARM}"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export XSHH_ALLT_CHUNK="${XSHH_ALLT_CHUNK:-2}" CUDA_VISIBLE_DEVICES="$GPU"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
tlog="$RES/smoke_${ARM}.log"; : > "$tlog"
echo "[smoke-$ARM] start GPU=$GPU steps=$STEPS"
python3 -u "$TRAIN" --qk-norm --attn-out-norm \
  --batch-size 1024 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed 20260520 \
  --save-every 100000 --save-dir "$OUT/smoke_runs" --run-name "$NAME" --log-every 5 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 6 "${EXTRA[@]}" \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt --pos-in-denominator --subtract-contrastive-floor \
  --stopgrad-positive-h \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.0078125 --crossfade-triplets 1 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
echo "[smoke-$ARM] python rc=$rc"
echo "--- params ---"; grep -E "Params:" "$tlog" || echo "MISSING PARAMS LINE"
echo "--- mix line ---"; grep -E "Data: MIX" "$tlog" || echo "MISSING MIX LINE"
echo "--- last steps ---"; grep -iE "step|loss|sps" "$tlog" | tail -4
if grep -qiE "nan|Traceback|Error|out of memory" "$tlog"; then echo "[smoke-$ARM] FAIL: nan/error/oom in log"; exit 1; fi
grep -q "Params: $WANT_PARAMS" "$tlog" \
  || { echo "[smoke-$ARM] FAIL: param count != $WANT_PARAMS (wrong architecture?)"; exit 1; }
[ $rc -eq 0 ] && grep -q "1 triplet(s)=3 rows" "$tlog" \
  && echo "[smoke-$ARM] PASS" || { echo "[smoke-$ARM] FAIL rc=$rc or mix line mismatch"; exit 1; }
