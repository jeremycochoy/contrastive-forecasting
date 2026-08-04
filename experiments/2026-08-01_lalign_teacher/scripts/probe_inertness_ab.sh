#!/bin/bash
# #390 review item 3 — is the latent-drift probe inert with respect to training?
#
# The probe runs inside every #390 wave and did not exist when #379's sweep
# ran (#379's logs print no "Latent-drift CSV" line and its runs/ holds no
# drift CSV). So the teacher cells and the copied student cells differ by the
# probe as well as by --align-target. If the probe perturbs the trained
# weights at all, the flag-for-flag launcher test cannot see it.
#
# This settles it on GPU rather than by reading the source: arm5's own
# command line, 200 steps, twice — probe on (what the waves ran) and
# --no-latent-drift-probe. Same seed, same data, same everything else. If
# the two [200] lines are identical the probe is inert.
#
# Usage:  WT=/home/jupyter/wt-cf-390-train GPU=1 bash probe_inertness_ab.sh
set -uo pipefail

WT="${WT:-$HOME/wt-cf-390-train}"
case "$WT" in
  /tmp/*|/tmp) echo "ABORT: WT=$WT is under /tmp — refusing." >&2; exit 2 ;;
esac
GPU="${GPU:-1}"
STEPS="${STEPS:-200}"

EXP="$WT/experiments/2026-08-01_lalign_teacher"
RES="$EXP/results"; mkdir -p "$RES"
SCRATCH="$EXP/runs_probe_ab"; mkdir -p "$SCRATCH"
OUT="$RES/probe_inertness_ab.txt"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK=64
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4 TEACHER_EMBED_CHUNK=16
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# arm5's line from run_arm.sh, with the teacher target the waves ran.
run_leg() {  # <name> [extra flags...]
  local name="$1"; shift
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u "$TRAIN" --qk-norm --attn-out-norm \
    --batch-size 64 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.98 --seed 20260520 \
    --save-every 25000 --extra-save-steps 2500 \
    --save-dir "$SCRATCH" --run-name "$name" --log-every 200 \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8 \
    --num-encoder-layers 3 --num-layers 3 \
    --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 \
    --loss-shape cosine_similarity_batch_rep_only \
    --align-loss-weight 1.0 --align-target teacher \
    --ema-embedding --ema-encoder --ema-tau 0.9 --cpc-infonce-weight 1.0 \
    --sigreg-embedding --sigreg-encoding --sigreg-n-chunk 2048 \
    --sigreg-embedding-weight 1.0 --sigreg-encoding-weight 1.0 \
    --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
    --synth-kind forked-arma --mix-ratio 0.0078125 --crossfade-triplets 1 \
    --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
    --log-attn-amplitude --log-attn-amplitude-every 200 \
    --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 \
    --patch-emb-dtype fp32 "$@" 2>&1
}

{
  echo "# #390 — latent-drift probe inertness A/B. elisa, RTX 4090 (GPU $GPU)."
  echo "# arm5's own command line, $STEPS steps, seed 20260520, back to back."
  echo "# The probe defaults ON; #379's sweep ran on code that had none."
  echo
} > "$OUT"

rm -f "$SCRATCH"/probe_ab_*
for leg in on off; do
  case "$leg" in
    on)  name=probe_ab_on;  extra=() ;;
    off) name=probe_ab_off; extra=(--no-latent-drift-probe) ;;
  esac
  log="$SCRATCH/${name}.log"
  run_leg "$name" "${extra[@]}" > "$log"
  rc=$?
  {
    echo "## probe $leg (rc=$rc)"
    grep -E "^Latent-drift CSV" "$log" || echo "(no latent-drift CSV line)"
    grep -E "^\[ *${STEPS}\]" "$log" || echo "(no [$STEPS] line — leg failed)"
    grep -E "^ +R²_rand" "$log" | tail -1
    echo
  } >> "$OUT"
done

# The verdict, computed rather than eyeballed.
a=$(grep -E "^\[ *${STEPS}\]" "$SCRATCH/probe_ab_on.log"  | tail -1)
b=$(grep -E "^\[ *${STEPS}\]" "$SCRATCH/probe_ab_off.log" | tail -1)
# sps and ETA are wall-clock, not training state — strip them before comparing.
strip(){ sed -E 's/[0-9.]+ sps//; s/ETA [0-9.]+h//'; }
if [ -n "$a" ] && [ "$(printf '%s' "$a" | strip)" = "$(printf '%s' "$b" | strip)" ]; then
  echo "VERDICT: identical at step $STEPS — the probe is inert w.r.t. training." >> "$OUT"
else
  echo "VERDICT: DIFFERENT at step $STEPS — the probe perturbs training." >> "$OUT"
fi
cat "$OUT"
