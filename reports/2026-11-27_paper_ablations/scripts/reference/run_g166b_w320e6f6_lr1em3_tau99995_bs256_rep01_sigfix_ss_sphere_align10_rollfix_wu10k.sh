#!/bin/bash
# g166b_w320e6f6_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k: g166b_w64e6f6_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k at width 320 (6 + 6 layers, 15,461,771 parameters; MPE 14,376,771) with the bf16 dtypes of the width-384 runs (the user, 11-25: 'we are going to do a 320-66 model', the best configuration of each family, with the warmup).
# g166b_w64e6f6_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k: g166b_w384e2f4_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k (green 2 + 4, warmup + constant) at width 64 with 6 encoder + 6 forecaster layers, the warmup 10k then a constant 1e-3 (the user, 11-24: 'So let's try to train light blue Grey at 64 latent and shape 6-6 [...] And also the tau 0.3 at 6-6! [...] And the green with warmup also at 6-6 but 64 latent', 'This 3 with the warmup OK?').
# g166b_w384e2f4_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k: g166b_w384e2f4_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix (green-384 with 2 encoder + 4 forecaster layers) with the Moirai warmup, linear from 0 to 1e-3 over
# 10k steps, then a constant 1e-3 (--lr-final 1e-3: the anneal runs from 1e-3 to 1e-3). The user, 11-23 about 12:15 UTC:
# "you can stop the purple 1e-3 cosntnt then cosine, and replace it by warm up then cosntant with configuration 2 encoder
# 4 forecast. using same machine."
# g166b_w384e2f4_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix: g166b_w384e1f5_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix (green-384) with 2 encoder and 4 forecaster layers in place of 1 and 5. The user, 11-22 about 16:10 UTC: 'one more la same as green 364 but with layers 2/4 instead of 1/5. Scoring and checkpoints saved in Elisa.' then 'On a new machine'. All other flags (muP bases 64, 3, 3 included) and the code (code_038ea26c) are green's. On the box cf-w384-e2f4 (vast 61999534).
# g166b_w384e1f5_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix: g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix at width 384 with 1 encoder and 5 forecaster layers (muP base 64, 3, 3), bf16 in place of fp16, 166k steps from step 0 (make_w384.py).
# g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix: D1 of the width-64 search of 11-08: A with L_align x 10 at its 4 depths (the user: 'we'll try a variation when we have 10 times Lcos or L-align'), 166k steps.
# Written by /workspace/hp64/make_run.py (the recipe and the changes of this run are there). Code: /workspace/hp64/code_038ea26c.
# A start with numbered checkpoints in the save folder resumes from the last one that loads, with its optimizer state
# (--resume); a resume writes ${NAME}_r2_<k>k.pth (then _r3, ...), and the glob takes them too. When the last save
# exists in any branch, this script writes the file stop (the supervisor then ends) and trains nothing.
# ABC_GIFT_DRY_RUN=1 prints the command and runs nothing.
set -uo pipefail
R="${R:-/workspace/hp64/g166b_w320e6f6_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k}"
WT="${WT:-/workspace/hp64/code_038ea26c}"
SAVE="$R/ckpt"
NAME="g166b_w320e6f6_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k"
TOTAL="${TOTAL:-166000}"

loads() {  # a checkpoint and its optimizer state both load
  python3 - "$1" <<'PY' >/dev/null 2>&1
import sys, torch
path = sys.argv[1]
torch.load(path, map_location="cpu", weights_only=False)
torch.load(path[:-4] + "_optimizer.pth", map_location="cpu", weights_only=False)
PY
}

last_checkpoint() {  # the numbered checkpoint of the highest step that loads
  ls "$SAVE"/${NAME}_[0-9]*k.pth "$SAVE"/${NAME}_r[0-9]*_[0-9]*k.pth 2>/dev/null | grep -v _optimizer |
    sed -E 's/.*_([0-9]+)k\.pth$/\1 &/' | sort -rn | cut -d' ' -f2 |
    while read -r ckpt; do
      if loads "$ckpt"; then echo "$ckpt"; break; fi
    done
}

ARGS=(
  --qk-norm
  --attn-out-norm
  --batch-size 256
  --device cuda
  --total-steps "$TOTAL"
  --lr 1e-3
  --lr-warmup-steps 10000
  --lr-final 1e-3
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --seed 20260520
  --save-every 5000
  --extra-save-steps 250,1000,166000
  --save-dir "$SAVE"
  --run-name "$NAME"
  --log-every 200
  --gift-pretrain
  --freq-vocab v2
  --t-raw 4096
  --n-channels 1
  --d-model 320
  --n-heads 8
  --num-encoder-layers 6
  --num-layers 6
  --encoder-dropkey 0.70
  --encoder-dropkey-share-heads
  --encoder-dropkey-share-layers
  --depthwise-conv 3
  --deprecated-depthwise-conv 0
  --loss-shape cosine_similarity_batch_rep_only
  --align-loss-weight 10.0
  --moco-rep-keys
  --tau-rep 1.0
  --align-target teacher
  --ema-embedding
  --ema-encoder
  --ema-tau 0.99995
  --ema-tau-end 1.0
  --ema-tau-ramp-steps "$TOTAL"
  --cpc-infonce-weight 0.0
  --sigreg-embedding
  --sigreg-encoding
  --sigreg-n-chunk 2048
  --sigreg-embedding-weight 1.0
  --sigreg-encoding-weight 1.0
  --tau 0.10
  --rev-norm-kind ewma
  --rev-norm-span 128
  --encoder-type gru
  --synth-kind forked-arma
  --mix-ratio 0.0078125
  --crossfade-triplets 1
  --mixup-p 0.3
  --freq-emb-dim 3
  --seasonality-emb-dim 3
  --log-attn-amplitude
  --log-attn-amplitude-every 200
  --residual-dtype fp32
  --attn-dtype bf16
  --ffn-dtype bf16
  --conv-dtype bf16
  --patch-emb-dtype fp32
  --train-rollout-depth 3
  --train-rollout-reduce sum
  --rep-loss-weight 0.1
  --sigreg-latent-scale inv_sqrt_d
  --mup-base-d-model 64
  --mup-base-num-layers 3
  --mup-base-num-encoder-layers 3
  --mup-init-output 0
  --mup-init-input 0.3
  --latent-sphere
)
RESUME="$(last_checkpoint)"
[ -n "$RESUME" ] && ARGS=(--resume "$RESUME" "${ARGS[@]}")

# The environment of the grid runs (run_grid40k_elisa_ln.sh).
ENVS=(PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
      FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=8 CPC_CB_CHUNK=256
      PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=1 TEACHER_EMBED_CHUNK=1
      CUDA_VISIBLE_DEVICES="${BB_GPU:?BB_GPU: the elisa GPU of the run}")
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
if [ -n "${ABC_GIFT_DRY_RUN:-}" ]; then
  echo "${ENVS[*]} python3 -u $TRAIN ${ARGS[*]}"
  exit 0
fi
FINAL="$(ls "$SAVE"/${NAME}_$((TOTAL / 1000))k.pth "$SAVE"/${NAME}_r[0-9]*_$((TOTAL / 1000))k.pth 2>/dev/null | head -1)"
if [ -n "$FINAL" ]; then
  echo "$(date -u +%FT%TZ) run: $FINAL exists, nothing to train: the file stop ends the supervisor"
  echo "reached the last step: $(basename "$FINAL")" > "$R/stop"
  exit 0
fi
mkdir -p "$SAVE" || exit 2
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt")" HUGGING_FACE_HUB_TOKEN="$(cat "$WT/experiments/hf_token.txt")"
exec env "${ENVS[@]}" python3 -u "$TRAIN" "${ARGS[@]}"
