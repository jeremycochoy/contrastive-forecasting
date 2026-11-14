#!/bin/bash
# #462: MPE (#421ew, our copy of Moirai) at the size of the width-64 runs.
#
# The trainer command is the MPE command (results/mpe_command.txt, line 1 of
# run_cf421ew_moirai_k3.log on S3) with these changes:
#   --d-model 64 --num-encoder-layers 1   MPE: 384 and 3. Both have 8 heads
#                                         and 3 forecaster layers.
#   --total-steps 166000                  in one process. MPE trained in 8
#                                         legs to the same stops.
#   --extra-save-steps (the 8 stops of MPE), --save-dir, --run-name
#   --keep-start-window-targets           the value objective of MPE, from
#                                         before 289b2fc8
# Each save also writes <name>_<k>k_optimizer.pth.
#
# The trainer runs from a copy of the code: $R/code, a git archive of HEAD at
# the first start. So a change of the checkout does not reach a restart. A
# start resumes from the last checkpoint that loads with its optimizer file,
# and the trainer then names its files <name>_r2_* (r3, ...). After a crash
# the script starts the trainer again, at most MAX_STARTS times. After the
# last stop it trains nothing.
#
# The memory knobs give the same numbers (src/encoders.py, src/blocks.py).
# The GRU patch encoders need them at batch 256 on a shared card.
#
# Usage: bash train.sh                   # the run. GPU: BB_GPU, else the one
#                                        # with the most free memory
#        SMOKE_STEPS=300 bash train.sh   # a short run in $R/smoke
#        DRY_RUN=1 bash train.sh         # print the command, run nothing
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)" || exit 2
R="${CF462_ROOT:-/home/jupyter/cf_runs/cf462}"
NAME=cf462_mpe_w64
TOTAL=166000
STOPS=10000,25000,50000,75000,100000,125000,150000,166000
if [ -n "${SMOKE_STEPS:-}" ]; then
  R="$R/smoke"; TOTAL="$SMOKE_STEPS"; STOPS="$SMOKE_STEPS"
fi
CODE="$R/code"
SAVE="$R/ckpt"
LOGS="$R/logs"
TRAIN="$CODE/experiments/2026-04-27_freq-embedding/scripts/train.py"
MAX_STARTS="${MAX_STARTS:-5}"

ARGS=(
  --qk-norm --attn-out-norm --batch-size 256 --device cuda --lr 1e-3
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 --lr-final 0
  --lr-cosine-steps 166000 --lr-warmup-steps 10000 --grad-clip 1.0
  --seed 20260520 --save-every 5000 --run-name "$NAME" --log-every 200
  --gift-pretrain --freq-vocab v2 --t-raw 4096 --n-channels 1
  --d-model 64 --n-heads 8 --num-encoder-layers 1 --num-layers 3
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads
  --encoder-dropkey-share-layers --depthwise-conv 3
  --deprecated-depthwise-conv 0 --rev-norm-kind ewma --rev-norm-span 128
  --encoder-type gru --synth-kind forked-arma --mix-ratio 0.0078125
  --crossfade-triplets 1 --mixup-p 0.3 --freq-emb-dim 3
  --seasonality-emb-dim 3 --log-attn-amplitude
  --log-attn-amplitude-every 200 --residual-dtype fp32 --attn-dtype fp32
  --ffn-dtype fp32 --conv-dtype fp32 --patch-emb-dtype fp32
  --value-space-objective --train-rollout-depth 3
  --train-rollout-reduce sum --multi-patch-sizes 8,16,32,64,128
  --skip-nan-samples --skip-spike-samples 10
  --total-steps "$TOTAL" --extra-save-steps "$STOPS" --save-dir "$SAVE"
  --keep-start-window-targets
)

. "$HERE/checkpoints.sh"

last_stop() {  # the save of the last step that loads, if any
  stop_save $((TOTAL / 1000))
}

pick_gpu() {  # the GPU with the most free memory
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    sort -t, -k2 -nr | head -1 | cut -d, -f1
}

hf_token_file() {  # the read-only HF token (gitignored): this checkout's, else the main one
  local f
  for f in "$REPO/experiments/hf_token.txt" \
           "$HOME/workspaces/contrastive-forecasting/experiments/hf_token.txt"; do
    if [ -s "$f" ]; then echo "$f"; return; fi
  done
}

snapshot_code() {  # the code of HEAD in $CODE, made once: the trainer and the eval
  [ -d "$CODE" ] && return 0
  rm -rf "$CODE.tmp" && mkdir -p "$CODE.tmp" &&
    git -C "$REPO" archive HEAD src experiments/2026-04-27_freq-embedding \
        experiments/2026-04-13_gift-eval reports/2026-08-08_rollout_depth/scripts \
        reports/2026-08-08_rollout_depth/results/config_costs.csv |
      tar -x -C "$CODE.tmp" &&
    git -C "$REPO" rev-parse HEAD > "$CODE.tmp/CODE_COMMIT" &&
    mv "$CODE.tmp" "$CODE"
}

start() {  # one start of the trainer: resume from the last checkpoint that loads
  local resume gpu
  resume="$(last_checkpoint)"
  gpu="${BB_GPU:-$(pick_gpu)}"
  local run_args=("${ARGS[@]}")
  [ -n "$resume" ] && run_args=(--resume "$resume" "${ARGS[@]}")
  local envs=(PYTHONPATH="$CODE" CUDA_VISIBLE_DEVICES="$gpu"
              PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
              PATCH_ENC_CKPT="${PATCH_ENC_CKPT:-1}"
              PATCH_ENC_CHUNK="${PATCH_ENC_CHUNK:-4}"
              FCST_GRAD_CKPT="${FCST_GRAD_CKPT:-1}"
              BACKBONE_CKPT="${BACKBONE_CKPT:-1}"
              OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3)
  if [ -n "${DRY_RUN:-}" ]; then
    printf '%q ' "${envs[@]}" python3 -u "$TRAIN" "${run_args[@]}"; echo
    return 0
  fi
  echo "$(date -u +%FT%TZ) train: GPU $gpu, ${resume:-a new run}, code $(cat "$CODE/CODE_COMMIT")"
  env "${envs[@]}" python3 -u "$TRAIN" "${run_args[@]}" >> "$LOGS/train.log" 2>&1
}

if [ -z "${DRY_RUN:-}" ]; then
  token="$(hf_token_file)"
  [ -n "$token" ] || { echo "ABORT: no experiments/hf_token.txt" >&2; exit 2; }
  export HF_TOKEN="$(cat "$token")" HUGGING_FACE_HUB_TOKEN="$(cat "$token")"
  mkdir -p "$SAVE" "$LOGS" || exit 2
  echo $$ > "$LOGS/train.pid"
  snapshot_code || { echo "ABORT: no code copy" >&2; exit 2; }
fi
for n in $(seq 1 "$MAX_STARTS"); do
  done_ckpt="$(last_stop)"
  if [ -n "$done_ckpt" ]; then
    echo "$(date -u +%FT%TZ) train: $done_ckpt exists, nothing to train"; exit 0
  fi
  start
  rc=$?
  [ -n "${DRY_RUN:-}" ] && exit 0
  echo "$(date -u +%FT%TZ) train: start $n of $MAX_STARTS ended with code $rc"
  [ -n "$(last_stop)" ] || sleep "${RESTART_WAIT:-120}"
done
[ -n "$(last_stop)" ] && exit 0
echo "$(date -u +%FT%TZ) train: $MAX_STARTS starts used, the last stop is missing" >&2
exit 1
