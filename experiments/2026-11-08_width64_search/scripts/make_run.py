"""Writes the run script and the supervisor of one run of the width-64 search of 11-08 (the user, about 13:10 UTC:
"we'll go back in smaller scale, we'll go 64 hidden with the red recipe ... the baseline will be this red, shrunk
back"). The red recipe is SPH36 (grn_ema099995_f6_rep01_ss256_sphere). At width 64 with 1 encoder and 3 forecaster
layers it is the box run g80b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_rollfix (SPH, 80k steps): its flags,
with the total steps of the run, the EMA ramp over them and a save every 5k steps (and at 250 and 1k steps).
The user, about 13:25 UTC: "Maybe we could just train 80k with more intermediary checkpoint then? Because if we use
the small red we already have, we already have the baseline computed": the variants train 80k steps. SPH is A.

Usage: python3 make_run.py <name> <code folder> <total steps> <why> [flag value ...]
  a flag that the recipe has gets the new value. A new flag is added at the end (a flag with no value: "")
"""
import os
import sys
from pathlib import Path

HP = Path(os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64"))
RECIPE = [  # (flag, value or None)
    ("--qk-norm", None), ("--attn-out-norm", None), ("--batch-size", "256"), ("--device", "cuda"),
    ("--total-steps", '"$TOTAL"'), ("--lr", "1e-3"), ("--weight-decay", "0.1"), ("--adam-beta1", "0.9"),
    ("--adam-beta2", "0.98"), ("--seed", "20260520"), ("--save-every", "5000"),
    ("--extra-save-steps", "250,1000"), ("--save-dir", '"$SAVE"'), ("--run-name", '"$NAME"'),
    ("--log-every", "200"), ("--gift-pretrain", None), ("--freq-vocab", "v2"), ("--t-raw", "4096"),
    ("--n-channels", "1"), ("--d-model", "64"), ("--n-heads", "8"), ("--num-encoder-layers", "1"),
    ("--num-layers", "3"), ("--encoder-dropkey", "0.70"), ("--encoder-dropkey-share-heads", None),
    ("--encoder-dropkey-share-layers", None), ("--depthwise-conv", "3"), ("--deprecated-depthwise-conv", "0"),
    ("--loss-shape", "cosine_similarity_batch_rep_only"), ("--align-loss-weight", "1.0"),
    ("--moco-rep-keys", None), ("--tau-rep", "1.0"), ("--align-target", "teacher"), ("--ema-embedding", None),
    ("--ema-encoder", None), ("--ema-tau", "0.99995"), ("--ema-tau-end", "1.0"),
    ("--ema-tau-ramp-steps", '"$TOTAL"'), ("--cpc-infonce-weight", "0.0"), ("--sigreg-embedding", None),
    ("--sigreg-encoding", None), ("--sigreg-n-chunk", "2048"), ("--sigreg-embedding-weight", "1.0"),
    ("--sigreg-encoding-weight", "1.0"), ("--tau", "0.10"), ("--rev-norm-kind", "ewma"),
    ("--rev-norm-span", "128"), ("--encoder-type", "gru"), ("--synth-kind", "forked-arma"),
    ("--mix-ratio", "0.0078125"), ("--crossfade-triplets", "1"), ("--mixup-p", "0.3"),
    ("--freq-emb-dim", "3"), ("--seasonality-emb-dim", "3"), ("--log-attn-amplitude", None),
    ("--log-attn-amplitude-every", "200"), ("--residual-dtype", "fp32"), ("--attn-dtype", "fp16"),
    ("--ffn-dtype", "fp16"), ("--conv-dtype", "fp16"), ("--patch-emb-dtype", "fp32"),
    ("--train-rollout-depth", "3"), ("--train-rollout-reduce", "sum"), ("--rep-loss-weight", "0.1"),
    ("--sigreg-latent-scale", "inv_sqrt_d"), ("--mup-base-d-model", "64"), ("--mup-base-num-layers", "3"),
    ("--mup-base-num-encoder-layers", "3"), ("--mup-init-output", "0"), ("--mup-init-input", "0.3"),
    ("--latent-sphere", None),
]

RUN = r'''#!/bin/bash
# @NAME@: @WHY@
# Written by make_run.py (experiments/2026-11-08_width64_search/scripts) (the recipe and the changes of this run are there). Code: @WT@.
# A start with numbered checkpoints in the save folder resumes from the last one that loads, with its optimizer state
# (--resume); a resume writes ${NAME}_r2_<k>k.pth (then _r3, ...), and the glob takes them too. When the last save
# exists in any branch, this script writes the file stop (the supervisor then ends) and trains nothing.
# ABC_GIFT_DRY_RUN=1 prints the command and runs nothing.
set -uo pipefail
R="${R:-@R@}"
WT="${WT:-@WT@}"
SAVE="$R/ckpt"
NAME="@NAME@"
TOTAL="${TOTAL:-@TOTAL@}"

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
@ARGS@
)
RESUME="$(last_checkpoint)"
[ -n "$RESUME" ] && ARGS=(--resume "$RESUME" "${ARGS[@]}")

# The environment of the grid runs (run_grid40k_elisa_ln.sh).
ENVS=(PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
      FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK=64
      PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4 TEACHER_EMBED_CHUNK=16
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
'''

SUPERVISE = r'''#!/bin/bash
# Keeps @NAME@ alive: it starts run_@NAME@.sh again (which resumes from the last checkpoint) when the trainer
# ends before the last step. It stops at the last checkpoint, at the file $R/stop, or after MAX_STARTS starts.
# Written by make_run.py (experiments/2026-11-08_width64_search/scripts) (the supervisor of the 166k runs).
#   cd @R@ && BB_GPU=<gpu> setsid nohup bash supervise_@NAME@.sh >> logs/supervise.log 2>&1 < /dev/null &
set -uo pipefail
R="${R:-@R@}"
MAX_STARTS="${MAX_STARTS:-5}"
LAST_K="${LAST_K:-@LAST_K@}"
for start in $(seq 1 "$MAX_STARTS"); do
  if [ -n "$(ls "$R"/ckpt/@NAME@_"${LAST_K}"k.pth "$R"/ckpt/@NAME@_r[0-9]*_"${LAST_K}"k.pth 2>/dev/null)" ]; then
    echo "$(date -u +%FT%TZ) supervise: the ${LAST_K}k checkpoint exists, end"; exit 0
  fi
  if [ -e "$R/stop" ]; then
    echo "$(date -u +%FT%TZ) supervise: stop file, end"; exit 0
  fi
  echo "$(date -u +%FT%TZ) supervise: start $start of $MAX_STARTS"
  bash "$R/run_@NAME@.sh" >>"$R/logs/train.log" 2>&1
  echo "$(date -u +%FT%TZ) supervise: the trainer ended with code $?"
  sleep 120
done
echo "$(date -u +%FT%TZ) supervise: $MAX_STARTS starts used, end"; exit 1
'''


def args_of(changes):
    """The recipe with the changes: a new value for a flag of the recipe, else a new flag at the end."""
    args = dict(RECIPE)
    for flag, value in changes:
        args[flag] = value or None
    return "\n".join(f"  {flag}" + (f" {value}" if value is not None else "") for flag, value in args.items())


def main(name, code, total, why, *pairs):
    changes = list(zip(pairs[::2], pairs[1::2]))
    folder = HP / name
    (folder / "logs").mkdir(parents=True, exist_ok=True)
    (folder / "ckpt").mkdir(exist_ok=True)
    fill = {"@NAME@": name, "@R@": str(folder), "@WT@": code, "@WHY@": why, "@ARGS@": args_of(changes),
            "@TOTAL@": str(int(total)), "@LAST_K@": str(int(total) // 1000)}
    for template, path in ((RUN, folder / f"run_{name}.sh"), (SUPERVISE, folder / f"supervise_{name}.sh")):
        text = template
        for key, value in fill.items():
            text = text.replace(key, value)
        assert not path.exists(), f"{path} exists"
        path.write_text(text)
        print("wrote", path)


if __name__ == "__main__":
    main(*sys.argv[1:])
