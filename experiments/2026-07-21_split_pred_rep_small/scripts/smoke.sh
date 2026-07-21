#!/bin/bash
# #379 end-to-end smoke — runs one arm's full chain (backbone train →
# extra-save snapshot → q-head train → tiny GIFT-Eval) in ~10 min so any
# breakage in the checkpoint-naming, load_state_dict, or eval path
# surfaces BEFORE 35 hours of GPU time is committed.
#
# Exercises:
#   1. `--extra-save-steps 150` snapshot lands as `_0k.pth` and is
#      discoverable by `run_arm.sh`'s `ckpt_path 0` glob.
#   2. `train_forecasting_head.py` auto-detects freq_emb_dim,
#      seasonality_emb_dim, num_encoder_layers, qk_norm, attn_out_norm,
#      depthwise_conv from the small-model backbone state_dict — the
#      exact path that fails at 35 h if any arch flag disagrees.
#   3. `eval_gift_eval_official.py` runs one downstream cell end-to-end.
#
# Any failure short-circuits with a non-zero exit and a clear message.
# Success prints `SMOKE OK` and a one-line summary.
#
# Usage (from the checkout root):
#   ARM=arm1 GPU=0 bash experiments/2026-07-21_split_pred_rep_small/scripts/smoke.sh
#
# Any of {arm1, arm3, arm4, arm5, arm6_v2, bimoco} is valid. bimoco is
# the harshest recipe (--moco-negatives + --moco-rep-keys); use it once
# the arm1 smoke passes.
set -euo pipefail

ARM="${ARM:-arm1}"
GPU="${GPU:-0}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

# WT for the smoke — under $HOME so it passes run_arm.sh's under-/tmp
# guard and, in the same stroke, is not the throwaway agent worktree that
# real launches ban.
SMOKE_WT="${HOME}/.cache/cf-379-smoke-$$"
cleanup(){ rm -rf "$SMOKE_WT"; }
trap cleanup EXIT
mkdir -p "$SMOKE_WT"

# Layer the smoke WT over the real repo: symlink every top-level dir so
# the launcher finds train.py, the HF token, safe_pull.sh, etc., but
# override the experiment dir with a fresh one so runs/ and results/ are
# empty and writable without polluting the real checkout.
for d in src scripts docs tests reports; do
  [ -e "$REPO_ROOT/$d" ] && ln -s "$REPO_ROOT/$d" "$SMOKE_WT/$d"
done
mkdir -p "$SMOKE_WT/experiments"
for exp in "$REPO_ROOT/experiments"/*; do
  base="$(basename "$exp")"
  [ "$base" = "2026-07-21_split_pred_rep_small" ] || ln -s "$exp" "$SMOKE_WT/experiments/$base"
done
mkdir -p "$SMOKE_WT/experiments/2026-07-21_split_pred_rep_small"
ln -s "$REPO_ROOT/experiments/2026-07-21_split_pred_rep_small/scripts" \
      "$SMOKE_WT/experiments/2026-07-21_split_pred_rep_small/scripts"
ln -s "$REPO_ROOT/experiments/2026-07-21_split_pred_rep_small/sync" \
      "$SMOKE_WT/experiments/2026-07-21_split_pred_rep_small/sync"
[ -e "$REPO_ROOT/experiments/2026-07-21_split_pred_rep_small/README.md" ] && \
  ln -s "$REPO_ROOT/experiments/2026-07-21_split_pred_rep_small/README.md" \
        "$SMOKE_WT/experiments/2026-07-21_split_pred_rep_small/README.md"

# HF token: prefer the real one, fall back to $HF_TOKEN env, else fail.
TOK="$REPO_ROOT/experiments/hf_token.txt"
[ -f "$TOK" ] || {
  echo "[smoke-379] ABORT: HF token missing at $TOK — set HF_TOKEN in the env" \
       "and echo it into that file before running smoke.sh." >&2
  exit 2
}

echo "[smoke-379] SMOKE_WT=$SMOKE_WT ARM=$ARM GPU=$GPU"
echo "[smoke-379] 200-step backbone + 200-step head + 1-cell eval — ~10-15 min on 4090"

# 200-step backbone. save-every=100 (regular snapshot at 100, 200);
# extras=150 (exercises the union rule + sub-1000 filename `_0k.pth`).
# Downstream restricted to the extra-save cell only (BB_STEPS_K="0")
# so the smoke tests the off-cadence load path — the exact one that
# BB_STEPS_K=2 (extra_save=2500 → `_2k.pth`) exercises in the real run.
STEPS=200 SAVE_EVERY=100 EXTRA_SAVES="150" \
HEAD_STEPS=200 HEAD_WARMUP=20 \
BB_STEPS_K="0" \
QEVAL_EXTRA_ARGS="--config-filter ett1/15T" \
WT="$SMOKE_WT" BB_GPU="$GPU" GPU_2L="$GPU" GPU_6L="$GPU" \
  bash "$SMOKE_WT/experiments/2026-07-21_split_pred_rep_small/scripts/run_arm.sh" "$ARM" \
  > "$SMOKE_WT/smoke.log" 2>&1 || {
    echo "[smoke-379] SMOKE FAILED — tail:" >&2
    tail -30 "$SMOKE_WT/smoke.log" >&2
    exit 1
  }

# Verify the artefacts we expect are on disk.
RUNS="$SMOKE_WT/experiments/2026-07-21_split_pred_rep_small/runs"
RES="$SMOKE_WT/experiments/2026-07-21_split_pred_rep_small/results"
ls "$RUNS"/*_FINAL.pth >/dev/null 2>&1 || {
  echo "[smoke-379] SMOKE FAILED — no backbone FINAL.pth in $RUNS." >&2
  tail -20 "$SMOKE_WT/smoke.log" >&2
  exit 1
}
ls "$RUNS"/*_0k.pth >/dev/null 2>&1 || {
  echo "[smoke-379] SMOKE FAILED — extra-save snapshot _0k.pth missing." >&2
  tail -20 "$SMOKE_WT/smoke.log" >&2
  exit 1
}
ls "$RUNS"/qhead_*_FINAL.pth >/dev/null 2>&1 || {
  echo "[smoke-379] SMOKE FAILED — no q-head FINAL.pth produced." >&2
  tail -20 "$SMOKE_WT/smoke.log" >&2
  exit 1
}
ls "$RES"/gift_eval_full_*/summary.txt >/dev/null 2>&1 || {
  echo "[smoke-379] SMOKE FAILED — no GIFT-Eval summary.txt produced." >&2
  tail -20 "$SMOKE_WT/smoke.log" >&2
  exit 1
}
gm=$(grep -h 'Aggregate GM-Relative MASE' "$RES"/gift_eval_full_*/summary.txt | head -1 || true)

echo "[smoke-379] SMOKE OK — arm=$ARM"
echo "[smoke-379]   backbone: $(ls "$RUNS"/*_FINAL.pth | head -1 | xargs basename)"
echo "[smoke-379]   extra:    $(ls "$RUNS"/*_0k.pth | head -1 | xargs basename)"
echo "[smoke-379]   q-head:   $(ls "$RUNS"/qhead_*_FINAL.pth | head -1 | xargs basename)"
[ -n "$gm" ] && echo "[smoke-379]   $gm"
