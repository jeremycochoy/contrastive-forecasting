#!/bin/bash
# Resumes the multi-experiment sequence after EXP1 (RevIN reproduction).
# Reordered to put the user's primary new ask (EXP4: patch-stats) ahead
# of the lower-priority queued items (EXP3: span sweep, EXP2: synth-only).
# Span-sweep evals are reduced to a 6-config periodic-focus screen via
# --config-filter to fit in budget.
#
# Stages (run order):
#   EXP4  Patch-stats backbone + qhead + full GIFT-Eval
#   EXP3  Span sweep (64, 128) — backbones + qheads + cheap 6-config screen
#         (drop span=256 if budget tight)
#   EXP3W Span sweep winner full eval (only if budget permits)
#   EXP2  Synth-only 30k+60k backbones + qheads (qualitative grids only)
set -e
cd /workspace/app

# Append to the existing run_all.log so the sync_loop pulls one stream.
exec >> >(tee -a /workspace/app/run_all.log) 2>&1
echo "" && echo "=== run_remaining: starting reordered tail ===" && date

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export GIFT_EVAL=/workspace/gift-eval-data

HF_REPO="jeremycochoy/contrastive-training-base-bundles"
HF_PATH="base_mixed_v1"

run_train_backbone() {
    local NAME=$1; shift
    # --save-every 2000 so a crash never costs more than ~6 min of compute
    # (backbone is ~6 sps; 2000 steps = ~5.5 min). Heads use 1000.
    python3 -u experiments/freq-embedding/scripts/train.py \
        --device cuda --total-steps 30000 --batch-size 24 --lr 1e-4 \
        --save-every 2000 \
        --save-dir checkpoints --run-name "$NAME" \
        --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
        "$@"
}

run_qhead() {
    local NAME=$1; shift
    local BB=$1; shift
    # --save-every 1000 (heads ~4 sps → ~4 min between snapshots) so a
    # crash never costs more than ~4 min of head training compute.
    # _best.pth is also updated every 500 steps when ema_loss improves,
    # but late in training those improvements are rare so we want
    # explicit periodic snapshots as well.
    python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
        --backbone-path "$BB" --forecast-len 16 --quantile-head \
        --total-steps 30000 --batch-size 24 --lr 3e-4 \
        --save-every 1000 \
        --save-dir checkpoints --run-name "$NAME" \
        --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
        "$@"
}

run_eval() {
    # $1=run_name, $2=backbone, $3=head, $4..=extra args
    local NAME=$1; shift; local BB=$1; shift; local HEAD=$1; shift
    mkdir -p "results/${NAME}"
    python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
        --backbone-path "$BB" --head-path "$HEAD" \
        --forecast-len 16 --strategy B4 \
        --output-dir "results/${NAME}" --device cuda \
        "$@"
}

# 6-config periodic-focus regex (covers ett1/15T/*, ett2/W/*, solar/10T/*,
# solar/H/short, m4_hourly/H/*). Expanded "short" suffix -> all terms so
# the regex matches /short, /medium, /long.
PERIODIC_FILTER='ett[12]/(15T|W)|solar/(10T|H/short)|m4_hourly/H'

# ============================================================
# EXP4 — Patch-stats (user's new ask, highest priority)
# ============================================================
echo "" && echo "=== EXP4 STAGE 1: patch-stats backbone (fe+mu, span=32, patch_stats=diff) ===" && date
run_train_backbone tiny_femu_pstats \
    --mix-ratio 0.5 --freq-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 32 \
    --patch-stats diff
cp -f checkpoints/tiny_femu_pstats_best_gap.pth checkpoints/tiny_femu_pstats_FINAL.pth
echo "=== EXP4 STAGE 1 DONE ===" && date

echo "" && echo "=== EXP4 STAGE 2: patch-stats qhead ===" && date
run_qhead R1q_femu_pstats checkpoints/tiny_femu_pstats_FINAL.pth \
    --rev-norm-kind ewma --rev-norm-span 32
cp -f checkpoints/R1q_femu_pstats_best.pth checkpoints/R1q_femu_pstats_FINAL.pth
echo "=== EXP4 STAGE 2 DONE ===" && date

echo "" && echo "=== EXP4 STAGE 3: patch-stats GIFT-Eval (full 97) ===" && date
run_eval R1q_femu_pstats \
    checkpoints/tiny_femu_pstats_FINAL.pth \
    checkpoints/R1q_femu_pstats_FINAL.pth \
    --rev-norm-kind ewma --rev-norm-span 32
echo "=== EXP4 STAGE 3 DONE ===" && date

# ============================================================
# EXP3 — Span sweep (64, 128) with cheap-screen evals
# span=32 reuses the existing fe+mu+qh local result, no need to retrain.
# span=256 deferred — budget guard. Add a third loop iter if budget OK.
# ============================================================
for SPAN in 64 128; do
    NAME="tiny_femu_span${SPAN}"
    echo "" && echo "=== EXP3 STAGE: span=${SPAN} backbone ===" && date
    run_train_backbone "$NAME" \
        --mix-ratio 0.5 --freq-emb-dim 3 --mixup-p 0.3 \
        --rev-norm-kind ewma --rev-norm-span "$SPAN"
    cp -f "checkpoints/${NAME}_best_gap.pth" "checkpoints/${NAME}_FINAL.pth"
    echo "=== EXP3 STAGE: span=${SPAN} backbone DONE ===" && date

    QNAME="R1q_femu_span${SPAN}"
    echo "" && echo "=== EXP3 STAGE: span=${SPAN} qhead ===" && date
    run_qhead "$QNAME" "checkpoints/${NAME}_FINAL.pth" \
        --rev-norm-kind ewma --rev-norm-span "$SPAN"
    cp -f "checkpoints/${QNAME}_best.pth" "checkpoints/${QNAME}_FINAL.pth"
    echo "=== EXP3 STAGE: span=${SPAN} qhead DONE ===" && date

    # Cheap screen: 6 periodic configs, ~10 min vs ~2.2h for full 97.
    echo "" && echo "=== EXP3 STAGE: span=${SPAN} screen (6 periodic configs) ===" && date
    run_eval "${QNAME}_screen" "checkpoints/${NAME}_FINAL.pth" "checkpoints/${QNAME}_FINAL.pth" \
        --rev-norm-kind ewma --rev-norm-span "$SPAN" \
        --config-filter "$PERIODIC_FILTER"
    echo "=== EXP3 STAGE: span=${SPAN} screen DONE ===" && date
done

# After both spans screened, pick the winner manually (offline) and
# launch a full eval just for that one. The driver script can't decide
# the winner mid-run because we'd need to parse the screen results;
# easier to inspect locally.
echo ""
echo "=== EXP3 SCREENS DONE — inspect results/{R1q_femu_span64_screen,"\
"R1q_femu_span128_screen}/all_results.csv to pick winner ==="
date

# ============================================================
# EXP2 — Synth-only (qualitative grid plots only, no GIFT-Eval)
# ============================================================
echo "" && echo "=== EXP2 STAGE 1: synth-only backbone 30k ===" && date
run_train_backbone tiny_femu_synthonly_30k \
    --mix-ratio 1.0 --freq-emb-dim 3 --mixup-p 0.3 --rev-norm-kind ewma
cp -f checkpoints/tiny_femu_synthonly_30k_best_gap.pth \
      checkpoints/tiny_femu_synthonly_30k_FINAL.pth
echo "=== EXP2 STAGE 1 DONE ===" && date

echo "" && echo "=== EXP2 STAGE 2: synth-only backbone 60k ===" && date
python3 -u experiments/freq-embedding/scripts/train.py \
    --device cuda --total-steps 60000 --batch-size 24 --lr 1e-4 \
    --save-dir checkpoints --run-name tiny_femu_synthonly_60k \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
    --mix-ratio 1.0 --freq-emb-dim 3 --mixup-p 0.3 --rev-norm-kind ewma
cp -f checkpoints/tiny_femu_synthonly_60k_best_gap.pth \
      checkpoints/tiny_femu_synthonly_60k_FINAL.pth
echo "=== EXP2 STAGE 2 DONE ===" && date

echo "" && echo "=== EXP2 STAGE 3: synth-only qhead 30k ===" && date
run_qhead R1q_femu_synthonly_30k checkpoints/tiny_femu_synthonly_30k_FINAL.pth \
    --mix-ratio 1.0 --rev-norm-kind ewma
cp -f checkpoints/R1q_femu_synthonly_30k_best.pth \
      checkpoints/R1q_femu_synthonly_30k_FINAL.pth
echo "=== EXP2 STAGE 3 DONE ===" && date

echo "" && echo "=== EXP2 STAGE 4: synth-only qhead 60k ===" && date
run_qhead R1q_femu_synthonly_60k checkpoints/tiny_femu_synthonly_60k_FINAL.pth \
    --mix-ratio 1.0 --rev-norm-kind ewma
cp -f checkpoints/R1q_femu_synthonly_60k_best.pth \
      checkpoints/R1q_femu_synthonly_60k_FINAL.pth
echo "=== EXP2 STAGE 4 DONE ===" && date

echo "" && echo "=== run_remaining: ALL TAIL EXPERIMENTS COMPLETE ===" && date
ls -la checkpoints/*_FINAL.pth results/*/all_results.csv 2>/dev/null | head -20
