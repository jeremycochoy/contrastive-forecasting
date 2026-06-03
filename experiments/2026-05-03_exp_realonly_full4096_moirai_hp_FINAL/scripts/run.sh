#!/bin/bash
# 2026-05-03_2026-05-02_exp_realonly_full4096_moirai_hp_FINAL (#10) — final 1-epoch retrain of the
# contrastive-forecasting backbone, resuming from the #9 30k checkpoint.
#
# Why this run exists:
#   #9 (MOIRAI-style optimizer HP: lr=1e-3, wd=0.1, β2=0.98) won the 30k
#   full-4096 comparison vs #6 (default HP) on the 2026-05-03 GIFT-Eval:
#     metric          | #6 default | #9 MOIRAI HP
#     GM-MASE         |   1.8043   |   1.6391    ← #9 wins
#     GM-MAPE_SN      |   1.3698   |   1.1850    ← #9 wins
#     GM-CRPS_SN      |   1.1000   |   1.0155    ← #9 wins
#   #6/#9 had only seen ~6.78% of one epoch (30k × 96 = 2.88M of 42.5M
#   windows). Step starvation was the dominant suspected limiter. This run
#   resumes from #9's 30k checkpoint and continues to a FULL EPOCH on
#   gift-pretrain-full-4096 (small_v1 path = full data despite the name).
#
# Total steps:
#   N_windows / batch_size = total_steps for one full epoch.
#   - Project codebase consistently documents 42.5M windows for
#     `jeremycochoy/gift-pretrain-full-4096` (path=small_v1) — see
#     experiments/2026-05-02_exp_realonly_full4096_learnable_tau/exp_realonly_full4096_learnable_tau.md and
#     experiments/2026-05-02_exp_realonly_full4096_learnable_tau/run.sh which both
#     state "42.5M windows, 619 GB, 4274 zstd parquet shards".
#   - The task brief mentioned 47.8M as an alternate estimate.
#   - 42.5M / 96 = 442,708 → ceil = 442,709
#   - 47.8M / 96 = 497,917 → ceil = 497,917
#   - To guarantee we cover one full epoch (undertraining > overtraining
#     by a small margin in cost), we use the upper-bound estimate:
#       --total-steps 498000
#     If 42.5M is the true number, we'd overshoot by ~12.5% (~7h extra).
#     If 47.8M is the true number, 498k is exactly one epoch.
#
# Resume mechanism (per train.py):
#   --resume <path> loads model state via torch.load(path), then
#   load_training_state(optimizer, args.resume, ...) reads
#   <path-without-.pth>_optimizer.pth for step counter, RNG state, AdamW
#   momentum, and hf_rows_consumed. The trainer continues from
#   start_step=30000 and the HF stream replays past consumed rows so the
#   epoch boundary lands at step ~498k (model sees each window roughly
#   once over the run, given hf_rows_per_step replay is correct).
#
# RESUME BUNDLE (4 files; push to fresh instance BEFORE invoking run.sh):
#   1. checkpoints/tiny_realonly_full4096_moirai_hp_30k.pth
#   2. checkpoints/tiny_realonly_full4096_moirai_hp_30k_optimizer.pth
#   3. checkpoints/tiny_realonly_full4096_moirai_hp_losses.csv  (so loss
#      history extends; trainer opens with mode="a")
#   4. run_full4096_moirai_hp.log → /workspace/app/run_full4096_moirai_hp_FINAL.log
#      (so τ-trajectory grep continues — this run's log is a NEW file but
#      the CSV+log of #9 must be on the box if you want one continuous
#      trajectory file. Optional but recommended.)
#
#   Use scripts/push_resume_bundle.sh — but note: it expects
#   <bb_run_name>_FINAL.pth or <bb_run_name>_best_loss.pth, not
#   <bb_run_name>_30k.pth. For this run, push the four files manually
#   with safe_pull.sh-style atomic scp, OR rename the local
#   tiny_realonly_full4096_moirai_hp_30k.pth → 30k_pinned/ ... hand-rolled.
#   The CLEANEST option:
#     scp -P <port> -o StrictHostKeyChecking=no \
#       sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/tiny_realonly_full4096_moirai_hp_30k.pth \
#       sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/tiny_realonly_full4096_moirai_hp_30k_optimizer.pth \
#       sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/tiny_realonly_full4096_moirai_hp_losses.csv \
#       sync_realonly_full4096_moirai_hp/moirai_hp/run.log \
#       root@ssh<N>.vast.ai:/workspace/app/checkpoints/  # then mv the .log up
#   Then on the remote: mv /workspace/app/checkpoints/run.log \
#                          /workspace/app/run_full4096_moirai_hp_FINAL.log
#
# Sync wiring (operator-launched, not by run.sh):
#   - Local sync target dir: sync_realonly_full4096_moirai_hp_FINAL/moirai_hp_FINAL/
#     (must NOT collide with #9's sync_realonly_full4096_moirai_hp/moirai_hp/).
#   - Run names use the _FINAL suffix throughout (BB and QH) to give
#     safe_save_path() in train.py a clean target — no collision with
#     #9's existing checkpoints/ dir on the same instance.
#   - Wire sync_loop.sh after vastrun-provision returns the SSH host/port,
#     pointing at sync_realonly_full4096_moirai_hp_FINAL/moirai_hp_FINAL/.
#
# Expected wallclock (5090, ~1.7 sps for backbone, similar to #9):
#   - Backbone: 498k - 30k = 468k new steps × ~0.6 s/step ≈ 78 hours
#     (#9 was ~1.7 sps but the 5090 may push it; budget conservatively).
#   - Q-head:   30k × ~0.6 s/step ≈ 5 hours
#   - GIFT-Eval: ~5 hours (per #9)
#   - Total:    ≈ 88 hours ≈ $34 at $0.39/h on a 5090.
#   The brief estimate of 67h/$26 assumes 1.7 sps consistently and the
#   42.5M lower-bound step count (442k); using the 498k upper bound
#   pushes it to ~88h/$34. User must add ≥ $30 of credit before launch.
#
# HP unchanged from #9 (all four optimizer settings carry through the
# resume because load_training_state reads the optimizer .pth which
# embeds the AdamW state — but we re-pass the flags so the saved arg
# manifest reflects the run intent):
#   --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
# NO grad-clip (project rule).
# NO warmup, NO cosine annealing — flat lr (same as #9, for τ-policy
# stability and to keep cross-#9 comparison clean).
#
# Save-every change: 2500 → 10000.
#   468k new steps at save-every=2500 = 187 periodic saves
#   × 230 MB (model+optim) ≈ 43 GB on disk. Bumping to 10000 cuts to
#   47 saves × 230 MB ≈ 11 GB — still gives crash-recovery granularity
#   of ~ every hour at 1.7 sps.
set -e
cd /workspace/app

LOG="/workspace/app/run_full4096_moirai_hp_FINAL.log"
exec >> >(tee -a "$LOG") 2>&1
echo "" && echo "=== run_full4096_moirai_hp_FINAL: starting ===" && date

SETUP_MARKER="/workspace/app/.setup_done_realonly_full4096_FINAL"
if [ ! -f "$SETUP_MARKER" ]; then
    echo "=== SETUP ===" && date
    apt-get update -qq
    apt-get install -y -qq python3-pip rsync > /dev/null 2>&1 || true
    pip install --break-system-packages "torch>=2.8,<2.9" \
        --index-url https://download.pytorch.org/whl/cu128 > /dev/null 2>&1 || true
    pip install --break-system-packages 'numpy<2' pandas pyarrow statsmodels \
        matplotlib datasets huggingface_hub tqdm gluonts > /dev/null 2>&1
    pip install --break-system-packages \
        "salesforce-gift-eval @ git+https://github.com/SalesforceAIResearch/gift-eval.git" \
        > /dev/null 2>&1
    python3 -c "import torch; print(f'torch {torch.__version__} | CUDA {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
    if [ ! -d /workspace/gift-eval-data ] || [ -z "$(ls -A /workspace/gift-eval-data 2>/dev/null)" ]; then
        export HF_TOKEN_TMP=$(cat experiments/hf_token.txt)
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_TMP"
        python3 -c "
from huggingface_hub import snapshot_download
import os, shutil
path = snapshot_download('jeremycochoy/gift-pretrain-small-4096',
                         repo_type='dataset', allow_patterns='eval/**',
                         local_dir='/workspace/gift-eval-download')
src = os.path.join(path, 'eval')
dst = '/workspace/gift-eval-data'
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f'GIFT-Eval data ready: {dst}')
"
    fi
    touch "$SETUP_MARKER"
fi

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export GIFT_EVAL=/workspace/gift-eval-data

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
LOSS="cosine_similarity_batch"

# New, distinct run names — never reuse #9's names (CLAUDE.md "Never reuse
# --save-path when resuming"; safe_save_path() in train.py is the safety
# net but explicit is clearer).
BB="tiny_full4096_moirai_hp_FINAL"
QH="R1q_full4096_moirai_hp_FINAL"

# Resume target: pre-staged on this fresh instance by the operator
# before launch (see RESUME BUNDLE comment above).
RESUME_BB="checkpoints/tiny_realonly_full4096_moirai_hp_30k.pth"

RES_DIR="experiments/2026-05-03_2026-05-02_exp_realonly_full4096_moirai_hp_FINAL/results/gift_eval"
mkdir -p "$RES_DIR"

# Preflight: verify resume bundle was pushed before continuing.
if [ ! -f "$RESUME_BB" ]; then
    echo "ERROR: resume backbone not found at $RESUME_BB" >&2
    echo "       Push the resume bundle to /workspace/app/checkpoints/ first:" >&2
    echo "         tiny_realonly_full4096_moirai_hp_30k.pth" >&2
    echo "         tiny_realonly_full4096_moirai_hp_30k_optimizer.pth" >&2
    exit 1
fi
if [ ! -f "${RESUME_BB%.pth}_optimizer.pth" ]; then
    echo "ERROR: resume optimizer not found at ${RESUME_BB%.pth}_optimizer.pth" >&2
    echo "       Without it, resume loses step counter, RNG, AdamW momentum, hf_rows_consumed." >&2
    exit 1
fi

echo "" && echo "=== STAGE B: $BB (resume from #9 30k → 1 full epoch, MOIRAI HP) ===" && date
# NOTE: as of PR #107, train.py always logs a fixed-τ=0.07 reference loss in
# the column `loss_tau_ref` of the losses CSV (computed under torch.no_grad,
# so no gradient impact and <5% step-time overhead). Useful as a cross-run
# baseline since the training τ is learnable here. No CLI flag — always on.
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --resume "$RESUME_BB" \
    --device cuda --total-steps 498000 --batch-size 96 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 10000 --save-dir checkpoints --run-name "$BB" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --tau 0.07 --learnable-tau \
    --loss-shape "$LOSS"
cp -f "checkpoints/${BB}_best_loss.pth" "checkpoints/${BB}_FINAL.pth"
echo "=== STAGE B DONE ===" && date

echo "" && echo "=== STAGE H: $QH (FRESH qhead on FINAL backbone, default optim) ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 --quantile-head \
    --total-steps 30000 --batch-size 96 --lr 3e-4 \
    --save-every 1000 --save-dir checkpoints --run-name "$QH" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster
cp -f "checkpoints/${QH}_best.pth" "checkpoints/${QH}_FINAL.pth"
echo "=== STAGE H DONE ===" && date

echo "" && echo "=== STAGE E: gift_eval ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" \
    --head-path "checkpoints/${QH}_FINAL.pth" \
    --output-dir "$RES_DIR" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda
echo "=== STAGE E DONE ===" && date

echo "" && echo "=== run_full4096_moirai_hp_FINAL: ALL DONE ===" && date
echo ""
if [ -f "$RES_DIR/summary.txt" ]; then
    head -30 "$RES_DIR/summary.txt"
fi
