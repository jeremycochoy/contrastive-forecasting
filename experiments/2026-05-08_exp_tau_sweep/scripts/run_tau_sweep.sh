#!/bin/bash
# τ sweep: 5 fixed-τ fine-tunes resumed from backbone-beta (167k steps).
#
# backbone-beta = the best_loss checkpoint of the FRESH_RESUME50k learnable-τ
# run (PR #112 era), where log_inv_tau converged to ~2.6313 → τ ≈ 0.0720.
# This sweep asks: holding everything else fixed, does a sharper (0.03/0.05)
# or softer (0.10/0.20) fixed τ improve downstream AUC after a short 20k-step
# fine-tune? The control is τ=0.07 (closest fixed value to the converged
# learnable τ).
#
# τ values: 0.03, 0.05, 0.07, 0.10, 0.20 (5 arms)
#
# Recipe per arm (identical to backbone-beta except --learnable-tau is OFF):
#   T_RAW=4096, C=1, H=384, num_layers=6, nhead=6
#   freq_emb_dim=3, seasonality_emb_dim=3, rev_norm_kind=ewma span=128
#   AdamW lr=1e-3 wd=0.1 β1=0.9 β2=0.98
#   mixup_p=0.3, mix_ratio=0.0, loss_shape=cosine_similarity_batch
#   batch_size=256, total_steps=20000, save_every=5000
#
# Resume strategy:
#   (1) backbone-beta's state_dict has `log_inv_tau` (it was learnable). With
#       --learnable-tau OFF the trainer's model has no such parameter, so a
#       strict load_state_dict() raises. We pre-strip the key once into
#       checkpoints/backbone_beta_167k_stripped.pth (idempotent).
#   (2) Optimizer companion: STRATEGY A. We deliberately do NOT supply
#       <stripped>_optimizer.pth. The trainer's load_training_state() falls
#       back gracefully (src/checkpoint.py:114), starting with a fresh
#       AdamW + step=0. For a 20k-step fine-tune from already-converged
#       weights, fresh optimizer state is acceptable; preserving the model
#       weights is the only thing that matters here. This also sidesteps the
#       shape mismatch from the saved Adam moments for log_inv_tau.
#
# After each arm: cp <run>_best_loss.pth <run>_FINAL.pth
# Pre-flight: drop checkpoints/backbone_beta_167k.pth on the instance first
# (early-exit at line ~55 has the source path).

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
LOSS="cosine_similarity_batch"

RESUME_RAW="checkpoints/backbone_beta_167k.pth"
RESUME_BB="checkpoints/backbone_beta_167k_stripped.pth"

if [ ! -f "${RESUME_RAW}" ]; then
    echo "ERROR: backbone-beta not found at ${RESUME_RAW}" >&2
    echo "       scp it from local sync_realonly_full4096_moirai_hp_FRESH_RESUME50k/" \
         "moirai_hp_FRESH_RESUME50k/checkpoints/tiny_full4096_moirai_hp_FRESH_RESUME50k_best_loss.pth" >&2
    exit 1
fi

# Strip log_inv_tau (idempotent — skip if already stripped).
if [ ! -f "${RESUME_BB}" ]; then
    echo "=== Stripping log_inv_tau from ${RESUME_RAW} → ${RESUME_BB} ===" && date
    python3 - <<PY
import torch
src = "${RESUME_RAW}"
dst = "${RESUME_BB}"
sd = torch.load(src, map_location="cpu", weights_only=False)
removed = sd.pop("log_inv_tau", None)
print(f"  removed log_inv_tau = "
      f"{removed.item() if removed is not None and removed.numel()==1 else removed}")
torch.save(sd, dst)
print(f"  wrote {dst} ({len(sd)} keys)")
PY
else
    echo "=== Stripped checkpoint already exists at ${RESUME_BB}, skipping ==="
fi

# Sanity: there must NOT be a companion optimizer at the stripped path.
# Strategy A: fresh optimizer for the 20k fine-tune. The trainer logs
# "[checkpoint] No optimizer state at ... starting fresh." — that's expected.
if [ -f "${RESUME_BB%.pth}_optimizer.pth" ]; then
    echo "WARNING: ${RESUME_BB%.pth}_optimizer.pth exists; would resume optimizer." >&2
    echo "         Strategy A wants a fresh optimizer. Remove it or this is wrong." >&2
fi

# τ sweep loop. Period replaced with underscore in run names so the file
# system stays sane (best_loss → bb_beta_tau_fixed_0_03_best_loss.pth, etc.).
TAUS=(0.03 0.05 0.07 0.10 0.20)

for TAU in "${TAUS[@]}"; do
    SAFE_TAU=${TAU//./_}
    BB="bb_beta_tau_fixed_${SAFE_TAU}"

    echo "" && echo "=== ARM τ=${TAU} → run_name=${BB} ===" && date

    python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
        --resume "${RESUME_BB}" \
        --device cuda --total-steps 20000 --batch-size 256 \
        --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
        --save-every 5000 --save-dir checkpoints --run-name "${BB}" \
        --hf-repo "${HF_REPO}" --hf-path "${HF_PATH}" \
        --t-raw 4096 --n-channels 1 \
        --d-model 384 --n-heads 6 --num-layers 6 \
        --mix-ratio 0.0 \
        --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
        --rev-norm-kind ewma --rev-norm-span 128 \
        --tau "${TAU}" \
        --loss-shape "${LOSS}"

    cp -f "checkpoints/${BB}_best_loss.pth" "checkpoints/${BB}_FINAL.pth"
    echo "=== ARM τ=${TAU} DONE — saved ${BB}_FINAL.pth ===" && date
done

echo "" && echo "=== τ sweep ALL DONE ===" && date
exit 0
