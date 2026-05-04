#!/bin/bash
# FOLLOWUP-1 v2: cosine_similarity_batch (paper-matching), best_loss FINAL.
set -e
cd /workspace/app
exec >> >(tee -a /workspace/app/run_all.log) 2>&1
echo "" && echo "=== run_wtn_v2: cosine_similarity_batch ===" && date
export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="tiny_femu_span512_synth30k_csb"
QNAME="R1q_femu_span512_synth30k_csb"

echo "=== STAGE B: backbone with cosine_similarity_batch ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py     --device cuda --total-steps 30000 --batch-size 24 --lr 1e-4     --save-every 2000 --save-dir checkpoints --run-name "$NAME"     --hf-repo "jeremycochoy/contrastive-training-base-bundles" --hf-path "base_mixed_v1"     --mix-ratio 1.0 --freq-emb-dim 3 --mixup-p 0.3     --rev-norm-kind ewma --rev-norm-span 512     --loss-shape cosine_similarity_batch
# best_loss FINAL (gap saturates early; loss is the better selector)
cp -f checkpoints/${NAME}_best_loss.pth checkpoints/${NAME}_FINAL.pth
echo "=== STAGE B DONE ===" && date

echo "=== STAGE H: qhead ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py     --backbone-path "checkpoints/${NAME}_FINAL.pth" --forecast-len 16 --quantile-head     --total-steps 30000 --batch-size 24 --lr 3e-4     --save-every 1000 --save-dir checkpoints --run-name "$QNAME"     --hf-repo "jeremycochoy/contrastive-training-base-bundles" --hf-path "base_mixed_v1" --device cuda     --mix-ratio 1.0 --rev-norm-kind ewma --rev-norm-span 512
cp -f checkpoints/${QNAME}_best.pth checkpoints/${QNAME}_FINAL.pth
echo "=== STAGE H DONE ===" && date

mkdir -p results/synth_eval
echo "=== STAGE E: synth eval ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/synth_eval.py     --backbone "checkpoints/${NAME}_FINAL.pth" --head "checkpoints/${QNAME}_FINAL.pth"     --arm "fe+mu @ 30k span=512 +cosine_similarity_batch"     --n-samples 1024 --batch-size 64     --out-csv results/synth_eval/all_results.csv     --device cuda --rev-norm-span 512
echo "=== STAGE E DONE ===" && date
echo "=== run_wtn_v2: ALL DONE ===" && date
tail -3 results/synth_eval/all_results.csv
