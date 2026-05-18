#!/bin/bash
# Triage eval helper for R10 proxy: head ↔ backbone are paired by name.
# Usage: ./run_eval_proxy.sh <NAME>   where NAME ∈ {moirai_hp_early, FRESH_50k,
#                                                   backbone_beta_167k,
#                                                   moirai_hp_FINAL_run1,
#                                                   learnable_tau}

set -e
ROOT="/home/jupyter/contrastive-forecasting"
cd "$ROOT"

NAME="${1:?usage: $0 <backbone_name>}"
case "$NAME" in
    moirai_hp_early)
        BB="$ROOT/sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/tiny_realonly_full4096_moirai_hp_best_loss.pth";;
    FRESH_50k)
        BB="$ROOT/sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/checkpoints/tiny_full4096_moirai_hp_FRESH_best_loss.pth";;
    backbone_beta_167k)
        BB="$ROOT/sync_realonly_full4096_moirai_hp_FRESH_RESUME50k/moirai_hp_FRESH_RESUME50k/checkpoints/tiny_full4096_moirai_hp_FRESH_RESUME50k_best_loss.pth";;
    moirai_hp_FINAL_run1)
        BB="$ROOT/sync_realonly_full4096_moirai_hp_FINAL_run1/moirai_hp_FINAL/checkpoints/tiny_full4096_moirai_hp_FINAL_180k.pth";;
    learnable_tau)
        BB="$ROOT/sync_realonly_full4096_learnable_tau/learnable/checkpoints/tiny_realonly_full4096_learnable_tau_best_loss.pth";;
    *) echo "ERROR: unknown backbone $NAME" >&2; exit 2;;
esac

HEAD="$ROOT/sync_qhead_beta_rd10/checkpoints/R10_proxy_${NAME}_FINAL.pth"
OUT="$ROOT/experiments/2026-05-05_exp_qhead_improvements/results/R10_proxy_${NAME}_triage"

[ ! -f "$BB" ] && { echo "ERROR: backbone $BB missing" >&2; exit 1; }
[ ! -f "$HEAD" ] && { echo "ERROR: head $HEAD missing" >&2; exit 1; }
mkdir -p "$OUT"

GPU_FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
echo "[eval-proxy] $NAME on GPU $GPU_FREE"

PYTHONPATH=. \
HF_TOKEN=$(cat experiments/hf_token.txt) \
HUGGING_FACE_HUB_TOKEN=$(cat experiments/hf_token.txt) \
GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
CUDA_VISIBLE_DEVICES=$GPU_FREE \
python3 -u experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "$BB" --head-path "$HEAD" \
    --output-dir "$OUT" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda \
    --head-causal true \
    --config-filter 'bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'

echo "[eval-proxy] $NAME DONE"
[ -f "$OUT/summary.txt" ] && grep -E "Aggregate|Ours" "$OUT/summary.txt" | head
