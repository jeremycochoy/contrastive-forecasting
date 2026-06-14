#!/bin/bash
# #344 — run ONE GIFT-Eval cell (arm × head-layers × checkpoint). The eval
# command is byte-identical to downstream_cpc.sh's do_eval (same flags ⇒
# comparable to the baselines); factored out so cells can run concurrently
# across/within GPUs (the eval is CPU-bound, so several fit per GPU).
#   eval_cell.sh <arm: enc3|enc6> <hl: 2|6> <ckpt: best|last> <gpu>
set -uo pipefail
ARM="${1:?arm}"; HL="${2:?hl}"; CK="${3:?best|last}"; GPU="${4:?gpu}"
TAG="allt08_xftrip_nobn_${ARM}_sgpos_qk_aon_b1024_cpc"
WT="${WT:-/home/jupyter/contrastive-forecasting/.claude/worktrees/exp+cpc-infonce-344}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-13_cpc_infonce_aux}"
RUNS="$OUT/runs"; RES="$OUT/results"
export PYTHONPATH="$WT" OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [eval-cell $ARM ${HL}L $CK g$GPU] $*"; }
if [ "$CK" = best ]; then
  BB="$RUNS/bb_${TAG}_FINAL.pth"; qf="$RUNS/qhead_${HL}L_${TAG}_FINAL.pth"
  out="$RES/gift_eval_full_${TAG}_${HL}L"; lg="$RES/run_eval_full_${TAG}_${HL}L.log"
else
  BB="$RUNS/bb_${TAG}_final.pth"; qf="$RUNS/qhead_${HL}L_${TAG}_last_FINAL.pth"
  out="$RES/gift_eval_full_${TAG}_last_${HL}L"; lg="$RES/run_eval_full_${TAG}_last_${HL}L.log"
fi
[ -f "$out/summary.txt" ] && { log "skip (summary exists, GM=$(grep 'Aggregate GM-Relative MASE' "$out/summary.txt" | grep -oE '[0-9]+\.[0-9]+$' | tail -1))"; exit 0; }
[ -f "$BB" ] || { log "ABORT backbone missing $BB"; exit 1; }
[ -f "$qf" ] || { log "ABORT head missing $qf"; exit 1; }
mkdir -p "$out"; log "start"
python3 -u "$QEVAL" --backbone-path "$BB" --head-path "$qf" --output-dir "$out" --strategy B4 \
  --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 6 \
  --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
  >>"$lg" 2>&1 || { log "FAILED (tail: $(tail -2 "$lg"|tr '\n' ' '))"; exit 1; }
log "done GM=$(grep 'Aggregate GM-Relative MASE' "$out/summary.txt" | grep -oE '[0-9]+\.[0-9]+$' | tail -1)"
