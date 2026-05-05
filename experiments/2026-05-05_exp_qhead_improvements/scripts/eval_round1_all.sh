#!/bin/bash
# Once E1 and E2 are synced locally, this runs triage evals on both heads
# in sequence and prints the aggregate GM-Relative MASE for quick triage.
#
# Usage: ./eval_round1_all.sh [--full]
#   --full forces full 97-config eval instead of triage.

set -e
ROOT="/home/jupyter/contrastive-forecasting"
cd "$ROOT"

MODE="${1:---triage}"

E1="R1_E1_linear_quant_lr3e4"
E2="R1_E2_gru_quant_moirai_wsd"
SYNC="${ROOT}/sync_qhead_beta_rd1/checkpoints"

for HEAD in "$E1" "$E2"; do
    if [ ! -f "${SYNC}/${HEAD}_FINAL.pth" ]; then
        echo "[eval-rd1] skipping $HEAD (no FINAL.pth synced yet)"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "[eval-rd1] running $HEAD ($MODE)"
    echo "================================================================"
    bash experiments/2026-05-05_exp_qhead_improvements/scripts/run_eval_elisa.sh \
        "$HEAD" "$MODE"
done

echo ""
echo "================================================================"
echo "[eval-rd1] DONE — summary table"
echo "================================================================"
for HEAD in "$E1" "$E2"; do
    SUM="${ROOT}/experiments/2026-05-05_exp_qhead_improvements/results/${HEAD}_${MODE#--}/summary.txt"
    if [ -f "$SUM" ]; then
        echo ""
        echo "=== $HEAD ==="
        # Aggregate GM-Relative MASE line
        grep -i "Aggregate\|** Ours" "$SUM" | head -5
    fi
done
