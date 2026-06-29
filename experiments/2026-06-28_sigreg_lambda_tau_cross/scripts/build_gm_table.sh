#!/bin/bash
# #366 — emit reports/gm_table.csv summarising the 8-cell cross
# (arm × head × ckpt → GM-Rel MASE). Schema matches #363's table so
# the two can stack cleanly for the verdict.
#
#   bash scripts/build_gm_table.sh
#
# Reads each results/gift_eval_full_..._<HL>L/summary.txt and pulls the
# 'Aggregate GM-Relative MASE (97 configs): <gm>' line. Writes to
# results/gm_table.csv. Idempotent.
set -euo pipefail
EXP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RES="$EXP_DIR/results"
OUT="$RES/gm_table.csv"
TAG="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc"

gm(){ grep 'Aggregate GM-Relative MASE' "$1" 2>/dev/null \
  | grep -oE '[0-9]+\.[0-9]+$' | tail -1; }

declare -A ARM_LABEL=(
  [lA_emb100_enc10_tau090]="cross_A: λ_e=10, λ_h=1, τ=0.90"
  [lB_emb10000_enc10_tau090]="cross_B: λ_e=1000, λ_h=1, τ=0.90"
)

{
  echo "arm,label,head,ckpt,gm,n"
  for arm in lA_emb100_enc10_tau090 lB_emb10000_enc10_tau090; do
    for HL in 2 6; do
      for ckpt in best last; do
        if [ "$ckpt" = best ]; then
          suffix="${TAG}_${arm}_${HL}L"
        else
          suffix="${TAG}_${arm}_last_${HL}L"
        fi
        s="$RES/gift_eval_full_${suffix}/summary.txt"
        val=$(gm "$s")
        if [ -z "$val" ]; then
          echo "MISSING: $s" >&2
          val=""
        fi
        echo "cross_${arm:1:1},\"${ARM_LABEL[$arm]}\",${HL}L,${ckpt},${val},97"
      done
    done
  done
} >"$OUT"
echo "Wrote $OUT"
cat "$OUT"
