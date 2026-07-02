#!/bin/bash
# #363 — aggregate raw artefacts from the experiments tree into the reports tree
# and run build_report.py. Intended to run AFTER the sweep finishes.
set -euo pipefail
WT="${WT:-/tmp/contrastive-forecasting-363}"
EXP="$WT/experiments/2026-06-24_sigreg_lambda_sweep"
REP="$WT/reports/2026-06-24_sigreg_lambda_sweep"
mkdir -p "$REP/runs" "$REP/results" "$REP/plots"

# Backbones: losses CSVs only (full .pth checkpoints stay in experiments/runs/
# — they're hundreds of MB each and not required for the report)
for s in emb100_enc01 emb100_enc10 emb100_enc100 emb10_enc10 emb1000_enc01 emb10000_enc10; do
  src="$EXP/runs/bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_${s}_losses.csv"
  [ -f "$src" ] && cp -u "$src" "$REP/runs/"
done

# Per-arm logs (backbone train log + downstream logs + GIFT-Eval logs)
for f in "$EXP"/results/run_bb_*.log \
         "$EXP"/results/sweep_*.log \
         "$EXP"/results/dl_*.log \
         "$EXP"/results/run_qhead_*.log \
         "$EXP"/results/run_eval_full_*.log \
         "$EXP"/results/launcher.log \
         "$EXP"/results/manual_bb_*.log; do
  [ -f "$f" ] && cp -u "$f" "$REP/results/"
done

# GIFT-Eval summary trees (one per (arm,head,ckpt))
for d in "$EXP"/results/gift_eval_full_*; do
  [ -d "$d" ] && cp -r -u "$d" "$REP/results/"
done

# Retroactive u_batchtime CSV (final-step pooled dim-usage per backbone)
[ -f "$EXP/results/u_batchtime_retro.csv" ] && cp -u "$EXP/results/u_batchtime_retro.csv" "$REP/results/"

# Anchor loss CSVs from prior reports — symlink, do not duplicate
ln -sf "$WT/reports/2026-06-20_lejepa_sigreg/runs/bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_losses.csv" "$REP/runs/anchor_sigreg01.csv" 2>/dev/null || true
ln -sf "$WT/reports/2026-06-22_lejepa_sigreg_emb10/runs/bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_emb10_losses.csv" "$REP/runs/anchor_sigreg10.csv" 2>/dev/null || true

# Build the report (gm_table.csv, plots, final_trajectories.txt)
python3 "$EXP/scripts/build_report.py" \
  --report-dir "$REP" \
  --sig01-csv "$REP/runs/anchor_sigreg01.csv" \
  --sig10-csv "$REP/runs/anchor_sigreg10.csv"

echo "REP=$REP"
ls -lh "$REP/results/gm_table.csv" "$REP/plots/" "$REP/results/final_trajectories.txt"
