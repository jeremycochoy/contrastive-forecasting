#!/bin/bash
# #341 — evaluate ONE (arm, checkpoint, head) cell end-to-end: shard the full-97
# GIFT-Eval, mop up any missing configs, merge into a summary with GM-Relative
# MASE. Idempotent (skips if the merged summary already exists).
#   run_eval_cell.sh <tag> <best|last> <head_layers> [n_shards] [gpu_csv]
#     tag = e.g. allt08_xftrip_bn_enc6_sgpos_qk_aon_b1024
set -uo pipefail
TAG="${1:?tag}"; CK="${2:?best|last}"; HL="${3:?head_layers}"; NS="${4:-6}"; GPUS="${5:-0}"
SD="$(cd "$(dirname "$0")" && pwd)"
RES=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity/results
case "$CK" in
  best)      QN="qhead_${HL}L_${TAG}";          BBF="bb_${TAG}_FINAL.pth"; OUT="${TAG}" ;;
  last)      QN="qhead_${HL}L_${TAG}_last";      BBF="bb_${TAG}_final.pth"; OUT="${TAG}_last" ;;
  # lastfresh: a head trained FRESH (30k) on the last backbone — avoids the
  # 10k-re-adapt confound when best-loss is very early (these arms: step ~1k).
  lastfresh) QN="qhead_${HL}L_${TAG}_lastfresh"; BBF="bb_${TAG}_final.pth"; OUT="${TAG}_lastfresh" ;;
  *) echo "bad checkpoint '$CK' (want best|last|lastfresh)"; exit 2 ;;
esac
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [evalcell $OUT ${HL}L] $*"; }
summ="$RES/gift_eval_full_${OUT}_${HL}L/summary.txt"
[ -f "$summ" ] && { log "skip (summary exists, GM=$(grep -oE '[0-9]+\.[0-9]+$' "$summ" | tail -1))"; exit 0; }
log "shard start ns=$NS gpus=$GPUS qn=$QN bb=$BBF"
python3 "$SD/shard_evals.py" "$QN" "$BBF" "$OUT" "$HL" "$NS" "$GPUS" || log "shard rc=$?"
python3 "$SD/mopup_evals.py" "$QN" "$BBF" "$OUT" "$HL" "$NS" "$GPUS" || log "mopup rc=$?"
if python3 "$SD/merge_shards.py" "$OUT" "$HL"; then
  log "DONE GM=$(grep -oE '[0-9]+\.[0-9]+$' "$summ" | tail -1)"; touch "$RES/evalcell_${OUT}_${HL}L.done"
else log "merge INCOMPLETE — re-run to mop up remaining configs"; fi
