#!/bin/bash
# One sync tick for one arm: re-resolve ssh, atomically pull artifacts
# FROM the vast box INTO the MAIN checkout (CLAUDE.md rule 4). Uses the
# audited safe_pull.sh (atomic .tmp + per-class size floor + 1-deep
# rotation). Never raw scp to a live destination.
#   sync_arm.sh <arm>
set -uo pipefail
WT=/home/jupyter/cf-wt-crossed-loss
EXP="$WT/experiments/2026-05-19_crossed_loss_xbranch_ablation"
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_xbranch_ablation
SAFE="$WT/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"
source "$EXP/scripts/_ssh.sh"
ARM="${1:?arm}"; ENV="$EXP/scripts/state/$ARM.env"
[ -f "$ENV" ] || { echo "no state for $ARM"; exit 1; }
. "$ENV"
NAME="cl_${ARM}_50k"; QN="${NAME}_qhead_xfmr2L_quant_30k"
LOC="$MAIN/sync_${ARM}"; mkdir -p "$LOC/runs" "$LOC/results"
read -r H P < <(ssh_coords "$INST") || { H="$HOST"; P="$PORT"; }
[ -n "$H" ] && [ -n "$P" ] || { echo "[$ARM] ssh unresolved"; exit 1; }
echo "=== sync $ARM tick $(date '+%m-%d %H:%M:%S') inst=$INST @ $H:$P ==="
RR="/workspace/app/runs"; RS="/workspace/app/results"
pull(){ bash "$SAFE" "$H" "$P" "$1" "$2" "${3:-50}" 2>&1 | sed "s/^/[$ARM] /"; }
remote_ls(){ ssh $SSHO -p "$P" "root@$H" "ls $1 2>/dev/null" 2>/dev/null; }

# training log + losses csv (small, always)
pull "$RS/run_${NAME}.log"      "$LOC/results/run_${NAME}.log"      50
pull "$RR/${NAME}_losses.csv"   "$LOC/runs/${NAME}_losses.csv"      50
# best + FINAL backbone (+ optimizer) — model ~47MB, optimizer ~95MB
for s in _best_loss _best_gap _FINAL; do
  pull "$RR/${NAME}${s}.pth"            "$LOC/runs/${NAME}${s}.pth"            20000000
  pull "$RR/${NAME}${s}_optimizer.pth" "$LOC/runs/${NAME}${s}_optimizer.pth"  40000000
done
# periodic Nk checkpoints (+ optimizer)
for f in $(remote_ls "$RR/${NAME}_*k.pth"); do
  b=$(basename "$f"); pull "$RR/$b" "$LOC/runs/$b" 20000000
  pull "$RR/${b%.pth}_optimizer.pth" "$LOC/runs/${b%.pth}_optimizer.pth" 40000000
done
# q-head artifacts (head ~ a few MB)
pull "$RS/run_${QN}.log"        "$LOC/results/run_${QN}.log"        50
for s in _best _final _FINAL; do
  pull "$RR/${NAME}_qhead${s}.pth" "$LOC/runs/${NAME}_qhead${s}.pth" 1000000
  pull "$RR/${QN}${s}.pth"         "$LOC/runs/${QN}${s}.pth"         1000000
done
# eval outputs (summary.txt + per-config CSVs)
for tag in triage full; do
  d="$RS/gift_eval_${tag}_${NAME}"
  for f in $(remote_ls "$d/*"); do
    b=$(basename "$f"); pull "$d/$b" "$LOC/results/gift_eval_${tag}_${NAME}/$b" 5
  done
done
# health
LG="$LOC/results/run_${NAME}.log"
if [ -f "$LG" ]; then
  grep -qiE 'nan loss|loss is nan|diverge|CUDA error|out of memory' "$LG" && echo "[$ARM] ⚠️ ANOMALY in training log"
  grep -qiE 'Training complete|Finished training|BB DONE' "$LG" 2>/dev/null && echo "[$ARM] ✓ backbone training-complete marker seen"
fi
echo "[$ARM] local backbone: $(ls -lh "$LOC/runs/${NAME}"_best_loss.pth 2>/dev/null | awk '{print $5}') | losses rows: $(wc -l < "$LOC/runs/${NAME}_losses.csv" 2>/dev/null || echo 0)"
