#!/bin/bash
# Variance sync — one tick per (arm,seed): re-resolve ssh, atomically
# pull backbone + losses + log from a vast box into the MAIN checkout's
# variance/<arm>_seed<seed>/ dir. CLAUDE.md: sync into main, not worktree.
#   sync_variance.sh <arm> <seed>
set -uo pipefail
WT=/home/jupyter/cf-wt-crossed-loss
EXP="$WT/experiments/2026-05-19_crossed_loss_xbranch_ablation"
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_xbranch_ablation
SAFE="$WT/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"
source "$EXP/scripts/_ssh.sh"
ARM="${1:?arm}"; SEED="${2:?seed}"
SS="s${SEED:(-2)}"; NAME="cl_${ARM}_50k_${SS}"
ENV="$EXP/scripts/state/variance_${ARM}_${SS}.env"
[ -f "$ENV" ] || { echo "no state $ENV"; exit 1; }
. "$ENV"
LOC="$MAIN/variance/${ARM}_seed${SEED}"; mkdir -p "$LOC/runs" "$LOC/results"
read -r H P < <(ssh_coords "$INST") || { H="$HOST"; P="$PORT"; }
[ -n "$H" ] && [ -n "$P" ] || { echo "[$ARM-$SS] ssh unresolved"; exit 1; }
echo "=== sync $ARM/$SS tick $(date '+%m-%d %H:%M:%S') inst=$INST @ $H:$P ==="
RR="/workspace/app/runs"; RS="/workspace/app/results"
pull(){ bash "$SAFE" "$H" "$P" "$1" "$2" "${3:-50}" 2>&1 | sed "s/^/[$ARM-$SS] /"; }
remote_ls(){ ssh $SSHO -p "$P" "root@$H" "ls $1 2>/dev/null" 2>/dev/null; }

# log + losses (small)
pull "$RS/run_${NAME}.log"      "$LOC/results/run_${NAME}.log"      50
pull "$RR/${NAME}_losses.csv"   "$LOC/runs/${NAME}_losses.csv"      50
# best + FINAL backbone + optimizer
for s in _best_loss _best_gap _FINAL; do
  pull "$RR/${NAME}${s}.pth"            "$LOC/runs/${NAME}${s}.pth"            20000000
  pull "$RR/${NAME}${s}_optimizer.pth" "$LOC/runs/${NAME}${s}_optimizer.pth"  40000000
done
# periodic Nk
for f in $(remote_ls "$RR/${NAME}_*k.pth"); do
  b=$(basename "$f"); pull "$RR/$b" "$LOC/runs/$b" 20000000
  pull "$RR/${b%.pth}_optimizer.pth" "$LOC/runs/${b%.pth}_optimizer.pth" 40000000
done
# health
LG="$LOC/results/run_${NAME}.log"
if [ -f "$LG" ]; then
  grep -qiE 'nan loss|loss is nan|diverge|CUDA error|out of memory' "$LG" && echo "[$ARM-$SS] ⚠️ ANOMALY"
  grep -qiE 'Training complete|Finished training|BB DONE' "$LG" 2>/dev/null && echo "[$ARM-$SS] ✓ BB-DONE seen"
fi
echo "[$ARM-$SS] local backbone: $(ls -lh "$LOC/runs/${NAME}_best_loss.pth" 2>/dev/null | awk '{print $5}') | losses rows: $(wc -l < "$LOC/runs/${NAME}_losses.csv" 2>/dev/null || echo 0)"
