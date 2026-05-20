#!/bin/bash
# #309 sync — one tick: re-resolve ssh, atomically pull backbone +
# losses + log + optimizer + periodic Nk for every arm under work into
# the MAIN checkout (CLAUDE.md: sync into main, not worktree).
#   sync.sh        — pull all arms (alpha/beta/gamma) once
#   sync.sh <arm>  — pull one arm
set -uo pipefail
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
EXP="$WT/experiments/2026-05-20_bottleneck_beta2_confound"
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-20_bottleneck_beta2_confound
SAFE="$WT/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"
source "$EXP/scripts/_ssh.sh"
ARM="${1:-}"
ENV="$EXP/scripts/state/box.env"
[ -f "$ENV" ] || { echo "no state $ENV"; exit 1; }
. "$ENV"
read -r H P < <(ssh_coords "$INST") || { H="$HOST"; P="$PORT"; }
[ -n "$H" ] && [ -n "$P" ] || { echo "ssh unresolved for inst $INST"; exit 1; }
echo "=== sync tick $(date '+%m-%d %H:%M:%S') inst=$INST @ $H:$P ==="
RR="/workspace/app/runs"; RS="/workspace/app/results"
pull(){ bash "$SAFE" "$H" "$P" "$1" "$2" "${3:-50}" 2>&1 | sed "s/^/[#309] /"; }
remote_ls(){ ssh $SSHO -p "$P" "root@$H" "ls $1 2>/dev/null" 2>/dev/null; }

# Always pull the serial driver log (small, free)
LOC="$MAIN"; mkdir -p "$LOC/runs" "$LOC/results"
pull "/workspace/app/box_serial.log" "$LOC/results/box_serial.log" 50
pull "$RS/serial.log"                "$LOC/results/serial.log"     50

ARMS=(alpha beta gamma)
[ -n "$ARM" ] && ARMS=("$ARM")
for A in "${ARMS[@]}"; do
  NAME="bb_${A}_50k"
  pull "$RS/run_${NAME}.log"      "$LOC/results/run_${NAME}.log"      50
  pull "$RR/${NAME}_losses.csv"   "$LOC/runs/${NAME}_losses.csv"      50
  # best + FINAL backbone + optimizer
  for s in _best_loss _best_gap _FINAL; do
    pull "$RR/${NAME}${s}.pth"            "$LOC/runs/${NAME}${s}.pth"            20000000
    pull "$RR/${NAME}${s}_optimizer.pth"  "$LOC/runs/${NAME}${s}_optimizer.pth"  40000000
  done
  # periodic Nk
  for f in $(remote_ls "$RR/${NAME}_*k.pth"); do
    b=$(basename "$f"); pull "$RR/$b" "$LOC/runs/$b" 20000000
    pull "$RR/${b%.pth}_optimizer.pth" "$LOC/runs/${b%.pth}_optimizer.pth" 40000000
  done
  # health
  LG="$LOC/results/run_${NAME}.log"
  if [ -f "$LG" ]; then
    grep -qiE 'nan loss|loss is nan|diverge|CUDA error|out of memory' "$LG" && echo "[#309 $A] ⚠️ ANOMALY"
    grep -qiE 'Training complete|Finished training|BB DONE' "$LG" 2>/dev/null && echo "[#309 $A] ✓ BB-DONE seen"
  fi
  echo "[#309 $A] local backbone: $(ls -lh "$LOC/runs/${NAME}_best_loss.pth" 2>/dev/null | awk '{print $5}') | losses rows: $(wc -l < "$LOC/runs/${NAME}_losses.csv" 2>/dev/null || echo 0)"
done
