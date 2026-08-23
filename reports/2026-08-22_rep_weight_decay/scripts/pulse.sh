#!/bin/bash
# #409 — one line per live leg, one line per card. For the hourly heartbeat.
R=/home/jupyter/checkpoints_backup/cf-409
R2=/tmp/contrastive-forecasting-409/reports/2026-08-22_rep_weight_decay/results
echo "PULSE $(date '+%m-%d %H:%M') | $(nvidia-smi --query-gpu=index,memory.free \
  --format=csv,noheader,nounits | awk -F', ' '{printf "gpu %s: %s MiB free  ", $1, $2}')"
for f in "$R"/*/arm6_v2_combab_alignT/leg_40k/*_losses.csv; do
  [ -e "$f" ] || continue
  arm=$(echo "$f" | sed -E 's#.*/cf-409/([^/]+)/.*#\1#')
  n=$(( $(wc -l < "$f") - 1 ))
  age=$(( ( $(date +%s) - $(stat -c %Y "$f") ) / 60 ))
  [ "$age" -le 20 ] && printf '  %-14s step %6d  (last write %2d min ago)\n' "$arm" "$n" "$age"
done
printf '  scores: '; tail -n +2 "$R2/scores.csv" 2>/dev/null | awk -F, '{printf "%s=%s ", $1, $NF}'; echo
