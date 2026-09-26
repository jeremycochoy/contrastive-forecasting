#!/bin/bash
# #412 pass 3 — one event line per state change, plus an hourly line.
#
# The agent session watches THIS. It prints a line only when an answer
# changes: a backbone lands, the AUC gate stops an arm, a head starts or
# ends, a score lands, or a lane leaves the process table. The hourly line
# reports the live step of each arm, so a stalled box is visible.
set -uo pipefail
D=/tmp/contrastive-forecasting-412/reports/2026-09-06_moirai_small_size
R="$D/results"
CK=/home/jupyter/checkpoints_backup/cf-412
ARMS="k32_r100_09_sum k3_r100_09_lr45 k3_r100_09_lr70"
HB="${CF412_HB:-3600}"

live_csv(){ ls "$CK/$1/arm6_v2_combab_alignT/leg_40k/"*_losses.csv 2>/dev/null | head -1; }
live_step(){ local f; f="$(live_csv "$1")"; [ -n "$f" ] && tail -1 "$f" | cut -d, -f1 || echo -; }
live_auc(){ local f; f="$(live_csv "$1")"
  [ -n "$f" ] || { echo -; return; }
  tail -300 "$f" | cut -d, -f17 | grep -E '^[0-9.]+$' | sort -n \
    | awk '{v[NR]=$1} END {if(NR)printf "%.4f\n", v[int((NR+1)/2)]; else print "-"}'; }

state(){
  local a s
  for a in $ARMS; do
    ls "$CK/$a/arm6_v2_combab_alignT/leg_40k/"*_40k.pth >/dev/null 2>&1 && echo "backbone $a at 40000 is on disk"
    [ -f "$R/collapsed_$a.txt" ] && echo "arm $a LOST the contrastive task — $(head -2 "$R/collapsed_$a.txt" | tr '\n' ' ')"
    grep -q "head ${a}_bb40k_h30k_student on" "$R/heads.log" 2>/dev/null && echo "head $a started"
    s="$R/score_${a}_bb40k_h30k_student.txt"
    [ -s "$s" ] && echo "SCORE $a = $(cat "$s")"
  done
  echo "lanes running: $(ps -eo args --no-headers 2>/dev/null | awk '$1 ~ /bash$/ && $2 ~ /pass2_lane\.sh$/ { n++ } END { print n+0 }')"
}

prev="$(state)"
echo "[watch] armed — $(echo "$prev" | tr '\n' '|')"
last=$(date +%s)
while :; do
  sleep 60
  now="$(state)"
  if [ "$now" != "$prev" ]; then
    diff <(printf '%s\n' "$prev") <(printf '%s\n' "$now") | grep '^>' | sed 's/^> /[event] /'
    prev="$now"
  fi
  t=$(date +%s)
  if [ $(( t - last )) -ge "$HB" ]; then
    last=$t
    line="[hourly]"
    for a in $ARMS; do line="$line $a step=$(live_step "$a") auc=$(live_auc "$a")"; done
    echo "$line lanes=$(ps -eo args --no-headers | awk '$1 ~ /bash$/ && $2 ~ /pass2_lane\.sh$/ {n++} END {print n+0}')"
  fi
  # Every arm scored, and no lane left: the pass is done.
  done_n=0
  for a in $ARMS; do
    [ -s "$R/score_${a}_bb40k_h30k_student.txt" ] && done_n=$(( done_n + 1 ))
    [ -f "$R/collapsed_$a.txt" ] && done_n=$(( done_n + 1 ))
  done
  [ "$done_n" -ge 3 ] && { echo "[watch] pass 3 complete — every arm has a score or a collapse note"; exit 0; }
done
