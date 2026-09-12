#!/bin/bash
# #412 pass 5 — one event line per state change, plus an hourly line.
#
# The agent session watches THIS, and not a lane's stdout. It prints a line
# only when an answer changes: a leg starts, a backbone lands, the AUC gate
# stops an arm, a head starts, a score lands, or the queue leaves the process
# table. The hourly line reports the live step and the AUC of each arm, so a
# stalled box is visible without a poll.
#
# SILENCE IS NOT SUCCESS. The queue holds three arms behind pass 4 for more
# than a day, so "no event" is the normal state for hours. The hourly line is
# what separates a healthy wait from a dead queue, and the `queue running`
# line fires the moment the lane dies.
set -uo pipefail
D=/tmp/contrastive-forecasting-412/reports/2026-09-06_moirai_small_size
R="$D/results"
CK=/home/jupyter/checkpoints_backup/cf-412
CELL=arm6_v2_combab_alignT
ARMS="${CF412_WATCH_ARMS:-k32_r200_08_lr56 k32_r100_09_dec_lr56 k8_r100_09_lr56}"
HB="${CF412_HB:-3600}"

live_csv(){ ls -t "$CK/$1/$CELL/leg_40k/"*_losses.csv 2>/dev/null | head -1; }
live_step(){ local f; f="$(live_csv "$1")"
  [ -n "$f" ] && tail -1 "$f" | cut -d, -f1 || echo -; }
# The AUC as the GATE reads it: a rolling median, never one raw row. A raw row
# swings by 0.2 between neighbours, and three sessions have compared a raw row
# with a median and read the difference as a disagreement. The column is taken
# from the header, so a trainer that adds a column does not move this reading.
live_auc(){ local f c; f="$(live_csv "$1")"
  [ -n "$f" ] || { echo -; return; }
  c="$(head -1 "$f" | tr ',' '\n' | grep -n '^auc$' | cut -d: -f1)"
  [ -n "$c" ] || { echo -; return; }
  tail -500 "$f" | cut -d, -f"$c" | grep -E '^[0-9.]+$' | sort -n \
    | awk '{v[NR]=$1} END {if(NR)printf "%.4f\n", v[int((NR+1)/2)]; else print "-"}'; }

# The card one arm's trainer runs on, or nothing.
arm_gpu(){ local pid dev
  for pid in $(ps -eo pid,args --no-headers 2>/dev/null \
       | awk -v n="_cf412_${1} " '$2 ~ /python/ && index($0 " ", n) { print $1 }'); do
    dev="$(tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null \
           | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)"
    [ -n "$dev" ] && { printf '%s\n' "$dev"; return 0; }
  done
  return 1
}

queue_n(){ ps -eo args --no-headers 2>/dev/null \
  | awk '$1 ~ /bash$/ && $2 ~ /pass5_lane\.sh$/ { n++ } END { print n+0 }'; }

state(){
  local a g s
  for a in $ARMS; do
    g="$(arm_gpu "$a")" && echo "leg $a trains on gpu $g"
    ls "$CK/$a/$CELL/leg_40k/"*_40k.pth >/dev/null 2>&1 \
      && echo "backbone $a at 40000 is on disk"
    [ -f "$R/collapsed_$a.txt" ] \
      && echo "arm $a LOST the contrastive task — $(head -2 "$R/collapsed_$a.txt" | tr '\n' ' ')"
    grep -q "head ${a}_bb40k_h30k_student on" "$R/heads.log" 2>/dev/null \
      && echo "head $a started"
    s="$R/score_${a}_bb40k_h30k_student.txt"
    [ -s "$s" ] && echo "SCORE $a = $(cat "$s")"
  done
  echo "queue running: $(queue_n)"
}

prev="$(state)"
echo "[watch] armed — $(echo "$prev" | tr '\n' '|')"
last=$(date +%s)
while :; do
  sleep 60
  now="$(state)"
  if [ "$now" != "$prev" ]; then
    diff <(printf '%s\n' "$prev") <(printf '%s\n' "$now") \
      | grep -E '^[<>]' | sed 's/^> /[event] /; s/^< /[event] ENDED: /'
    prev="$now"
  fi
  t=$(date +%s)
  if [ $(( t - last )) -ge "$HB" ]; then
    last=$t
    line="[hourly]"
    for a in $ARMS; do
      line="$line $a step=$(live_step "$a") auc=$(live_auc "$a")"
    done
    echo "$line queue=$(queue_n) gpu_free=$(nvidia-smi \
      --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/')"
  fi
  # Every arm holds a score or a collapse note: the pass is done.
  done_n=0
  for a in $ARMS; do
    [ -s "$R/score_${a}_bb40k_h30k_student.txt" ] && done_n=$(( done_n + 1 ))
    [ -f "$R/collapsed_$a.txt" ] && done_n=$(( done_n + 1 ))
  done
  [ "$done_n" -ge 3 ] && {
    echo "[watch] pass 5 complete — every arm has a score or a collapse note"
    exit 0; }
done
