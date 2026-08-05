#!/bin/bash
# #393 event stream: one line per thing worth acting on. Emits new ladder
# scores, new extend-rule decisions, dead drivers and crash signatures.
# Silence means every stream is still climbing.
#
# The box list is the one in results/machines.txt. Six vast.ai boxes now,
# one or two cells each; a cell that finishes leaves its box idle, which
# `queue idle` reports so the box can be reused or destroyed.
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
EXP=/tmp/contrastive-forecasting-393/experiments/2026-08-04_ema_sched_ladder
STATE=/tmp/cf393_watch.state
BOXES="a ssh2.vast.ai 11448
b ssh4.vast.ai 13146
c ssh6.vast.ai 18762
d ssh7.vast.ai 18862
e ssh5.vast.ai 18856
f ssh1.vast.ai 18914"
: > "$STATE"

emit(){ grep -vxF -f "$STATE" <<<"$1" | while read -r l; do [ -n "$l" ] && echo "$l"; done
        printf '%s\n' "$1" >> "$STATE"; sort -u "$STATE" -o "$STATE"; }

while true; do
  buf=""
  for f in "$EXP/results/ladder.csv" "$EXP/results/decisions.csv"; do
    [ -f "$f" ] && buf+=$(sed 's/^/elisa '"$(basename "$f" .csv)"': /' "$f")$'\n'
  done
  while read -r lbl host port; do
    [ -n "$lbl" ] || continue
    r=$(ssh -n $SSH_OPTS -p "$port" "root@$host" \
      'R=/root/cf/experiments/2026-08-04_ema_sched_ladder/results;
       for f in ladder decisions; do [ -f "$R/$f.csv" ] && sed "s/^/$f: /" "$R/$f.csv"; done;
       grep -hE "Traceback|CUDA error|out of memory|Killed|nan|NaN" "$R"/run_cf393_*.log 2>/dev/null | tail -n 2;
       grep -hE "HOLD:|ABORT:|rc=[1-9]" "$R"/leg_*.log 2>/dev/null | tail -n 2;
       pgrep -f "ladder.py --cells" >/dev/null || echo "queue idle";
       tail -n 1 /root/queue.log 2>/dev/null' 2>/dev/null)
    [ -n "$r" ] && buf+=$(sed "s/^/vast $lbl /" <<<"$r")$'\n'
  done <<<"$BOXES"
  c=$(grep -hE "Traceback|CUDA error|out of memory|Killed|HOLD:|ABORT:|rc=[1-9]" \
      "$EXP"/results/ladder_*.log "$EXP"/results/leg_*.log 2>/dev/null | tail -n 4)
  [ -n "$c" ] && buf+=$(sed 's/^/elisa ERR: /' <<<"$c")$'\n'
  n=$(pgrep -fc "ladder.py --cells" 2>/dev/null || echo 0)
  buf+="elisa drivers alive: $n"$'\n'
  emit "$(grep -v '^$' <<<"$buf")"
  sleep 600
done
