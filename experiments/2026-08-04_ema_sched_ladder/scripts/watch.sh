#!/bin/bash
# #393 event stream: one line per thing worth acting on. Emits new ladder
# scores, new extend-rule decisions, dead drivers and crash signatures.
# Silence means every stream is still climbing.
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
EXP=/tmp/contrastive-forecasting-393/experiments/2026-08-04_ema_sched_ladder
STATE=/tmp/cf393_watch.state
: > "$STATE"

emit(){ grep -vxF -f "$STATE" <<<"$1" | while read -r l; do [ -n "$l" ] && echo "$l"; done
        printf '%s\n' "$1" >> "$STATE"; sort -u "$STATE" -o "$STATE"; }

while true; do
  buf=""
  # scores and decisions, all three machines
  for f in "$EXP/results/ladder.csv" "$EXP/results/decisions.csv"; do
    [ -f "$f" ] && buf+=$(sed 's/^/elisa '"$(basename "$f" .csv)"': /' "$f")$'\n'
  done
  for spec in "A ssh2.vast.ai 11448" "B ssh4.vast.ai 13146"; do
    set -- $spec
    r=$(ssh $SSH_OPTS -p "$3" "root@$2" \
      'R=/root/cf/experiments/2026-08-04_ema_sched_ladder/results;
       for f in ladder decisions; do [ -f "$R/$f.csv" ] && sed "s/^/$f: /" "$R/$f.csv"; done;
       grep -hE "Traceback|CUDA error|out of memory|Killed|nan|NaN" "$R"/run_cf393_*.log 2>/dev/null | tail -n 2;
       tail -n 2 /root/queue.log 2>/dev/null' 2>/dev/null)
    [ -n "$r" ] && buf+=$(sed "s/^/vast $1 /" <<<"$r")$'\n'
  done
  # elisa crash signatures and driver deaths
  c=$(grep -hE "Traceback|CUDA error|out of memory|Killed|rc=[1-9]" \
      "$EXP"/results/ladder_*.log "$EXP"/results/leg_*.log 2>/dev/null | tail -n 4)
  [ -n "$c" ] && buf+=$(sed 's/^/elisa ERR: /' <<<"$c")$'\n'
  n=$(pgrep -fc "ladder.py --cells" 2>/dev/null || echo 0)
  buf+="elisa drivers alive: $n"$'\n'
  emit "$(grep -v '^$' <<<"$buf")"
  sleep 600
done
