#!/bin/bash
# One-line status per #393 stream, across elisa and the six vast.ai boxes.
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
EXP=/tmp/contrastive-forecasting-393/experiments/2026-08-04_ema_sched_ladder
BOXES="a ssh2.vast.ai 11448
b ssh4.vast.ai 13146
c ssh6.vast.ai 18762
d ssh7.vast.ai 18862
e ssh5.vast.ai 18856
f ssh1.vast.ai 18914"
echo "===== $(date '+%m-%d %H:%M:%S') ====="
echo "--- elisa ---"
for c in arm6_v2_combab_alignS arm6_v2_combab_alignT arm6_v2_nse_alignS arm4_combab arm6_v2_nse_alignT arm1_nse; do
  f="$EXP/results/run_cf393_$c.log"
  [ -f "$f" ] || continue
  printf '%-24s %s\n' "$c" "$(grep -E '^\[ *[0-9]+\]' "$f" | tail -n 1 | cut -c1-90)"
done
[ -f "$EXP/results/ladder.csv" ] && { echo "--- ladder.csv ---"; cat "$EXP/results/ladder.csv"; }
[ -f "$EXP/results/decisions.csv" ] && { echo "--- decisions.csv ---"; cat "$EXP/results/decisions.csv"; }
while read -r lbl host port; do
  [ -n "$lbl" ] || continue
  echo "--- vast $lbl ($host:$port) ---"
  ssh $SSH_OPTS -p "$port" "root@$host" 'R=/root/cf/experiments/2026-08-04_ema_sched_ladder/results;
    for f in "$R"/run_cf393_*.log; do [ -f "$f" ] || continue;
      n=$(basename "$f" .log); n=${n#run_cf393_};
      printf "%-24s %s\n" "$n" "$(grep -E "^\[ *[0-9]+\]" "$f" | tail -n 1 | cut -c1-90)"; done
    for f in /root/cf393_runs/*/eval/bb*/eval.log; do [ -f "$f" ] || continue;
      printf "  head %-28s %s\n" "$(basename "$(dirname "$f")")" \
        "$(grep -E "^\[ *[0-9]+\]|gift-eval rc|DONE|configs" "$f" | tail -n 1 | cut -c1-70)"; done
    [ -f "$R/ladder.csv" ] && { echo "ladder.csv:"; cat "$R/ladder.csv"; }
    [ -f "$R/decisions.csv" ] && { echo "decisions.csv:"; cat "$R/decisions.csv"; }
    tail -n 2 /root/queue.log 2>/dev/null' 2>/dev/null
done <<<"$BOXES"
