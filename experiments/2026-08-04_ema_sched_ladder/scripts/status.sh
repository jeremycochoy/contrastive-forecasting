#!/bin/bash
# One-line status per #393 stream, across elisa and both vast.ai boxes.
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
EXP=/tmp/contrastive-forecasting-393/experiments/2026-08-04_ema_sched_ladder
echo "===== $(date '+%m-%d %H:%M:%S') ====="
echo "--- elisa ---"
for c in arm6_v2_combab_alignS arm6_v2_combab_alignT arm6_v2_nse_alignS arm4_combab arm6_v2_nse_alignT arm1_nse; do
  f="$EXP/results/run_cf393_$c.log"
  [ -f "$f" ] || continue
  printf '%-24s %s\n' "$c" "$(grep -E '^\[ *[0-9]+\]' "$f" | tail -n 1 | cut -c1-90)"
done
[ -f "$EXP/results/ladder.csv" ] && { echo "--- ladder.csv ---"; cat "$EXP/results/ladder.csv"; }
[ -f "$EXP/results/decisions.csv" ] && { echo "--- decisions.csv ---"; cat "$EXP/results/decisions.csv"; }
for spec in "A ssh2.vast.ai 11448" "B ssh4.vast.ai 13146"; do
  set -- $spec
  echo "--- vast $1 ($2:$3) ---"
  ssh $SSH_OPTS -p "$3" "root@$2" 'R=/root/cf/experiments/2026-08-04_ema_sched_ladder/results;
    for f in "$R"/run_cf393_*.log; do [ -f "$f" ] || continue;
      n=$(basename "$f" .log); n=${n#run_cf393_};
      printf "%-24s %s\n" "$n" "$(grep -E "^\[ *[0-9]+\]" "$f" | tail -n 1 | cut -c1-90)"; done
    [ -f "$R/ladder.csv" ] && { echo "ladder.csv:"; cat "$R/ladder.csv"; }
    [ -f "$R/decisions.csv" ] && { echo "decisions.csv:"; cat "$R/decisions.csv"; }
    cat /root/queue.log 2>/dev/null | tail -n 3' 2>/dev/null
done
