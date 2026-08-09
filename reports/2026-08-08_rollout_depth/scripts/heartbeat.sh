#!/bin/bash
# #373 — one status line per box, plus the things worth waking up for.
#
# Usage: bash heartbeat.sh          # one pass, prints and exits
#
# Prints, per box: the newest step line the trainer wrote, whether a python
# process is alive on it, and whether the local sync loop for that box is
# alive. Then the vast.ai balance, because the study's ceiling is money and
# not time.
#
# Anything on a line starting `ALERT` is a thing to act on now: a dead
# trainer with jobs left, a dead sync loop, a NaN, or a balance that will
# not cover the boxes still running.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
BOXES="${BOXES_FILE:-$STUDY/results/boxes.tsv}"
SYNC_BASE="${CF373_SYNC_BASE:-$HOME/cf373_sync}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=10 -n)
[ -f "$BOXES" ] || { echo "ALERT no box table at $BOXES"; exit 1; }

while IFS=$'\t' read -r lbl id host port jobs; do
  case "$lbl" in ''|'#'*) continue ;; esac

  info=$(timeout 45 ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" '
      last=$(grep -hE "sps  ETA" /root/cf/reports/2026-08-08_rollout_depth/results/run_*.log 2>/dev/null | tail -1)
      alive=$(pgrep -c -f "train.py" 2>/dev/null || echo 0)
      done_f=$(test -f /root/cf/reports/2026-08-08_rollout_depth/results/QUEUE_DONE && echo yes || echo no)
      nan=$(grep -lc -iE "nan|inf" /root/cf/reports/2026-08-08_rollout_depth/results/run_*.log 2>/dev/null | wc -l)
      ck=$(ls /root/cf373_runs/*/*_40k.pth /root/cf373_runs/*/*/*_40k.pth 2>/dev/null | grep -vc optimizer || echo 0)
      echo "LAST|$last"; echo "ALIVE|$alive"; echo "DONE|$done_f"; echo "NAN|$nan"; echo "CK|$ck"
    ' 2>/dev/null)

  if [ -z "$info" ]; then
    echo "ALERT [$lbl] unreachable ($host:$port), jobs: $jobs"
    continue
  fi
  last=$(sed -n 's/^LAST|//p' <<<"$info")
  alive=$(sed -n 's/^ALIVE|//p' <<<"$info")
  qdone=$(sed -n 's/^DONE|//p' <<<"$info")
  nan=$(sed -n 's/^NAN|//p' <<<"$info")
  ck=$(sed -n 's/^CK|//p' <<<"$info")

  loop=""
  for p in $(pgrep -f "bash .*sync_loop.sh" 2>/dev/null); do
    [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$SYNC_BASE/$lbl" ] && { loop="$p"; break; }
  done

  printf '[%s] %s | train=%s sync=%s ckpts=%s | %s\n' \
    "$lbl" "$jobs" "${alive:-?}" "${loop:-DEAD}" "${ck:-0}" \
    "$(sed 's/^ *//' <<<"${last:-<no step line yet>}")"

  [ "${alive:-0}" = "0" ] && [ "$qdone" != "yes" ] && \
    echo "ALERT [$lbl] no trainer running and the queue is not drained"
  [ -z "$loop" ] && echo "ALERT [$lbl] sync loop is dead"
  [ "${nan:-0}" != "0" ] && echo "ALERT [$lbl] a run log mentions nan/inf"
  [ "$qdone" = "yes" ] && echo "[$lbl] QUEUE DRAINED — destroy the box once its checkpoints are synced"
done < "$BOXES"

# elisa's half: the two head/eval drivers, and what they have produced.
drv=$(pgrep -fc "bash .*stops_driver.sh" 2>/dev/null || echo 0)
heads=$(pgrep -fc "train_forecasting_head.py" 2>/dev/null || echo 0)
evals=$(pgrep -fc "eval_gift_eval_official.py" 2>/dev/null || echo 0)
scores=$(ls "$STUDY/results"/score_*.txt 2>/dev/null | wc -l)
printf '[elisa] drivers=%s heads=%s eval-shards=%s scores=%s/10\n' \
  "$drv" "$heads" "$evals" "$scores"
[ "${drv:-0}" -lt 2 ] && echo "ALERT [elisa] fewer than 2 stops drivers running"
tail -n 3 "$STUDY/results/stops_driver.log" 2>/dev/null | sed 's/^/        /'

bal=$(timeout 90 vastrun-balance 2>/dev/null | awk '/Credit/ {print $2}')
n_up=$(timeout 120 vastrun-status 2>/dev/null | grep -c 'cf373-rollout' || echo 0)
echo "balance=${bal:-?} instances=${n_up}"
# ~$0.37/h per box. Under an hour of runway left on what is up is a thing
# to act on, not a thing to notice tomorrow.
case "${bal:-}" in
  \$*) v=${bal#\$}; awk -v b="$v" -v n="$n_up" \
        'BEGIN { if (n > 0 && b < 0.40 * n) print "ALERT balance $" b " is under an hour for " n " box(es)" }' ;;
esac
