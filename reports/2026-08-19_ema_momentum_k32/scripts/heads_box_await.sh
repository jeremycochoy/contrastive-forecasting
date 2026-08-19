#!/bin/bash
# #404 — wait for the four heads on the box, and report every tick.
#
# `heads_box.sh` runs on the box and detaches. This watches it from elisa over
# ssh, so a dropped session does not take the watcher with it. It ends when
# the four head checkpoints are on the box, or when no trainer is left alive.
#
# Usage: nohup setsid bash scripts/heads_box_await.sh > results/heads_box_await.log 2>&1 &
set -uo pipefail

HOST="${HOST:-ssh1.vast.ai}"
PORT="${PORT:-29998}"
POLL="${POLL:-300}"
BOX_STUDY="${BOX_STUDY:-/root/cf/reports/2026-08-19_ema_momentum_k32}"
BOX_RUNS="${BOX_RUNS:-/root/cf404_runs}"
ARMS="${ARMS:-a08 a09 s08 s09}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o ServerAliveInterval=15)

say(){ echo "[$(date -u '+%m-%d %H:%M:%S')Z] [#404 heads await] $*"; }
rsh(){ timeout 90 ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@" 2>/dev/null; }

n=0
while :; do
  n=$(( n + 1 ))
  out="$(rsh "
    for a in $ARMS; do
      t=\${a}_bb40k_h30k_student
      ck=$BOX_RUNS/\$a/eval/\$t/qhead_\${t}_s20260722_final.pth
      if [ -f \"\$ck\" ]; then echo \"\$a DONE \$(wc -c <\"\$ck\")\"
      else echo \"\$a \$(grep -c '^' $BOX_RUNS/\$a/eval/\$t/stop.log 2>/dev/null || echo 0) \$(tail -1 $BOX_RUNS/\$a/eval/\$t/stop.log 2>/dev/null | tr -d '\n' | tail -c 90)\"
      fi
    done
    echo \"LIVE \$(pgrep -c -f train_forecasting_head 2>/dev/null || echo 0)\"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' '|'
  ")"
  if [ -z "$out" ]; then
    say "tick $n: no answer from the box"
  else
    say "tick $n"
    echo "$out" | sed 's/^/    /'
  fi

  done_n="$(printf '%s\n' "$out" | grep -c ' DONE ')"
  live="$(printf '%s\n' "$out" | awk '/^LIVE /{print $2}')"
  if [ "$done_n" -ge 4 ]; then say "ALL FOUR HEADS ON THE BOX"; exit 0; fi
  if [ -n "${live:-}" ] && [ "$live" = "0" ] && [ "$n" -gt 2 ]; then
    say "NO TRAINER ALIVE and $done_n of 4 heads present — stopping the watch"
    exit 1
  fi
  sleep "$POLL"
done
