#!/bin/bash
# Runs ON the box. Cmdline of THIS process is just "bash _hardkill.sh"
# (no train.py/torchrun substring) so pkill -f cannot self-match.
# Kills every GPU compute proc + its ancestors, then any backbone
# wrapper, until nvidia-smi shows zero compute apps.
set -u
for round in 1 2 3 4 5 6; do
  PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  WRAP=$(pgrep -f 'box_run[.]sh' ; pgrep -f 'bin/torchru[n]' ; pgrep -f 'distributed.run' ; pgrep -f '/trai[n][.]py')
  ALL=$(echo "$PIDS $WRAP" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u)
  [ -z "$ALL" ] && { echo "round $round: clean (no GPU/backbone procs)"; break; }
  for p in $ALL; do
    pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    kill -9 "$p" 2>/dev/null
    [ -n "$pp" ] && [ "$pp" != 1 ] && kill -9 "$pp" 2>/dev/null
  done
  echo "round $round: killed [$ALL]"
  sleep 4
done
echo "FINAL gpu_compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
cd /workspace/app && rm -rf runs results box_*.log _restart* _remote_restart.sh
echo "cleaned artifacts; ls: $(ls /workspace/app | tr '\n' ' ')"
