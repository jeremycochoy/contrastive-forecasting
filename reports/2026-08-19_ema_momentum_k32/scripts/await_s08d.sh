#!/bin/bash
# #404 — wait for the s08d 97-config GIFT-Eval, and report when it lands.
#
# The eval runs detached under `evals_elisa.sh`. This waiter blocks on its
# SCORE FILE, which is the artefact collect.sh reads, and not on a pid: an
# eval that dies leaves no score, and one that finishes writes one.
#
# It logs progress every 10 min as a row count out of 97, off the shard CSVs.
set -uo pipefail
RES=/tmp/contrastive-forecasting-404/reports/2026-08-19_ema_momentum_k32/results
GIFT=/home/jupyter/cf404_sync/box_a/sync/s08d/eval/s08d_bb40k_h30k_student/gift
SCORE="$RES/score_s08d_bb40k_h30k_student.txt"
LOG="$RES/await_s08d.log"
TIMEOUT="${TIMEOUT:-10800}"   # 3 h against 1 h 14 measured on s08c
POLL=60

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [await s08d] $*" | tee -a "$LOG"; }

done_rows(){  # configs finished, over the four shards
  local n=0 f
  for f in "$GIFT"/shard_*/all_results.csv; do
    [ -s "$f" ] || continue
    n=$(( n + $(( $(wc -l <"$f") - 1 )) ))
  done
  echo "$n"
}

# An eval process of THIS backbone. Read off /proc, never off a pattern
# alone: a pattern also matches the shell that carries it, and elisa runs
# other sessions' work.
shards_up(){
  local p cl n=0
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    cl="$(tr '\0' ' ' <"/proc/$p/cmdline" 2>/dev/null)" || continue
    case "$cl" in *eval_gift_eval_official.py*) ;; *) continue ;; esac
    case "$cl" in *"$GIFT"*) n=$(( n + 1 )) ;; esac
  done
  echo "$n"
}

say "START — waiting for $SCORE"
waited=0
while [ ! -s "$SCORE" ]; do
  if [ "$waited" -ge "$TIMEOUT" ]; then
    say "TIMEOUT after ${waited}s — $(done_rows)/97 configs, $(shards_up) shard(s) up"
    exit 1
  fi
  if [ "$(shards_up)" -eq 0 ] && [ ! -s "$SCORE" ]; then
    sleep 30
    [ -s "$SCORE" ] && break
    say "NO SHARD IS UP and no score — the eval died at $(done_rows)/97"
    tail -5 "$GIFT"/shard_*/shard.log 2>/dev/null | tee -a "$LOG"
    exit 2
  fi
  [ $(( waited % 600 )) -eq 0 ] && \
    say "  $(done_rows)/97 configs, $(shards_up) shard(s) up, ${waited}s"
  sleep "$POLL"; waited=$(( waited + POLL ))
done

say "SCORE s08d $(tr -d ' \t\r\n' <"$SCORE") after ${waited}s"
grep -h 'Aggregate GM-Relative MASE' "$GIFT/summary.txt" 2>/dev/null | tee -a "$LOG"
say "DONE"
