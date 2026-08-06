#!/bin/bash
# #393 — watch the last leg of the card: the bb200k stops on elisa.
#
# Three cells reached 200,000 steps. arm5_combab_alignT is already scored and
# stopped on the rule (none_down). The other two are evaluating now:
#
#   arm5_combab_alignS   student only — its bb100k branch was one_down, and
#                        the rule carries only the head that improved forward
#   arm6_v2_nse_alignT   student and teacher — bb100k was both_down
#
# HOLD_ABOVE is 200000, so ladder.py records session_end and the driver exits
# rather than starting a leg to 300k. This prints one line per change and
# exits when both drivers are gone, which is the end of the experiment's
# compute.
#
# Emits on failure too. An eval that dies leaves the driver blocked in
# eval_stop.sh's sleep loop with nothing running, and silence from a watcher
# that greps only for scores looks exactly like an eval still in progress.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
RES="$EXP/results"
RUNS="${RUNS:-/home/jupyter/checkpoints_backup/cf-393}"
POLL="${POLL:-300}"

# cell:head pairs still due at bb200k, from each cell's bb100k branch.
DUE=(arm5_combab_alignS:student arm6_v2_nse_alignT:student arm6_v2_nse_alignT:teacher)

say(){ echo "[$(date -u '+%m-%d %H:%M:%SZ')] [bb200k] $*"; }

last=""
while :; do
  now=""
  scored=0
  for d in "${DUE[@]}"; do
    cell="${d%%:*}"; enc="${d##*:}"
    f="$RUNS/$cell/eval/score_bb200k_$enc.txt"
    if [ -s "$f" ]; then
      now+=" $cell/$enc=$(tr -d '[:space:]' <"$f")"
      scored=$(( scored + 1 ))
    else
      # rows finished across this eval's four shards, out of 97 configs
      n=$(cat "$RUNS/$cell/eval/bb200k_$enc"/gift/shard_*/all_results.csv 2>/dev/null \
          | grep -cv '^dataset' || echo 0)
      now+=" $cell/$enc=$n/97"
    fi
  done

  # Anchored, and it has to be. The shell that launched arm5_combab_alignS is
  # still alive with the whole launch command in its argv, so an unanchored
  # `ladder.py --cells` counts a driver that does not exist and the exit
  # condition below can never be met — the watcher would run to the end of the
  # session reporting 3/3 scored and never say so. `^python3` matches the
  # driver itself and not the /bin/bash that spawned it (nor pgrep's own argv,
  # which starts with "pgrep").
  drivers=$(pgrep -fc '^python3 -u scripts/ladder\.py --cells' 2>/dev/null; true)
  drivers="${drivers:-0}"
  evals=$(pgrep -fc '[e]val_gift_eval_official\.py' 2>/dev/null; true)
  evals="${evals:-0}"
  heads=$(pgrep -fc '[t]rain_forecasting_head\.py' 2>/dev/null; true)
  heads="${heads:-0}"

  # A driver alive with nothing running under it is a dead eval, not progress.
  stall=""
  [ "$drivers" -gt 0 ] && [ "$evals" -eq 0 ] && [ "$heads" -eq 0 ] && [ "$scored" -lt 3 ] \
    && stall=" STALLED: driver alive, no eval and no head running"

  line="$scored/3 scored |$now | drivers=$drivers evals=$evals heads=$heads$stall"
  [ "$line" != "$last" ] && say "$line"
  last="$line"

  if [ "$scored" -ge 3 ] && [ "$drivers" -eq 0 ]; then
    say "ALL bb200k STOPS SCORED, both drivers exited"
    for d in "${DUE[@]}"; do
      cell="${d%%:*}"
      grep -h "^\[ladder\] $cell @200000" "$RES/ladder_$cell.log" 2>/dev/null | tail -1
      grep -h "HOLD_ABOVE" "$RES/ladder_$cell.log" 2>/dev/null | tail -1
    done
    exit 0
  fi
  sleep "$POLL"
done
