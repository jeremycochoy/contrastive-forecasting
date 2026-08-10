#!/bin/bash
# #373 round 2 — bring a cell back when its box dies.
#
# Usage: bash r2_watchdog.sh [poll seconds]
#
# CLAUDE.md: assume the machine can crash at any time. Fourteen rented boxes
# run for eight to ten hours each here, so at least one of them dying is the
# expected case, not the exception.
#
# A cell is DEAD when its box fails to answer ssh, or answers and holds
# neither a backbone nor a head nor a CELL_<cell>_DONE marker, on
# STRIKES consecutive polls. Strikes rather than one reading, because the
# vast.ai proxy drops connections that come back.
#
# The recovery is a relaunch. `r2_launch_cell.sh` stages the furthest
# checkpoint in the cell's own sync tree, with its optimizer companion, so
# the run continues from the last step the sync loop saved rather than from
# zero. What is lost is at most one sync interval of training.
set -uo pipefail

POLL="${1:-600}"
STRIKES="${STRIKES:-3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
BOXES="$RES/r2_boxes.tsv"
SYNC_BASE="${CF373_R2_SYNC:-/home/jupyter/cf373_r2}"
VASTRUN_DIR="${VASTRUN_DIR:-$(cd "$HERE" && git rev-parse --show-toplevel)}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15)
STOPS="${STOPS:-40000 100000}"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [watchdog] $*" | tee -a "$RES/r2_watchdog.log"; }
declare -A strike=()

log "start poll=${POLL}s strikes=$STRIKES"
while :; do
  [ -f "$BOXES" ] || { sleep "$POLL"; continue; }
  while IFS=$'\t' read -r cell id host port label stops; do
    [ -n "${cell:-}" ] || continue
    case "$cell" in \#*) continue;; esac
    [ -f "$RES/r2_reaped_$cell" ] && continue
    [ -f "$RES/HOLD_$cell" ] && continue

    alive=$(timeout 60 ssh "${SSH_OPTS[@]}" -n -p "$port" "root@$host" '
      all=$(ps -eo args 2>/dev/null)
      n=$(printf "%s\n" "$all" | grep -c -- "[f]req-embedding/scripts/train.py")
      h=$(printf "%s\n" "$all" | grep -c -- "[t]rain_forecasting_head.py")
      d=0; ls /root/cf/reports/2026-08-08_rollout_depth/results/CELL_*_DONE >/dev/null 2>&1 && d=1
      [ "$n" -gt 0 ] || [ "$h" -gt 0 ] || [ "$d" = 1 ] && echo yes || echo no' 2>/dev/null | tail -1)

    if [ "$alive" = "yes" ]; then
      [ "${strike[$cell]:-0}" -gt 0 ] && log "$cell recovered after ${strike[$cell]} strike(s)"
      strike[$cell]=0
      continue
    fi
    strike[$cell]=$(( ${strike[$cell]:-0} + 1 ))
    log "$cell strike ${strike[$cell]}/$STRIKES (box $id at $host:$port answered '${alive:-nothing}')"
    [ "${strike[$cell]}" -ge "$STRIKES" ] || continue

    log "$cell DEAD — destroying $id and relaunching from its own sync tree"
    (cd "$VASTRUN_DIR" && timeout 300 vastrun-destroy "$id" --force) >/dev/null 2>&1
    # Stop THIS cell's sync loop before relaunching, by the pid the launcher
    # recorded. Two loops pointed at one local tree would race on the same
    # `.tmp` names, and a pattern kill would take all fourteen: their command
    # lines are identical and the host they poll is in the environment.
    if [ -f "$RES/r2_syncpid_$cell" ]; then
      kill "$(cat "$RES/r2_syncpid_$cell")" 2>/dev/null || true
      rm -f "$RES/r2_syncpid_$cell"
    fi
    tmp="$BOXES.tmp.$$"
    grep -v -P "^$cell\t" "$BOXES" > "$tmp" && mv -f "$tmp" "$BOXES"
    strike[$cell]=0
    nohup bash "$HERE/r2_launch_cell.sh" "$cell" $STOPS \
      > "$RES/r2_relaunch_$cell.out" 2>&1 &
    log "$cell relaunch started (pid $!)"
  done < "$BOXES"
  sleep "$POLL"
done
