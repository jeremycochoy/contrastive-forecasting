#!/bin/bash
# #373 round 2 — keep one box running per cell until every cell has one.
#
# Usage: bash r2_fleet.sh <cell> [cell...]
#
# vast.ai does not hand out 14 boxes on demand. The kit's safety floors cut
# a 64-offer listing to 7-12 rows, the cheap ones churn in seconds, and a
# provisioner that loses the race has to search again. So this is a loop,
# not a fan-out: it walks the cell list in the card's priority order, tries
# the ones that still have no box, and comes back for the rest.
#
# Order: B5 and B9 first, then A3 and A4, then A1 and A2, then the rest of
# group B. If the credit or the supply runs short, the cells that answer the
# card's rule-2 question are already running.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
BOXES="$RES/r2_boxes.tsv"
WAVE="${WAVE:-3}"            # launchers started at once
ROUND_SLEEP="${ROUND_SLEEP:-300}"
STOPS="${STOPS:-40000 100000}"
mkdir -p "$RES"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [fleet] $*" | tee -a "$RES/r2_boxes.log"; }

[ $# -gt 0 ] || { echo "usage: r2_fleet.sh <cell> [cell...]" >&2; exit 2; }
CELLS=("$@")
log "fleet target: ${CELLS[*]} (stops: $STOPS)"

placed(){ grep -qP "^$1\t" "$BOXES" 2>/dev/null; }

# The four cells that carry a round-1 k = 3 checkpoint at 40k AND both of
# that stop's head scores. Their boxes resume the backbone and skip the 40k
# heads: re-training them would spend an hour of a rented card to reproduce
# a number the study already holds.
RESUMED=" A3 B1 B5 B9 "
skip_heads(){ case "$RESUMED" in *" $1 "*) echo 40000 ;; *) echo "" ;; esac; }

round=0
while :; do
  round=$(( round + 1 ))
  todo=()
  for c in "${CELLS[@]}"; do placed "$c" || todo+=("$c"); done
  if [ "${#todo[@]}" -eq 0 ]; then
    log "every cell has a box; fleet complete after $round round(s)"; break
  fi
  log "round $round: ${#todo[@]} cell(s) still without a box: ${todo[*]}"

  pids=()
  for c in "${todo[@]:0:$WAVE}"; do
    SKIP_HEAD_STOPS="$(skip_heads "$c")" \
      nohup bash "$HERE/r2_launch_cell.sh" "$c" $STOPS > "$RES/r2_launch_$c.out" 2>&1 &
    pids+=($!)
    log "  launcher for $c (pid $!)"
    sleep 20
  done
  for p in "${pids[@]}"; do wait "$p" || true; done

  for c in "${todo[@]:0:$WAVE}"; do
    placed "$c" && log "  $c PLACED" || log "  $c not placed this round"
  done
  placed "${todo[0]}" || sleep "$ROUND_SLEEP"
done
log "fleet loop done"
