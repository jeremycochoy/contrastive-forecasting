#!/bin/bash
# #373 round 2 — the 200k round: one box per cell the extend rule kept.
#
# Usage: bash r2_extend_fleet.sh [cell...]
#   With no argument it reads results/r2_extend.tsv and runs every cell the
#   rule marked `extend`, in the card's priority order.
#
# Each box resumes that cell's own bb100k checkpoint from the local sync
# tree, with its optimizer companion, and runs one 100k leg to the card's
# 200k ceiling. It then trains the heads the rule KEPT, and only those: on a
# cell where one head went up, that head is not a deliverable at 200k, and
# half an hour of a rented card is the price of training it anyway.
#
# The order is the bb100k standing, best first. A box can be lost at any
# hour, and the credit is finite, so the cells whose 200k number carries the
# most weight go out first: B10, A2, A4, then the rest of the extend list.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
STAGGER="${STAGGER:-90}"
# The credit buys about nine box-hours per dollar at this class. A bid
# ceiling below the launcher's own keeps a price spike from turning eight
# boxes into a budget the study cannot finish on.
export VAST_SEARCH_ARGS="${VAST_SEARCH_ARGS:---gpu-model RTX_5090,RTX_4090 --min-vram 24 --max-bid 0.45}"
export WT="${WT:-/home/jupyter/wt-cf-373-run2}"
export CF373_R2_SYNC="${CF373_R2_SYNC:-/home/jupyter/cf373_r2}"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [extfleet] $*" | tee -a "$RES/r2_boxes.log"; }

ORDER="B10 A2 A4 A3 B6 B4 B2 B1 A1 B3 B5 B7 B9"

if [ $# -gt 0 ]; then
  CELLS=("$@")
else
  [ -f "$RES/r2_extend.tsv" ] || { echo "no $RES/r2_extend.tsv — run r2_extend.py --write" >&2; exit 2; }
  CELLS=()
  for c in $ORDER; do
    awk -F'\t' -v c="$c" '$1==c && $4=="extend" {found=1} END {exit !found}' \
      "$RES/r2_extend.tsv" && CELLS+=("$c")
  done
fi
[ "${#CELLS[@]}" -gt 0 ] || { log "no cell extends; nothing to launch"; exit 0; }
log "200k round: ${CELLS[*]}"

for c in "${CELLS[@]}"; do
  # The heads this cell keeps, from the rule's own verdict file.
  encs="$(awk -F'\t' -v c="$c" '$1==c && $4=="extend" {print $3}' \
          "$RES/r2_extend.tsv" | sort | tr '\n' ' ' | sed 's/ $//')"
  [ -n "$encs" ] || { log "$c: no head kept, skipping"; continue; }

  # A cell whose 200k checkpoint is already in the sync tree is done.
  if find "$CF373_R2_SYNC/$c/sync" -name "*_200k.pth" ! -name "*optimizer*" \
       2>/dev/null | grep -q .; then
    log "$c: bb200k already on disk, skipping"; continue
  fi
  # The launcher records every box in one table, so "already placed" means
  # a row for this cell whose stops column is the 200k leg.
  if awk -F'\t' -v c="$c" '$1==c && $6 ~ /200000/ {found=1} END {exit !found}' \
       "$RES/r2_boxes.tsv" 2>/dev/null; then
    log "$c: already has a 200k box, skipping"; continue
  fi

  log "$c: launching 200k leg, heads: $encs"
  # Through the retry wrapper, not the launcher: a box that fails to
  # bootstrap must cost one offer, not the cell.
  HEAD_ENCS="$encs" nohup bash "$HERE/r2_box_retry.sh" "$c" 200000 \
    >"$RES/r2_launch_${c}_200k.out" 2>&1 &
  sleep "$STAGGER"
done

wait
log "every 200k launcher has returned"
