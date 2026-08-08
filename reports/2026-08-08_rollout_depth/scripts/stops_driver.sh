#!/bin/bash
# #373 — elisa's half of the study: heads on the GPU, GIFT-Eval on the cores.
#
# Usage: bash stops_driver.sh <cell>:<k>:<steps> [...]
#
# Waits for each backbone to land from a box's sync loop, adopts it onto the
# durable root, then runs stop_k.sh twice — student head and teacher head —
# per #393's protocol.
#
# Why this is on elisa and not on the box that trained the backbone: the
# GIFT-Eval is CPU-bound (58.7 s on a 4090 against 97.3 s on ONE core for
# six configs, and it then takes no VRAM at all), elisa has 32 cores, and
# the study's whole GPU budget is $7.31 of vast.ai credit. Head training
# stays on a GPU because it is 15,000 optimizer steps.
#
# The driver polls rather than being triggered: the sync loop ticks every
# 15 minutes and a box can die between ticks, so "the file is here now" is
# the only condition worth acting on.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
LOCAL_BASE="${CF373_SYNC_BASE:-$HOME/cf373_sync}"
export WT="${WT:-/home/jupyter/wt-cf-373-train}"
POLL="${POLL:-180}"
# A backbone to bb40k is ~1.3 h on a 5090 at k = 0 and ~3 h at k = 3, plus
# the sync tick. Ten hours covers a box that had to be re-provisioned once.
WAIT_MAX="${WAIT_MAX:-36000}"
mkdir -p "$RES"
. "$HERE/cell_paths.sh"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [stops] $*" | tee -a "$RES/stops_driver.log"; }

# Bring one backbone (and its optimizer companion) from a sync root onto the
# durable root, where cell_paths.sh looks for it. Copy, never move: the sync
# loop compares by size on every tick and would re-pull a file it no longer
# finds.
adopt(){ # <cell> <k> <steps> -> 0 if the checkpoint is now on the durable root
  local cell="$1" k="$2" steps="$3" name dir found
  name="$(cf373_run_name "$cell" "$k")" || return 1
  dir="$(cf373_runs_dir "$cell" "$k" "$steps")" || return 1
  [ -n "$(cf373_bb_ckpt "$cell" "$k" "$steps")" ] && return 0

  found="$(find "$LOCAL_BASE" -type f -name "$name*_$(( steps / 1000 ))k.pth" \
           ! -name '*_optimizer.pth' 2>/dev/null | head -1)"
  [ -n "$found" ] || return 1
  # A checkpoint still landing is a partial file. safe_pull.sh writes to
  # .tmp and renames, so the name only appears whole — but a torch.load of
  # a truncated file is the one failure that costs a head's GPU time, so it
  # is checked rather than assumed.
  python3 -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu')" \
    "$found" >/dev/null 2>&1 || { log "  $(basename "$found") does not load yet"; return 1; }
  mkdir -p "$dir"
  cp -f "$found" "$dir/" || return 1
  [ -f "${found%.pth}_optimizer.pth" ] && cp -f "${found%.pth}_optimizer.pth" "$dir/"
  log "adopted $(basename "$found") -> $dir"
  return 0
}

for job in "$@"; do
  IFS=: read -r cell k steps <<<"$job"
  [ -n "${steps:-}" ] || { log "ABORT: bad job '$job'"; exit 2; }

  waited=0
  until adopt "$cell" "$k" "$steps"; do
    if [ "$waited" -ge "$WAIT_MAX" ]; then
      log "GIVE UP on $cell k=$k bb$(( steps / 1000 ))k after ${waited}s"
      continue 2
    fi
    [ $(( waited % 1800 )) -eq 0 ] && \
      log "waiting for $cell k=$k bb$(( steps / 1000 ))k (${waited}s)"
    sleep "$POLL"; waited=$(( waited + POLL ))
  done

  # ENCODERS restricts one driver to one head, so two drivers can share the
  # card: the study needs 2 heads x 5 backbones x ~1 h, and elisa has one
  # usable GPU. Measured at batch 256 the head fits twice over in what is
  # free. Left unset, one driver does both in turn.
  for enc in ${ENCODERS:-student teacher}; do
    log "START $cell k=$k bb$(( steps / 1000 ))k $enc"
    BB_GPU="${BB_GPU:-1}" bash "$HERE/stop_k.sh" "$cell" "$k" "$steps" "$enc"
    log "END   $cell k=$k bb$(( steps / 1000 ))k $enc rc=$?"
  done
done

log "driver drained: $*"
