#!/bin/bash
# #320 — downstream poller. For each arm, wait until its backbone FINAL lands
# (trained by the backbone lane on the other GPU), then evaluate it with a 2L and
# a 6L q-head (triage-11 + full-97). Overlaps the heavy q-head/eval work with
# backbone training. Idempotent. Arm spec = "arm:mix:tag".
#
#   downstream_poll.sh <gpu> <arm:mix:tag> [<arm:mix:tag> ...]
set -uo pipefail
GPU="${1:?gpu}"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-26_forked_6Lf
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [DS-POLL g$GPU] $*"; }
log "start: $*"
for spec in "$@"; do
  IFS=':' read -r arm mix tag <<<"$spec"
  case "$arm" in
    beta)    NAME="bb_beta_${tag}_6Lf_50k";      DTAG="beta_${tag}_6Lf_50k" ;;
    alltime) NAME="bb_xshh_allt_${tag}_6Lf_50k"; DTAG="xshh_allt_${tag}_6Lf_50k" ;;
    *) log "unknown arm $arm — skip"; continue ;;
  esac
  BB="$OUT/runs/${NAME}_FINAL.pth"
  log ">>> waiting for $BB"
  while [ ! -f "$BB" ]; do sleep 60; done
  log ">>> backbone ready: $NAME — running downstream"
  bash "$HERE/downstream_6Lf.sh" "$BB" 2 "$DTAG" "$GPU" "triage full" || log "$DTAG 2L nonzero (continuing)"
  bash "$HERE/downstream_6Lf.sh" "$BB" 6 "$DTAG" "$GPU" "triage full" || log "$DTAG 6L nonzero (continuing)"
  log ">>> downstream done for $DTAG"
done
log "=== ALL DOWNSTREAM (g$GPU) DONE ==="
