#!/bin/bash
# #322 supervision watcher. Converts the detached (non-harness-tracked) b1024 qk+aon
# sweep into a single wake event for the agent: this script blocks, polling every 120s,
# and EXITS (which re-invokes the agent) on the next meaningful event or an hourly
# heartbeat. Read-only — it never touches the runs. Prints one WAKE: line on exit.
#
# Exit (=wake) conditions:
#   - a new bb_*qk_aon*FINAL.pth appears (an arm finished)
#   - all 5 qk_aon FINALs present (sweep complete)
#   - collapse signature on the active arm (loss NaN/inf, cross_batch>0.1, or gap<0.3),
#     gated on step>1000 + numeric row so a fresh arm's warmup never false-triggers
#   - orchestrator process gone while <5 FINALs (unexpected death -> relaunch)
#   - HEARTBEAT seconds elapsed with no event (default 3600 = hourly check-in)
set -uo pipefail
RUNS=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024/runs
HEARTBEAT="${HEARTBEAT:-3600}"
isnum(){ echo "$1" | grep -qE '^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$'; }
finals(){ ls -1 $RUNS/bb_*qk_aon*FINAL.pth 2>/dev/null | wc -l; }
active(){ ls -t $RUNS/bb_*qk_aon*_losses.csv 2>/dev/null | head -1; }
snap(){ local f last; f=$(active); [ -z "$f" ] && { echo "no-active-arm finals=$(finals)"; return; }
  last=$(tail -1 "$f" 2>/dev/null)
  printf "arm=%s step=%s loss=%s gap=%s cb=%s ff=%s finals=%s" \
    "$(basename "$f" _losses.csv | sed 's/^bb_//;s/_6Lf_b1024$//')" \
    "$(echo "$last"|cut -d, -f1)" "$(echo "$last"|cut -d, -f2)" "$(echo "$last"|cut -d, -f4)" \
    "$(echo "$last"|cut -d, -f9)" "$(echo "$last"|cut -d, -f6)" "$(finals)"; }
BASE=$(finals); T0=$SECONDS
while true; do
  sleep 120
  N=$(finals)
  [ "$N" -gt "$BASE" ] && { echo "WAKE: FINAL landed ($BASE->$N) | $(snap)"; exit 0; }
  [ "$N" -ge 5 ] && { echo "WAKE: SWEEP COMPLETE 5/5 FINALs | $(snap)"; exit 0; }
  f=$(active); last=$(tail -1 "$f" 2>/dev/null)
  step=$(echo "$last"|cut -d, -f1); loss=$(echo "$last"|cut -d, -f2)
  gap=$(echo "$last"|cut -d, -f4); cb=$(echo "$last"|cut -d, -f9)
  if isnum "$step" && [ "$step" -gt 1000 ] 2>/dev/null && isnum "$gap" && isnum "$cb"; then
    case "$loss" in *nan*|*inf*|*NaN*|*Inf*) echo "WAKE: COLLAPSE loss=$loss | $(snap)"; exit 0;; esac
    if awk -v g="$gap" -v c="$cb" 'BEGIN{exit !((g+0)<0.3 || (c+0)>0.1)}'; then
      echo "WAKE: COLLAPSE gap=$gap cb=$cb (fix regressed) | $(snap)"; exit 0; fi
  fi
  pgrep -f orchestrate_qkaon.sh >/dev/null || { [ "$N" -lt 5 ] && { echo "WAKE: ORCHESTRATOR DIED finals=$N | $(snap)"; exit 0; }; }
  [ $((SECONDS-T0)) -ge "$HEARTBEAT" ] && { echo "WAKE: heartbeat ${HEARTBEAT}s | $(snap)"; exit 0; }
done
