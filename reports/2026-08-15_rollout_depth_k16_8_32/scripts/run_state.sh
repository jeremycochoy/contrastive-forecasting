#!/bin/bash
# #401 — RUN_STATE.md, rebuilt from the artefacts on disk.
#
# What a re-dispatched session reads first. One file, overwritten, so it is
# never a log to scroll.
#
# It lived inside `launch_elisa.sh` as a shell function, and the launcher's
# reporting loop called it every 30 minutes. That made the file only as fresh
# as the launcher: when the launcher exited and another driver took the queue,
# the file froze at "Scores so far: (none yet)" while `scores.csv` filled up
# with seven cells. A reader then took a stale file for the run.
#
# So it is a script now. `launch_elisa.sh` calls it, and so can anybody, at
# any moment, with no launcher running.
#
# Every line comes from a file or from /proc. Nothing is passed in but the
# note, so the state cannot disagree with the artefacts it is built from.
#
# Usage:  bash scripts/run_state.sh ["a note about the driver"]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF401_ROOT_GIVEN="${CF401_ROOT:-}"
. "$HERE/study.sh"

# The same root `launch_elisa.sh` and `heads_watch.sh` read: the tree the sync
# loop lands the box's checkpoints in. The study default points at a local
# checkpoints directory that holds only the legs elisa trained before the
# handover, so a state file built from it lists two backbones for a study that
# has twenty.
cf401_use_root "$CF401_SYNC_ROOT"

NOTE="${1:-rebuilt by scripts/run_state.sh}"
STATE="$CF401_RESULTS/RUN_STATE.md"
mkdir -p "$CF401_RESULTS"

# The cells the eval finished, newest last, out of the eval's own DONE lines.
# `stops.log` is the authority: it names the tag and the number the eval
# printed, so no cell picks up a score from a run it did not have.
done_cells(){
  grep -oE '\[[a-z0-9_]+\] DONE — GM-Relative MASE [0-9.]+' \
    "$CF401_RESULTS/stops.log" 2>/dev/null \
    | sed -E 's/^\[([a-z0-9_]+)\] DONE — GM-Relative MASE /\1  /'
}

# Every head or eval of THIS study that runs now, whoever started it. The
# driver changes over a multi-day study, so the process list is read, not a
# pid a launcher wrote down once.
running_cells(){
  local pid cmd
  for pid in $(pgrep -f 'train_forecasting_head\.py|eval_gift_eval_official\.py' \
               2>/dev/null); do
    cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)"
    case "$cmd" in
      *"$CF401_ROOT"*|*"$CF401_SYNC_ROOT"*) ;;
      *) continue ;;
    esac
    case "$cmd" in
      *train_forecasting_head.py*) printf 'head-train  %s\n' \
        "$(printf '%s\n' "$cmd" | grep -oE '\-\-run-name [^ ]+' \
           | cut -d' ' -f2)" ;;
      *) printf 'eval        %s\n' \
        "$(printf '%s\n' "$cmd" | grep -oE '\-\-output-dir [^ ]+' \
           | cut -d' ' -f2 | sed 's#.*/eval/##; s#/gift/.*##')" ;;
    esac
  done | sort -u
}

{
  echo "# #401 run state — the mean objective"
  echo
  echo "- updated: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "- note: $NOTE"
  echo "- objective: \`--train-rollout-reduce $CF401_REDUCE\`, depths $CF401_DEPTHS"
  echo "- root (the sync loop lands the box's tree here): \`$CF401_ROOT\`"
  echo "- results: \`$CF401_RESULTS\`"
  echo
  echo "## Scores so far"
  echo
  echo '```'
  cat "$CF401_RESULTS/scores.csv" 2>/dev/null || echo "(none yet)"
  echo '```'
  echo
  echo "## Cells the eval finished"
  echo
  echo '```'
  done_cells || true
  echo '```'
  echo
  echo "## Running now"
  echo
  echo '```'
  running_cells | grep -q . && running_cells || echo "(nothing on this machine)"
  echo '```'
  echo
  echo "## Backbone stops on this side"
  echo
  echo '```'
  ls -1 "$CF401_ROOT"/k*/*/leg_*k/*k.pth 2>/dev/null \
    | grep -v optimizer | sed "s#$CF401_ROOT/##" || echo "(none yet)"
  echo '```'
} >"$STATE.tmp" && mv -f "$STATE.tmp" "$STATE"

echo "wrote $STATE"
