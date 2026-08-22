#!/bin/bash
# #409 — stop the leg that lost the contrastive task.
#
# WHY THIS SCRIPT EXISTS. The decay ends at step 10,000 and every arm trains to
# 40,000. Four arms fall to weight 0.0 and cross the known-dead 1:9
# repel-to-pull ratio near step 5,600, so a collapsed arm climbs about 30,000
# dead steps to a checkpoint whose score is already known to be bad. The Fable
# opinion (scripts/fable_opinion.md, section 5) asks for a gate on the
# trainer's own AUC column. This is that gate.
#
# HOW IT READS. `auc_watch.py` gives the verdict: the rolling median of the
# `auc` column over CF409_AUC_WINDOW steps, against CF409_AUC_THRESHOLD. Rows
# at or below CF409_AUC_WARMUP do not count — the AUC of a fresh run starts
# near 0.5 and climbs, and every arm still holds a weight of 0.9 or more there,
# so no arm can collapse from the decay inside the warmup.
#
# WHAT IT DOES. It reads the live CSV every CF409_AUC_POLL seconds while the
# leg runs. On a `lost` verdict it stops the whole process tree and writes
# `results/collapsed_<arm>.txt`, which names the step. `run_arm.sh` then exits
# CF409_RC_COLLAPSED (4) and `phase1.sh` does NOT re-fire the arm: a re-fire
# trains the same collapse.
#
# The report still reads the arm. A stopped arm has its whole AUC curve, its
# loss by term to the step it reached, and no 40,000-step score. That is the
# answer to "does any run lose the contrastive task, and at which step".
#
# CF409_AUC_WATCH=0 turns the gate off. The report must then say that a human
# watched the AUC.
#
# Usage:  auc_guard.sh <arm> <stop steps> <leg pid>
set -uo pipefail

ARM="${1:?usage: auc_guard.sh <arm> <stop steps> <leg pid>}"
STOP="${2:?usage: auc_guard.sh <arm> <stop steps> <leg pid>}"
PID="${3:?usage: auc_guard.sh <arm> <stop steps> <leg pid>}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"
cf409_require_arm "$ARM" || exit $?
cf409_require_stop "$STOP" || exit $?

WATCH="$HERE/auc_watch.py"
[ -f "$WATCH" ] || { echo "ABORT: no watch at $WATCH" >&2; exit 2; }
mkdir -p "$CF409_RESULTS"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#409 auc] $*" \
  | tee -a "$CF409_RESULTS/auc_guard.log"; }

while kill -0 "$PID" 2>/dev/null; do
  csv="$(cf409_live_losses_csv "$ARM" "$STOP")"
  if [ -n "$csv" ] && [ -s "$csv" ]; then
    line="$(python3 "$WATCH" "$csv" --window "$CF409_AUC_WINDOW" \
              --threshold "$CF409_AUC_THRESHOLD" \
              --warmup "$CF409_AUC_WARMUP" 2>&1)"
    rc=$?
    if [ "$rc" -eq 1 ]; then
      log "arm $ARM LOST the contrastive task — stopping the leg"
      log "  $line"
      { echo "arm $ARM lost the contrastive task."
        echo "verdict: $line"
        echo "window: $CF409_AUC_WINDOW steps, threshold: $CF409_AUC_THRESHOLD"
        echo "warmup: $CF409_AUC_WARMUP steps"
        echo "csv: $csv"
        echo "stopped: $(date '+%Y-%m-%d %H:%M:%S')"
      } >"$(cf409_collapse_file "$ARM")"
      cf409_kill_tree "$PID"
      exit 1
    fi
  fi
  # `kill -0` again before the sleep, so a leg that finished during the read
  # does not hold this loop for another poll.
  kill -0 "$PID" 2>/dev/null || break
  sleep "$CF409_AUC_POLL"
done
exit 0
