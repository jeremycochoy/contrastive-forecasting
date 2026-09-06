#!/bin/bash
# #412 — stop the leg that lost the contrastive task.
#
# WHY THIS SCRIPT EXISTS. An arm of this card costs 2.6 to 6.5 GPU-hours to
# 40,000 steps and up to 32.6 to 200,000. An arm that lost the contrastive
# task climbs all of that to a checkpoint whose score is already known to be
# bad. The card asks for the AUC of every run for the same reason: #404 saw
# one backbone of this cell lose the task at seed 20260521.
#
# The two decay arms carry the higher risk. `L_rep` holds the negatives of
# this objective, and their weight reaches 0.0 at step 2,000. Past that step
# nothing pushes the representations apart.
#
# HOW IT READS. `scripts/auc_watch.py` of the main checkout gives the verdict:
# the rolling median of the `auc` column over CF412_AUC_WINDOW rows, against
# CF412_AUC_THRESHOLD. Rows at or below CF412_AUC_WARMUP do not count, because
# the AUC of a fresh run starts near 0.5 and climbs. Without that warm-up the
# gate would stop every arm in its first minute.
#
# WHAT IT DOES. It reads the live CSV every CF412_AUC_POLL seconds while the
# leg runs. On a `lost` verdict it stops the whole process tree and writes
# `results/collapsed_<arm>.txt`, which names the step. `run_arm.sh` then exits
# CF412_RC_COLLAPSED (4), and `phase1.sh` gives the arm no further stop: a
# re-fire trains the same collapse.
#
# WHICH ROWS IT READS. This leg's own, and no other. One arm can hold rows
# from more than one leg:
#
#   * a re-fired leg with a checkpoint on disk opens a FRESH CSV, because
#     train.py branches its run name to `<name>_r2`.
#   * a re-fired leg with no checkpoint keeps its run name, and train.py then
#     APPENDS to the CSV the dead leg wrote.
#
# The trainer flushes every 100 rows, so a fresh leg holds no row of its own
# for its first minutes. A verdict on the rows below would stop a leg that has
# trained nothing yet. So the gate counts the rows of every CSV of this arm
# BEFORE it reads, and passes that count to `auc_watch.py --skip-rows`.
#
# The report still reads a stopped arm. It has its whole AUC curve, its loss
# by term to the step it reached, and no score. That is the answer to "does
# any run lose the contrastive task, and at which step".
#
# CF412_AUC_WATCH=0 turns the gate off. The report must then say that a human
# watched the AUC.
#
# Usage:  auc_guard.sh <arm> <stop steps> <leg pid>
set -uo pipefail

ARM="${1:?usage: auc_guard.sh <arm> <stop steps> <leg pid>}"
STOP="${2:?usage: auc_guard.sh <arm> <stop steps> <leg pid>}"
PID="${3:?usage: auc_guard.sh <arm> <stop steps> <leg pid>}"

. "$(dirname "${BASH_SOURCE[0]}")/study.sh"
cf412_require_arm "$ARM" || exit $?
cf412_require_stop "$STOP" || exit $?

WATCH="$CF412_AUC_WATCH_PY"
[ -f "$WATCH" ] || { echo "ABORT: no watch at $WATCH" >&2; exit 2; }
mkdir -p "$CF412_RESULTS"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412 auc] $*" \
  | tee -a "$CF412_RESULTS/auc_guard.log"; }

# The rows on disk before this leg wrote one. `run_arm.sh` starts the gate
# right after the trainer logs its command line, which is minutes before the
# first flush, so the count is this leg's floor and not part of its run.
declare -A BASE
while IFS= read -r f; do
  [ -n "$f" ] || continue
  BASE["$f"]="$(cf412_csv_rows "$f")"
  log "arm $ARM skips ${BASE[$f]} row(s) of an earlier leg in $f"
done < <(cf412_losses_csvs "$ARM" "$STOP")
log "arm $ARM gate on — window $CF412_AUC_WINDOW," \
    "threshold $CF412_AUC_THRESHOLD, warmup $CF412_AUC_WARMUP," \
    "poll ${CF412_AUC_POLL}s"

reading=""
while kill -0 "$PID" 2>/dev/null; do
  csv="$(cf412_live_losses_csv "$ARM" "$STOP")"
  skip=0
  [ -n "$csv" ] && skip="${BASE[$csv]:-0}"
  # A CSV this leg has not reached yet gives no verdict. That is the wait.
  if [ -n "$csv" ] && [ "$(cf412_csv_rows "$csv")" -gt "$skip" ]; then
    if [ "$reading" != "$csv" ]; then
      reading="$csv"
      log "arm $ARM reads $csv from row $(( skip + 1 ))"
    fi
    line="$(python3 "$WATCH" "$csv" --window "$CF412_AUC_WINDOW" \
              --threshold "$CF412_AUC_THRESHOLD" \
              --warmup "$CF412_AUC_WARMUP" --skip-rows "$skip" 2>&1)"
    rc=$?
    if [ "$rc" -eq 1 ]; then
      log "arm $ARM LOST the contrastive task — stopping the leg"
      log "  $line"
      { echo "arm $ARM lost the contrastive task."
        echo "verdict: $line"
        echo "window: $CF412_AUC_WINDOW rows, threshold: $CF412_AUC_THRESHOLD"
        echo "warmup: $CF412_AUC_WARMUP steps"
        echo "csv: $csv"
        echo "skipped: $skip row(s), which an earlier leg wrote"
        echo "stopped: $(date '+%Y-%m-%d %H:%M:%S')"
      } >"$(cf412_collapse_file "$ARM")"
      cf412_kill_tree "$PID"
      exit 1
    fi
  fi
  # `kill -0` again before the sleep, so a leg that finished during the read
  # does not hold this loop for another poll.
  kill -0 "$PID" 2>/dev/null || break
  sleep "$CF412_AUC_POLL"
done
exit 0
