#!/bin/bash
# #373 round 3 — the budget stop. The card sets a floor of $5 credit and says
# to stop there rather than redefine the deliverable. A run left alone for
# hours can pass a floor nobody is watching, so this watches it.
#
# Usage: BOX_ID=<id> bash q_guard.sh [floor dollars] [poll seconds]
#
# At the floor it does three things, in this order: stop the dispatcher so no
# new job starts, destroy the box so the meter stops, and write BLOCKED with
# the reason. It does NOT kill a running job first — the sync loop's last
# tick is what saves the work, and the destroy waits for it.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$(dirname "$HERE")/results"
BOX_ID="${BOX_ID:?BOX_ID}"
BOX_LABEL="${BOX_LABEL:-cf373-dual}"
FLOOR="${1:-5.50}"
POLL="${2:-600}"
VDIR="${VDIR:-/home/jupyter/wt-cf-373-run2}"
PR="${PR:-400}"
R3_ROOT="${CF373_R3:-/home/jupyter/cf373_r3/sync}"
export PATH="$HOME/.local/bin:$PATH"

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [guard] $*" | tee -a "$RES/q_guard.log"; }

log "watching credit, floor \$$FLOOR, box $BOX_ID"
while :; do
  credit="$(cd "$VDIR" && timeout 120 vastrun-balance 2>/dev/null \
            | awk '/Credit/{gsub(/\$/,"",$2); print $2}')"
  if [ -z "$credit" ]; then
    log "credit unreadable; will retry"
    sleep "$POLL"; continue
  fi
  if awk -v c="$credit" -v f="$FLOOR" 'BEGIN{exit !(c < f)}'; then
    log "credit \$$credit below floor \$$FLOOR — stopping"
    pkill -f "bash scripts/q_run.sh" 2>/dev/null
    log "dispatcher stopped; letting the sync loop take one last tick"
    sleep 300
    (cd "$VDIR" && timeout 300 vastrun-destroy "$BOX_ID" "$BOX_LABEL" --force) 2>&1 | tail -3
    printf 'credit $%s fell below the $%s floor at %s. Box %s destroyed.\n' \
      "$credit" "$FLOOR" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$BOX_ID" \
      > "$RES/BLOCKED_BUDGET"
    log "BLOCKED_BUDGET written"

    # The card asks for a blocking comment on the PR, not a file on one
    # machine. This loop outlives the session that started it, so it posts
    # the comment itself rather than leaving it to a session that may be
    # gone by the time the floor is reached.
    {
      printf '«Agent ExperimentRunner claude-opus-5 writing»\n\n'
      printf '## BLOCKED — the budget floor stopped the round\n\n'
      cat "$RES/BLOCKED_BUDGET"
      printf '\nThe dispatcher is stopped and the box is destroyed. No job\n'
      printf 'was killed first: the sync loop took a last tick before the\n'
      printf 'destroy, so every artefact on the box is in `%s`.\n\n' "$R3_ROOT"
      printf 'Coverage at the stop:\n\n```\n'
      cat "$RES/coverage.txt" 2>/dev/null || echo "(no coverage table on disk)"
      printf '```\n\nThe job is not redefined. It needs credit to finish.\n'
    } > "$RES/BLOCKED_BUDGET.md"
    if (cd "$VDIR" && timeout 180 gh pr comment "$PR" \
          --body-file "$RES/BLOCKED_BUDGET.md") >>"$RES/q_guard.log" 2>&1; then
      log "blocking comment posted on PR #$PR"
    else
      log "could not post the blocking comment on PR #$PR — $RES/BLOCKED_BUDGET.md holds it"
    fi
    exit 0
  fi
  sleep "$POLL"
done
