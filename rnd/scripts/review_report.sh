#!/usr/bin/env bash
# Independent report-quality review against experiments/REPORT_STANDARD.md.
#
# Launches a FRESH Claude agent (headless, no authoring context) and feeds it the
# report + the REPORT_STANDARD checklist + the experiment's own data (results/, plots/,
# notes/). The agent independently verifies — recomputes numbers, views plots — and
# reports which checklist boxes don't tick and which statements aren't supported by data.
#
# Usage:   rnd/scripts/review_report.sh experiments/<YYYY-MM-DD>_<name>/<name>.md
# Output:  the reviewer's verdict (GREEN, or NEEDS-FIX with a precise list) on stdout.
#
# Requires the `claude` CLI on PATH. The point is independence: run it in a clean
# session so the reviewer has not seen how the report was written.
set -euo pipefail

REPORT="${1:?usage: review_report.sh <path/to/report.md>}"
[ -f "$REPORT" ] || { echo "no such report: $REPORT" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STD="$ROOT/experiments/REPORT_STANDARD.md"
DIR="$(cd "$(dirname "$REPORT")" && pwd)"

read -r -d '' PROMPT <<EOF || true
You are an independent report-standard reviewer who has NOT seen how this report was written.

1. Read the checklist: $STD
2. Read the report:    $REPORT
3. Read its data:      $DIR/results/  $DIR/plots/  $DIR/notes/  $DIR/logs/  $DIR/scripts/

Independently VERIFY — recompute every number from the data yourself, and VIEW each
plot the report embeds. Do not trust the prose.

For EACH checklist item in REPORT_STANDARD.md, state tick or ✗ with one line of evidence.
Then list EVERY statement in the report that is not directly supported by the data:
quote the sentence, name the contradicting or absent source, and give the correct value.

End with a one-word VERDICT on its own line: GREEN (every box ticks and every claim is
supported) or NEEDS-FIX (followed by the prioritised list).

IMPORTANT — output channel: you are in read-only plan mode. Do NOT call ExitPlanMode and
do NOT defer any content to a plan file or a "plan" artifact. Emit your ENTIRE review —
every checklist tick/✗, every unsupported statement with its correct value, and the final
VERDICT plus the full prioritised fix list — directly as your final assistant text
message. That final message is the whole deliverable; it is what gets captured on stdout,
so nothing may be withheld for a later approval step.
EOF

# Run from the report's own worktree root so the reviewer's project tree is the right
# checkout (a sibling git worktree is otherwise outside the sandbox), and allow-list it.
cd "$ROOT"
exec claude -p "$PROMPT" \
  --append-system-prompt "You are a skeptical, independent scientific reviewer. Verify against data, never against the report's own claims. Be exact and terse." \
  --permission-mode plan \
  --add-dir "$ROOT"
