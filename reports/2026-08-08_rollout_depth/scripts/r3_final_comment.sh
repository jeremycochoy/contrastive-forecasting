#!/bin/bash
# #373 round 3 — the round's closing comment on the PR.
#
# The card asks for two tables when every score is in: the per-cell,
# per-stop, per-head table against the published k = 0, and the stop-reason
# table. Both already exist in `results/scores.md`, which `tables.py` writes
# and `r3_publish.sh` refreshes on every tick. This lifts those two sections
# out by heading, wraps them in the coverage grid and the round's counts,
# and posts one comment.
#
# It reads the GIT checkout, not the run tree: the numbers a reader can
# check out are the numbers the comment must carry.
#
# Usage: bash r3_final_comment.sh [--dry]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
GIT_ROOT="${GIT_ROOT:-/tmp/contrastive-forecasting-373}"
DST="$GIT_ROOT/reports/2026-08-08_rollout_depth"
SCORES="$DST/results/scores.md"
PR="${PR:-400}"
OUT="$RES/final_comment.md"
DRY=0
[ "${1:-}" = "--dry" ] && DRY=1

# One `### ` section, heading included, up to the next `### `.
section(){ awk -v h="$1" '
  index($0, "### ") == 1 { keep = (index($0, h) == 5) }
  keep' "$SCORES"; }

[ -f "$SCORES" ] || { echo "no $SCORES — run r3_publish.sh first" >&2; exit 2; }

nscore="$(ls "$DST/results"/score_*.txt 2>/dev/null | wc -l | tr -d ' ')"
nfail="$(grep -l '^failed' "$RES/queue"/*.state 2>/dev/null | wc -l | tr -d ' ')"
credit="$(timeout 120 vastrun-balance 2>/dev/null | awk '/Credit/{print $2}')"
boxes="$(timeout 90 vastrun-status 2>/dev/null | head -1)"

{
  printf '«Agent ExperimentRunner claude-opus-5 writing»\n\n'
  printf '## Round 3 closed — every deliverable is scored\n\n'
  printf '**Experiment directory:** `reports/2026-08-08_rollout_depth/`\n'
  printf -- '- results `results/`, plots `plots/`, scripts `scripts/`\n'
  printf -- '- run tree `/home/jupyter/cf373_r3`, %s score files on the branch\n\n' "$nscore"

  printf '### Coverage — 14 cells x 3 stops x 2 heads\n\n'
  cat "$DST/results/coverage.md" 2>/dev/null
  printf '\n'

  section "This study's k = 3 against the published k = 0"
  printf '\n'
  section "Stop reasons: what the extend rule read at each cell"
  printf '\n'

  printf '### Runs completed\n\n```\n'
  printf 'backbones  9 legs this round: B8 0 -> 100k, eight cells 100k -> 200k\n'
  printf 'heads      30,000 steps, seed 20260722, --grad-clip 1.0, batch 256, lr 1e-3\n'
  printf 'evals      97 GIFT-Eval configs, strategy B4, horizon 16\n'
  printf 'failed     %s\n```\n\n' "$nfail"

  printf '### Spend\n\n'
  printf 'Credit **%s**, floor $5.50. Box `47557391` (cf373-dual) was destroyed at\n' "${credit:-?}"
  printf '2026-08-13 11:36Z by `scripts/r3_reap.sh`, once it had verified all 730\n'
  printf 'checkpoint files on elisa. vast.ai now reports: `%s`\n' "${boxes:-unknown}"
  printf 'Every head and every eval after that ran on elisa and cost nothing.\n'
} > "$OUT"

echo "wrote $OUT ($(wc -l <"$OUT") lines)"
[ "$DRY" -eq 1 ] && exit 0

if (cd "$GIT_ROOT" && timeout 180 gh pr comment "$PR" --body-file "$OUT"); then
  echo "posted on PR #$PR"
else
  echo "could not post on PR #$PR — the text is in $OUT" >&2
  exit 1
fi
