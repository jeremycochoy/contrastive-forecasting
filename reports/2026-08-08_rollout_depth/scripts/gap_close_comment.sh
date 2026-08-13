#!/bin/bash
# #373 — the comment that closes the full-study review's gap list.
#
# Two runs answered the two items that cost GPU time. Everything else was
# closed without compute. This lifts the finished tables out of
# `results/scores.md`, wraps them in the item list and the coverage grid,
# and posts one comment.
#
# It reads the GIT checkout, like `r3_final_comment.sh` does, and for the
# same reason: the numbers a reader can check out are the numbers the
# comment must carry. Nothing here is typed by hand except the two prose
# files it includes, and those cite the numbers beside them.
#
# Usage: bash gap_close_comment.sh [--dry]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
GIT_ROOT="${GIT_ROOT:-/tmp/contrastive-forecasting-373}"
DST="$GIT_ROOT/reports/2026-08-08_rollout_depth"
SCORES="$DST/results/scores.md"
PR="${PR:-400}"
OUT="$RES/gap_close_comment.md"
DRY=0
[ "${1:-}" = "--dry" ] && DRY=1

section(){ awk -v h="$1" '
  index($0, "### ") == 1 { keep = (index($0, h) == 5) }
  keep' "$SCORES"; }

sc(){ cat "$DST/results/score_$1.txt" 2>/dev/null || echo "MISSING"; }

[ -f "$SCORES" ] || { echo "no $SCORES — run make_report_assets.sh first" >&2; exit 2; }

nscore="$(ls "$DST/results"/score_*.txt 2>/dev/null | wc -l | tr -d ' ')"
credit="$(timeout 120 vastrun-balance 2>/dev/null | awk '/Credit/{print $2}')"
boxes="$(timeout 90 vastrun-status 2>/dev/null | head -1)"

{
  printf '«Agent ExperimentRunner claude-opus-5 writing»\n\n'
  printf '## The full-study review closes: ten items, two of them measured\n\n'
  printf '**Experiment directory:** `reports/2026-08-08_rollout_depth/`\n'
  printf -- '- results `results/`, plots `plots/`, scripts `scripts/`\n'
  printf -- '- %s score files on the branch, each with its own 97-config eval\n' "$nscore"
  printf -- '- run tree `/home/jupyter/cf373_r3`, controls under `checkpoints_backup/cf-373/`\n\n'

  printf '### Runs completed this session\n\n```\n'
  printf 'backbones  1  G_B1_k0_aw4, 40,000 steps, elisa, seed 20260520, L_align x4 at k = 0\n'
  printf 'heads      3  2 on that backbone (15,000 steps, seed 20260722)\n'
  printf '              1 A3 bb200k student redraw (30,000 steps, seed 20260723)\n'
  printf 'evals      3  97 GIFT-Eval configs each, strategy B4, horizon 16\n'
  printf 'failed     0\n'
  printf 'rented     nothing\n```\n\n'

  printf '### Headline numbers\n\n```\n'
  printf 'item 3  B1 k = 0            %s student   %s teacher\n' "$(sc G6_B1_k0_bb40k_student)"    "$(sc G6_B1_k0_bb40k_teacher)"
  printf 'item 3  B1 k = 0, L_align x4  %s student   %s teacher\n' "$(sc G_B1_k0_aw4_bb40k_student)" "$(sc G_B1_k0_aw4_bb40k_teacher)"
  printf 'item 3  B1 k = 3            %s student   %s teacher\n' "$(sc G6_B1_k3_bb40k_student)"    "$(sc G6_B1_k3_bb40k_teacher)"
  printf 'item 6  A3 bb200k student, draw 1 seed 20260722   %s\n' "$(sc A3_k3_bb200k_student)"
  printf 'item 6  A3 bb200k student, draw 2 seed 20260723   %s\n' "$(sc A3_k3_bb200k_student_s20260723)"
  printf 'item 6  A3 bb200k teacher,        seed 20260722   %s\n```\n\n' "$(sc A3_k3_bb200k_teacher)"

  cat "$RES/gap_close_items.md"
  printf '\n'

  printf '## Item 3 — the decisive control\n\n'
  section "B1: is the win the depth, or the weight?"
  printf '\n'
  cat "$RES/gap_close_item3.md" 2>/dev/null
  printf '\n'

  printf '## Item 6 — the redrawn head\n\n'
  section "A3's bb200k student, drawn twice"
  printf '\n'
  cat "$RES/gap_close_item6.md" 2>/dev/null
  printf '\n'

  printf '## Coverage — 14 cells x 3 stops x 2 heads\n\n'
  cat "$DST/results/coverage.md" 2>/dev/null
  printf '\n'

  cat "$RES/gap_close_verdict.md"
  printf '\n'

  printf '### Spend\n\n'
  printf 'Credit **%s**, floor $5.50. Nothing was rented this session: the ' "${credit:-?}"
  printf 'backbone, the three heads and the three evals all ran on elisa.\n'
  printf 'vast.ai reports: `%s`\n' "${boxes:-unknown}"
} > "$OUT"

echo "wrote $OUT ($(wc -l <"$OUT") lines)"
grep -n 'MISSING' "$OUT" && { echo "ABORT: a score file is missing" >&2; exit 3; }
# The verdict file carries @@...@@ markers for the two bullets that only the
# item-3 result can write. An unfilled marker must never reach the PR.
grep -n '@@' "$OUT" && { echo "ABORT: an unfilled marker is still in the text" >&2; exit 4; }
[ "$DRY" -eq 1 ] && exit 0

if (cd "$GIT_ROOT" && timeout 180 gh pr comment "$PR" --body-file "$OUT"); then
  echo "posted on PR #$PR"
else
  echo "could not post on PR #$PR — the text is in $OUT" >&2
  exit 1
fi
