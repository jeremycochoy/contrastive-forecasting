#!/bin/bash
# #404 — the steps that follow the round 3c driver, detached from any session.
#
# `round3c.sh` trains the s08b head, scores it on elisa, tears the box down,
# rebuilds the tables and the figures, and posts the comment. It never touches
# git. A session that ends takes its own waiters with it, so this script holds
# the rest, and it holds it under `nohup setsid`:
#
#   1. Wait for the driver to exit, by pid. Never by a pattern.
#   2. Put the six arms' raw artefacts into the study directory.
#   3. Rebuild scores.csv, splits.csv, the three figures and the table.
#   4. Verify every artefact on elisa, by name and by size.
#   5. Destroy the box, if the driver did not.
#   6. Post the comment, if the driver did not.
#   7. Commit the study directory, and push.
#
# Steps 3, 5 and 6 repeat what the driver does. Each one is idempotent, and
# each one is the safety net for a driver that dies before its own stage 7.
#
# Usage:
#   nohup setsid bash scripts/finish_round3c.sh \
#     > results/finish_round3c.out 2>&1 < /dev/null &
#
#   ONLY=verify bash scripts/finish_round3c.sh   # steps 2 to 4 alone, so the
#                                                # checks can be read while the
#                                                # driver still runs
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
. "$HERE/study.sh"

# Every arm of this card scored against ONE tree. The a095 eval wrote its
# shards under box_a, not box_r3, so box_a holds all six and the figures read
# one root. See the note in round3c.sh.
MAIN_DIR="${MAIN_DIR:-$HOME/cf404_sync/box_a}"
MAIN_ROOT="$MAIN_DIR/sync"
cf404_use_root "$MAIN_ROOT"
STOP="${STOP:-$CF404_STOPS}"
STUDY_REL="reports/$(basename "$CF404_STUDY")"
PR="${PR:-405}"
AGENT="${AGENT:-ExperimentRunner claude-opus-5}"
ENVF="${ENVF:-$CF404_RESULTS/round3.env}"
PIDF="${PIDF:-$CF404_RESULTS/round3c.pid}"
WAIT_MAX="${WAIT_MAX:-43200}"   # 12 h, against 3 h of head plus eval left
ONLY="${ONLY:-all}"             # `verify` runs steps 2 to 4 and stops
LOG="$CF404_RESULTS/finish_round3c.log"
VERIFY="$CF404_RESULTS/verify_round3c.txt"
BODY="$CF404_RESULTS/pr_comment_round3c.md"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 finish] $*" | tee -a "$LOG"; }

INSTANCE=""; VAST_LABEL="cf404-box-r3"
[ -s "$ENVF" ] && . "$ENVF"

say "START pid=$$ driver_pidfile=$PIDF box=${INSTANCE:-none}"

# ---- 1: wait for the driver, by pid -----------------------------------------
#
# `kill -0` asks whether the pid exists, and sends nothing. A pattern here
# would also match this script, and on 2026-08-19 a pattern for the sync loop
# matched four running eval shards.
DRIVER=""
[ -s "$PIDF" ] && DRIVER="$(tr -d ' \t\r\n' <"$PIDF")"
if [ "$ONLY" = verify ]; then
  say "ONLY=verify — the driver is not waited for"
elif [ -n "$DRIVER" ] && kill -0 "$DRIVER" 2>/dev/null; then
  say "waiting for the driver, pid $DRIVER"
  waited=0
  while kill -0 "$DRIVER" 2>/dev/null; do
    [ "$waited" -ge "$WAIT_MAX" ] && { say "the driver still runs after ${waited}s — going on anyway"; break; }
    [ $(( waited % 1800 )) -eq 0 ] && [ "$waited" -gt 0 ] && \
      say "  driver up ${waited}s, $(tail -1 "$CF404_RESULTS/round3c.log" 2>/dev/null | cut -c1-140)"
    sleep 60; waited=$(( waited + 60 ))
  done
  say "the driver is gone after ${waited}s"
else
  say "no live driver at pid '${DRIVER:-none}' — going on"
fi

# ---- 2: the raw artefacts ----------------------------------------------------
#
# One curves directory, named for the tree the curves came from. The driver
# labels its own run box_r3, and that would split six arms across two
# directories that hold one tree.
say "report assets"
CF404_BOX_LABEL=box_a CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" \
  CF404_SYNC_DIR="$MAIN_DIR" \
  bash "$HERE/report_assets.sh" >>"$CF404_RESULTS/report_assets_finish.out" 2>&1
say "  $(tail -1 "$CF404_RESULTS/report_assets_finish.out" 2>/dev/null)"

# A curve the driver wrote under box_r3 is a duplicate the moment box_a holds
# the same file at the same size. Delete only in that case.
if [ -d "$CF404_STUDY/curves/box_r3" ]; then
  for f in "$CF404_STUDY/curves/box_r3"/*; do
    [ -f "$f" ] || continue
    twin="$CF404_STUDY/curves/box_a/$(basename "$f")"
    if [ -f "$twin" ] && [ "$(wc -c <"$twin")" -ge "$(wc -c <"$f")" ]; then
      rm -f "$f"; say "  curves/box_r3/$(basename "$f") is a duplicate of box_a — removed"
    fi
  done
  rmdir "$CF404_STUDY/curves/box_r3" 2>/dev/null \
    && say "  curves/box_r3 removed, it is empty"
fi

# ---- 3: the tables and the figures -------------------------------------------
say "collect and draw"
CF404_ROOT="$MAIN_ROOT" bash "$HERE/make_plots.sh" \
  >>"$CF404_RESULTS/make_plots_finish.out" 2>&1
say "  scores.csv holds $(( $(wc -l <"$CF404_RESULTS/scores.csv") - 1 )) arm(s)"

# ---- 4: the verification, by name and by size --------------------------------
say "verify"
{
  echo "#404 round 3c — every artefact of the six arms, on elisa"
  echo "date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "tree: $MAIN_ROOT"
  echo "study: $CF404_STUDY"
  echo
  printf '%-6s %-46s %12s\n' arm artefact bytes
  missing=0
  for arm in $CF404_ARMS; do
    tag="$(cf404_tag "$arm" "$STOP" "$CF404_HEAD_STEPS")"
    name="$(cf404_run_name "$arm")"
    leg="$(cf404_leg_dir "$arm" "$STOP")"
    ev="$(cf404_eval_dir "$arm" "$tag")"
    kk=$(( STOP / 1000 ))
    for f in \
      "$leg/${name}_${kk}k.pth" \
      "$leg/${name}_losses.csv" \
      "$ev/gift/all_results.csv" \
      "$ev/gift/summary.txt" \
      "$ev/qhead_${tag}_s${HEAD_SEED:-20260722}_final.pth" \
      "$(cf404_score_file "$arm" "$STOP")" \
      "$CF404_RESULTS/eval/$tag/all_results.csv" \
      "$CF404_RESULTS/eval/$tag/backbone.txt" \
      "$CF404_RESULTS/run_${name}.log" \
      "$CF404_STUDY/curves/box_a/${name}_losses.csv" \
    ; do
      if [ -f "$f" ]; then
        printf '%-6s %-46s %12s\n' "$arm" "$(basename "$f")" "$(wc -c <"$f")"
      else
        printf '%-6s %-46s %12s\n' "$arm" "$(basename "$f")" MISSING
        missing=$(( missing + 1 ))
      fi
    done
    # The braces put the shell's own redirection error on /dev/null too. A
    # score file that does not exist is `none` in this table, not a stack of
    # "No such file" lines through the middle of it.
    printf '%-6s %-46s %12s\n' "$arm" "SCORE" \
      "$( { tr -d ' \t\r\n' <"$(cf404_score_file "$arm" "$STOP")"; } 2>/dev/null || echo none)"
    echo
  done
  for f in "$CF404_PLOTS/momentum.png" "$CF404_PLOTS/loss_curves.png" \
           "$CF404_PLOTS/domain_radar.png" "$CF404_RESULTS/scores.csv" \
           "$CF404_RESULTS/splits.csv" "$CF404_RESULTS/table.md"; do
    if [ -f "$f" ]; then
      printf '%-6s %-46s %12s\n' study "$(basename "$f")" "$(wc -c <"$f")"
    else
      printf '%-6s %-46s %12s\n' study "$(basename "$f")" MISSING
      missing=$(( missing + 1 ))
    fi
  done
  echo
  echo "missing: $missing"
} >"$VERIFY" 2>&1
say "  $(tail -1 "$VERIFY")"

[ "$ONLY" = verify ] && { say "ONLY=verify — stopping before the box"; exit 0; }

# ---- 5: the box --------------------------------------------------------------
#
# Only this round's instance, and only by the id its own `.env` file records.
# `vastrun-destroy` takes the id and the label together as a confirmation
# token. The vast.ai account is shared with other sessions.
if [ -n "${INSTANCE:-}" ]; then
  if timeout 120 vastrun-status 2>/dev/null | awk -v id="$INSTANCE" '$1 == id { found = 1 } END { exit !found }'; then
    say "the box $INSTANCE still runs — destroying it"
    timeout 300 vastrun-destroy "$INSTANCE" "$VAST_LABEL" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
  else
    say "the box $INSTANCE is already gone"
  fi
  say "vast: $(timeout 120 vastrun-status 2>&1 | tail -1)"
  say "credit: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"
fi

# ---- 6: the comment ----------------------------------------------------------
#
# The driver posts it at its own stage 7. This posts only when the PR carries
# no comment that names the six-arm table.
say "comment"
have="$(timeout 180 gh pr view "$PR" --json comments \
  --jq '[.comments[] | select(.body | contains("## The 6 scored arms"))] | length' 2>/dev/null)"
if [ "${have:-0}" -ge 1 ]; then
  say "  the PR already carries the six-arm comment ($have)"
else
  say "  no six-arm comment on PR #$PR — posting one"
  python3 "$HERE/pr_comment.py" --scores "$CF404_RESULTS/scores.csv" \
    --agent "$AGENT" --dir "$STUDY_REL" --runs 2 \
    --cost "\$$(cat "$CF404_RESULTS/box_spent.txt" 2>/dev/null || echo '?') on the round 3 box" \
    --out "$BODY" >>"$CF404_RESULTS/pr_comment_finish.out" 2>&1 \
    && awk 'NR==1{print; print ""; print "**Round 3 is complete: 6 arms scored.**"; next} {print}' \
         "$BODY" >"$BODY.tmp" && mv "$BODY.tmp" "$BODY" \
    && timeout 180 gh pr comment "$PR" --body-file "$BODY" >>"$LOG" 2>&1 \
    && say "  posted to PR #$PR" || say "  the comment did not post"
fi

# ---- 7: git ------------------------------------------------------------------
say "commit"
cd "$CF404_REPO" || { say "ABORT: no repo at $CF404_REPO"; exit 2; }
git add -A "$STUDY_REL" >>"$LOG" 2>&1
if git diff --cached --quiet; then
  say "  nothing to commit"
else
  git -c user.name="jeremycochoy-agent" -c user.email="jeremy@redstone.ee" \
    commit -q -F - <<'MSG' >>"$LOG" 2>&1
exp(#404): the s08b repeat closes round 3, and the six-arm tables

s08b is s08 at backbone seed 20260521. The distance between their two scores
is the run-to-run spread of THIS cell, at k = 32 against the teacher. Before
it, the card quoted #373's spread, which was measured at k = 3 against the
student.

scores.csv now holds six arms. The three figures and the table are redrawn
from it, and `verify_round3c.txt` lists every artefact by name and by size.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
  say "  $(git log --oneline -1)"
fi
say "push"
timeout 300 git push origin HEAD >>"$LOG" 2>&1 \
  && say "  pushed $(git rev-parse --short HEAD)" || say "  the push failed"

say "FINISH DONE"
