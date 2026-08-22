#!/bin/bash
# #404 — the steps that follow the round 5 driver, detached from any session.
#
# `round5.sh` trains `r100_09` and `r100_08`, pulls their artefacts, proves
# each one reads, destroys the box and then scores both on elisa. It never
# touches git, and it draws no figure. A session that ends takes its own
# waiters with it, so this script holds the rest, under `nohup setsid`:
#
#   1. Wait for the driver to exit, by pid. Never by a pattern.
#   2. Put every arm's raw artefacts into the study directory.
#   3. Rebuild scores.csv, splits.csv, EVERY figure and the table.
#   4. Verify every artefact on elisa, by name and by size.
#   5. Destroy the box, if the driver did not.
#   6. Post the ExperimentRunner comment.
#   7. Commit the study directory, and push.
#
# Step 5 repeats what the driver does at its stage 8. It is idempotent, and it
# is the safety net for a driver that dies before it.
#
# WHAT THE COMMENT MUST CARRY. This round asks one question: does a shorter
# ramp score lower at the 40,000-step stop? So the comment carries every
# scored arm WITH THE MOMENTUM IT HOLDS AT THE STOP, because two arms of this
# card start at 0.8 and two start at 0.9, and the start value no longer names
# an arm.
#
# WHAT THE VERIFY TABLE COUNTS. `s08c` and `s08d` trained a backbone and ran
# no head on purpose. Their eval artefacts are absent BY DESIGN, so the table
# checks a head and an eval only for an arm that carries a score, and checks a
# backbone for every arm. A `missing` count that included them would report a
# healthy round as broken.
#
# Usage:
#   nohup setsid bash scripts/finish_round5.sh \
#     > results/finish_round5.out 2>&1 < /dev/null &
#
#   ONLY=verify bash scripts/finish_round5.sh   # steps 2 to 4 alone
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
. "$HERE/study.sh"

# Every arm of this card scored against ONE tree. The evals write their shards
# under box_a, so box_a holds all eight and the figures read one root.
MAIN_DIR="${MAIN_DIR:-$HOME/cf404_sync/box_a}"
MAIN_ROOT="$MAIN_DIR/sync"
cf404_use_root "$MAIN_ROOT"
# The health figure and the seed report read the PARENT of the per-box roots:
# the contrastive AUC lives in each arm's backbone losses CSV, and the arms of
# this card were trained on five boxes.
SYNC_TREE="${SYNC_TREE:-$(dirname "$MAIN_DIR")}"
STOP="${STOP:-$CF404_STOPS}"
STUDY_REL="reports/$(basename "$CF404_STUDY")"
PR="${PR:-405}"
AGENT="${AGENT:-ExperimentRunner claude-opus-5}"
ROUND_ARMS="${ROUND_ARMS:-r100_09 r100_08}"
ENVF="${ENVF:-$CF404_RESULTS/round5.env}"
PIDF="${PIDF:-$CF404_RESULTS/round5.pid}"
WAIT_MAX="${WAIT_MAX:-50400}"   # 14 h, the driver's own deadline
ONLY="${ONLY:-all}"             # `verify` runs steps 2 to 4 and stops
LOG="$CF404_RESULTS/finish_round5.log"
VERIFY="$CF404_RESULTS/verify_round5.txt"
BODY="$CF404_RESULTS/pr_comment_round5.md"
BOX_LABEL_R4="${BOX_LABEL_R4:-box_r4}"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 finish5] $*" | tee -a "$LOG"; }

INSTANCE=""; VAST_LABEL="cf404-${BOX_LABEL_R4//_/-}"
# shellcheck disable=SC1090
[ -s "$ENVF" ] && . "$ENVF"

say "START pid=$$ driver_pidfile=$PIDF box=${INSTANCE:-none}"
say "arms this round: $ROUND_ARMS"

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
      say "  driver up ${waited}s, $(tail -1 "$CF404_RESULTS/round5.log" 2>/dev/null | cut -c1-140)"
    sleep 60; waited=$(( waited + 60 ))
  done
  say "the driver is gone after ${waited}s"
else
  say "no live driver at pid '${DRIVER:-none}' — going on"
fi

# ---- 2: the raw artefacts ----------------------------------------------------
#
# One curves directory, named for the tree the curves came from. The driver
# labels its own run box_r4, and that would split eight arms across two
# directories that hold one tree.
say "report assets"
CF404_BOX_LABEL=box_a CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" \
  CF404_SYNC_DIR="$MAIN_DIR" \
  bash "$HERE/report_assets.sh" >"$CF404_RESULTS/report_assets_finish5.out" 2>&1
say "  $(tail -1 "$CF404_RESULTS/report_assets_finish5.out" 2>/dev/null)"

# A curve the sync loop wrote under box_r4 is a duplicate the moment box_a
# holds the same file at the same size or larger. Delete only in that case.
if [ -d "$CF404_STUDY/curves/$BOX_LABEL_R4" ]; then
  for f in "$CF404_STUDY/curves/$BOX_LABEL_R4"/*; do
    [ -f "$f" ] || continue
    twin="$CF404_STUDY/curves/box_a/$(basename "$f")"
    if [ -f "$twin" ] && [ "$(wc -c <"$twin")" -ge "$(wc -c <"$f")" ]; then
      rm -f "$f"; say "  curves/$BOX_LABEL_R4/$(basename "$f") is a duplicate of box_a — removed"
    fi
  done
  rmdir "$CF404_STUDY/curves/$BOX_LABEL_R4" 2>/dev/null \
    && say "  curves/$BOX_LABEL_R4 removed, it is empty"
fi

# ---- 3: the tables and EVERY figure ------------------------------------------
#
# The two `.out` files below are TRUNCATED, never appended. This script greps
# them for its own report, and an appended file makes a second run quote the
# first run's counts beside its own.
say "collect and draw"
CF404_ROOT="$MAIN_ROOT" CF404_SYNC_DIR="$MAIN_DIR" CF404_SYNC_TREE="$SYNC_TREE" \
  bash "$HERE/make_plots.sh" >"$CF404_RESULTS/make_plots_finish5.out" 2>&1
say "  scores.csv holds $(( $(wc -l <"$CF404_RESULTS/scores.csv") - 1 )) arm(s)"
grep -E '^(wrote|SKIP)' "$CF404_RESULTS/make_plots_finish5.out" 2>/dev/null \
  | sed 's/^/  /' | tee -a "$LOG"

# ---- 4: the verification, by name and by size --------------------------------
say "verify"
{
  echo "#404 round 5 — every artefact of every arm, on elisa"
  echo "date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "tree: $MAIN_ROOT"
  echo "study: $CF404_STUDY"
  echo
  printf '%-8s %-46s %12s\n' arm artefact bytes
  missing=0
  for arm in $CF404_ARMS; do
    tag="$(cf404_tag "$arm" "$STOP" "$CF404_HEAD_STEPS")"
    name="$(cf404_run_name "$arm")"
    leg="$(cf404_leg_dir "$arm" "$STOP")"
    ev="$(cf404_eval_dir "$arm" "$tag")"
    kk=$(( STOP / 1000 ))
    # A backbone is expected of EVERY arm. A head and an eval are expected
    # only of an arm that carries a score: `s08c` and `s08d` trained a
    # backbone and ran no head on purpose, and counting their absent eval as
    # missing would report this round as broken.
    files=( "$leg/${name}_${kk}k.pth"
            "$leg/${name}_losses.csv"
            "$CF404_RESULTS/run_${name}.log"
            "$CF404_STUDY/curves/box_a/${name}_losses.csv" )
    if [ -s "$(cf404_score_file "$arm" "$STOP")" ]; then
      files+=( "$ev/gift/all_results.csv"
               "$ev/gift/summary.txt"
               "$ev/qhead_${tag}_s${HEAD_SEED:-20260722}_final.pth"
               "$(cf404_score_file "$arm" "$STOP")"
               "$CF404_RESULTS/eval/$tag/all_results.csv"
               "$CF404_RESULTS/eval/$tag/backbone.txt" )
    else
      printf '%-8s %-46s %12s\n' "$arm" "(no score — head and eval skipped)" "-"
    fi
    for f in "${files[@]}"; do
      if [ -f "$f" ]; then
        printf '%-8s %-46s %12s\n' "$arm" "$(basename "$f")" "$(wc -c <"$f")"
      else
        printf '%-8s %-46s %12s\n' "$arm" "$(basename "$f")" MISSING
        missing=$(( missing + 1 ))
      fi
    done
    # The braces put the shell's own redirection error on /dev/null too. A
    # score file that does not exist is `none` in this table, not a stack of
    # "No such file" lines through the middle of it.
    printf '%-8s %-46s %12s\n' "$arm" "SCORE" \
      "$( { tr -d ' \t\r\n' <"$(cf404_score_file "$arm" "$STOP")"; } 2>/dev/null || echo none)"
    echo
  done
  for f in "$CF404_PLOTS/momentum.png" "$CF404_PLOTS/momentum_at_stop.png" \
           "$CF404_PLOTS/loss_curves.png" \
           "$CF404_PLOTS/domain_radar.png" "$CF404_PLOTS/backbone_health.png" \
           "$CF404_RESULTS/scores.csv" "$CF404_RESULTS/splits.csv" \
           "$CF404_RESULTS/table.md" "$CF404_RESULTS/seed_report.md" \
           "$CF404_RESULTS/seed_table.csv"; do
    if [ -f "$f" ]; then
      printf '%-8s %-46s %12s\n' study "$(basename "$f")" "$(wc -c <"$f")"
    else
      printf '%-8s %-46s %12s\n' study "$(basename "$f")" MISSING
      missing=$(( missing + 1 ))
    fi
  done
  echo
  echo "missing: $missing"
} >"$VERIFY" 2>&1
say "  $(tail -1 "$VERIFY")"
say "  seed table:"
sed 's/^/    /' "$CF404_RESULTS/seed_table.csv" 2>/dev/null | tee -a "$LOG"

[ "$ONLY" = verify ] && { say "ONLY=verify — stopping before the box"; exit 0; }

# ---- 5: the box --------------------------------------------------------------
#
# Only this round's instance, and only by the id its own `.env` file records.
# `vastrun-destroy` takes the id and the label together as a confirmation
# token. The vast.ai account is shared with other sessions.
SPENT="?"
if [ -n "${INSTANCE:-}" ]; then
  SPENT="$(timeout 120 vastrun-status 2>/dev/null \
    | awk -v id="$INSTANCE" '$1 == id { for (i = 1; i <= NF; i++) if ($i ~ /^\$/) v = $i; print substr(v, 2) }')"
  [ -n "$SPENT" ] && printf '%s\n' "$SPENT" >"$CF404_RESULTS/box_r4_spent.txt"
  [ -n "$SPENT" ] || SPENT="$(cat "$CF404_RESULTS/box_r4_spent.txt" 2>/dev/null || echo '?')"
  if timeout 120 vastrun-status 2>/dev/null | awk -v id="$INSTANCE" '$1 == id { found = 1 } END { exit !found }'; then
    say "the box $INSTANCE still runs, \$$SPENT spent — destroying it"
    timeout 300 vastrun-destroy "$INSTANCE" "$VAST_LABEL" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
  else
    say "the box $INSTANCE is already gone, \$$SPENT spent"
  fi
  say "vast: $(timeout 120 vastrun-status 2>&1 | tail -1)"
  say "credit: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"
fi

# ---- 6: the comment ----------------------------------------------------------
#
# It is built from `results/scores.csv` and the sync tree, so it cannot
# disagree with the figures. `--sync-root` is what makes the spread the one the
# card asks for: the spread over the seeds that did NOT collapse.
say "comment"
n_arms="$(( $(wc -l <"$CF404_RESULTS/scores.csv") - 1 ))"
n_round=0
for arm in $ROUND_ARMS; do
  [ -s "$(cf404_score_file "$arm" "$STOP")" ] && n_round=$(( n_round + 1 ))
done
# The key is this round's own heading, not round 4's. Round 4 already posted a
# comment carrying "## The repeat family, seed by seed", so that string would
# make this round think it had already posted.
have="$(timeout 180 gh pr view "$PR" --json comments \
  --jq '[.comments[] | select(.body | contains("Round 5 is complete"))] | length' 2>/dev/null)"
if [ "${have:-0}" -ge 1 ] && [ "${FORCE_COMMENT:-0}" != 1 ]; then
  say "  the PR already carries the seed-family comment ($have)"
else
  python3 "$HERE/pr_comment.py" --scores "$CF404_RESULTS/scores.csv" \
    --sync-root "$SYNC_TREE" --stop "$STOP" \
    --agent "$AGENT" --dir "$STUDY_REL" --runs "$n_round" \
    --cost "\$$SPENT on the round 4 box, one RTX 5090 with a Ryzen 7 7800X3D" \
    --out "$BODY" >"$CF404_RESULTS/pr_comment_round5.out" 2>&1 || {
      say "  pr_comment.py failed, see pr_comment_round5.out"; }
  if [ -s "$BODY" ]; then
    awk -v h="**Round 5 is complete: $n_round of 2 shorter-ramp arms scored, $n_arms arms in all.**" \
      'NR==1{print; print ""; print h; next} {print}' "$BODY" >"$BODY.tmp" \
      && mv "$BODY.tmp" "$BODY"
    timeout 180 gh pr comment "$PR" --body-file "$BODY" >>"$LOG" 2>&1 \
      && say "  posted to PR #$PR" || say "  the comment did not post"
    sed 's/^/  /' "$BODY" | tee -a "$LOG"
  else
    say "  no comment body — nothing posted"
  fi
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
exp(#404): a shorter ramp, and the momentum an arm holds at the stop

r100_09 and r100_08 raise the momentum to 1.0 over 100,000 steps instead of
200,000. The ramp length is the one axis this card had not moved, and the EMA
schedule ladder scored this ramp 0.0259 below the fixed 0.9 reference at this
stop.

The two arms hold 0.940 and 0.880 at 40,000 steps. With the four earlier arms
the card now reads one ordered row of held momenta: 0.800, 0.840, 0.880,
0.900, 0.920, 0.940, 0.950.

A ramp arm does not hold the momentum it names, and every reader now says so.
plots/momentum_at_stop.png puts the held value on an x axis, make_table.py and
pr_comment.py carry it as a column, and plot_momentum.py keys its series on
the schedule together with the ramp length, so two ramps that share a start
value are no longer averaged into one marker.

s08c and s08d trained a backbone to 40,000 steps and ran no head. Their
contrastive AUC says the backbone lived, and the verify table expects no eval
of them.

verify_round5.txt lists every artefact by name and by size.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
  say "  $(git log --oneline -1)"
fi
say "push"
timeout 300 git push origin HEAD >>"$LOG" 2>&1 \
  && say "  pushed $(git rev-parse --short HEAD)" || say "  the push failed"

say "FINISH DONE"
