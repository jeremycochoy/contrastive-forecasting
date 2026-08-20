#!/bin/bash
# #404 round 3c — the same two arms as round 3b, in an order that leaves the
# card no idle hour.
#
# WHY THIS SCRIPT EXISTS. `round3b.sh` runs one arm end to end before the next
# one starts: backbone, head, pull, EVAL, then the second arm. The eval is a
# 97-config GIFT-Eval on elisa's CPUs, and it takes one to two hours. The box
# holds one RTX 5090 and that card sits at 0% for every hour of it. The two
# stages use two machines, so they overlap:
#
#   the box, card 0    a095 head (already up), then the s08b backbone.
#   elisa, CPUs        the a095 97-config GIFT-Eval, at the same time.
#
# WHAT IT INHERITS. `round3b.sh` trained the a095 backbone to 40,000 steps and
# started the a095 head. This script picks that head up where it stands. Every
# stage is idempotent: an arm whose checkpoint is on disk is a no-op, and a
# score file that exists is never recomputed.
#
# NO THIRD ARM, and NO SECOND BOX. This script runs the two arms in ARMS and
# stops. The user picks the third momentum from the a095 score. The instance
# comes from `results/round3.env` and nothing here provisions.
#
# THE a095 SCORE IS THE DELIVERABLE THE USER WAITS ON. So the score is posted
# to PR #405 by THIS script, from the tables on disk, the minute it exists. A
# session that ends does not hold the number back.
#
# THE VERIFICATION. A box at 0% GPU with no run directory is a failed launch,
# not a slow start. So the s08b launch leaves nothing to a report: the guard
# line that carries alpha and the seed back off the trainer's own command line,
# the GPU memory in use, one compute app, the first rows of the losses CSV with
# its 33 depth columns, and the measured step rate.
#
# THE BUDGET. This round may spend $12, and round 3b's box has spent $1.51 of
# it. The watchdog tears the box down at MAX_SPEND dollars or DEADLINE_HOURS
# hours, whichever comes first, whatever stage is running.
#
# THE TEARDOWN COMES LAST. The box lives until every score exists.
#
# Usage:
#   nohup setsid bash scripts/round3c.sh > results/round3c.out 2>&1 &
#   CF404_DRY_RUN=1 bash scripts/round3c.sh    # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
export CF404_BOX_LABEL="${LABEL:-box_r3}"
. "$HERE/study.sh"

# The arm that is already up on the card, and the arm that follows it. The
# order is the user's: the 0.95 number first, the repeat under it.
LIVE_ARM="${LIVE_ARM:-a095}"
NEXT_ARM="${NEXT_ARM:-s08b}"
LABEL="$CF404_BOX_LABEL"
VAST_LABEL="cf404-${LABEL//_/-}"
STOP="${STOP:-$CF404_STOPS}"
STUDY_REL="reports/$(basename "$CF404_STUDY")"
BOX_GPU="${BOX_GPU:-0}"
PR="${PR:-405}"
AGENT="${AGENT:-ExperimentRunner claude-opus-5}"

# The canonical tree. Round 1's four arms are here, and the eval, the figures
# and `collect.sh` all read this one root. The box_r3 sync loop keeps its own
# tree as the safety net and never writes this one.
MAIN_DIR="${MAIN_DIR:-$HOME/cf404_sync/box_a}"
MAIN_ROOT="$MAIN_DIR/sync"
SAFE_PULL="$CF404_REPO/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"

POLL="${POLL:-300}"
HEAD_TIMEOUT="${HEAD_TIMEOUT:-10800}"  # 3 h on one head, against 1.1 h measured
BB_TIMEOUT="${BB_TIMEOUT:-32400}"      # 9 h on one backbone, against 3.9 h measured
EVAL_TIMEOUT="${EVAL_TIMEOUT:-21600}"  # 6 h on one 97-config eval, against 1.9 h
DEADLINE_HOURS="${DEADLINE_HOURS:-16}"
MAX_SPEND="${MAX_SPEND:-9.00}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

mkdir -p "$CF404_RESULTS" "$CF404_PLOTS"
LOG="$CF404_RESULTS/round3c.log"
ENVF="${ENVF:-$CF404_RESULTS/round3.env}"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 round3c] $*" | tee -a "$LOG"; }
rsh(){ timeout 180 ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@" 2>/dev/null; }

cf404_require_arm "$LIVE_ARM" || exit $?
cf404_require_arm "$NEXT_ARM" || exit $?
cf404_require_stop "$STOP" || exit $?

# ---- the box this round reuses ----------------------------------------------
INSTANCE=""; HOST=""; PORT=""
[ -s "$ENVF" ] || { say "ABORT: no box address at $ENVF"; exit 2; }
# shellcheck disable=SC1090
. "$ENVF"
[ -n "${HOST:-}" ] && [ -n "${PORT:-}" ] && [ -n "${INSTANCE:-}" ] \
  || { say "ABORT: $ENVF names no instance"; exit 2; }

KK=$(( STOP / 1000 ))
box_leg(){ printf '%s/%s/%s/leg_%dk\n' "$CF404_BOX_RUNS" "$1" "$CF404_CELL" "$KK"; }
box_bb(){    # <arm> — the backbone checkpoint on the box, or nothing
  rsh "ls -1 $(box_leg "$1")/$(cf404_run_name "$1")_${KK}k.pth 2>/dev/null | head -1"
}
box_head(){  # <arm> — the head checkpoint size on the box, or 0
  local tag; tag="$(cf404_tag "$1" "$STOP" "$CF404_HEAD_STEPS")"
  rsh "wc -c <$CF404_BOX_RUNS/$1/eval/$tag/qhead_${tag}_s${HEAD_SEED:-20260722}_final.pth 2>/dev/null" \
    | tr -d ' ' | grep -E '^[0-9]+$' || echo 0
}
box_log(){ printf '/root/cf/%s/results/run_%s.log\n' "$STUDY_REL" "$(cf404_run_name "$1")"; }
box_rate(){ rsh "grep -hoE '[0-9.]+ sps  ETA [0-9.]+h' $(box_log "$1") 2>/dev/null | tail -1"; }

if [ -n "${CF404_DRY_RUN:-}" ]; then
  echo "round3c box=$INSTANCE at $HOST:$PORT card=$BOX_GPU"
  echo "  live=$LIVE_ARM next=$NEXT_ARM stop=$STOP head=$CF404_HEAD_STEPS"
  echo "  head seed=${HEAD_SEED:-20260722} bb seed $LIVE_ARM=$(cf404_seed "$LIVE_ARM")" \
       "$NEXT_ARM=$(cf404_seed "$NEXT_ARM")"
  echo "  alpha $LIVE_ARM=$(cf404_alpha "$LIVE_ARM") $(cf404_schedule "$LIVE_ARM")," \
       "$NEXT_ARM=$(cf404_alpha "$NEXT_ARM") $(cf404_schedule "$NEXT_ARM")"
  echo "  canonical root=$MAIN_ROOT"
  echo "  score $LIVE_ARM=$(cf404_score_file "$LIVE_ARM" "$STOP")"
  echo "  score $NEXT_ARM=$(cf404_score_file "$NEXT_ARM" "$STOP")"
  echo "  deadline=${DEADLINE_HOURS}h max_spend=\$$MAX_SPEND pr=#$PR"
  exit 0
fi

say "START live=$LIVE_ARM next=$NEXT_ARM box=$VAST_LABEL card=$BOX_GPU"
say "  deadline=${DEADLINE_HOURS}h max_spend=\$$MAX_SPEND"
say "credit before: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"

# ---- the teardown, which every exit path runs -------------------------------
#
# Only the instance THIS round uses is destroyed, and only by the id its own
# `.env` file records. `vastrun-destroy` takes the id and the label together as
# a confirmation token. The vast.ai account is shared with other sessions.
TORN=0
teardown(){
  [ "$TORN" -eq 1 ] && return 0
  TORN=1
  say "teardown: destroying $INSTANCE ($VAST_LABEL)"
  timeout 300 vastrun-destroy "$INSTANCE" "$VAST_LABEL" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
  # By pid, from the working directory. NEVER a pattern: on 2026-08-19 a
  # pattern for this loop also matched four running eval shards.
  say "teardown: stopping the sync loop ($(cf404_stop_sync_loop "$CF404_SYNC_DIR") loop(s))"
  timeout 120 vastrun-status 2>&1 | sed 's/^/  /' | tee -a "$LOG"
}

box_spent(){
  timeout 120 vastrun-status 2>/dev/null \
    | awk -v id="$INSTANCE" '$1 == id { for (i = 1; i <= NF; i++) if ($i ~ /^\$/) v = $i; print substr(v, 2) }'
}

watchdog(){
  local secs waited=0 spent
  secs="$(awk -v h="$DEADLINE_HOURS" 'BEGIN{printf "%d", h*3600}')"
  while [ "$waited" -lt "$secs" ]; do
    sleep 600; waited=$(( waited + 600 ))
    spent="$(box_spent)"
    [ -n "$spent" ] || continue
    if awk -v s="$spent" -v m="$MAX_SPEND" 'BEGIN{exit !(s+0 >= m+0)}'; then
      say "WATCHDOG: the box has spent \$$spent of \$$MAX_SPEND — tearing it down"
      teardown
      return 0
    fi
  done
  say "WATCHDOG: ${DEADLINE_HOURS} h reached — tearing the box down"
  teardown
}
watchdog & WATCHDOG=$!
stop_watchdog(){ kill -TERM "$WATCHDOG" 2>/dev/null; }

rsh true || { say "ABORT: instance $INSTANCE at $HOST:$PORT does not answer."
              say "  This round reuses one box and rents none. Nothing was destroyed."
              stop_watchdog; exit 2; }
say "reusing instance $INSTANCE at $HOST:$PORT"

CARD="$(rsh "nvidia-smi --query-gpu=index,name,memory.total,compute_mode --format=csv,noheader")"
NCARD="$(printf '%s\n' "$CARD" | grep -c .)"
say "card(s): $(printf '%s' "$CARD" | tr '\n' '|')  count=$NCARD"
[ "$BOX_GPU" -lt "${NCARD:-0}" ] || {
  say "ABORT: every lane names card $BOX_GPU and the box carries $NCARD card(s)"
  teardown; stop_watchdog; exit 1; }

say "sync loop: $(cf404_sync_loops "$CF404_SYNC_DIR") for $CF404_SYNC_DIR," \
    "$(find "$CF404_SYNC_DIR" -type f 2>/dev/null | wc -l) file(s) here"

pull(){  # <remote> <local> <floor>
  local dst="$2"
  [ -f "$dst" ] && [ "$(wc -c <"$dst")" -ge "$3" ] && { say "  have $(basename "$dst")"; return 0; }
  mkdir -p "$(dirname "$dst")"
  SSH_USER=root bash "$SAFE_PULL" "$HOST" "$PORT" "$1" "$dst" "$3" >>"$LOG" 2>&1
  [ -f "$dst" ] || { say "  MISSING $(basename "$dst")"; return 1; }
  say "  $(basename "$dst") $(wc -c <"$dst") B"
}

pull_arm(){  # <arm>
  local arm="$1" NAME TAG RL LL missing=0
  NAME="$(cf404_run_name "$arm")"
  TAG="$(cf404_tag "$arm" "$STOP" "$CF404_HEAD_STEPS")"
  RL="$(box_leg "$arm")"; LL="$MAIN_ROOT/$arm/$CF404_CELL/leg_${KK}k"
  say "$arm: pulling into $MAIN_ROOT"
  pull "$RL/${NAME}_${KK}k.pth"           "$LL/${NAME}_${KK}k.pth"           3000000 || missing=1
  pull "$RL/${NAME}_${KK}k_optimizer.pth" "$LL/${NAME}_${KK}k_optimizer.pth" 4000000 || missing=1
  pull "$RL/${NAME}_losses.csv"           "$LL/${NAME}_losses.csv"           1000000 || missing=1
  pull "$RL/${NAME}_attn_amplitude.csv"   "$LL/${NAME}_attn_amplitude.csv"   1000     || missing=1
  pull "$RL/${NAME}_latent_drift.csv"     "$LL/${NAME}_latent_drift.csv"     100      || missing=1
  pull "$CF404_BOX_RUNS/$arm/eval/$TAG/qhead_${TAG}_s${HEAD_SEED:-20260722}_final.pth" \
       "$MAIN_ROOT/$arm/eval/$TAG/qhead_${TAG}_s${HEAD_SEED:-20260722}_final.pth" 200000 || missing=1
  pull "/root/cf/$STUDY_REL/results/run_${NAME}.log" \
       "$CF404_RESULTS/run_${NAME}.log" 1000 || missing=1
  [ "$missing" -eq 0 ]
}

# ---- the head of one arm ----------------------------------------------------
run_head(){  # <arm>
  local arm="$1" waited
  if [ "$(box_head "$arm")" -gt 200000 ]; then
    say "$arm: the head is already on the box"
    return 0
  fi
  if rsh "pgrep -f '$(cf404_pgrep_pattern train_forecasting_head)' >/dev/null"; then
    say "$arm: a head trainer already runs on the box"
  else
    say "$arm: starting the head on card $BOX_GPU"
    rsh "cd /root/cf/$STUDY_REL && ARMS='$arm' GPUS='$BOX_GPU' \
         nohup setsid bash scripts/heads_box.sh \
           > results/heads_${LABEL}_${arm}.out 2>&1 < /dev/null & echo started" \
      >>"$LOG" 2>&1
  fi
  waited=0
  while [ "$(box_head "$arm")" -le 200000 ]; do
    [ "$waited" -ge "$HEAD_TIMEOUT" ] && { say "$arm: no head after ${waited}s"; return 1; }
    [ $(( waited % 1800 )) -eq 0 ] && say "  $arm head $(box_head "$arm") B"
    sleep "$POLL"; waited=$(( waited + POLL ))
  done
  say "$arm: the head is done, $(box_head "$arm") B"
}

# ---- the backbone of one arm ------------------------------------------------
run_backbone(){  # <arm>
  local arm="$1" waited verdict stopped rows cols used apps rate
  if [ -n "$(box_bb "$arm")" ]; then
    say "$arm: the backbone is already on the box"
    return 0
  fi
  if rsh "pgrep -f '$(cf404_pgrep_pattern "run_leg_k.sh $CF404_CELL")' >/dev/null"; then
    say "$arm: a trainer already runs on the box"
  else
    say "$arm: starting the backbone on card $BOX_GPU"
    rsh "cd /root/cf/$STUDY_REL && ARMS='$arm' GPUS='$BOX_GPU' \
         nohup setsid bash scripts/launch_box.sh \
           > results/launch_${LABEL}_${arm}.out 2>&1 < /dev/null & echo started" \
      >>"$LOG" 2>&1
  fi

  # -- the verification. Four facts, all read off the box.
  say "$arm: waiting for the guard line (it reads alpha and the seed back off"
  say "  the trainer's own command line)"
  waited=0; verdict=""
  while [ "$waited" -lt 2400 ]; do
    verdict="$(rsh "grep -h 'arm $arm .*reached the trainer' /root/cf/$STUDY_REL/results/arms.log 2>/dev/null")"
    stopped="$(rsh "grep -h \"arm $arm STOPPED\" /root/cf/$STUDY_REL/results/arms.log 2>/dev/null | tail -1")"
    [ -n "$stopped" ] && { say "$arm: $stopped"; return 1; }
    [ -n "$verdict" ] && break
    sleep 30; waited=$(( waited + 30 ))
  done
  [ -n "$verdict" ] || { say "$arm: no trainer command line in ${waited}s"; return 1; }
  printf '%s\n' "$verdict" | sed 's/^/  /' | tee -a "$LOG"

  say "$arm: waiting for the first rows of the losses CSV"
  waited=0; rows=0
  while [ "$waited" -lt 1800 ]; do
    rows="$(rsh "csv=\$(ls $(box_leg "$arm")/*_losses.csv 2>/dev/null | head -1); \
                 [ -n \"\$csv\" ] && grep -c '^' \"\$csv\" || echo 0")"
    case "$rows" in ''|*[!0-9]*) rows=0 ;; esac
    [ "$rows" -ge 2 ] && break
    sleep 30; waited=$(( waited + 30 ))
  done

  {
    echo "=== $arm ==="
    rsh "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv"
    rsh "nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
    rsh "csv=\$(ls $(box_leg "$arm")/*_losses.csv 2>/dev/null | head -1); \
         if [ -n \"\$csv\" ]; then \
           echo \"csv \$csv\"; \
           echo \"depth_cols \$(head -1 \"\$csv\" | tr -d '\r' | tr ',' '\n' | grep -c '^cos_err_d[0-9]*\$')\"; \
           echo \"rows \$(grep -c '^' \"\$csv\")\"; head -3 \"\$csv\" | cut -c1-160; \
         else echo 'csv MISSING'; fi"
  } >"$CF404_RESULTS/round3c_verify_${arm}.txt" 2>&1
  sed 's/^/  /' "$CF404_RESULTS/round3c_verify_${arm}.txt" | tee -a "$LOG"

  used="$(awk -F', ' 'NR==2{gsub(/[^0-9]/,"",$2); print $2}' \
          <(rsh "nvidia-smi --query-gpu=index,memory.used --format=csv"))"
  apps="$(rsh "nvidia-smi --query-compute-apps=pid --format=csv,noheader" | grep -c .)"
  cols="$(awk '/^depth_cols /{print $2; exit}' "$CF404_RESULTS/round3c_verify_${arm}.txt")"
  rate="$(box_rate "$arm")"
  say "$arm: GPU memory in use ${used:-0} MiB, $apps compute app(s)," \
      "${cols:-0} depth columns, $rows CSV row(s)"
  say "$arm: STEP RATE ${rate:-unknown}"
  [ "${used:-0}" -ge 500 ] || { say "$arm: the card holds ${used:-0} MiB — no trainer is on it"; return 1; }
  [ "$apps" -ge 1 ] || { say "$arm: no compute app on the card"; return 1; }
  [ "${cols:-0}" -eq $(( CF404_K + 1 )) ] \
    || { say "$arm: ${cols:-0} depth columns, k=$CF404_K wants $(( CF404_K + 1 ))"; return 1; }
  [ "$rows" -ge 2 ] || { say "$arm: the losses CSV has $rows row(s)"; return 1; }
  say "$arm: VERIFIED — one trainer on card $BOX_GPU at $(( CF404_K + 1 )) depth columns"
  return 0
}

# ---- the 97-config GIFT-Eval of one arm, on elisa ---------------------------
#
# Detached, so it does not hold this driver: the box trains the next backbone
# through every hour of it. The score file is the completion signal.
start_eval(){  # <arm>
  local arm="$1" f
  f="$(cf404_score_file "$arm" "$STOP")"
  [ -s "$f" ] && { say "$arm: already scored $(tr -d ' \t\r\n' <"$f")"; return 0; }
  # The bracket class keeps this check off its own command line. See
  # `cf404_pgrep_pattern`.
  if pgrep -u "$(id -u)" -f "$(cf404_pgrep_pattern "head_eval.sh $arm")" >/dev/null 2>&1; then
    say "$arm: the eval already runs on elisa"
    return 0
  fi
  say "$arm: starting the 97-config GIFT-Eval on elisa CPUs, detached"
  ARMS="$arm" CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" \
    nohup setsid bash "$HERE/evals_elisa.sh" \
      >"$CF404_RESULTS/evals_round3c_${arm}.out" 2>&1 < /dev/null &
  sleep 20
  say "$arm: eval launched, $(pgrep -u "$(id -u)" -f "$(cf404_pgrep_pattern "head_eval")" | grep -c . ) eval process(es) on elisa"
}

# ---- the comment this card owes the user ------------------------------------
#
# Built from `results/scores.csv`, which `collect.sh` writes from the score
# files, so the comment cannot disagree with the figures.
POSTED_LIVE=0
post_comment(){  # <runs> <headline>
  local runs="$1" head="$2" body="$CF404_RESULTS/pr_comment_round3c.md" spent
  bash "$HERE/collect.sh" >>"$CF404_RESULTS/collect_round3c.out" 2>&1
  CF404_ROOT="$MAIN_ROOT" bash "$HERE/make_plots.sh" \
    >>"$CF404_RESULTS/make_plots_round3c.out" 2>&1
  spent="$(box_spent)"
  python3 "$HERE/pr_comment.py" --scores "$CF404_RESULTS/scores.csv" \
    --agent "$AGENT" --dir "$STUDY_REL" --runs "$runs" \
    --cost "\$${spent:-?} on the round 3 box" --out "$body" \
    >>"$CF404_RESULTS/pr_comment_round3c.out" 2>&1 || {
      say "comment: pr_comment.py failed"; return 1; }
  # The headline goes first, under the signature line the pipeline reads.
  awk -v h="$head" 'NR==1{print; print ""; print h; next} {print}' "$body" >"$body.tmp" \
    && mv "$body.tmp" "$body"
  timeout 180 gh pr comment "$PR" --body-file "$body" >>"$LOG" 2>&1 \
    && say "comment: posted to PR #$PR" || say "comment: gh refused PR #$PR"
  cat "$body" | sed 's/^/  /' | tee -a "$LOG"
}

# The score of the live arm, reported the minute it exists.
check_live_score(){
  local f
  [ "$POSTED_LIVE" -eq 1 ] && return 0
  f="$(cf404_score_file "$LIVE_ARM" "$STOP")"
  [ -s "$f" ] || return 1
  POSTED_LIVE=1
  say "SCORE $LIVE_ARM $(tr -d ' \t\r\n' <"$f")"
  post_comment 1 "**\`$LIVE_ARM\` (EMA momentum 0.95, fixed) scores $(tr -d ' \t\r\n' <"$f") GM-Relative MASE.** The \`s08b\` repeat trains on the box now."
}

# One poll tick: sleep, then look at the eval that runs beside the box.
tick(){ sleep "$POLL"; check_live_score >/dev/null 2>&1 || true; }

# ---- 1: the live arm's head, which is already on the card -------------------
say "==== $LIVE_ARM : alpha $(cf404_alpha "$LIVE_ARM") $(cf404_schedule "$LIVE_ARM") seed $(cf404_seed "$LIVE_ARM") ===="
if ! run_head "$LIVE_ARM"; then
  say "$LIVE_ARM: FAILED at the head"
  teardown; stop_watchdog; exit 1
fi

# The card is free the moment the head process leaves it. The next backbone
# takes it, and the eval of this arm runs on elisa beside it.
say "$LIVE_ARM: waiting for the head process to leave card $BOX_GPU"
waited=0
while rsh "pgrep -f '$(cf404_pgrep_pattern train_forecasting_head)' >/dev/null"; do
  [ "$waited" -ge 900 ] && { say "  the head process still holds the card after ${waited}s"; break; }
  sleep 30; waited=$(( waited + 30 ))
done
say "$LIVE_ARM: the card is free after ${waited}s"

# ---- 2: the next backbone, on the card the head just left -------------------
NEXT_OK=1
if ! run_backbone "$NEXT_ARM"; then
  say "$NEXT_ARM: FAILED at the backbone — the $LIVE_ARM score still lands below"
  NEXT_OK=0
fi

# ---- 3: the live arm's artefacts, and its eval on elisa ---------------------
pull_arm "$LIVE_ARM" || say "$LIVE_ARM: WARNING — an artefact did not land"
start_eval "$LIVE_ARM"

# ---- 4: the two machines run together ---------------------------------------
#
# The box climbs to 40,000 steps. elisa scores the arm above. The loop below
# watches both and posts the 0.95 number the minute the score file appears.
if [ "$NEXT_OK" -eq 1 ]; then
  say "$NEXT_ARM: waiting for $STOP steps, while $LIVE_ARM scores on elisa"
  waited=0
  while [ -z "$(box_bb "$NEXT_ARM")" ]; do
    [ "$waited" -ge "$BB_TIMEOUT" ] && { say "$NEXT_ARM: no backbone after ${waited}s"; NEXT_OK=0; break; }
    [ $(( waited % 1800 )) -eq 0 ] && say "  $NEXT_ARM $(rsh "grep -hoE '^\[ *[0-9]+\].*sps  ETA [0-9.]+h' $(box_log "$NEXT_ARM") 2>/dev/null | tail -1")"
    tick; waited=$(( waited + POLL ))
  done
  [ "$NEXT_OK" -eq 1 ] && say "$NEXT_ARM: the backbone is done"
fi

# The live arm's eval may still run. Wait for it before the box goes, because
# a score that misses can only be recomputed from the artefacts this box holds.
say "$LIVE_ARM: waiting for the score"
waited=0
while ! check_live_score >/dev/null 2>&1; do
  [ "$waited" -ge "$EVAL_TIMEOUT" ] && { say "$LIVE_ARM: no score after ${waited}s"; break; }
  [ $(( waited % 1800 )) -eq 0 ] && say "  $LIVE_ARM eval running, $(pgrep -u "$(id -u)" -f "$(cf404_pgrep_pattern head_eval)" | grep -c .) process(es)"
  sleep "$POLL"; waited=$(( waited + POLL ))
done
check_live_score >/dev/null 2>&1 || true

# ---- 5: the next arm's head, then its eval ----------------------------------
scored=0
[ -s "$(cf404_score_file "$LIVE_ARM" "$STOP")" ] && scored=1
if [ "$NEXT_OK" -eq 1 ]; then
  say "==== $NEXT_ARM : alpha $(cf404_alpha "$NEXT_ARM") $(cf404_schedule "$NEXT_ARM") seed $(cf404_seed "$NEXT_ARM") ===="
  if run_head "$NEXT_ARM"; then
    pull_arm "$NEXT_ARM" || say "$NEXT_ARM: WARNING — an artefact did not land"
    start_eval "$NEXT_ARM"
    say "$NEXT_ARM: waiting for the score"
    waited=0
    while [ ! -s "$(cf404_score_file "$NEXT_ARM" "$STOP")" ]; do
      [ "$waited" -ge "$EVAL_TIMEOUT" ] && { say "$NEXT_ARM: no score after ${waited}s"; break; }
      [ $(( waited % 1800 )) -eq 0 ] && say "  $NEXT_ARM eval running"
      sleep "$POLL"; waited=$(( waited + POLL ))
    done
    if [ -s "$(cf404_score_file "$NEXT_ARM" "$STOP")" ]; then
      scored=$(( scored + 1 ))
      say "SCORE $NEXT_ARM $(tr -d ' \t\r\n' <"$(cf404_score_file "$NEXT_ARM" "$STOP")")"
    else
      say "SCORE $NEXT_ARM MISSING"
    fi
  else
    say "$NEXT_ARM: FAILED at the head"
  fi
fi

# ---- 6: the teardown --------------------------------------------------------
#
# Every score that exists, exists now. The box outlived the scores.
say "$scored of 2 arm(s) scored — tearing the box down"
teardown
stop_watchdog
say "credit after: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"

# ---- 7: the artefacts, the figures and the closing comment ------------------
say "shard check"
python3 "$HERE/check_shards.py" --root "$MAIN_ROOT" \
  --out "$CF404_RESULTS/shard_check.txt" 2>&1 | tail -20 | tee -a "$LOG"
say "report assets"
CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" CF404_SYNC_DIR="$MAIN_DIR" \
  bash "$HERE/report_assets.sh" >>"$CF404_RESULTS/report_assets_round3c.out" 2>&1
POSTED_LIVE=1
post_comment "$scored" "**Round 3 is complete: $scored of 2 arms scored.**"
say "ROUND 3C DONE — $scored of 2 arm(s) scored"
