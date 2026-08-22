#!/bin/bash
# #404 round 3b — `a095` first, then `s08b`, on the box round 3 already rented.
#
# WHAT CHANGED FROM ROUND 3. The user read the momentum figure. With alpha held
# constant, 0.8 scores 1.2309 and 0.9 scores 1.1819. That segment falls steeply
# and it does not turn, so the next value belongs ABOVE 0.9, not between 0.8
# and 0.9. Round 3 planned three arms. This round runs two:
#
#   a095  alpha 0.95 held constant. The arm the user asks for.
#   s08b  `s08` again at backbone seed 20260521. It measures THIS cell's
#         repeat spread, which is what says whether the 0.95 result is a
#         result or noise.
#
# `a085` is DROPPED. It interpolates inside a segment that already falls in one
# direction.
#
# ONE ARM AT A TIME, IN THAT ORDER. Each arm runs end to end — backbone, head,
# pull, eval, score — before the next one starts. Two arms on one card would
# both land at the end. The user needs the 0.95 number first, because the third
# arm of the study is chosen from it.
#
# NO THIRD ARM. This script runs the two arms in ARMS and stops. The user picks
# the third value from `a095`.
#
# ONE BOX, ONE CARD. The instance comes from `results/round3.env` and nothing
# here provisions. A box that does not answer is an ABORT, never a second
# rental. Every lane names card 0, and `cf404_require_gpus` reads the real card
# count off the driver before a lane starts: round 3's plan print put `a095` on
# `gpu=1` while the box held one card, at index 0.
#
# THE VERIFICATION. A box at 0% GPU with no run directory is a failed launch,
# not a slow start. So stage 3 leaves no arm until it reads, off the box: the
# guard line that carries alpha and the seed back off the trainer's own command
# line, the GPU memory in use, one compute app, the first rows of the losses
# CSV with its 33 depth columns, and the measured step rate.
#
# THE BUDGET. This round may spend $12. The watchdog tears the box down at
# MAX_SPEND dollars or DEADLINE_HOURS hours, whichever comes first, whatever
# stage is running.
#
# THE TEARDOWN COMES LAST. The box lives until every score exists.
#
# Usage:
#   nohup setsid bash scripts/round3b.sh > results/round3b.out 2>&1 &
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
export CF404_BOX_LABEL="${LABEL:-box_r3}"
. "$HERE/study.sh"

ARMS="${ARMS:-a095 s08b}"
LABEL="$CF404_BOX_LABEL"
VAST_LABEL="cf404-${LABEL//_/-}"
STOP="${STOP:-$CF404_STOPS}"
STUDY_REL="reports/$(basename "$CF404_STUDY")"
BOX_GPU="${BOX_GPU:-0}"

# The canonical tree. Round 1's four arms are here, and the eval, the figures
# and `collect.sh` all read this one root.
MAIN_DIR="${MAIN_DIR:-$HOME/cf404_sync/box_a}"
MAIN_ROOT="$MAIN_DIR/sync"
SAFE_PULL="$CF404_REPO/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"

POLL="${POLL:-300}"
BB_TIMEOUT="${BB_TIMEOUT:-32400}"      # 9 h on one backbone, against 4.2 h measured
HEAD_TIMEOUT="${HEAD_TIMEOUT:-10800}"  # 3 h on one head, against 1 h measured
# Two arms are about 14 h of box life at $0.3611/h, which is $5.05. The two
# ceilings below are the ones that stop a stage that hangs on a dead box.
DEADLINE_HOURS="${DEADLINE_HOURS:-22}"
MAX_SPEND="${MAX_SPEND:-11.00}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

mkdir -p "$CF404_RESULTS" "$CF404_PLOTS"
LOG="$CF404_RESULTS/round3b.log"
# Round 3 provisioned the box and wrote its address here. This round reads that
# file and never writes it.
ENVF="${ENVF:-$CF404_RESULTS/round3.env}"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 round3b] $*" | tee -a "$LOG"; }
rsh(){ timeout 180 ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@" 2>/dev/null; }

read -r -a arm_list <<<"$ARMS"
for arm in "${arm_list[@]}"; do cf404_require_arm "$arm" || exit $?; done
cf404_require_stop "$STOP" || exit $?

say "START arms='$ARMS' box=$VAST_LABEL card=$BOX_GPU"
say "  deadline=${DEADLINE_HOURS}h max_spend=\$$MAX_SPEND"
say "credit before: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"

# ---- 1: the box this round reuses -------------------------------------------
INSTANCE=""; HOST=""; PORT=""
[ -s "$ENVF" ] || { say "ABORT: no box address at $ENVF"; exit 2; }
# shellcheck disable=SC1090
. "$ENVF"
[ -n "${HOST:-}" ] && [ -n "${PORT:-}" ] && [ -n "${INSTANCE:-}" ] \
  || { say "ABORT: $ENVF names no instance"; exit 2; }

# ---- the teardown, which every exit path runs -------------------------------
#
# Only the instance THIS round uses is destroyed, and only by the id its own
# `.env` file records. `vastrun-destroy` takes the id and the label together as
# a confirmation token. The vast.ai account is shared with other sessions.
teardown(){
  say "teardown: destroying $INSTANCE ($VAST_LABEL)"
  timeout 300 vastrun-destroy "$INSTANCE" "$VAST_LABEL" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
  # By pid, from the working directory. NEVER a pattern: on 2026-08-19 a
  # pattern for this loop also matched four running eval shards.
  say "teardown: stopping the sync loop ($(cf404_stop_sync_loop "$CF404_SYNC_DIR") loop(s))"
  timeout 120 vastrun-status 2>&1 | sed 's/^/  /' | tee -a "$LOG"
}

# What vast.ai has billed for THIS instance, in dollars. The status table
# carries two dollar columns, the rate and the spend, and the spend is the last
# one. Prints nothing when the row is gone, which a caller reads as "no cap".
box_spent(){
  timeout 120 vastrun-status 2>/dev/null \
    | awk -v id="$INSTANCE" '$1 == id { for (i = 1; i <= NF; i++) if ($i ~ /^\$/) v = $i; print substr(v, 2) }'
}

# ---- the watchdog -----------------------------------------------------------
#
# It holds no other state, so it survives every failure of the stages below. A
# stage that hangs on a dead box would otherwise bill until a person looks.
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

die(){ say "ABORT: $*"; teardown; stop_watchdog; exit 1; }

rsh true || { say "ABORT: instance $INSTANCE at $HOST:$PORT does not answer."
              say "  This round reuses one box and rents none. Nothing was destroyed."
              stop_watchdog; exit 2; }
say "reusing instance $INSTANCE at $HOST:$PORT"

# ---- 2: the card, against the index every lane names ------------------------
#
# Round 3's plan print read `arm a095 gpu=1` on a box with ONE card, at index 0.
# The print came from a dry run that passed no GPUS, so the launcher took its
# own default. The two checks below close both halves: the count is read off
# the driver here, and each launcher's own `cf404_require_gpus` runs ON THE BOX
# through the plan print at the end of stage 3.
CARD="$(rsh "nvidia-smi --query-gpu=index,name,memory.total,compute_mode --format=csv,noheader")"
NCARD="$(printf '%s\n' "$CARD" | grep -c .)"
say "card(s): $(printf '%s' "$CARD" | tr '\n' '|')  count=$NCARD"
[ "$NCARD" -ge 1 ] || die "no card on the box"
[ "$BOX_GPU" -lt "$NCARD" ] \
  || die "every lane names card $BOX_GPU and the box carries $NCARD card(s), indices 0 to $(( NCARD - 1 ))"
case "$CARD" in
  *Default*) ;;
  *) die "the card is not in Default compute mode" ;;
esac

# ---- 3: the payload, and the plan the box itself prints ---------------------
rsh "test -f /root/cf/$STUDY_REL/scripts/study.sh" \
  || die "the box carries no checkout of this study"
# The plan print with the REAL lane list. It runs every guard on the box,
# `cf404_require_gpus` among them, and starts nothing.
say "the box's own plan, at GPUS='$BOX_GPU'"
rsh "cd /root/cf/$STUDY_REL && ARMS='$ARMS' GPUS='$BOX_GPU' CF404_DRY_RUN=1 \
     bash scripts/launch_box.sh" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
rsh "cd /root/cf/$STUDY_REL && ARMS='$ARMS' GPUS='$BOX_GPU' CF404_DRY_RUN=1 \
     bash scripts/launch_box.sh >/dev/null" \
  || die "the box refuses '$ARMS' on card $BOX_GPU"
rsh "cd /root/cf/$STUDY_REL && ARMS='$ARMS' GPUS='$BOX_GPU' CF404_DRY_RUN=1 \
     bash scripts/heads_box.sh >/dev/null" \
  || die "the box refuses a head of '$ARMS' on card $BOX_GPU"

# ---- 4: the sync loop -------------------------------------------------------
#
# The safety net, beside the targeted pulls of stage 8. Every remote run of this
# project carries one for its whole duration.
if [ "$(cf404_sync_loops "$CF404_SYNC_DIR")" -ge 1 ]; then
  say "sync loop already up for $CF404_SYNC_DIR"
else
  REMOTE_HOST="$HOST" REMOTE_PORT="$PORT" SSH_USER=root \
  REMOTE_DIR="/root/cf/$STUDY_REL" \
    bash "$CF404_STUDY/sync/launch_sync.sh" "$LABEL" >>"$LOG" 2>&1
fi
say "sync loop: $(cf404_sync_loops "$CF404_SYNC_DIR") for $CF404_SYNC_DIR," \
    "$(find "$CF404_SYNC_DIR" -type f 2>/dev/null | wc -l) file(s) here"

# ---- the pieces one arm needs ----------------------------------------------
KK=$(( STOP / 1000 ))
box_leg(){ printf '%s/%s/%s/leg_%dk\n' "$CF404_BOX_RUNS" "$1" "$CF404_CELL" "$KK"; }
box_bb(){  # <arm> — the checkpoint on the box, or nothing
  rsh "ls -1 $(box_leg "$1")/$(cf404_run_name "$1")_${KK}k.pth 2>/dev/null | head -1"
}
box_head(){  # <arm> — the head checkpoint size on the box, or 0
  local tag; tag="$(cf404_tag "$1" "$STOP" "$CF404_HEAD_STEPS")"
  rsh "wc -c <$CF404_BOX_RUNS/$1/eval/$tag/qhead_${tag}_s${HEAD_SEED:-20260722}_final.pth 2>/dev/null" \
    | tr -d ' ' | grep -E '^[0-9]+$' || echo 0
}
box_log(){ printf '/root/cf/%s/results/run_%s.log\n' "$STUDY_REL" "$(cf404_run_name "$1")"; }
# The last step rate the trainer printed, as `<sps> sps  ETA <h>h`.
box_rate(){ rsh "grep -hoE '[0-9.]+ sps  ETA [0-9.]+h' $(box_log "$1") 2>/dev/null | tail -1"; }

pull(){  # <remote> <local> <floor>
  local dst="$2"
  [ -f "$dst" ] && [ "$(wc -c <"$dst")" -ge "$3" ] && { say "  have $(basename "$dst")"; return 0; }
  mkdir -p "$(dirname "$dst")"
  SSH_USER=root bash "$SAFE_PULL" "$HOST" "$PORT" "$1" "$dst" "$3" >>"$LOG" 2>&1
  [ -f "$dst" ] || { say "  MISSING $(basename "$dst")"; return 1; }
  say "  $(basename "$dst") $(wc -c <"$dst") B"
}

# ---- 5: the backbone of one arm --------------------------------------------
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
  } >"$CF404_RESULTS/round3b_verify_${arm}.txt" 2>&1
  sed 's/^/  /' "$CF404_RESULTS/round3b_verify_${arm}.txt" | tee -a "$LOG"

  used="$(awk -F', ' 'NR==2{gsub(/[^0-9]/,"",$2); print $2}' \
          <(rsh "nvidia-smi --query-gpu=index,memory.used --format=csv"))"
  apps="$(rsh "nvidia-smi --query-compute-apps=pid --format=csv,noheader" | grep -c .)"
  cols="$(awk '/^depth_cols /{print $2; exit}' "$CF404_RESULTS/round3b_verify_${arm}.txt")"
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

  # -- the climb
  say "$arm: waiting for $STOP steps"
  waited=0
  while [ -z "$(box_bb "$arm")" ]; do
    [ "$waited" -ge "$BB_TIMEOUT" ] && { say "$arm: no backbone after ${waited}s"; return 1; }
    [ $(( waited % 1800 )) -eq 0 ] && say "  $arm $(rsh "grep -hoE '^\[ *[0-9]+\].*sps  ETA [0-9.]+h' $(box_log "$arm") 2>/dev/null | tail -1")"
    sleep "$POLL"; waited=$(( waited + POLL ))
  done
  say "$arm: the backbone is done"
}

# ---- 6: the head of one arm -------------------------------------------------
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

# ---- 7: the artefacts of one arm, into the canonical tree -------------------
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

# ---- 8: the 97-config GIFT-Eval of one arm, on elisa ------------------------
eval_arm(){  # <arm>
  local arm="$1" f
  f="$(cf404_score_file "$arm" "$STOP")"
  if [ -s "$f" ]; then say "$arm: already scored $(tr -d ' \t\r\n' <"$f")"; return 0; fi
  say "$arm: starting the 97-config GIFT-Eval on elisa"
  ARMS="$arm" CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" \
    bash "$HERE/evals_elisa.sh" >>"$CF404_RESULTS/evals_round3b_${arm}.out" 2>&1
  say "$arm: eval rc=$?"
  [ -s "$f" ]
}

# ---- the two arms, one after the other --------------------------------------
scored=0
for arm in "${arm_list[@]}"; do
  say "==== $arm : alpha $(cf404_alpha "$arm") $(cf404_schedule "$arm") seed $(cf404_seed "$arm") ===="
  run_backbone "$arm" || { say "$arm: FAILED at the backbone"; continue; }
  run_head "$arm"     || { say "$arm: FAILED at the head"; continue; }
  pull_arm "$arm"     || say "$arm: WARNING — an artefact did not land"
  if eval_arm "$arm"; then
    scored=$(( scored + 1 ))
    say "SCORE $arm $(tr -d ' \t\r\n' <"$(cf404_score_file "$arm" "$STOP")")"
  else
    say "SCORE $arm MISSING"
  fi
  # The figures and the tables are redrawn after EVERY arm, so the user reads
  # the 0.95 number without waiting for the repeat below it.
  CF404_ROOT="$MAIN_ROOT" bash "$HERE/make_plots.sh" \
    >>"$CF404_RESULTS/make_plots_round3b.out" 2>&1
  say "$arm: DONE — figures and tables redrawn"
done

# ---- 9: the teardown --------------------------------------------------------
#
# Every score that exists, exists now. The box outlived the scores.
say "$scored of ${#arm_list[@]} arm(s) scored — tearing the box down"
teardown
stop_watchdog
say "credit after: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"

# ---- 10: the artefacts and the figures --------------------------------------
say "shard check"
python3 "$HERE/check_shards.py" --root "$MAIN_ROOT" \
  --out "$CF404_RESULTS/shard_check.txt" 2>&1 | tail -20 | tee -a "$LOG"
say "report assets"
CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" CF404_SYNC_DIR="$MAIN_DIR" \
  bash "$HERE/report_assets.sh" >>"$CF404_RESULTS/report_assets_round3b.out" 2>&1
say "plots"
CF404_ROOT="$MAIN_ROOT" bash "$HERE/make_plots.sh" \
  >>"$CF404_RESULTS/make_plots_round3b.out" 2>&1
say "plots rc=$?"
say "ROUND 3B DONE — $scored of ${#arm_list[@]} arm(s) scored"
