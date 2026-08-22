#!/bin/bash
# #404 — the w3_s08 head again, on one rented card, and then its eval.
#
# WHY THIS SCRIPT EXISTS. Round 6 trained three backbones and three heads on
# one box. Two heads landed. The third, `w3_s08`, was at about 26,000 steps of
# 30,000 when the round 6 driver destroyed the box. The driver logged the
# missing checkpoint as a WARNING and tore the box down anyway. So the head is
# lost and the backbone is not: `finish_round6.sh` pulled all three backbones
# first.
#
# THE RULE THIS SCRIPT ADDS. A missing final checkpoint is a STOP, not a
# warning. `destroy_box` refuses to run while the head is not on elisa's disk,
# by name and by size. Every other exit path leaves the box alive and says so.
# A box that stays alive costs money and a person can see it. A box that is
# destroyed early costs the run.
#
# WHAT IT DOES, in order:
#   1. gates      — the backbone is here, the head is not, the score is not
#   2. box        — one card, datacenter, a desktop CPU, under MAX_BID
#   3. payload    — #373's bootstrap, then this study, then the backbone
#   4. sync loop  — 15 min ticks for the whole run (CLAUDE.md)
#   5. head       — heads_box.sh, one arm, one card, detached on the box
#   6. watch      — the step count, the head size and the spend, every 5 min
#   7. teardown   — the head lands here FIRST, then the box goes
#   8. eval       — the 97-config GIFT-Eval on elisa's CPUs
#
# The head is 30,000 steps at head seed 20260722, on the surviving 40,000-step
# backbone, encoder `student`. Those are round 1's values and this script takes
# every one of them from study.sh, so no number here can drift from the arms
# that already scored.
#
# Usage:
#   nohup setsid bash scripts/recover_w3_head.sh \
#     > results/recover_w3_head.out 2>&1 &
#
#   CF404_DRY_RUN=1 bash scripts/recover_w3_head.sh   # print the plan
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
. "$HERE/study.sh"
cf404_use_root "$CF404_SYNC_ROOT"

ARM="${ARM:-w3_s08}"
STOP="${STOP:-$CF404_STOPS}"
HEAD_SEED="${HEAD_SEED:-20260722}"
LABEL="${CF404_BOX_LABEL:-box_a}"
VAST_LABEL="${VAST_LABEL:-cf404-w3-head}"
POLL="${POLL:-300}"
# The head took 1.2 h alone on elisa's card and 1.7 h on a box that also
# carried three trainers. This box carries the head alone. MAX_SPEND is the
# money gate and HEAD_TIMEOUT is the time gate. Neither one destroys the box.
MAX_SPEND="${MAX_SPEND:-1.20}"
HEAD_TIMEOUT="${HEAD_TIMEOUT:-14400}"
MIN_HEAD_BYTES="${MIN_HEAD_BYTES:-400000}"
MIN_BB_BYTES="${MIN_BB_BYTES:-5000000}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.99}"
MAX_BID="${MAX_BID:-0.45}"
VAST_CPU_RE="${VAST_CPU_RE:-7800X3D|9800X3D|7950X3D|9950X3D|7950X|9950X|7900X|9900X|7700X|9700X}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

STUDY_REL="reports/$(basename "$CF404_STUDY")"
mkdir -p "$CF404_RESULTS"
LOG="$CF404_RESULTS/recover_w3_head.log"
ENVF="${ENVF:-$CF404_RESULTS/recover_w3_head.env}"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 recover] $*" | tee -a "$LOG"; }
rsh(){ timeout 180 ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@" 2>/dev/null; }

cf404_require_arm "$ARM" || exit $?
cf404_require_stop "$STOP" || exit $?

TAG="$(cf404_tag "$ARM" "$STOP" "$CF404_HEAD_STEPS")"
BB_HERE="$(cf404_bb_ckpt "$ARM" "$STOP")"
HEAD_HERE="$(cf404_eval_dir "$ARM" "$TAG")/qhead_${TAG}_s${HEAD_SEED}_final.pth"
SCORE_HERE="$(cf404_score_file "$ARM" "$STOP")"
KK="$(cf404_steps_label "$STOP")"
BB_BOX="$CF404_BOX_RUNS/$ARM/$CF404_CELL/leg_${KK}/$(cf404_run_name "$ARM")_${KK}.pth"
HEAD_BOX="$CF404_BOX_RUNS/$ARM/eval/$TAG/qhead_${TAG}_s${HEAD_SEED}_final.pth"
STOPLOG_BOX="$CF404_BOX_RUNS/$ARM/eval/$TAG/stop.log"
SAFE_PULL="$CF404_REPO/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"

if [ -n "${CF404_DRY_RUN:-}" ]; then
  echo "recover arm=$ARM stop=$STOP head=$CF404_HEAD_STEPS seed=$HEAD_SEED"
  echo "  tag=$TAG enc=$CF404_ENC align_w=$(cf404_align_weight "$ARM" 2>/dev/null || echo '?')"
  echo "  bb here  = ${BB_HERE:-MISSING} $([ -n "$BB_HERE" ] && wc -c <"$BB_HERE")"
  echo "  bb box   = $BB_BOX"
  echo "  head here= $HEAD_HERE $([ -f "$HEAD_HERE" ] && wc -c <"$HEAD_HERE" || echo MISSING)"
  echo "  head box = $HEAD_BOX"
  echo "  score    = $SCORE_HERE"
  echo "  box: 1 card, datacenter, reliability >= $MIN_RELIABILITY, <= \$$MAX_BID/h"
  echo "  max_spend=\$$MAX_SPEND head_timeout=${HEAD_TIMEOUT}s label=$VAST_LABEL"
  exit 0
fi

# ---- 1: the gates ------------------------------------------------------------
say "START arm=$ARM tag=$TAG label=$VAST_LABEL max_spend=\$$MAX_SPEND"
[ -n "$BB_HERE" ] && [ -f "$BB_HERE" ] \
  || { say "ABORT: no bb${KK} backbone for $ARM under $(cf404_leg_dir "$ARM" "$STOP")"; exit 2; }
BB_BYTES="$(wc -c <"$BB_HERE")"
[ "$BB_BYTES" -ge "$MIN_BB_BYTES" ] \
  || { say "ABORT: the backbone is $BB_BYTES B, under $MIN_BB_BYTES B"; exit 2; }
say "backbone here: $(basename "$BB_HERE") $BB_BYTES B"
if [ -f "$HEAD_HERE" ]; then
  say "the head is already here — nothing to rent"; exit 0
fi
if [ -s "$SCORE_HERE" ]; then
  say "$ARM is already scored $(tr -d ' \t\r\n' <"$SCORE_HERE") — nothing to rent"; exit 0
fi
say "credit before: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"

# ---- the teardown, which ONE condition opens ---------------------------------
#
# The head has to be on elisa's disk, by name and by size, before the box can
# go. Round 6 destroyed a box on a missing checkpoint. This function cannot.
TORN=0
head_is_here(){
  [ -f "$HEAD_HERE" ] || return 1
  local n; n="$(wc -c <"$HEAD_HERE")"
  [ "${n:-0}" -ge "$MIN_HEAD_BYTES" ]
}
destroy_box(){
  local inst
  [ "$TORN" -eq 1 ] && return 0
  if ! head_is_here; then
    say "REFUSING to destroy: $HEAD_HERE is missing or too small"
    return 1
  fi
  inst="$(awk -F= '$1=="INSTANCE"{print $2}' "$ENVF" 2>/dev/null)"
  [ -n "$inst" ] || { say "teardown: no instance id in $ENVF"; return 1; }
  TORN=1
  say "teardown: the head is here — destroying $inst ($VAST_LABEL)"
  timeout 300 vastrun-destroy "$inst" "$VAST_LABEL" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
  timeout 120 vastrun-status 2>&1 | sed 's/^/  /' | tee -a "$LOG"
}
# The box stays up. This is the loud path, and it names the instance so a
# person can act on one line.
leave_box_alive(){  # <why>
  local inst; inst="$(awk -F= '$1=="INSTANCE"{print $2}' "$ENVF" 2>/dev/null)"
  say "STOP: $1"
  say "STOP: the head is NOT here, so the box STAYS ALIVE. Instance ${inst:-?} at ${HOST:-?}:${PORT:-?}"
  say "STOP: destroy it by hand with: vastrun-destroy ${inst:-?} $VAST_LABEL"
}

box_spent(){
  local inst
  inst="$(awk -F= '$1=="INSTANCE"{print $2}' "$ENVF" 2>/dev/null)"
  [ -n "$inst" ] || return 0
  timeout 120 vastrun-status 2>/dev/null \
    | awk -v id="$inst" '$1 == id { for (i = 1; i <= NF; i++) if ($i ~ /^\$/) v = $i; print substr(v, 2) }'
}

# ---- 2: the box --------------------------------------------------------------
INSTANCE=""; HOST=""; PORT=""
if [ -s "$ENVF" ]; then
  # shellcheck disable=SC1090
  . "$ENVF"
  if [ -n "${HOST:-}" ] && rsh true; then
    say "reusing instance $INSTANCE at $HOST:$PORT"
  else
    say "the box in $ENVF does not answer — provisioning another"
    HOST=""
  fi
fi
if [ -z "${HOST:-}" ]; then
  say "searching — 1 card, datacenter, reliability >= $MIN_RELIABILITY, <= \$$MAX_BID/h"
  out="$(VAST_SEARCH_ARGS="--num-gpus 1 --min-reliability $MIN_RELIABILITY --max-bid $MAX_BID" \
        VAST_SEARCH_LIMIT=40 VAST_CPU_RE="$VAST_CPU_RE" \
        bash "$CF404_PARENT/scripts/provision_box.sh" "$VAST_LABEL" 8 2>>"$LOG")"
  read -r INSTANCE HOST PORT <<<"$(printf '%s\n' "$out" | tail -1)"
  [ -n "${PORT:-}" ] || { say "ABORT: no box"; exit 2; }
  printf 'INSTANCE=%s\nHOST=%s\nPORT=%s\n' "$INSTANCE" "$HOST" "$PORT" >"$ENVF"
  say "instance $INSTANCE at $HOST:$PORT"
fi
say "card: $(rsh "nvidia-smi --query-gpu=index,name,memory.total,compute_mode --format=csv,noheader" | tr '\n' '|')"

# ---- 3: the payload ----------------------------------------------------------
if rsh "test -f /root/cf/$STUDY_REL/scripts/study.sh"; then
  say "the box already carries the study"
else
  say "bootstrap"
  WT="$CF404_REPO" bash "$HERE/bootstrap_box.sh" "$HOST" "$PORT" >>"$LOG" 2>&1 \
    || { leave_box_alive "bootstrap failed, see $LOG"; exit 3; }
  say "bootstrap OK"
fi

# The backbone. It trained on the box that is gone, so it goes UP this time.
if [ "$(rsh "wc -c <$BB_BOX 2>/dev/null" | tr -d ' ')" = "$BB_BYTES" ]; then
  say "the backbone is already on the box"
else
  say "pushing the backbone, $BB_BYTES B"
  rsh "mkdir -p $(dirname "$BB_BOX")" \
    || { leave_box_alive "cannot make the leg directory on the box"; exit 3; }
  timeout 600 scp -q "${SSH_OPTS[@]}" -P "$PORT" "$BB_HERE" "root@$HOST:$BB_BOX.tmp" \
    || { leave_box_alive "the backbone did not go up"; exit 3; }
  rsh "mv $BB_BOX.tmp $BB_BOX"
  got="$(rsh "wc -c <$BB_BOX 2>/dev/null" | tr -d ' ')"
  [ "$got" = "$BB_BYTES" ] \
    || { leave_box_alive "the backbone on the box is $got B, not $BB_BYTES B"; exit 3; }
  say "the backbone is on the box, $got B"
fi

# The box has to name the arm and the head the same way this machine does.
rsh "cd /root/cf/$STUDY_REL && ARMS='$ARM' GPUS=0 CF404_DRY_RUN=1 bash scripts/heads_box.sh" \
  | sed 's/^/  /' | tee -a "$LOG"
rsh "cd /root/cf/$STUDY_REL && ARMS='$ARM' GPUS=0 CF404_DRY_RUN=1 bash scripts/heads_box.sh" \
  >/dev/null 2>&1 \
  || { leave_box_alive "the box refuses arm $ARM — its arms table is stale"; exit 3; }

# ---- 4: the sync loop --------------------------------------------------------
REMOTE_HOST="$HOST" REMOTE_PORT="$PORT" SSH_USER=root \
REMOTE_DIR="/root/cf/$STUDY_REL" \
  bash "$CF404_STUDY/sync/launch_sync.sh" "$LABEL" >>"$LOG" 2>&1
say "sync loop: $(cf404_sync_loops "$CF404_SYNC_DIR") for $CF404_SYNC_DIR"

# ---- 5: the head -------------------------------------------------------------
if [ -n "$(rsh "pgrep -f 'train_forecasting_head[.]py' | head -1")" ]; then
  say "a head trainer already runs on the box"
else
  say "starting the head: $CF404_HEAD_STEPS steps, seed $HEAD_SEED, card 0"
  rsh "cd /root/cf/$STUDY_REL && mkdir -p results && \
       ARMS='$ARM' GPUS=0 nohup setsid bash scripts/heads_box.sh \
       > results/heads_box_${ARM}.out 2>&1 < /dev/null & echo started"
  sleep 90
fi
say "box process: $(rsh 'pgrep -af "train_forecasting_head[.]py" | head -2' | cut -c1-140)"
say "box card: $(rsh "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader")"

# ---- 6: the watch ------------------------------------------------------------
waited=0
while [ "$waited" -lt "$HEAD_TIMEOUT" ]; do
  size="$(rsh "wc -c <$HEAD_BOX 2>/dev/null" | tr -d ' ' | grep -E '^[0-9]+$' || echo 0)"
  if [ "${size:-0}" -ge "$MIN_HEAD_BYTES" ]; then
    say "the head is on the box: $size B"
    break
  fi
  spent="$(box_spent)"
  step="$(rsh "grep -aoE 'step [0-9]+' $STOPLOG_BOX 2>/dev/null | tail -1")"
  alive="$(rsh 'pgrep -c -f "train_forecasting_head[.]py"' | tr -d ' ')"
  say "  head=${size:-0} B ${step:-<no step line>} trainer=${alive:-0} spent=\$${spent:-?} waited=$(( waited / 60 )) min"
  if [ -n "${spent:-}" ] && awk -v s="$spent" -v m="$MAX_SPEND" 'BEGIN{exit !(s+0 >= m+0)}'; then
    leave_box_alive "the box has spent \$$spent of \$$MAX_SPEND and the head is not here"
    exit 4
  fi
  if [ "${alive:-0}" -eq 0 ] && [ "$waited" -gt 600 ]; then
    say "the trainer is not in the box process table any more"
    say "$(rsh "tail -5 $STOPLOG_BOX 2>/dev/null")"
    leave_box_alive "the head trainer died before the final checkpoint"
    exit 5
  fi
  sleep "$POLL"; waited=$(( waited + POLL ))
done
if [ "$waited" -ge "$HEAD_TIMEOUT" ]; then
  leave_box_alive "the head did not finish inside $(( HEAD_TIMEOUT / 3600 )) h"
  exit 6
fi

# ---- 7: the head comes here, and only then does the box go -------------------
#
# The sync loop is stopped FIRST, by pid, so that two writers cannot land on
# one `.tmp` name. `cf404_stop_sync_loop` matches on the working directory of
# the loop, never on a pattern: elisa carries other sessions' evals.
say "stopping the sync loop ($(cf404_stop_sync_loop "$CF404_SYNC_DIR") loop(s))"
mkdir -p "$(dirname "$HEAD_HERE")"
for try in 1 2 3; do
  SSH_USER=root bash "$SAFE_PULL" "$HOST" "$PORT" "$HEAD_BOX" "$HEAD_HERE" \
    "$MIN_HEAD_BYTES" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
  head_is_here && break
  say "pull attempt $try did not land the head"
  sleep 30
done
# The head's own losses CSV and log, which the report reads. Small files, so
# they take a small floor, and a miss on them is not a stop.
for f in "qhead_${TAG}_s${HEAD_SEED}_losses.csv" "stop.log"; do
  SSH_USER=root bash "$SAFE_PULL" "$HOST" "$PORT" \
    "$CF404_BOX_RUNS/$ARM/eval/$TAG/$f" "$(dirname "$HEAD_HERE")/$f" 200 \
    >>"$LOG" 2>&1
done
if ! head_is_here; then
  leave_box_alive "the head is on the box and three pulls did not bring it here"
  exit 7
fi
say "the head is HERE: $HEAD_HERE $(wc -c <"$HEAD_HERE") B"
destroy_box || { say "teardown refused — see the line above"; exit 8; }
say "credit after: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"

# ---- 8: the eval -------------------------------------------------------------
say "starting the 97-config GIFT-Eval for $ARM on elisa's CPUs"
ARMS="$ARM" nohup setsid bash "$HERE/evals_elisa.sh" \
  >"$CF404_RESULTS/evals_elisa_${ARM}.out" 2>&1 < /dev/null &
sleep 30
say "eval process: $(pgrep -af "head_eval[.]sh $ARM" | head -1 | cut -c1-120)"
say "DONE — the head is here and its eval runs"
